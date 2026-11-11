"""Bounded scan-image registration for display overlays; never moves hardware."""
from dataclasses import dataclass
import time
import cv2
import numpy as np


@dataclass(frozen=True)
class ImageFitConfig:
    max_dimension: int = 1280
    max_features: int = 4500
    ratio: float = .80
    tolerance_px: float = 4.
    minimum_matches: int = 12
    minimum_inliers: int = 9
    minimum_coverage: float = .015
    phase_fallback: bool = True
    expected_native_scale: float = 1.


def _preview(scan,maximum):
    pixels=np.asarray(scan['pixels'])
    if pixels.ndim==3:pixels=cv2.cvtColor(pixels[:,:,:3],cv2.COLOR_RGB2GRAY)
    if pixels.ndim!=2 or min(pixels.shape)<16 or not np.isfinite(pixels).all():raise ValueError('A finite grayscale or RGB scan image is required.')
    if pixels.dtype!=np.uint8:
        low,high=np.percentile(pixels,[.5,99.5]);pixels=np.clip((pixels-low)*255/max(high-low,1e-6),0,255).astype(np.uint8)
    factor=min(1.,maximum/max(pixels.shape));shape=(max(16,round(pixels.shape[1]*factor)),max(16,round(pixels.shape[0]*factor)))
    gray=cv2.resize(pixels,shape,interpolation=cv2.INTER_AREA) if shape!=pixels.shape[::-1] else np.ascontiguousarray(pixels)
    scale=np.array([float(scan['width'])/shape[0],float(scan['height'])/shape[1]])
    if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Positive native scan dimensions are required.')
    native=np.eye(3);native[:2,:2]=np.diag(scale);native[:2,2]=(scale-1)/2
    return gray,native


def _features(gray,config):
    mask=cv2.erode((gray>2).astype(np.uint8)*255,np.ones((7,7),np.uint8))
    enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
    return cv2.SIFT_create(nfeatures=config.max_features,contrastThreshold=.02).detectAndCompute(enhanced,mask)


def _coverage(points,shape):
    return float(cv2.contourArea(cv2.convexHull(np.float32(points)))/(shape[0]*shape[1])) if len(points)>=3 else 0.


def _sift(a,b,na,nb,c):
    ka,da=_features(a,c);kb,db=_features(b,c);metrics=dict(source_features=len(ka),reference_features=len(kb))
    if da is None or db is None or min(len(ka),len(kb))<3:return None,metrics,'Fewer than three image features are available.'
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64))
    def ratio(x,y):return {p[0].queryIdx:p[0].trainIdx for p in matcher.knnMatch(x,y,k=2) if len(p)==2 and p[0].distance<c.ratio*p[1].distance}
    forward,backward=ratio(da,db),ratio(db,da);pairs=[(i,j) for i,j in forward.items() if backward.get(j)==i]
    metrics['matches']=len(pairs)
    if len(pairs)<3:return None,metrics,'Fewer than three reciprocal image-feature matches are available.'
    source=np.float32([ka[i].pt for i,j in pairs]);target=np.float32([kb[j].pt for i,j in pairs])
    order=np.random.default_rng(0).permutation(len(pairs));held=order[::4];train=np.delete(order,np.arange(0,len(order),4))
    fit,_=cv2.estimateAffinePartial2D(source[train],target[train],method=cv2.RANSAC,ransacReprojThreshold=c.tolerance_px*.75,maxIters=3000,confidence=.995,refineIters=10)
    if fit is None:return None,metrics,'No consistent image similarity transform.'
    matrix=np.eye(3);matrix[:2]=fit;method='image_sift_similarity'
    # Prefer fixed native scale when it explains the training observations as well.
    expected=np.diag([c.expected_native_scale,c.expected_native_scale,1.]);expected[:2,2]=(c.expected_native_scale-1)/2
    fixed=np.linalg.inv(nb)@expected@na;translation=target[train]-(source[train]@fixed[:2,:2].T+fixed[:2,2]);delta=np.median(translation,axis=0)
    fixed[:2,2]+=delta
    errors=lambda m,indices:np.linalg.norm(source[indices]@m[:2,:2].T+m[:2,2]-target[indices],axis=1)
    if np.sum(errors(fixed,train)<=c.tolerance_px)>=np.sum(errors(matrix,train)<=c.tolerance_px)*.95:
        matrix=fixed;method='image_sift_translation'
    native_candidate=nb@matrix@np.linalg.inv(na)
    if np.isfinite(native_candidate).all() and np.linalg.det(native_candidate[:2,:2])!=0:
        metrics['candidate_scan_to_reference_native']=native_candidate.tolist()
        metrics['candidate_method']=method
    residual=errors(matrix,np.arange(len(source)));good=residual<=c.tolerance_px;heldgood=good[held]
    metrics.update(inliers=int(good.sum()),inlier_fraction=float(good.mean()),heldout_matches=len(held),heldout_inliers=int(heldgood.sum()),
        heldout_inlier_fraction=float(heldgood.mean()),source_coverage=_coverage(source[good],a.shape),reference_coverage=_coverage(target[good],b.shape),
        median_error_overview_px=float(np.median(residual[good])) if good.any() else None)
    if len(pairs)<c.minimum_matches:return None,metrics,'Image candidate has fewer than the preferred reciprocal feature matches.'
    if good.sum()<c.minimum_inliers or good.mean()<.3 or heldgood.sum()<3 or heldgood.mean()<.35:return None,metrics,'Transform failed held-out image-feature checks.'
    if min(metrics['source_coverage'],metrics['reference_coverage'])<c.minimum_coverage:return None,metrics,'Feature matches cover too little of the images.'
    return (nb@matrix@np.linalg.inv(na),method),metrics,''


def _ncc(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64);a-=a.mean();b-=b.mean();den=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/den) if den>1e-9 else 0.


def _phase(a,b,na,nb,c):
    # Equal native sampling, not equal image dimensions. This is translation only.
    native_scale=c.expected_native_scale
    sample=max(float(na[0,0])*native_scale,float(na[1,1])*native_scale,float(nb[0,0]),float(nb[1,1]))
    def resample(gray,native,scale):
        factors=np.diag(native)[:2]*scale/sample
        size=np.maximum(16,np.rint(np.array(gray.shape[::-1])*factors).astype(int))
        transform=np.column_stack((np.diag(factors),(factors-1)/2))
        return cv2.warpAffine(gray,transform.astype(np.float32),tuple(size),flags=cv2.INTER_LINEAR)
    aa,bb=resample(a,na,native_scale),resample(b,nb,1.);height=max(aa.shape[0],bb.shape[0]);width=max(aa.shape[1],bb.shape[1])
    pa,pb=np.zeros((height,width),np.float32),np.zeros((height,width),np.float32)
    ma,mb=np.zeros((height,width),np.uint8),np.zeros((height,width),np.uint8)
    for gray,canvas,mask in [(aa,pa,ma),(bb,pb,mb)]:
        signal=gray.astype(np.float32);signal=cv2.GaussianBlur(signal,(0,0),1)-cv2.GaussianBlur(signal,(0,0),8)
        valid=cv2.erode((gray>2).astype(np.uint8),np.ones((9,9),np.uint8))
        canvas[:gray.shape[0],:gray.shape[1]]=signal*valid;mask[:gray.shape[0],:gray.shape[1]]=valid
    if min(float(pa.std()),float(pb.std()))<.1:return None,dict(phase_reason='Insufficient texture for phase correlation.')
    shift,response=cv2.phaseCorrelate(pa,pb);matrix=np.float32([[1,0,-shift[0]],[0,1,-shift[1]]])
    aligned=cv2.warpAffine(pb,matrix,(width,height));valid=(ma>0)&(cv2.warpAffine(mb,matrix,(width,height),flags=cv2.INTER_NEAREST)>0)
    scores=[];centers=[]
    for gy in range(4):
        for gx in range(4):
            region=np.zeros(valid.shape,bool);region[gy*height//4:(gy+1)*height//4,gx*width//4:(gx+1)*width//4]=True;region&=valid
            if region.sum()>=400 and pa[region].std()>=.5 and aligned[region].std()>=.5:
                score=_ncc(pa[region],aligned[region]);scores.append(score)
                if score>=.40:centers.append([gx,gy])
    score=_ncc(pa[valid],aligned[valid]) if valid.sum()>=2000 else 0.
    metrics=dict(phase_response=float(response),phase_ncc=score,phase_overlap_pixels=int(valid.sum()),phase_supported_regions=len(centers),phase_region_ncc=scores)
    spread=len(centers)>=3 and np.ptp(np.asarray(centers),axis=0).min()>=1
    if response<.12 or score<.45 or not spread:return None,metrics
    # Compare fixed nearby alternatives on the same overlap to reject broad ridges.
    margins=[]
    for dx,dy in [(24,0),(-24,0),(0,24),(0,-24)]:
        alternative=np.float32([[1,0,-shift[0]-dx],[0,1,-shift[1]-dy]])
        test=cv2.warpAffine(pb,alternative,(width,height))
        shared=valid&(cv2.warpAffine(mb,alternative,(width,height),flags=cv2.INTER_NEAREST)>0)
        if shared.sum()<2000:return None,dict(metrics,phase_reason='Insufficient common support for translation alternatives.')
        margins.append(_ncc(pa[shared],aligned[shared])-_ncc(pa[shared],test[shared]))
    metrics['phase_peak_margin']=min(margins)
    metrics['phase_alternative_margins']=margins
    if metrics['phase_peak_margin']<.10:return None,metrics
    native=np.diag([native_scale,native_scale,1.]);native[:2,2]=np.array(shift)*sample+(native_scale-1)/2
    return (native,'image_phase_translation'),metrics


def register_scan_images(scan,reference,*,config=None):
    started=time.perf_counter();c=config or ImageFitConfig()
    result=dict(version=1,accepted=False,reason='',method='bounded_scan_image_fit',metrics={},
        image_validation=dict(accepted=False,method='bounded_scan_image_validation',raw_frames_used=False,hardware_coordinates_changed=False))
    try:
        if not 256<=c.max_dimension<=1600 or not .2<=c.expected_native_scale<=5:raise ValueError('A bounded256–1600pixel preview and positive supported native scale are required.')
        a,na=_preview(scan,c.max_dimension);b,nb=_preview(reference,c.max_dimension)
        result['metrics'].update(source_preview_size=list(a.shape[::-1]),reference_preview_size=list(b.shape[::-1]),max_dimension=c.max_dimension,expected_native_scale=c.expected_native_scale)
        candidate,metrics,reason=_sift(a,b,na,nb,c);result['metrics'].update(metrics)
        for key in ('candidate_scan_to_reference_native','candidate_method'):
            if key in result['metrics']:result[key]=result['metrics'].pop(key)
        if candidate is None and c.phase_fallback:
            candidate,phase_metrics=_phase(a,b,na,nb,c);result['metrics'].update(phase_metrics)
        if candidate is None:result['reason']=reason or 'Image alignment failed independent texture and spatial checks.'
        else:
            matrix,method=candidate;scales=np.linalg.svd(matrix[:2,:2],compute_uv=False)
            if np.isfinite(matrix).all() and np.linalg.det(matrix[:2,:2])!=0:
                result.update(candidate_scan_to_reference_native=matrix.tolist(),candidate_method=method)
            if not np.isfinite(matrix).all() or np.linalg.det(matrix[:2,:2])<=0 or scales.min()<.2 or scales.max()>5:
                result['reason']='The fitted native image scale is outside the supported range.'
            else:
                result.update(accepted=True,method=method,reason='Scan-image alignment passed bounded image checks.',scan_to_reference_native=matrix.tolist())
                result['metrics'].update(native_scale_xy=np.linalg.norm(matrix[:2,:2],axis=0).tolist(),native_singular_values=scales.tolist(),
                    rotation_degrees=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))))
                result['image_validation']['accepted']=True
    except (ValueError,TypeError,KeyError,cv2.error) as error:result['reason']=f'Image fitting unavailable: {error}'
    result['metrics']['elapsed_seconds']=time.perf_counter()-started
    return result


def _dense_inputs(scan, reference, expected_native_scale, max_dimension):
    """Common optical sampling with exact native pixel-center transforms."""
    scale=float(expected_native_scale)
    if not np.isfinite(scale) or scale<=0 or not 128<=max_dimension<=1280:
        raise ValueError('Dense scoring requires positive optical scale and bounded previews.')
    sample=max(1.,max(float(scan['width'])*scale,float(scan['height'])*scale,
               float(reference['width']),float(reference['height']))/max_dimension)
    common=np.diag([sample,sample,1.]);common[:2,2]=(sample-1)/2
    optical=np.diag([scale,scale,1.]);optical[:2,2]=(scale-1)/2
    prepared=[]
    for image,physical in ((scan,optical),(reference,np.eye(3))):
        gray,native=_preview(image,max_dimension)
        shape=np.maximum(16,np.rint(np.array([image['width'],image['height']])*physical[0,0]/sample).astype(int))
        transform=np.linalg.inv(common)@physical@native
        gray=cv2.warpAffine(gray,transform[:2],tuple(shape),flags=cv2.INTER_LINEAR)
        mask=cv2.erode((gray>3).astype(np.float32),np.ones((5,5),np.uint8))
        values=gray.astype(np.float32)
        signal=(cv2.GaussianBlur(values,(0,0),1)-cv2.GaussianBlur(values,(0,0),8))*mask
        prepared.append((signal,mask,np.linalg.inv(physical)@common))
    return prepared


def _dense_score(prepared, matrix):
    (a,ma,na),(b,mb,nb)=prepared
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
        return dict(defined=False,reason='No finite invertible image transform.')
    transform=np.linalg.inv(nb)@matrix@na
    warped=cv2.warpAffine(a,transform[:2],b.shape[::-1])
    mask=cv2.warpAffine(ma,transform[:2],b.shape[::-1])>.99
    valid=mask&(mb>0)
    pixels=int(valid.sum())
    # Account for the transformed source area, so different scales share one score.
    area=min(float(ma.sum())*abs(np.linalg.det(transform[:2,:2])),float(mb.sum()))
    overlap=min(1.,pixels/max(area,1.))
    if pixels<64 or min(float(warped[valid].std()),float(b[valid].std()))<1e-5:
        return dict(defined=False,reason='No overlapping textured image support.',dense_overlap_pixels=pixels)
    ncc=_ncc(warped[valid],b[valid])
    return dict(defined=True,dense_ncc=ncc,dense_overlap_fraction=overlap,
                dense_overlap_pixels=pixels,dense_ranking_score=ncc*np.sqrt(overlap))


def score_image_transform(scan, reference, matrix, max_dimension=600, expected_native_scale=1.):
    """Rank image proposals on identical masked bandpass pixels; no quality veto."""
    return _dense_score(_dense_inputs(scan,reference,expected_native_scale,max_dimension),matrix)


def propose_dense_translation(scan, reference, expected_native_scale, max_dimension=600):
    """Bounded all-translation FFT search at the known optical scale, without rotation."""
    from scipy.signal import fftconvolve
    started=time.perf_counter()
    prepared=_dense_inputs(scan,reference,expected_native_scale,max_dimension)
    (a,ma,na),(b,mb,nb)=prepared
    correlate=lambda x,y:fftconvolve(x,y[::-1,::-1],mode='full')
    n=np.maximum(correlate(mb,ma),0.);safe=np.maximum(n,1.)
    sa,sb=correlate(mb,a),correlate(b,ma)
    va=np.maximum(correlate(mb,a*a)-sa*sa/safe,0.)
    vb=np.maximum(correlate(b*b,ma)-sb*sb/safe,0.)
    covariance=correlate(b,a)-sa*sb/safe
    variance=va*vb;support=min(float(ma.sum()),float(mb.sum()))
    valid=(n>=max(64.,support*.25))&(variance>1e-5)
    if not valid.any():
        return dict(available=False,reason='No overlapping textured image support.',metrics={})
    ncc=np.clip(covariance/np.sqrt(np.maximum(variance,1e-5)),-1.,1.)
    ranks=np.where(valid,ncc*np.sqrt(np.minimum(n/max(support,1.),1.)),-np.inf)
    y,x=np.unravel_index(np.argmax(ranks),ranks.shape)
    translation=np.eye(3);translation[:2,2]=[x-a.shape[1]+1,y-a.shape[0]+1]
    matrix=nb@translation@np.linalg.inv(na)
    metrics=_dense_score(prepared,matrix)
    metrics.update(dense_elapsed_seconds=time.perf_counter()-started,dense_max_dimension=max_dimension,
                   expected_native_scale=float(expected_native_scale))
    return dict(available=metrics['defined'],method='image_dense_translation',
                scan_to_reference_native=matrix.tolist(),metrics=metrics)
