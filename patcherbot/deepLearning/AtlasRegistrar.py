"""Automatic Allen CCFv3 registration for a stitched coronal mouse scan.

Requires Python 3.10+, PyTorch, h5py, NumPy, SciPy, Pillow, OpenCV, and SimpleITK.
Install the CUDA build of PyTorch to run DeepSlice inference on an NVIDIA GPU.
The same module runs on CPU with device="cpu". Run registration once per scan.

    registrar = AtlasRegistrar(device="cuda:0")  # or device="auto" (default)
    result = registrar.register("scan.png")
    print(registrar.locate(result, 400, 300))
    result.save_quicknii("registration.json", "scan.png")

The five atlas/model files are downloaded to cache_dir on first use. Supply
download_assets=False for offline use with an already populated cache.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from urllib.request import urlopen
import gzip
import hashlib
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import SimpleITK as sitk


STEP = 3
PADDING = 100
ATLAS_SHAPE = (528, 320, 456)
ATLAS_TARGET = "ABA_Mouse_CCFv3_2017_25um.cutlas"
ATLAS_URL = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/"
WEIGHTS_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/deepslice/weights/"
ASSETS = {
    "annotation_25.nrrd": (
        ATLAS_URL + "annotation/ccf_2017/annotation_25.nrrd",
        "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42",
    ),
    "ara_nissl_25.nrrd": (
        ATLAS_URL + "ara_nissl/ara_nissl_25.nrrd",
        "16596c2c03935b7ff58f1fed41bac2aa21423f39d2c17d3fc400387f054b3bdf",
    ),
    "Allen_Mixed_Best.h5": (
        WEIGHTS_URL + "Allen_Mixed_Best.h5",
        "da34a5bca0245314a68daff30c1446758c0851b21370ef9797d5288018b08717",
    ),
    "Synthetic_data_final.hdf5": (
        WEIGHTS_URL + "Synthetic_data_final.hdf5",
        "b85b7325158d117b2ac7559495c0b50dfcf3545aa29281a345f9ef1e33542e42",
    ),
    "xception_weights_tf_dim_ordering_tf_kernels.h5": (
        WEIGHTS_URL + "xception_weights_tf_dim_ordering_tf_kernels.h5",
        "c5bf1c05b020c4177164039b854c3ef92b16b73384734647c1f0ad5cf79f6975",
    ),
}
WEIGHTS = {"primary": "Allen_Mixed_Best.h5", "secondary": "Synthetic_data_final.hdf5"}
XCEPTION_WEIGHTS = "xception_weights_tf_dim_ordering_tf_kernels.h5"

def elliptical_kernel(radius):
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def largest_component(mask, interior_only):
    height, width = mask.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    x, y, w, h, area = stats[1:].T
    eligible = area >= 0.02 * height * width
    if interior_only:
        eligible &= (x > 0) & (y > 0) & (x + w < width) & (y + h < height)
    candidates = np.flatnonzero(eligible)
    if not candidates.size:
        raise ValueError("No isolated tissue component found in the scan.")
    winner = candidates[np.argmax(area[candidates])] + 1
    return ndi.binary_fill_holes(labels == winner).astype(np.uint8)


def detect_tissue(image, max_side=1800):
    """Return a uint8 0/1 tissue mask in original image coordinates.

    The input is a grayscale or OpenCV BGR uint8 image. A global intensity
    threshold and morphology provide GrabCut with an initial tissue mask.
    """
    height, width = image.shape[:2]
    work_scale = min(1.0, max_side / max(height, width))
    work = cv2.resize(image, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    side = min(gray.shape)
    smooth = cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.0028 * side))
    _, dark = cv2.threshold(smooth, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    opening = elliptical_kernel(max(1, round(0.008 * side)))
    coarse = largest_component(cv2.morphologyEx(dark, cv2.MORPH_OPEN, opening), interior_only=True)

    band = elliptical_kernel(max(1, round(0.02 * side)))
    definite_tissue = cv2.erode(coarse, band) > 0
    possible_tissue = cv2.dilate(coarse, band) > 0
    seeds = np.where(coarse, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    seeds[definite_tissue] = cv2.GC_FGD
    seeds[~possible_tissue] = cv2.GC_BGD
    smooth_bgr = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
    cv2.setRNGSeed(3)
    cv2.grabCut(smooth_bgr, seeds, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    mask = largest_component((seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD), interior_only=True)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, elliptical_kernel(max(1, round(0.004 * side))))
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def read_nrrd(path):
    blob = Path(path).read_bytes()
    sep = b"\n\n"
    pos = blob.find(sep)
    if pos < 0:
        sep = b"\r\n\r\n"
        pos = blob.find(sep)
    if pos < 0:
        raise ValueError("NRRD header separator not found")
    header = {}
    for line in blob[:pos].decode("ascii").splitlines():
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()
    types = {"unsigned int": "u4", "uint": "u4", "uint32": "u4",
             "unsigned short": "u2", "ushort": "u2", "uint16": "u2",
             "unsigned char": "u1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "float": "f4", "double": "f8"}
    dt = np.dtype(types[header["type"]])
    if dt.itemsize > 1:
        dt = dt.newbyteorder("<" if header.get("endian", "little") == "little" else ">")
    data = blob[pos + len(sep):]
    if header["encoding"] in ("gzip", "gz"):
        data = gzip.decompress(data)
    elif header["encoding"] != "raw":
        raise ValueError("Unsupported NRRD encoding: " + header["encoding"])
    shape = tuple(map(int, header["sizes"].split()))
    arr = np.frombuffer(data, dtype=dt).reshape(shape, order="F")
    return arr, header


def quicknii_to_ccf25(q):
    """QuickNII RAS voxel coords to Allen PIR 25um voxel indices."""
    return np.stack((527.0 - q[1], 319.0 - q[2], q[0]))


def normalize_template(arr):
    vals = arr[arr > 0]
    if vals.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(vals, (1, 99))
    return (np.clip((arr - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def atlas_view(section, annotation, nissl, step=STEP, padding=PADDING):
    origin = (-padding*step, -padding*step)
    xs = np.arange(origin[0], section['width'] + padding*step, step)
    ys = np.arange(origin[1], section['height'] + padding*step, step)
    x, y = np.meshgrid(xs, ys)
    o, u, v = np.array(section['anchoring']).reshape(3, 3)
    coords = quicknii_to_ccf25(o[:, None, None] + u[:, None, None]*x/section['width'] + v[:, None, None]*y/section['height'])
    labels = ndi.map_coordinates(annotation, coords, order=0, mode='constant', cval=0, prefilter=False)
    template = ndi.map_coordinates(nissl, coords, order=1, mode='constant', cval=0, prefilter=False)
    return labels, normalize_template(template), origin


def itk_image(array, origin=(0, 0), spacing=STEP):
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(tuple(float(x) for x in origin))
    image.SetSpacing((float(spacing), float(spacing)))
    return image


def bbox(mask, origin=(0, 0)):
    rows, cols = np.nonzero(mask)
    lo = np.array([cols.min(), rows.min()])*STEP + origin
    hi = np.array([cols.max(), rows.max()])*STEP + origin
    return (lo+hi)/2, hi-lo


def distance_image(mask):
    dist = ndi.distance_transform_edt(~mask)-ndi.distance_transform_edt(mask)
    return np.clip(dist*STEP, -100, 100).astype(np.float32)


def affine_parts(transform):
    matrix = np.asarray(transform.GetMatrix()).reshape(2,2)
    center = np.asarray(transform.GetCenter())
    offset = center + np.asarray(transform.GetTranslation()) - matrix @ center
    return matrix, offset


def evaluate(tissue, tissue_mask, atlas, atlas_mask, transform, origin):
    reference = itk_image(tissue.astype(np.float32))
    moved_mask = sitk.GetArrayFromImage(sitk.Resample(itk_image(atlas_mask.astype(np.uint8), origin), reference, transform, sitk.sitkNearestNeighbor, 0)) > 0
    overlap = tissue_mask & moved_mask
    dice = 2*overlap.sum()/(tissue_mask.sum()+moved_mask.sum())
    values = sitk.GetArrayFromImage(sitk.Resample(itk_image(atlas.astype(np.float32), origin), reference, transform))
    core = ndi.binary_erosion(overlap, iterations=6)
    histogram, _, _ = np.histogram2d(tissue[core], values[core], bins=24, range=[[0, 256], [0, 256]])
    joint = histogram/max(histogram.sum(), 1)
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    independent = px[:, None]*py[None, :]
    nonzero = joint > 0
    mi = float(np.sum(joint[nonzero]*np.log(joint[nonzero]/independent[nonzero])))
    return {'dice': float(dice), 'interior_mutual_information': mi}


def fit(tissue, tissue_mask, atlas, atlas_mask, origin, threads):
    center, size = bbox(tissue_mask)
    atlas_center, atlas_size = bbox(atlas_mask, origin)
    start = sitk.AffineTransform(2)
    start.SetCenter(center.tolist())
    start.SetMatrix(np.diag(atlas_size/size).ravel().tolist())
    start.SetTranslation((atlas_center-center).tolist())
    fixed_distance = distance_image(tissue_mask)
    moving_distance = distance_image(atlas_mask)
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(threads)
    registration.SetMetricAsMeanSquares()
    registration.SetMetricFixedMask(itk_image((np.abs(fixed_distance)<65).astype(np.uint8)))
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=2., minStep=.02, numberOfIterations=150)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(start, inPlace=True)
    registration.Execute(itk_image(fixed_distance), itk_image(moving_distance, origin))
    shape = evaluate(tissue, tissue_mask, atlas, atlas_mask, start, origin)

    adjusted = sitk.AffineTransform(start)
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(threads)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=24)
    registration.SetMetricFixedMask(itk_image(ndi.binary_erosion(tissue_mask, iterations=8).astype(np.uint8)))
    registration.SetMetricMovingMask(itk_image(ndi.binary_erosion(atlas_mask, iterations=5).astype(np.uint8), origin))
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(.7, seed=37)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1., minStep=.04, numberOfIterations=120)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([3, 0])
    registration.SetInitialTransform(adjusted, inPlace=True)
    registration.Execute(itk_image(tissue.astype(np.float32)), itk_image(atlas.astype(np.float32), origin))
    intensity = evaluate(tissue, tissue_mask, atlas, atlas_mask, adjusted, origin)
    # Require the intensity step to preserve gross tissue agreement.
    use_intensity = intensity['dice'] >= shape['dice']-.02 and intensity['interior_mutual_information'] > shape['interior_mutual_information']
    selected, metric = (adjusted, intensity) if use_intensity else (start, shape)
    matrix, offset = affine_parts(selected)
    return matrix, offset, {'unrefined': evaluate(tissue, tissue_mask, atlas, atlas_mask, sitk.AffineTransform(2), origin),
                            'shape': shape, 'intensity': intensity,
                            'selected': 'intensity' if use_intensity else 'shape',
                            'score': metric['dice'] + .2*metric['interior_mutual_information']}


@dataclass(frozen=True)
class AtlasRegistration:
    """A fit expressed in the pixel coordinates of the original stitched scan."""

    width: int
    height: int
    anchoring: tuple[float, ...]
    rotation_degrees: int
    deepslice_model: str
    silhouette_dice_before: float
    silhouette_dice: float
    candidate_scores: tuple[dict, ...]

    def pixel_to_ccf_um(self, x: float, y: float) -> tuple[float, float, float]:
        """Map a scan pixel to Allen CCF (AP, DV, ML) in micrometers."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f'Pixel ({x}, {y}) is outside the {self.width} × {self.height} scan.')
        origin, u, v = np.asarray(self.anchoring).reshape(3, 3)
        q = origin + (x/self.width)*u + (y/self.height)*v
        return tuple(float(value) for value in quicknii_to_ccf25(q)*25)

    def save_quicknii(self, path: str | Path, image_filename: str) -> None:
        """Write a QuickNII mapping for the supplied original scan filename."""
        data = {
            'name': 'Automatic Allen CCFv3 registration',
            'target': ATLAS_TARGET,
            'aligner': 'DeepSlice 1.2.8 (PyTorch) + SimpleITK',
            'slices': [{'filename': Path(image_filename).name, 'width': self.width,
                        'height': self.height, 'anchoring': list(self.anchoring), 'nr': 1, 'markers': []}],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def _validate_registration_archive(result, angle, transform, rgba, bounds, source_size):
    transform = np.asarray(transform, dtype=float)
    rgba = np.asarray(rgba)
    source_size = tuple(source_size)
    if (len(source_size) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in source_size)
            or not isinstance(result.width, int) or not isinstance(result.height, int)
            or min(result.width, result.height) < 1):
        raise ValueError('Invalid saved scan dimensions.')
    if not np.isfinite(angle) or not -180 <= angle <= 180:
        raise ValueError('Invalid saved rotation angle.')
    if (transform.shape != (3, 3) or not np.isfinite(transform).all()
            or not np.allclose(transform[2], [0, 0, 1])
            or abs(np.linalg.det(transform[:2, :2])) < 1e-12):
        raise ValueError('Invalid saved scan-to-upright transform.')
    if np.asarray(result.anchoring).shape != (9,) or not np.isfinite(result.anchoring).all():
        raise ValueError('Invalid saved atlas anchoring.')
    if rgba.ndim != 3 or rgba.shape[2] != 4 or min(rgba.shape[:2]) < 1 or rgba.dtype != np.uint8:
        raise ValueError('Invalid saved atlas overlay.')
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError('Invalid saved Allen coordinate bounds.')
    return transform, rgba, bounds, source_size


def save_scan_registration(scan_path, result, rotation_degrees_clockwise,
                           transform, rgba, ccf_bounds, source_size):
    """Atomically save mapping, rotation and overlay beside the original scan.

    The original image is untouched. Transform coordinates are pixel edges;
    the registration's anchoring uses upright pixel centers.
    """
    scan_path = Path(scan_path).resolve()
    transform, rgba, bounds, source_size = _validate_registration_archive(
        result, rotation_degrees_clockwise, transform, rgba, ccf_bounds, source_size)
    source = scan_path.stat()
    metadata = {
        'schema_version': 1, 'atlas_target': ATLAS_TARGET,
        'image_filename': scan_path.name, 'source_bytes': source.st_size,
        'source_mtime_ns': source.st_mtime_ns,
        'source_width': int(source_size[0]), 'source_height': int(source_size[1]),
        'rotation_degrees_clockwise': float(rotation_degrees_clockwise),
        'transform_convention': 'original_pixel_edges_to_upright_pixel_edges',
        'registration': asdict(result),
    }
    destination = scan_path.with_name(scan_path.stem + '_atlas_registration.npz')
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=scan_path.parent, prefix=destination.name + '.',
                                suffix='.part', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
                                transform=transform, rgba=rgba,
                                ccf_bounds=np.asarray([]) if bounds is None else bounds)
            temporary.flush()
            os.fsync(temporary.fileno())
        current = scan_path.stat()
        if (current.st_size, current.st_mtime_ns) != (source.st_size, source.st_mtime_ns):
            raise ValueError('Scan changed while saving atlas registration.')
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_scan_registration(scan_path):
    """Restore a complete sidecar without model inference; reject changed scans."""
    scan_path = Path(scan_path).resolve()
    saved_path = scan_path.with_name(scan_path.stem + '_atlas_registration.npz')
    if not saved_path.is_file():
        return None
    with np.load(saved_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        source = scan_path.stat()
        if (metadata.get('schema_version') != 1 or metadata.get('atlas_target') != ATLAS_TARGET
                or metadata.get('image_filename') != scan_path.name
                or metadata.get('source_bytes') != source.st_size
                or metadata.get('source_mtime_ns') != source.st_mtime_ns):
            raise ValueError('Saved atlas registration does not match this scan.')
        fields = dict(metadata['registration'])
        fields['anchoring'] = tuple(fields['anchoring'])
        fields['candidate_scores'] = tuple(fields['candidate_scores'])
        result = AtlasRegistration(**fields)
        angle = float(metadata['rotation_degrees_clockwise'])
        bounds = archive['ccf_bounds']
        bounds = None if bounds.size == 0 else bounds
        transform, rgba, bounds, source_size = _validate_registration_archive(
            result, angle, archive['transform'], archive['rgba'], bounds,
            (metadata['source_width'], metadata['source_height']))
    return {'result': result, 'rotation_degrees_clockwise': angle, 'transform': transform,
            'rgba': rgba, 'ccf_bounds_um_ap_dv_ml': bounds, 'saved_path': str(saved_path),
            'source_size': source_size}


def _as_rgb(scan):
    if isinstance(scan, (str, Path)):
        with Image.open(scan) as image:
            if image.mode in ('I', 'F') or image.mode.startswith('I;16'):
                raise ValueError('Pass an 8-bit preview of the stitched scan.')
            return image.convert('RGB')
    if isinstance(scan, Image.Image):
        if scan.mode in ('I', 'F') or scan.mode.startswith('I;16'):
            raise ValueError('Pass an 8-bit preview of the stitched scan.')
        return scan.convert('RGB')
    scan = np.asarray(scan)
    if scan.dtype != np.uint8:
        raise ValueError('Scan arrays must be uint8. Scale 16-bit scans to 8-bit first.')
    if scan.ndim == 2:
        return Image.fromarray(scan).convert('RGB')
    if scan.ndim == 3 and scan.shape[2] in (3, 4):
        # PatcherBot and cv2 use BGR/BGRA ordering for color arrays.
        return Image.fromarray(np.ascontiguousarray(scan[..., :3][..., ::-1]))
    raise ValueError('Expected an image path, PIL image, or grayscale/BGR/BGRA scan array.')


def _torch_xception(torch):
    nn, functional = torch.nn, torch.nn.functional

    class SeparableConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.depthwise = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
            self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)

        def forward(self, x):
            return self.pointwise(self.depthwise(x))

    class Xception(nn.Module):
        def __init__(self):
            super().__init__()
            layers = self.layers = nn.ModuleDict()
            self.conv_bn_pairs = []

            def conv_bn(name, convolution, channels, bn_name=None):
                layers[name] = convolution
                bn_name = bn_name or f"{name}_bn"
                layers[bn_name] = nn.BatchNorm2d(channels, eps=1e-3)
                self.conv_bn_pairs.append((name, bn_name))

            conv_bn("block1_conv1", nn.Conv2d(3, 32, 3, stride=2, bias=False), 32)
            conv_bn("block1_conv2", nn.Conv2d(32, 64, 3, bias=False), 64)
            cin = 64
            for block, outputs in [(2, (128, 128)), (3, (256, 256)), (4, (728, 728)),
                                   *((n, (728, 728, 728)) for n in range(5, 13)),
                                   (13, (728, 1024)), (14, (1536, 2048))]:
                if block in (2, 3, 4, 13):
                    suffix = "" if block == 2 else f"_{3 if block == 13 else block - 2}"
                    conv_bn(f"conv2d{suffix}", nn.Conv2d(cin, outputs[-1], 1, stride=2, bias=False),
                            outputs[-1], f"batch_normalization{suffix}")
                for index, cout in enumerate(outputs, 1):
                    conv_bn(f"block{block}_sepconv{index}", SeparableConv(cin, cout), cout)
                    cin = cout

            layers["predictions"] = nn.Linear(2048, 1000)
            layers["dense"] = nn.Linear(1000, 256)
            layers["dense_1"] = nn.Linear(256, 256)
            layers["dense_2"] = nn.Linear(256, 9)

        def forward(self, x):
            layers = self.layers

            def conv_bn(name, values, bn_name=None):
                return layers[bn_name or f"{name}_bn"](layers[name](values))

            x = functional.relu(conv_bn("block1_conv1", x))
            x = functional.relu(conv_bn("block1_conv2", x))
            for block in range(2, 14):
                downsample = block in (2, 3, 4, 13)
                if downsample:
                    suffix = "" if block == 2 else f"_{3 if block == 13 else block - 2}"
                    residual = conv_bn(f"conv2d{suffix}", x, f"batch_normalization{suffix}")
                else:
                    residual = x
                for index in range(1, 3 if downsample else 4):
                    if block != 2 or index != 1:
                        x = functional.relu(x)
                    x = conv_bn(f"block{block}_sepconv{index}", x)
                if downsample:
                    # TensorFlow SAME for kernel 3/stride 2 adds one extra
                    # pixel on bottom/right when the input size is even.
                    height, width = x.shape[-2:]
                    if height % 2 and width % 2:
                        x = functional.max_pool2d(x, 3, 2, padding=1)
                    else:
                        x = functional.pad(x, (width % 2, 1, height % 2, 1), value=-float("inf"))
                        x = functional.max_pool2d(x, 3, 2)
                x = x + residual
            x = functional.relu(conv_bn("block14_sepconv1", x))
            x = functional.relu(conv_bn("block14_sepconv2", x))
            x = layers["predictions"](x.mean(dim=(2, 3))).softmax(dim=1)
            x = functional.relu(layers["dense"](x))
            x = functional.relu(layers["dense_1"](x))
            return layers["dense_2"](x)

    return Xception().eval(), SeparableConv


class _TorchDeepSlice:
    """Load the two mouse regressors once and batch the two image orientations."""

    def __init__(self, xception_path: str | Path, weight_paths: dict[str, str | Path], device: str):
        import h5py
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.models = {}
        self.memory_format = torch.channels_last if self.device.type == "cuda" else torch.contiguous_format

        def copy(parameter, dataset, axes=None):
            array = np.asarray(dataset)
            if axes is not None:
                array = array.transpose(axes)
            if tuple(parameter.shape) != array.shape:
                raise ValueError(f"Unexpected DeepSlice weight shape for {dataset.name}: {array.shape}")
            parameter.copy_(torch.from_numpy(array))

        with h5py.File(xception_path, "r") as stock, torch.no_grad():
            for model_name, path in weight_paths.items():
                model, separable_conv = _torch_xception(torch)
                with h5py.File(path, "r") as trained:
                    self._load(model, separable_conv, stock, trained, copy)
                # Merge each inference batch normalization into its preceding
                # convolution to reduce CUDA kernel launches and tensor traffic.
                for name, bn_name in model.conv_bn_pairs:
                    convolution, bn = model.layers[name], model.layers[bn_name]
                    if isinstance(convolution, separable_conv):
                        convolution.pointwise = torch.nn.utils.fuse_conv_bn_eval(convolution.pointwise, bn)
                    else:
                        model.layers[name] = torch.nn.utils.fuse_conv_bn_eval(convolution, bn)
                    model.layers[bn_name] = torch.nn.Identity()
                self.models[model_name] = model.to(self.device, memory_format=self.memory_format)

    def _load(self, model, separable_conv, stock, trained, copy):
        nn = self.torch.nn
        xception = trained["xception"]
        dense_names = ("dense", "dense_1", "dense_2")
        expected = set(model.layers) - set(dense_names) - {"predictions"}
        if set(xception) != expected or set(trained) != {"xception", *dense_names}:
            raise ValueError("Unexpected DeepSlice mouse weight groups; expected the DeepSlice 1.2.8 assets.")

        for name, layer in model.layers.items():
            if name == "predictions":
                # The mouse checkpoints omit the ImageNet classifier. Its
                # softmax output is the input to the DeepSlice regression head.
                copy(layer.weight, stock["dense_2"]["dense_2_W:0"], (1, 0))
                copy(layer.bias, stock["dense_2"]["dense_2_b:0"])
                continue
            weights = trained[name][name] if name in dense_names else xception[name]
            if isinstance(layer, separable_conv):
                copy(layer.depthwise.weight, weights["depthwise_kernel:0"], (2, 3, 0, 1))
                copy(layer.pointwise.weight, weights["pointwise_kernel:0"], (3, 2, 0, 1))
            elif isinstance(layer, nn.Conv2d):
                copy(layer.weight, weights["kernel:0"], (3, 2, 0, 1))
            elif isinstance(layer, nn.BatchNorm2d):
                copy(layer.weight, weights["gamma:0"])
                copy(layer.bias, weights["beta:0"])
                copy(layer.running_mean, weights["moving_mean:0"])
                copy(layer.running_var, weights["moving_variance:0"])
            elif isinstance(layer, nn.Linear):
                copy(layer.weight, weights["kernel:0"], (1, 0))
                copy(layer.bias, weights["bias:0"])
            else:
                raise TypeError(f"Unsupported DeepSlice layer {name}: {type(layer).__name__}")

    def predict(self, batch_nhwc: np.ndarray):
        with self.torch.inference_mode():
            batch = self.torch.from_numpy(batch_nhwc).permute(0, 3, 1, 2)
            batch = batch.to(self.device, dtype=self.torch.float32, memory_format=self.memory_format)
            # Collect outputs inside the inference context, then return. A
            # generator here could leak inference mode into the caller.
            return [(name, model(batch).cpu().numpy()) for name, model in self.models.items()]


class AtlasRegistrar:
    """Register a complete, near-coronal mouse slice to Allen CCFv3.

    The detector expects dark tissue on a brighter background. Orientation
    selection tests upright and inverted views. Scores describe image fit;
    use the overlay/landmarks to review anatomy before assigning precise sites.
    Reuse this instance across scans to keep the atlas and network in memory.
    The threads argument budgets CPU threads for the independent atlas fits.
    Set segmentation_side=900 for faster mask generation at lower resolution.
    """

    def __init__(self, cache_dir: str | Path | None = None, *, download_assets: bool = True,
                 threads: int = 4, max_working_side: int = 1600, device: str = 'auto',
                 segmentation_side: int | None = None):
        if threads < 1 or max_working_side < 299 or (segmentation_side is not None and segmentation_side < 299):
            raise ValueError('threads must be positive and image size limits at least 299.')
        self.cache_dir = (Path(cache_dir).expanduser() if cache_dir is not None
                          else Path.home() / '.cache' / 'patcherbot_atlas' / 'allen_ccfv3_25um')
        self.download_assets = download_assets
        self.threads = threads
        self.max_working_side = max_working_side
        self.segmentation_side = segmentation_side
        self.device = device
        self.inference_device: str | None = None
        self._lock = RLock()
        self._network = None
        self._annotation = None
        self._nissl = None

    def _asset(self, name):
        url, expected_hash = ASSETS[name]
        path = self.cache_dir / name

        def validate(file_path):
            digest = hashlib.sha256()
            with file_path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f'Unexpected bytes in {file_path}. Check the file or choose a fresh cache directory.')

        if path.is_file():
            validate(path)
            return path
        if not self.download_assets:
            raise FileNotFoundError(f'Missing atlas asset {path}. Download it from {url} or enable downloads.')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(dir=self.cache_dir, prefix=name+'.', suffix='.part', delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with urlopen(url, timeout=90) as response:
                    shutil.copyfileobj(response, temporary, length=1024*1024)
            validate(temporary_path)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def prepare(self) -> None:
        """Populate the cache and load both DeepSlice models outside acquisition."""
        with self._lock:
            if self._network is not None:
                return
            try:
                import torch
            except ImportError as exc:
                raise ImportError('AtlasRegistrar requires PyTorch. Install a CUDA build for GPU use.') from exc
            requested = self.device
            if requested == 'auto':
                requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            device = torch.device(requested)
            if device.type not in ('cpu', 'cuda'):
                raise ValueError('Choose device="auto", "cpu", or a CUDA device such as "cuda:0".')
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and the CUDA build of PyTorch.')
                if device.index is not None and device.index >= torch.cuda.device_count():
                    raise ValueError(f'CUDA device {device.index} is unavailable.')

            weight_paths = {name: str(self._asset(file)) for name, file in WEIGHTS.items()}
            xception = str(self._asset(XCEPTION_WEIGHTS))
            def load_volume(name):
                return read_nrrd(self._asset(name))[0]

            if self._annotation is None and self.threads > 1:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    self._annotation, self._nissl = executor.map(load_volume, ('annotation_25.nrrd', 'ara_nissl_25.nrrd'))
            else:
                if self._annotation is None:
                    self._annotation = load_volume('annotation_25.nrrd')
                self._nissl = load_volume('ara_nissl_25.nrrd')
            if self._annotation.shape != ATLAS_SHAPE or self._nissl.shape != ATLAS_SHAPE:
                raise ValueError('Expected the 25 µm Allen CCFv3 volume dimensions.')
            self._network = _TorchDeepSlice(xception, weight_paths, str(device))
            self.inference_device = str(device)

    def _planes(self, views):
        # Match DeepSlice's ImageDataGenerator: nearest resize, grayscale,
        # samplewise centering and scaling, then broadcast to three channels.
        inputs = []
        for image in views:
            resized = image.resize((299, 299), Image.Resampling.NEAREST)
            rgb = np.asarray(resized, dtype=np.float32)
            gray = (rgb @ np.array([.2125, .7154, .0721], dtype=np.float32))[..., None]
            gray -= np.mean(gray)
            gray /= np.std(gray) + 1e-6
            inputs.append(np.repeat(gray, 3, axis=2))
        for model_name, predictions in self._network.predict(np.stack(inputs)):
            for index, anchoring in enumerate(predictions):
                yield model_name, index, anchoring

    def estimate_upright_rotation(self, scan) -> float:
        """Estimate clockwise tilt and resolve the 180-degree orientation.

        Search +/-15 degrees using bilateral tissue symmetry, then compare
        atlas fits for upright/inverted views. Return an angle for review;
        registration of the final rotated scan remains a separate operation.
        """
        image = _as_rgb(scan)
        if min(image.size) < 64:
            raise ValueError('Provide a whole-section scan at least 64 pixels on each side.')
        image.thumbnail((self.max_working_side, self.max_working_side), Image.Resampling.LANCZOS)
        rgb = np.asarray(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = detect_tissue(gray)
        scale = min(1.0, 420 / max(gray.shape))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        small = np.pad(small, 80, constant_values=1)
        small_mask = np.pad(small_mask, 80)
        height, width = small.shape
        best_score, best_angle = -np.inf, None
        for clockwise in np.arange(-15, 15.01, .25):
            matrix = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), -float(clockwise), 1)
            moved_mask = cv2.warpAffine(small_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
            moved_gray = cv2.warpAffine(small, matrix, (width, height), borderValue=1)
            moments = cv2.moments(moved_mask)
            if moments['m00'] <= 0:
                continue
            center_x = round(moments['m10'] / moments['m00'])
            for center in range(max(1, center_x-8), min(width-1, center_x+9)):
                extent = min(center, width-center-1)
                left = moved_mask[:, center-extent:center][:, ::-1].astype(bool)
                right = moved_mask[:, center+1:center+extent+1].astype(bool)
                overlap = left & right
                if not overlap.any():
                    continue
                dice = 2 * overlap.sum() / (left.sum() + right.sum())
                left_gray = moved_gray[:, center-extent:center][:, ::-1]
                right_gray = moved_gray[:, center+1:center+extent+1]
                difference = float(np.mean(np.abs(left_gray[overlap] - right_gray[overlap])))
                score = float(dice - .5*difference)
                if np.isfinite(score) and score > best_score:
                    best_score, best_angle = score, float(clockwise)
        if best_angle is None:
            raise ValueError('Could not estimate a bilateral tissue axis.')

        # Apply the chosen tilt with the same expanded pixel-edge bounds as
        # the scan viewer. No display transform or source pixels are mutated.
        height, width = gray.shape
        matrix = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), -best_angle, 1)
        corners = np.array([[-.5, -.5], [width-.5, -.5],
                            [width-.5, height-.5], [-.5, height-.5]])
        bounds = corners @ matrix[:, :2].T + matrix[:, 2]
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        output_size = tuple(np.ceil(high-low-1e-9).astype(int))
        matrix[:, 2] -= low + .5
        level = cv2.warpAffine(rgb, matrix, output_size, flags=cv2.INTER_LINEAR,
                               borderValue=(255, 255, 255))
        orientation = self.register(Image.fromarray(level), upright=False)
        return float((best_angle + orientation.rotation_degrees + 180) % 360 - 180)

    def register(self, scan: str | Path | Image.Image | np.ndarray, *,
                 upright: bool = False) -> AtlasRegistration:
        """Register a scan; upright=True trusts the user-confirmed orientation."""
        image = _as_rgb(scan)
        if min(image.size) < 64:
            raise ValueError('Provide a stitched whole-section scan at least 64 pixels on each side.')
        work = image.copy()
        work.thumbnail((self.max_working_side, self.max_working_side), Image.Resampling.LANCZOS)
        scale = np.asarray(work.size)/np.asarray(image.size)
        gray = np.asarray(work.convert('L'))
        mask_side = self.segmentation_side or self.max_working_side
        try:
            mask = detect_tissue(gray, mask_side)
        except ValueError:
            if max(work.size) <= mask_side:
                raise
            mask = detect_tissue(gray, self.max_working_side)
        x, y, w, h = cv2.boundingRect(mask)
        if w < 32 or h < 32:
            raise ValueError('The automatic tissue mask is too small for registration.')
        pad_x, pad_y = round(.05*w), round(.05*h)
        left, top = max(0, x-pad_x), max(0, y-pad_y)
        right, bottom = min(work.width, x+w+pad_x), min(work.height, y+h+pad_y)
        cropped = work.crop((left, top, right, bottom))
        views = ((cropped,) if upright else
                 (cropped, cropped.transpose(Image.Transpose.ROTATE_180)))
        cropped_mask = mask[top:bottom, left:right] > 0
        masks = (cropped_mask, cropped_mask[::-1, ::-1])
        tissues = [np.asarray(view.convert('L'))[::STEP, ::STEP] for view in views]
        with self._lock:
            self.prepare()
            planes = list(self._planes(views))
            workers = min(self.threads, len(planes))
            threads_per_fit = max(1, self.threads // workers)

            def fit_candidate(plane):
                model_name, index, anchoring = plane
                diagnostics = {'model': model_name, 'rotation_degrees': index*180}
                section = {'width': cropped.width, 'height': cropped.height, 'anchoring': anchoring}
                try:
                    if not np.isfinite(anchoring).all():
                        raise ValueError('Non-finite DeepSlice plane.')
                    labels, template, atlas_origin = atlas_view(section, self._annotation, self._nissl)
                    atlas_mask = ndi.binary_fill_holes(labels > 0)
                    matrix, offset, report = fit(tissues[index], masks[index][::STEP, ::STEP], template,
                                                 atlas_mask, atlas_origin, threads_per_fit)
                    if np.linalg.det(matrix) <= 0 or not np.isfinite(report['score']):
                        raise ValueError('Invalid affine transform.')
                except (ValueError, RuntimeError) as exc:
                    return {**diagnostics, 'error': str(exc)}, None
                return {**diagnostics, **report}, (report['score'], index, model_name, anchoring, matrix, offset, report)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(fit_candidate, planes))
        scores = [report for report, _ in results]
        candidates = [candidate for _, candidate in results if candidate is not None]
        if not candidates:
            raise RuntimeError(f'No atlas candidate could be registered. {scores}')
        _, index, model_name, anchoring, matrix, offset, report = max(candidates, key=lambda item: item[0])

        # Compose original pixel centers -> working image -> oriented crop ->
        # primary projection -> 3D QuickNII anchor. Preserve the original scan.
        sign = -1 if index else 1
        crop_offset = np.array([right-1, bottom-1] if index else [-left, -top])
        work_offset = (scale-1)/2
        origin, u, v = anchoring.astype(float).reshape(3, 3)
        per_pixel = np.column_stack([u/cropped.width, v/cropped.height])
        original_offset = matrix @ (sign*work_offset + crop_offset) + offset
        basis = per_pixel @ (sign*matrix) @ np.diag(scale*np.asarray(image.size))
        original_anchoring = np.concatenate([origin+per_pixel @ original_offset, basis[:, 0], basis[:, 1]])
        return AtlasRegistration(
            width=image.width, height=image.height, anchoring=tuple(original_anchoring.tolist()),
            rotation_degrees=index*180, deepslice_model=model_name,
            silhouette_dice_before=report['unrefined']['dice'],
            silhouette_dice=report[report['selected']]['dice'], candidate_scores=tuple(scores),
        )

    def render_overlay(self, result: AtlasRegistration, max_side: int = 1600) -> dict:
        """Render annotation colors in the registered image's pixel frame.

        Scale the returned RGBA raster to result.width/result.height in the
        viewer. Pixel-center sampling preserves alignment when downsampled.
        Bounds are sampled Allen CCF AP/DV/ML ranges in micrometers over
        nonzero atlas labels, not stereotaxic coordinates or confidence bounds.
        An empty atlas intersection returns transparent pixels and None bounds.
        """
        if not isinstance(max_side, (int, np.integer)) or max_side < 1:
            raise ValueError('max_side must be a positive integer.')
        ratio = min(1.0, max_side / max(result.width, result.height))
        width = max(1, round(result.width * ratio))
        height = max(1, round(result.height * ratio))
        xs = (np.arange(width) + .5) * result.width / width - .5
        ys = (np.arange(height) + .5) * result.height / height - .5
        x, y = np.meshgrid(xs, ys)
        origin, u, v = np.asarray(result.anchoring).reshape(3, 3)
        coords = quicknii_to_ccf25(
            origin[:, None, None] + u[:, None, None] * x / result.width
            + v[:, None, None] * y / result.height)
        with self._lock:
            if self._annotation is None:
                self._annotation, _ = read_nrrd(self._asset('annotation_25.nrrd'))
            labels = ndi.map_coordinates(self._annotation, coords, order=0,
                                         mode='constant', cval=0, prefilter=False)
        valid = labels != 0
        edges = np.zeros(labels.shape, dtype=bool)
        edges[1:] |= labels[1:] != labels[:-1]
        edges[:-1] |= labels[:-1] != labels[1:]
        edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        # Stable display colors; these are not the Allen ontology's palette.
        hashed = labels.astype(np.uint64) * 2654435761
        rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
        for channel, shift in enumerate((0, 8, 16)):
            rgba[..., channel] = 64 + ((hashed >> shift) % 192).astype(np.uint8)
        rgba[..., 3] = np.where(valid, np.where(edges, 220, 85), 0)
        bounds = None
        if valid.any():
            covered = coords[:, valid] * 25
            bounds = np.column_stack((covered.min(axis=1), covered.max(axis=1)))
        return {'rgba': rgba, 'labels': labels,
                'ccf_bounds_um_ap_dv_ml': bounds}

    def locate(self, result: AtlasRegistration, x: float, y: float) -> dict:
        """Look up Allen annotation ID and CCF coordinates for a scan pixel."""
        ccf_um = result.pixel_to_ccf_um(x, y)
        indices = np.floor(np.asarray(ccf_um)/25 + .5).astype(int)
        if np.any(indices < 0) or np.any(indices >= np.asarray(ATLAS_SHAPE)):
            return {'ccf_um_ap_dv_ml': ccf_um, 'region_id': 0}
        with self._lock:
            if self._annotation is None:
                self._annotation, _ = read_nrrd(self._asset('annotation_25.nrrd'))
            region_id = int(self._annotation[tuple(indices)])
        return {'ccf_um_ap_dv_ml': ccf_um, 'region_id': region_id}
