"""Saved source-image registration and reproducible image-anchor setup.

Global coordinates are anchor-relative microns in the preserved common map.
Loading never performs matching. Anchor rebuilding does not rebuild the map.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading

import cv2
import numpy as np


class ObjectiveRegistrar:
    REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_FOLDER = Path(__file__).resolve().parent / "globalModel" / "reference"
    DEFAULT_CONFIG = REPOSITORY_ROOT / "rig_setup" / "cal_configs" / "moscow_cal.yaml"
    _shared = None
    _lock = threading.RLock()

    def __init__(self):
        self.anchor = None
        self.registration = None
        self.templates = {}
        self.donors = {}
        self.reference_image_path = None
        self.reference_image_transform = None
        self.reference_manifest_path = None
        self.render = None
        self._frames = {}

    @classmethod
    def shared(cls, config=None):
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls().load_default(config=config)
            return cls._shared

    @property
    def map(self):
        return self.registration

    def _index(self):
        self._frames = {o: {f["index"]: f for f in self.registration[f"frame{o}"]} for o in (4, 40)}

    def load_default(self, config=None):
        """Load the portable saved map; donor paths are only needed for rebuilding.

        An active config object or mapping can override objective_donor_4x and
        objective_donor_40x. Without one, read the Moscow calibration YAML.
        Relative configured paths are resolved against the repository root.
        """
        if config is None:
            import yaml
            config = (yaml.safe_load(self.DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
                      if self.DEFAULT_CONFIG.is_file() else {})
        donors = {}
        for objective in (4, 40):
            name = f"objective_donor_{objective}x"
            value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
            if value:
                path = Path(value).expanduser()
                donors[objective] = (self.REPOSITORY_ROOT / path).resolve()
        return self.load(self.DEFAULT_FOLDER, donors=donors)

    @staticmethod
    def _image(image):
        value = image if isinstance(image, np.ndarray) else cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if value is None or value.ndim != 2 or value.dtype != np.uint8:
            raise ValueError("Expected readable native uint8 grayscale pixels")
        return value

    @staticmethod
    def _objective(objective):
        if objective not in (4, 40):
            raise ValueError("objective must be 4 or 40")
        return objective

    def calibration(self, objective):
        return np.array(self.anchor["objective_calibrations"][f"M{self._objective(objective)}"], dtype=float)

    def reference_frame(self, objective):
        objective = self._objective(objective)
        ref = self.anchor["references"][str(objective)]
        record = copy.deepcopy(self._frames[objective][ref["frame_index"]])
        record["pixel_to_global_um"] = self._matrix(objective, ref["frame_index"])
        record["image_size"] = ref.get("image_size", [1280, 1280])
        return record

    def _matrix(self, objective, frame_index=None):
        objective = self._objective(objective)
        if frame_index is None:
            frame_index = self.anchor["references"][str(objective)]["frame_index"]
        frame = self._frames[objective][frame_index]
        return np.array(self.anchor["global4_to_anchor_microns"]) @ np.array(frame[f"pixel{objective}_to_global4"])

    def pixel_to_micron_matrix(self, objective):
        return self._matrix(objective)[:2, :2].copy()

    @staticmethod
    def _transform(points, matrix):
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (2,) or not np.isfinite(points).all():
            raise ValueError("Expected finite XY coordinates")
        return points @ matrix[:2, :2].T + matrix[:2, 2]

    def pixel_to_global(self, pixels, objective, frame_index=None):
        return self._transform(pixels, self._matrix(objective, frame_index))

    def global_to_pixel(self, coordinates, objective, frame_index=None):
        return self._transform(coordinates, np.linalg.inv(self._matrix(objective, frame_index)))

    @property
    def reference_pixel_to_global_um(self):
        if self.reference_image_transform is not None:
            return self.reference_image_transform.copy()
        render_to_global = np.eye(3)
        render_to_global[:2, :2] /= self.render["render_pixels_per_native4_pixel"]
        render_to_global[:2, 2] = self.render["render_origin_global4"]
        return np.array(self.anchor["global4_to_anchor_microns"]) @ render_to_global

    def anchor_settings(self):
        """Explicit seed for rebuilding this anchor from the two donor folders."""
        refs = {}
        for o in (4, 40):
            ref = self.anchor["references"][str(o)]
            refs[str(o)] = dict(filename=str(Path(ref["source_path"]).relative_to(self.donors[o])),
                                frame_index=ref["frame_index"])
        return dict(references=refs, pixel40_to_pixel4=self.anchor["pixel40_to_pixel4"],
                    reference4_origin=self.reference_frame(4)["origin_4x_pixels"],
                    calibration=self.anchor["objective_calibrations"], contour_roi=[100, 180, 1080, 520],
                    smoothing_width=61, fit_radius=200, template_sizes={"4": [240, 160], "40": [400, 240]})

    def build_anchor(self, donor4, donor40, settings):
        """Recompute the boundary and original-pixel templates, without match caches.

        Settings identify both donor-relative files, the saved optical seed,
        reference origin, calibration and contour/template extraction parameters.
        The existing source map is preserved; full cold map rebuilding is separate.
        """
        donors = {4: Path(donor4).resolve(), 40: Path(donor40).resolve()}
        paths = {}
        for o in (4, 40):
            path = (donors[o] / settings["references"][str(o)]["filename"]).resolve()
            path.relative_to(donors[o])
            paths[o] = path
        images = {o: self._image(p) for o, p in paths.items()}
        x, y, width, height = settings["contour_roi"]
        if min(x, y) < 0 or min(width, height) <= 0 or x+width > images[40].shape[1] or y+height > images[40].shape[0]:
            raise ValueError("Contour rectangle is outside the native 40x image")
        derivative = cv2.Sobel(cv2.GaussianBlur(images[40].astype(np.float32), (0, 0), 3), cv2.CV_32F, 0, 1)
        xs = np.arange(x, x+width)
        ys = y + np.argmax(derivative[y:y+height, x:x+width], axis=0)
        window = settings["smoothing_width"]
        if window < 3 or window % 2 != 1 or window >= width:
            raise ValueError("Smoothing width must be odd and smaller than contour width")
        half = window//2
        middle = xs[half:-half][np.argmin(np.convolve(ys, np.ones(window)/window, mode="valid"))]
        radius = settings["fit_radius"]
        keep = np.abs(xs-middle) <= radius
        coefficients = np.polyfit(xs[keep]-middle, ys[keep], 2)
        vertex = -coefficients[1]/(2*coefficients[0])
        if coefficients[0] <= 0 or abs(vertex) > radius:
            raise ValueError("Boundary crest is not a stable interior minimum")
        point40 = np.array([middle+vertex, np.polyval(coefficients, vertex)])
        local = np.array(settings["pixel40_to_pixel4"], dtype=float)
        if local.shape != (3, 3) or not np.isfinite(local).all() or abs(np.linalg.det(local)) < 1e-12:
            raise ValueError("Optical seed must be a finite invertible 3x3 transform")
        point4 = self._transform(point40, local)
        origin = np.array(settings["reference4_origin"]) + point4
        physical = np.eye(3)
        physical[:2, :2] = np.linalg.inv(-np.array(settings["calibration"]["M4"]))
        physical[:2, 2] = -physical[:2, :2] @ origin
        references, templates = {}, {}
        for o, point in ((4, point4), (40, point40)):
            size = np.array(settings["template_sizes"][str(o)], dtype=int)
            if np.any(size <= 0) or np.any(size > images[o].shape[::-1]):
                raise ValueError("Template does not fit donor image")
            start = np.maximum(0, np.minimum(np.floor(point-size/2).astype(int), np.array(images[o].shape[::-1])-size))
            templates[o] = images[o][start[1]:start[1]+size[1], start[0]:start[0]+size[0]].copy()
            references[str(o)] = dict(frame_index=settings["references"][str(o)]["frame_index"],
                source_path=str(paths[o]), source_sha256=hashlib.sha256(paths[o].read_bytes()).hexdigest(),
                crop_xywh=[*start.tolist(), *size.tolist()], anchor_pixel=point.tolist(),
                anchor_in_template=(point-start).tolist(), image_size=list(images[o].shape[::-1]))
        anchor = dict(status="reusable_image_template_anchor", references=references,
            global4_anchor_pixel=origin.tolist(), global4_to_anchor_microns=physical.tolist(),
            anchor_microns_to_global4=np.linalg.inv(physical).tolist(), pixel40_to_pixel4=local.tolist(),
            pixel4_to_pixel40=np.linalg.inv(local).tolist(), objective_calibrations=copy.deepcopy(settings["calibration"]),
            build_settings=copy.deepcopy(settings), physical_accuracy_verified=False,
            boundary_fit_rms40_pixels=float(np.sqrt(np.mean((np.polyval(coefficients,xs[keep]-middle)-ys[keep])**2))))
        self.anchor, self.templates, self.donors = anchor, templates, donors
        return self

    def locate_anchor(self, image, objective):
        objective = self._objective(objective)
        source, template = self._image(image), self.templates[objective]
        if np.any(np.array(source.shape) < template.shape):
            raise ValueError("Image is smaller than the anchor template")
        def gradient(value):
            return cv2.Sobel(cv2.GaussianBlur(value.astype(np.float32), (0, 0), 2), cv2.CV_32F, 0, 1)
        scores = cv2.matchTemplate(gradient(source), gradient(template), cv2.TM_CCOEFF_NORMED)
        _, score, _, xy = cv2.minMaxLoc(scores)
        position = np.array(xy, dtype=float)
        x, y = xy
        for axis, index in enumerate(xy):
            if 0 < index < scores.shape[1-axis]-1:
                values = scores[y, x-1:x+2] if axis == 0 else scores[y-1:y+2, x]
                denominator = float(values[0]-2*values[1]+values[2])
                if denominator < -1e-7:
                    position[axis] += np.clip(.5*(values[0]-values[2])/denominator, -.5, .5)
        second = scores.copy()
        second[max(0,y-8):y+9, max(0,x-8):x+9] = -1
        margin = float(score-second.max())
        boundary = x == 0 or y == 0 or x == scores.shape[1]-1 or y == scores.shape[0]-1
        return dict(anchor_pixel=(position+self.anchor["references"][str(objective)]["anchor_in_template"]).tolist(),
                    matched=bool(score >= .85 and margin > .005 and not boundary),
                    template_correlation=float(score), separated_peak_margin=margin,
                    search_boundary=boundary, objective=objective, image_size=list(source.shape[::-1]))

    def register_pair(self, image4, image40):
        detections = {"4": self.locate_anchor(image4, 4), "40": self.locate_anchor(image40, 40)}
        if not all(d["matched"] for d in detections.values()):
            raise ValueError("Shared anchor was not confidently found in both images")
        p4, p40 = [np.array(detections[str(o)]["anchor_pixel"]) for o in (4, 40)]
        transform = np.array(self.anchor["pixel40_to_pixel4"], dtype=float)
        transform[:2, 2] = p4-transform[:2, :2]@p40
        global4 = np.eye(3)
        global4[:2, 2] = np.array(self.anchor["global4_anchor_pixel"])-p4
        micron4 = np.array(self.anchor["global4_to_anchor_microns"])@global4
        micron40 = micron4@transform
        w, h = detections["40"]["image_size"]
        corners = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
        return dict(status="image_anchor_registered", detections=detections,
            pixel40_to_pixel4=transform.tolist(), pixel4_to_pixel40=np.linalg.inv(transform).tolist(),
            pixel4_to_global4=global4.tolist(), pixel40_to_global4=(global4@transform).tolist(),
            pixel4_to_anchor_microns=micron4.tolist(), pixel40_to_anchor_microns=micron40.tolist(),
            anchor_microns_to_pixel4=np.linalg.inv(micron4).tolist(), anchor_microns_to_pixel40=np.linalg.inv(micron40).tolist(),
            footprint40_in_pixel4=self._transform(corners,transform).tolist(), physical_accuracy_verified=False,
            anchor_sha256=hashlib.sha256(json.dumps(self.anchor,sort_keys=True).encode()).hexdigest())

    def register_session_anchor(self, image40, session_scan_path, session_metadata, calibration40):
        """Locate a fresh 40x field in a session mosaic without changing calibration.

        Fixed measured optics, a bounded whole-image search, then native-pixel
        refinement. The best textured candidate is returned with alternatives;
        correlation is diagnostic, not an automatic acceptance threshold.
        """
        import time
        started = time.perf_counter()
        source = self._image(image40)
        scan = cv2.imread(str(session_scan_path), cv2.IMREAD_GRAYSCALE)
        if scan is None:
            raise ValueError("The session scan image could not be read")
        m4 = np.asarray(session_metadata["calibration"]["stage"]["M"], dtype=float)
        m40 = np.asarray(calibration40, dtype=float)
        if any(m.shape != (2, 2) or not np.isfinite(m).all()
               or abs(np.linalg.det(m)) < 1e-12 for m in (m4, m40)):
            raise ValueError("Both objective calibrations must be finite invertible XY matrices")
        full = np.asarray(session_metadata.get("canvas_shape", scan.shape)[:2], dtype=float)[::-1]
        sampling = np.asarray(session_metadata["pixel_scale_xy"], dtype=float) * full / scan.shape[::-1]
        if sampling.shape != (2,) or not np.isfinite(sampling).all() or np.any(sampling <= 0):
            raise ValueError("Session sampling must contain positive XY scales")
        optics = np.diag(1. / sampling) @ m4 @ np.linalg.inv(m40)
        h, w = source.shape
        corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]) @ optics.T
        lower = np.floor(corners.min(0))
        shape = (np.ceil(corners.max(0)) - lower + 1).astype(int)
        border = max(2, int(np.ceil(max(abs(optics[0, 1])*h, abs(optics[1, 0])*w))) + 1)
        if np.any(shape - 2*border < 16) or np.any(shape > np.array(scan.shape[::-1])):
            raise ValueError("The calibrated 40x field does not fit inside this session scan")
        template = cv2.warpAffine(source, np.column_stack([optics, -lower]), tuple(shape),
            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)[border:-border, border:-border]
        center = optics @ ((np.array([w, h])-1)/2) - lower - border
        if float(np.std(template)) < 1e-6:
            raise ValueError("The captured image has no intensity variation to match")

        def prepare(value, method):
            value = value.astype(np.float32)
            if method == "highpass":
                return cv2.GaussianBlur(value, (0, 0), 1) - cv2.GaussianBlur(value, (0, 0), 8)
            value = cv2.GaussianBlur(value, (0, 0), 1.5)
            return cv2.magnitude(cv2.Sobel(value, cv2.CV_32F, 1, 0),
                                 cv2.Sobel(value, cv2.CV_32F, 0, 1))

        reports = []
        factor = min(.5, 2400. / max(scan.shape))
        for method in ("gradient", "highpass"):
            target, needle = prepare(scan, method), prepare(template, method)
            small = cv2.resize(target, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            tiny = cv2.resize(needle, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            response = cv2.matchTemplate(small, tiny, cv2.TM_CCOEFF_NORMED)
            # Constant regions and empty mosaic borders cannot locate a template.
            mean = cv2.boxFilter(small, cv2.CV_32F, tiny.shape[::-1], anchor=(0, 0), normalize=True)
            mean2 = cv2.boxFilter(small*small, cv2.CV_32F, tiny.shape[::-1], anchor=(0, 0), normalize=True)
            variance = mean2[:response.shape[0], :response.shape[1]] - mean[:response.shape[0], :response.shape[1]]**2
            response[(variance < 1e-6) | ~np.isfinite(response)] = -2
            radius = max(8, int(round(200 * factor)))
            reach = max(12, int(np.ceil(3/factor)))
            candidates = []
            for _ in range(12):
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                if score <= -1:
                    break
                x0, y0 = max(0, int(x/factor)-reach), max(0, int(y/factor)-reach)
                x1 = min(target.shape[1], int(x/factor)+reach+needle.shape[1])
                y1 = min(target.shape[0], int(y/factor)+reach+needle.shape[0])
                roi = target[y0:y1, x0:x1]
                local = cv2.matchTemplate(roi, needle, cv2.TM_CCOEFF_NORMED)
                local[~np.isfinite(local)] = -2
                _, score, _, (dx, dy) = cv2.minMaxLoc(local)
                point = np.array([x0+dx, y0+dy]) + center
                candidates.append(dict(score=float(score), center_session_native_px=point.tolist()))
                response[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -2
            if not candidates:
                raise ValueError("No textured session image support is available for matching")
            candidates.sort(key=lambda value: -value["score"])
            reports.append(dict(method=method, candidates=candidates))
        selected = reports[0]["candidates"][0]
        point = np.array(selected["center_session_native_px"])
        x, y = np.rint(point-center).astype(int)
        crop = scan[y:y+template.shape[0], x:x+template.shape[1]].copy()
        agreement = np.array(reports[1]["candidates"][0]["center_session_native_px"]) - point
        return dict(center_session_native_px=point.tolist(), score=selected["score"],
            method="fixed_optics_gradient_native_refinement", candidates=reports,
            fixed_camera40_to_camera4=optics.tolist(),
            highpass_disagreement_native_px=agreement.tolist(),
            image40_preview=template.copy(), match_preview=crop,
            elapsed_seconds=time.perf_counter()-started, physical_accuracy_verified=False)

    def save(self, folder):
        """Write a portable geometry/template bundle only to the explicit folder."""
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        registration, anchor = copy.deepcopy(self.registration), copy.deepcopy(self.anchor)
        for o in (4, 40):
            for frame in registration[f"frame{o}"]:
                frame["path"] = os.path.relpath(frame["path"], self.donors[o])
            anchor["references"][str(o)]["source_path"] = os.path.relpath(anchor["references"][str(o)]["source_path"], self.donors[o])
            anchor["references"][str(o)].pop("template_path", None)
            if not cv2.imwrite(str(folder/f"template_{o}.png"), self.templates[o]):
                raise OSError("Could not write anchor template")
        reference = None
        if self.reference_image_path is not None:
            destination = folder/"reference.tif"
            if Path(self.reference_image_path).resolve() != destination:
                shutil.copyfile(self.reference_image_path,destination)
            reference = destination.name
        bundle = dict(schema_version=1, registration=registration, anchor=anchor, render=self.render,
            donors={str(o):os.path.relpath(p,folder) for o,p in self.donors.items()}, reference_image=reference,
            reference_pixel_to_global_um=self.reference_pixel_to_global_um.tolist(),
            template_sha256={str(o):hashlib.sha256((folder/f"template_{o}.png").read_bytes()).hexdigest() for o in (4,40)})
        (folder/"registration.json").write_text(json.dumps(bundle,indent=2,allow_nan=False),encoding="utf-8")
        return folder

    def load(self, folder, donors=None):
        """Load saved geometry/templates, optionally relocating donor roots.

        Donors are not opened here. Relative overrides use the bundle folder.
        """
        folder = Path(folder).resolve()
        bundle = json.loads((folder/"registration.json").read_text(encoding="utf-8"))
        if bundle["schema_version"] != 1:
            raise ValueError("Unsupported registration bundle schema")
        templates = {}
        for o in (4,40):
            path = folder/f"template_{o}.png"
            if hashlib.sha256(path.read_bytes()).hexdigest() != bundle["template_sha256"][str(o)]:
                raise ValueError("Anchor template checksum mismatch")
            templates[o] = self._image(path)
        self.registration, self.anchor = bundle["registration"], bundle["anchor"]
        self.donors = {int(o):(folder/p).resolve() for o,p in bundle["donors"].items()}
        for objective, path in (donors or {}).items():
            self.donors[self._objective(int(objective))] = (folder / Path(path)).resolve()
        for o in (4,40):
            for frame in self.registration[f"frame{o}"]:
                frame["path"] = str(self.donors[o]/frame["path"])
            self.anchor["references"][str(o)]["source_path"] = str(self.donors[o]/self.anchor["references"][str(o)]["source_path"])
        self.templates, self.render = templates, bundle["render"]
        self.reference_image_path = folder/bundle["reference_image"] if bundle["reference_image"] else None
        transform = bundle.get("reference_pixel_to_global_um")
        if transform is not None:
            transform = np.asarray(transform, dtype=float)
            if (transform.shape != (3, 3) or not np.isfinite(transform).all()
                    or not np.allclose(transform[2], [0., 0., 1.])
                    or np.linalg.matrix_rank(transform) != 3):
                raise ValueError("Reference image transform must be a finite invertible XY affine matrix")
        self.reference_image_transform = transform
        self.reference_manifest_path = folder / "scan_reference.json"
        self._index()
        return self
