# Reference center measurement — 2026-11-04

The user selected the **second / middle visible rim**, outside the thin inner rim
and inside the outermost dark boundary. Its adopted center is **(1952, 1888)
pixels** in the 4380 × 5279 reference TIFF. Image coordinates start at the top
left, with X rightward and Y downward.

The estimated recorded stage position that centers this point in the camera is
**X = 28133.0927 µm, Y = -49116.8871 µm**. Report it as approximately
**(28130, -49120) µm**, with a practical rim-selection uncertainty of about
**±150 µm**. This is not a statistical confidence interval or validated absolute
accuracy bound. The six local frame estimates agree within 1.15 µm; that measures
internal consistency, not the accuracy of the physical center.

## Saved files

- `scan_reference.tif`: unchanged copy of the user-designated 2× downsampled TIFF.
- `scan_reference_center.png`: marked preview showing the middle rim and center.
- `scan_reference.json`: source paths, image hashes, measured center, calibration,
  coordinate transform, and the six local frame observations.
- This document: measurement method and executable reproduction code.

The original full-resolution TIFF and recordings remain at the source paths in
the JSON. The saved TIFF here is the user's chosen downsampled reference, not a
copy of the larger full-resolution output. No runtime code or calibration matrix
was changed to install or apply this reference.

## Finding the middle rim

1. Convert the reference to grayscale and resize to one-quarter width and height
   using area interpolation (1095 × 1319 analysis pixels).
2. Subtract Gaussian-smoothed images with sigma 1 and sigma 8 analysis pixels to
   emphasize dark ridges.
3. Sample 720 radial directions around the approximate image-coordinate seed
   (1847, 1872). Search source-image radii 1850–2150 pixels. These bounds select
   the middle rim rather than the inner or outermost boundary.
4. Choose the darkest ridge on each ray. Exclude image-edge samples, weak
   responses (strength ≤3), and radial-search endpoints.
5. Fit a circle with robust `soft_l1` least squares, reject points with residuals
   ≥40 source pixels, and refit. The canonical run retained 481 of 561 points;
   its median absolute radial residual was 11.19 source pixels.
6. Cross-check edge-selection settings and an ellipse fit. The rim is thick and
   partly clipped; allow roughly ±40 source pixels for boundary selection.

The canonical nominal-factor-four fit gives center (1951.4575, 1887.8401) and
radius 1973.4776 pixels. The reported working estimate was rounded to (1952,1888),
radius 1974. Exact pixel-center back-mapping of the quarter-size analysis yields
(1952.9575,1890.4147); this less-than-three-pixel difference is recorded rather
than silently changing the previously reported center and stage coordinates.

Run the following code from the repository root to reproduce the fit. It reads
the saved TIFF and prints results; it does not write files or operate hardware.

```python
from pathlib import Path
import cv2
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

image = tifffile.imread(Path("patcherbot/utils/scan_reference.tif"))
height, width = image.shape[:2]
gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                  (width // 4, height // 4),
                  interpolation=cv2.INTER_AREA).astype(float)
seed = np.array([1847.0, 1872.0]) / 4
angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
detail = (cv2.GaussianBlur(gray, (0, 0), 1)
          - cv2.GaussianBlur(gray, (0, 0), 8))
radii = np.arange(1850 / 4, 2150 / 4, 0.5)
x = seed[0] + np.cos(angles[:, None]) * radii
y = seed[1] + np.sin(angles[:, None]) * radii
response = map_coordinates(detail, [y, x], order=1, mode="constant", cval=0)
valid = ((x > 12) & (x < gray.shape[1] - 12)
         & (y > 12) & (y < gray.shape[0] - 12))
response[~valid] = 0
indices = response.argmin(axis=1)
strength = -response[np.arange(len(angles)), indices]
selected_radius = radii[indices]
good = ((strength > 3) & (indices > 2) & (indices < len(radii) - 3))
points = np.column_stack((
    seed[0] + np.cos(angles[good]) * selected_radius[good],
    seed[1] + np.sin(angles[good]) * selected_radius[good]))

def residual(parameters, selected_points):
    return (np.linalg.norm(selected_points - parameters[:2], axis=1)
            - parameters[2])

initial = least_squares(residual, [*seed, np.median(selected_radius[good])],
                        args=(points,), loss="soft_l1", f_scale=5)
keep = np.abs(residual(initial.x, points)) < 10
final = least_squares(residual, initial.x, args=(points[keep],),
                      loss="soft_l1", f_scale=3)
print("Nominal center X/Y and radius:", final.x * 4)
print("Candidate / retained points:", len(points), keep.sum())
resize_scale = np.array([width / gray.shape[1], height / gray.shape[0]])
print("Exact resize-center mapping:",
      (final.x[:2] + 0.5) * resize_scale - 0.5)
```

## Converting the center to recorded stage microns

The stage calibration maps stage displacement in microns to image pixels. The
stitch uses both coordinate flips, so `A = -M`. A frame's stitched position marks
its **top-left pixel**, not its camera center. The camera frame is 1280 × 1280;
subtract (640,640) before converting the center's displacement into stage motion.

For the adopted downsampled center `p`, map through the actual OpenCV resize:

`p_full = (p + 0.5) * (full_width,full_height) / (saved_width,saved_height) - 0.5`

Then `world = output_origin + p_full * pixel_scale`. For each of the six base
frames covering that point, use its actual rounded tile position, including its
saved image-registration correction. The stage estimate is:

`stage_center_i = recorded_stage_i + solve(A, world - tile_origin_i - (640,640))`

Combine these estimates with the same triangular feather weights used for the
stitched image. The local frame observations and resulting coordinates are saved
explicitly in the JSON. They can reproduce the stage conversion without relying
on an evolving camera-frame directory or importing application controllers.

```python
import json
from pathlib import Path
import numpy as np

record = json.loads(Path("patcherbot/utils/scan_reference.json").read_text(
    encoding="utf-8"))
transform = record["coordinate_transform"]
flips = np.diag([-1 if transform["flip_x"] else 1,
                 -1 if transform["flip_y"] else 1])
A = flips @ np.asarray(transform["calibration_M"])
p = np.asarray(record["center"]["image_xy_pixels"])
full_size = np.asarray(transform["full_shape_yxc"][1::-1])
saved_size = np.asarray(transform["downsampled_shape_yxc"][1::-1])
p_full = (p + 0.5) * full_size / saved_size - 0.5
world = (np.asarray(transform["output_origin_xy_pixels"])
         + p_full * np.asarray(transform["pixel_scale_xy"]))
camera_center = np.asarray(transform["camera_center_xy_pixels"])
positions, weights = [], []
for observation in record["local_frame_observations"]:
    uv = world - np.asarray(observation["tile_origin_world_pixels"])
    position = (np.asarray(observation["stage_xy_um"])
                + np.linalg.solve(A, uv - camera_center))
    positions.append(position)
    weights.append(observation["feather_weight_unnormalized"])
estimate = np.average(positions, axis=0, weights=weights)
np.testing.assert_allclose(estimate, record["center"]["stage_xy_um"], atol=1e-6)
print("Estimated recorded stage X/Y, microns:", estimate)
```

Source movement rows were matched to frame timestamps by nearest time, choosing
the earlier row on ties, as in `ScanStitcher._align_all_frames_to_movement`.
The original stitch indexed 4890 frames; 4926 were present during later analysis.
All files through the six relevant indices (1342–1810) had been written more than
154 seconds before scan end. Their local ordering therefore predates the late
writes; the explicit saved observations preserve the conversion evidence.

These are coordinates in the recorded experiment's stage frame. The image and
rim estimate do not by themselves establish a drift-independent hardware origin.
