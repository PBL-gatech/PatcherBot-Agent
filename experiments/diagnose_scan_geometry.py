"""CPU-only stage/image motion diagnostics; edit constants, no CLI required."""
from pathlib import Path
import json
import runpy
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/Data/stitch_benchmarks/2026_10_30-13_39_05_815409/geometry_diagnostic"
PAIRS = [(i, i + 20) for i in range(60, 741, 80)] + [(770, 790), (785, 815), (800, 840), (810, 850), (830, 870), (860, 900), (940, 980), (1100, 1140)]
DOWNSAMPLE = 4


def correlate(a, b):
    size = (a.shape[1] // DOWNSAMPLE, a.shape[0] // DOWNSAMPLE)
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(b, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a, b, window)
    return np.array(shift) * DOWNSAMPLE, float(response)


def predicted_overlap(a, b, delta):
    dx, dy = np.rint(delta).astype(int)
    h, w = a.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        raise ValueError("No predicted overlap")
    return (a[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)],
            b[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    setup = runpy.run_path(str(ROOT / "experiments/benchmark_scan_stitcher.py"), run_name="geometry_setup")
    legacy = setup["load_module"]("geometry_legacy", "scanStitcher.py")
    source = setup["new_stitcher"](legacy.ScanStitcher)
    timestamps, raw, paths = source._align_all_frames_to_movement()
    xy = source._aligned_coords_from_raw(raw)
    cache = {}

    def image(index):
        if index not in cache:
            cache[index] = cv2.imread(paths[index], cv2.IMREAD_GRAYSCALE)
        return cache[index]

    records = []
    panels = []
    for i, j in PAIRS:
        a, b = image(i), image(j)
        expected = xy[j] - xy[i]
        image_shift, full_response = correlate(a, b)
        overlap_a, overlap_b = predicted_overlap(a, b, expected)
        residual, overlap_response = correlate(overlap_a, overlap_b)
        measured = np.rint(expected) - residual
        records.append({
            "i": i, "j": j, "seconds": float(timestamps[j] - timestamps[i]),
            "expected_tile_delta_xy": expected.tolist(),
            "full_image_shift_xy": image_shift.tolist(), "full_response": full_response,
            "overlap_residual_image_shift_xy": residual.tolist(), "overlap_response": overlap_response,
            "measured_tile_delta_xy": measured.tolist(),
            "tile_delta_error_pixels": float(np.linalg.norm(measured - expected)),
            "overlap_mean_intensity_a_b": [float(overlap_a.mean()), float(overlap_b.mean())],
        })
        if (i, j) in ((60, 80), (700, 720), (800, 840), (860, 900)):
            size = (480, round(overlap_a.shape[0] * 480 / overlap_a.shape[1]))
            left, right = [cv2.resize(item, size, interpolation=cv2.INTER_AREA) for item in (overlap_a, overlap_b)]
            pair = cv2.cvtColor(np.hstack((left, right)), cv2.COLOR_GRAY2BGR)
            banner = np.full((52, pair.shape[1], 3), 28, np.uint8)
            label = f"Frames {i}/{j} | expected XY={expected.round(1)} | residual={residual.round(1)} | response={overlap_response:.2f}"
            cv2.putText(banner, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(banner, "Same world overlap according to stage coordinates; uncorrected image pixels", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .46, (200, 200, 200), 1, cv2.LINE_AA)
            panels.append(np.vstack((banner, pair)))
    accepted = [record for record in records if record["overlap_response"] >= .25]
    diagnostics = {"pairs": records, "accepted_pair_count": len(accepted), "response_threshold": .25,
                   "convention": "Image content displacement should be the negative of tile/world translation. Residuals are measured on stage-predicted overlap crops."}
    if len(accepted) >= 4:
        expected = np.array([item["expected_tile_delta_xy"] for item in accepted])
        measured = np.array([item["measured_tile_delta_xy"] for item in accepted])
        correction = np.linalg.lstsq(expected, measured, rcond=None)[0]
        diagnostics["measured_from_expected_row_vector_matrix"] = correction.tolist()
        diagnostics["median_translation_error_pixels"] = float(np.median(np.linalg.norm(measured - expected, axis=1)))
        diagnostics["median_error_after_linear_fit_pixels"] = float(np.median(np.linalg.norm(measured - expected @ correction, axis=1)))
        movement = source.coord_data[np.argsort(source.coord_data[:, 0])]
        lag_results = []
        for lag in np.arange(-1.0, 1.001, .02):
            frame_ids = np.array([(item["i"], item["j"]) for item in accepted])
            times = timestamps[frame_ids] + lag
            interpolated = np.column_stack([np.interp(times.ravel(), movement[:, 0], movement[:, column]) for column in source.coord_cols])
            mapped = source._aligned_coords_from_raw(interpolated).reshape(-1, 2, 2)
            predicted = mapped[:, 1] - mapped[:, 0]
            lag_results.append((float(lag), float(np.median(np.linalg.norm(predicted - measured, axis=1)))))
        best = min(lag_results, key=lambda item: item[1])
        diagnostics["best_timestamp_shift_seconds"] = best[0]
        diagnostics["best_timestamp_shift_median_error_pixels"] = best[1]
        diagnostics["timestamp_shift_caveat"] = "Exploratory coarse lag scan; motion calibration and image-correlation error are confounded. Not a validated correction."
    cv2.imwrite(str(OUTPUT / "predicted_overlap_pairs.png"), np.vstack(panels))
    (OUTPUT / "geometry.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)
    return diagnostics


if __name__ == "__main__":
    main()
