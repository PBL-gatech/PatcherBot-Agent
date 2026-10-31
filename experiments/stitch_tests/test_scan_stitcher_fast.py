"""Small image fixtures that verify fast stitching geometry and CPU/GPU parity."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "patcherbot" / "utils"
sys.path.insert(0, str(UTILS))
from scanStitcher import ScanStitcher
from scanStitcherFast import (
    FastStitchConfig, preserve_source_coverage, select_keyframes, stitch_fast,
)


@pytest.fixture(params=["numpy", "cupy"])
def backend(request):
    if request.param == "cupy":
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy is not installed")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device is available")
        # A present GPU with a broken CUDA setup must fail, not silently skip.
        cp.zeros(1).sum().item()
        cp.cuda.Stream.null.synchronize()
    return request.param


def make_session(tmp_path, positions, colors, shape=(4, 4)):
    frames = tmp_path / "camera_frames"
    frames.mkdir()
    rows = ["timestamp;st_x;st_y"]
    for index, (position, color) in enumerate(zip(positions, colors)):
        timestamp = 1000 + index
        rows.append(f"{timestamp};{position[0]};{position[1]}")
        rgb = np.empty((*shape, 3), dtype=np.uint8)
        rgb[:] = color
        assert cv2.imwrite(str(frames / f"{index}_{timestamp}.png"), rgb[:, :, ::-1])
    (tmp_path / "movement_recording.csv").write_text("\n".join(rows), encoding="utf-8")
    return ScanStitcher(
        date_time_folder=str(tmp_path), use_calibration=False,
        flip_x=False, flip_y=False, prefer_exts=(".png",), downsample=1,
    )


def run_image(stitcher, backend, **options):
    config = FastStitchConfig(backend=backend, select_frames=False, **options)
    output = stitch_fast(stitcher, output_path="result.tif", config=config)
    assert isinstance(output, str)
    assert Path(output).is_file()
    return tifffile.imread(output)


def coverage_mask(positions, shape):
    """Independent exact integer-grid coverage for these small fixtures."""
    positions = np.asarray(positions, dtype=int)
    lower = positions.min(axis=0)
    extent = positions.max(axis=0) - lower + np.array(shape[::-1])
    mask = np.zeros(tuple(extent[::-1]), dtype=bool)
    for x, y in positions - lower:
        mask[y:y + shape[0], x:x + shape[1]] = True
    return mask


def test_selection_preserves_first_turn_and_complete_serpentine_coverage():
    positions = np.array(
        [(x, 0) for x in range(21)]
        + [(20, y) for y in range(1, 7)]
        + [(x, 6) for x in range(19, -1, -1)], dtype=float,
    )
    selected = select_keyframes(positions, (10, 10), overlap=0.4)
    assert len(selected) < len(positions)
    assert {0, 20, 26, len(positions) - 1}.issubset(set(selected))
    np.testing.assert_array_equal(
        coverage_mask(positions[selected], (10, 10)),
        coverage_mask(positions, (10, 10)),
    )


def test_selection_deduplicates_pauses_without_losing_slow_accumulated_motion():
    positions = np.column_stack((np.linspace(0, 20, 4001), np.zeros(4001)))
    positions = np.repeat(positions, 2, axis=0)
    selected = select_keyframes(positions, (10, 10), overlap=0.4)
    assert selected[0] == 0
    assert positions[selected[-1], 0] == 20
    assert len(selected) < 10
    assert np.max(np.diff(positions[selected, 0])) <= 6 + 1e-9
    np.testing.assert_array_equal(select_keyframes(np.zeros((20, 2)), (10, 10)), [0])


def test_selection_does_not_invent_frames_across_sparse_gaps():
    positions = np.array([[0, 0], [30, 0], [60, 0]], dtype=float)
    selected = select_keyframes(positions, (10, 10), overlap=0.4)
    np.testing.assert_array_equal(selected, [0, 1, 2])
    assert not coverage_mask(positions[selected], (10, 10))[:, 10:30].any()


def test_coverage_repair_restores_diagonal_boundary_slivers_exactly():
    positions = np.array([(i - 10, i + 20) for i in range(13)], dtype=int)
    base = select_keyframes(positions, (8, 8), overlap=0.4)
    source_mask = coverage_mask(positions, (8, 8))
    base_mask = coverage_mask(positions[base], (8, 8))
    assert np.any(source_mask & ~base_mask), "Fixture must expose boundary loss"
    repaired, report = preserve_source_coverage(positions, (8, 8), base)
    np.testing.assert_array_equal(positions[repaired].min(axis=0), positions.min(axis=0))
    np.testing.assert_array_equal(positions[repaired].max(axis=0), positions.max(axis=0))
    np.testing.assert_array_equal(coverage_mask(positions[repaired], (8, 8)), source_mask)
    assert not source_mask.all(), "Unacquired bounding-box corners must remain empty"
    assert set(base).issubset(set(repaired))
    assert report["missing_pixels_before"] == np.count_nonzero(source_mask & ~base_mask)
    assert report["missing_pixels_after"] == 0
    assert report["source_covered_pixels"] == np.count_nonzero(source_mask)


def test_coverage_repair_avoids_redundant_paused_frames_without_filling_source_holes():
    positions = np.array([(x, 0) for x in (0, 0, 1, 1, 2, 2, 30, 30, 31, 31)], dtype=int)
    base = np.array([0, 6], dtype=int)
    repaired, report = preserve_source_coverage(positions, (6, 6), base)
    source_mask = coverage_mask(positions, (6, 6))
    np.testing.assert_array_equal(coverage_mask(positions[repaired], (6, 6)), source_mask)
    assert len(repaired) < len(positions)
    assert len(np.unique(positions[repaired], axis=0)) == len(repaired)
    assert not source_mask[:, 8:30].any()
    assert report["missing_pixels_after"] == 0


def test_coverage_repair_respects_its_memory_limit_before_large_grid_allocation():
    with pytest.raises(MemoryError):
        preserve_source_coverage(np.array([[0, 0], [1000, 1000]]), (8, 8), [0], max_pixels=100)


@pytest.mark.parametrize("positions, shape, base", [
    ([[0, 0], [1.5, 0]], (8, 8), [0]),
    ([[0, 0]], (8, 8), [1]),
    ([[0, 0]], (8, 8), [-1]),
    ([[0, 0]], (0, 8), [0]),
])
def test_coverage_repair_rejects_invalid_render_geometry(positions, shape, base):
    with pytest.raises(ValueError):
        preserve_source_coverage(positions, shape, base)


def test_overwrite_preserves_pixel_geometry_and_rgb_channels(tmp_path, backend):
    first, second = [20, 60, 100], [100, 140, 180]
    stitcher = make_session(tmp_path, [(-2, 5), (0, 5)], [first, second])
    actual = run_image(stitcher, backend, blend="overwrite")
    expected = np.empty((4, 6, 3), dtype=np.uint8)
    expected[:, :2] = first
    expected[:, 2:] = second
    np.testing.assert_array_equal(actual, expected)


def test_feather_averages_coincident_frames_without_dark_edges(tmp_path, backend):
    stitcher = make_session(tmp_path, [(0, 0), (0, 0)], [[20, 60, 100], [100, 140, 180]])
    actual = run_image(stitcher, backend, blend="feather")
    expected = np.empty((4, 4, 3), dtype=np.uint8)
    expected[:] = [60, 100, 140]
    np.testing.assert_array_equal(actual, expected)


def test_explicit_all_frame_processing_keeps_pauses_and_feathers_overlap(tmp_path):
    # A stationary revisit must remain eligible: spatial selection would drop it.
    stitcher = make_session(tmp_path, [(0, 0), (0, 0)],
                            [[20, 60, 100], [100, 140, 180]])
    output = stitcher.stitch_fast(output_path="defaults.tif",
                                 config=FastStitchConfig(backend="numpy", select_frames=False))
    expected = np.empty((4, 4, 3), dtype=np.uint8)
    expected[:] = [60, 100, 140]
    np.testing.assert_array_equal(tifffile.imread(output), expected)
    assert stitcher._fast_stitch_report["selected_frame_count"] == 2


def test_default_selection_preserves_exact_source_footprints_in_written_image(tmp_path):
    positions = np.repeat(np.array([(i - 10, i + 20) for i in range(13)]), 2, axis=0)
    color = [50, 90, 130]
    stitcher = make_session(tmp_path, positions, [color] * len(positions), shape=(8, 8))
    output = stitcher.stitch_fast(output_path="coverage.tif",
                                 config=FastStitchConfig(backend="numpy"))
    actual = tifffile.imread(output)
    source_mask = coverage_mask(positions, (8, 8))
    expected = np.zeros((*source_mask.shape, 3), dtype=np.uint8)
    expected[source_mask] = color
    np.testing.assert_array_equal(actual, expected)
    assert not source_mask.all()
    assert stitcher._fast_stitch_report["selected_frame_count"] < len(positions)
    assert stitcher._fast_stitch_report["source_frame_count"] == len(positions)


def test_coverage_repair_preserves_every_original_feather_pixel(tmp_path, backend):
    positions = np.array([(i - 10, i + 20) for i in range(13)])
    colors = [[20 + 10 * i, 50 + 7 * i, 200 - 8 * i] for i in range(13)]
    stitcher = make_session(tmp_path, positions, colors, shape=(8, 8))
    base = select_keyframes(positions, (8, 8), overlap=0.4)
    base_mask = coverage_mask(positions[base], (8, 8))
    source_mask = coverage_mask(positions, (8, 8))
    assert np.any(source_mask & ~base_mask)
    options = dict(backend=backend, select_frames=True, blend="feather")
    baseline_path = stitcher.stitch_fast(
        output_path="base_feather.tif",
        config=FastStitchConfig(**options, preserve_coverage=False),
    )
    baseline = tifffile.imread(baseline_path)
    repaired_path = stitcher.stitch_fast(
        output_path="repaired_feather.tif",
        config=FastStitchConfig(**options, preserve_coverage=True),
    )
    repaired = tifffile.imread(repaired_path)
    # Differing frame brightness exposes reweighting bands hidden by uniform fixtures.
    np.testing.assert_array_equal(repaired[base_mask], baseline[base_mask])
    np.testing.assert_array_equal(np.any(repaired != 0, axis=2), source_mask)
    np.testing.assert_array_equal(repaired.shape, baseline.shape)


def test_sparse_mosaic_preserves_empty_space_instead_of_shrinking_positions(tmp_path, backend):
    stitcher = make_session(tmp_path, [(-20, -7), (20, 9)], [[17, 43, 89], [101, 151, 201]])
    actual = run_image(stitcher, backend, blend="overwrite")
    expected = np.zeros((20, 44, 3), dtype=np.uint8)
    expected[:4, :4] = [17, 43, 89]
    expected[16:20, 40:44] = [101, 151, 201]
    np.testing.assert_array_equal(actual, expected)


def test_downsampling_odd_sized_tiles_scales_positions_with_actual_resize(tmp_path, backend):
    stitcher = make_session(tmp_path, [(0, 0), (5, 0)],
                            [[20, 60, 100], [100, 140, 180]], shape=(5, 5))
    actual = run_image(stitcher, backend, blend="overwrite", output_downsample=2)
    expected = np.empty((3, 6, 3), dtype=np.uint8)
    expected[:, :3] = [20, 60, 100]
    expected[:, 3:] = [100, 140, 180]
    np.testing.assert_array_equal(actual, expected)


def test_output_memory_limit_rejects_geometry_without_writing(tmp_path):
    stitcher = make_session(tmp_path, [(0, 0), (10000, 10000)], [[1, 2, 3], [4, 5, 6]])
    original = stitcher.coord_data.copy()
    config = FastStitchConfig(backend="numpy", select_frames=False, max_output_gb=0.000001)
    with pytest.raises(MemoryError):
        stitch_fast(stitcher, output_path="must_not_exist.tif", config=config)
    assert not (tmp_path / "must_not_exist.tif").exists()
    np.testing.assert_array_equal(stitcher.coord_data, original)


def test_frame_limit_excludes_later_frames_before_spatial_selection(tmp_path):
    stitcher = make_session(tmp_path, [(0, 0), (2, 0), (20, 0)],
                            [[20, 60, 100], [100, 140, 180], [200, 220, 240]])
    config = FastStitchConfig(backend="numpy", select_frames=True, blend="overwrite")
    output = stitch_fast(stitcher, output_path="limited.tif", config=config, frame_limit=2)
    expected = np.empty((4, 6, 3), dtype=np.uint8)
    expected[:, :2] = [20, 60, 100]
    expected[:, 2:] = [100, 140, 180]
    np.testing.assert_array_equal(tifffile.imread(output), expected)
    assert stitcher._fast_stitch_report["source_frame_count"] == 3
    assert stitcher._fast_stitch_report["eligible_frame_count"] == 2


@pytest.mark.parametrize("options", [
    {"overlap": 1}, {"overlap": -0.1}, {"workers": 0},
    {"output_downsample": 0}, {"batch_size": 0},
    {"max_output_gb": 0}, {"max_gpu_gb": 0},
    {"backend": "unknown"}, {"blend": "unknown"},
])
def test_invalid_configuration_rejected_before_output(tmp_path, options):
    stitcher = make_session(tmp_path, [(0, 0)], [[1, 2, 3]])
    with pytest.raises((ValueError, TypeError)):
        config = FastStitchConfig(**dict({"backend": "numpy"}, **options))
        stitch_fast(stitcher, output_path="invalid.tif", config=config)
    assert not (tmp_path / "invalid.tif").exists()
