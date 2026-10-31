"""Measure full-scan runs in fresh Python processes; configure the constants below."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiments" / "Data" / "stitch_benchmarks"
TARGET_SECONDS = 10.0
CASES = [
    ("coverage_feather_1", {"select_frames": True, "preserve_coverage": True, "blend": "feather"}),
    ("coverage_feather_2", {"select_frames": True, "preserve_coverage": True, "blend": "feather"}),
    ("coverage_feather_3", {"select_frames": True, "preserve_coverage": True, "blend": "feather"}),
]
CHILD = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'experiments'))
import benchmark_scan_stitcher as benchmark
job = json.load(sys.stdin)
benchmark.FAST_OPTIONS.update(job['options'])
legacy = benchmark.load_module('fresh_scan_stitcher', 'scanStitcher.py')
fast = benchmark.load_module('scanStitcherFast', 'scanStitcherFast.py')
output = Path(job['output'])
result = benchmark.run_stage(job['name'], legacy.ScanStitcher, fast, output)
(output / (job['name'] + '_timing.json')).write_text(json.dumps(result, indent=2))
"""


def main():
    output = OUTPUT_ROOT / (datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f") + "_fresh")
    output.mkdir(parents=True, exist_ok=False)
    report = {"python": sys.executable, "target_seconds": TARGET_SECONDS,
              "timing_scope": "fresh process launch, imports, initialization, full scan, TIFF/metadata, process exit",
              "cache_note": "Fresh processes; filesystem and CUDA disk caches are not cleared.", "runs": {}}
    for name, options in CASES:
        start = time.perf_counter()
        process = subprocess.run(
            [sys.executable, "-u", "-c", CHILD], cwd=ROOT, text=True,
            input=json.dumps({"name": name, "options": options, "output": str(output)}),
            capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        elapsed = time.perf_counter() - start
        (output / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        result = {"process_seconds": elapsed, "exit_code": process.returncode,
                  "under_target": process.returncode == 0 and elapsed < TARGET_SECONDS}
        if process.returncode == 0:
            result["stage"] = json.loads((output / (name + "_timing.json")).read_text())
        report["runs"][name] = result
        (output / "fresh_process_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{name}: {elapsed:.3f}s, exit={process.returncode}, under_target={result['under_target']}", flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
    print(f"Report: {output / 'fresh_process_validation.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
