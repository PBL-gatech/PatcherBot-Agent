"""
Simple Pickle Extractor (no argparse)
-------------------------------------
- Loads one or more pickle files
- Prints a quick summary to stdout
- Exports a JSON preview for *any* Python object
- If the object looks tabular, also exports CSV

How to use
----------
1) Set `PICKLE_PATH` below to your main pickle.
2) (Optional) Add paths to `ADDITIONAL_PICKLES` to process more.
3) Run: `python simple_pickle_extractor.py`

Outputs are written next to each input file (same base name, .json and maybe .csv).
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from typing import Any, Iterable, List

# --- Configuration -----------------------------------------------------------
# Main pickle to process
PICKLE_PATH: str = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\calibration_data\2026_03_06-15_38\calibration.pickle"


# Add more pickle files here if you want to process several at once
ADDITIONAL_PICKLES: List[str] = [
    # Example: "/path/to/another_file.pickle",
]

# Optional deps (used if available for nicer handling of arrays/dataframes)
try:  # numpy is optional
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pandas is optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# --- Core utilities ----------------------------------------------------------

def load_pickle(path: str) -> Any:
    """Load a pickle file, with a fallback encoding for older pickles."""
    with open(path, "rb") as f:  # type: ignore[call-arg]
        try:
            return pickle.load(f)
        except Exception as e1:
            f.seek(0)
            try:
                return pickle.load(f, encoding="latin1")
            except Exception as e2:  # pragma: no cover
                raise RuntimeError(
                    "Failed to load pickle with default and latin1 encodings.\n"
                    f"Default error: {e1}\nlatin1 error: {e2}"
                )


def summarize(obj: Any) -> str:
    """Human-friendly one-line summary of an object."""
    try:
        if pd is not None and isinstance(obj, pd.DataFrame):
            return f"pandas.DataFrame shape={obj.shape} columns={list(obj.columns)[:10]}{'...' if obj.shape[1] > 10 else ''}"
        if np is not None and isinstance(obj, np.ndarray):
            return f"numpy.ndarray shape={obj.shape} dtype={obj.dtype}"
        if isinstance(obj, dict):
            return f"dict with {len(obj)} keys: {list(obj.keys())[:10]}{'...' if len(obj) > 10 else ''}"
        if isinstance(obj, list):
            t = type(obj[0]).__name__ if obj else 'N/A'
            return f"list (len={len(obj)}) first_item_type={t}"
        if isinstance(obj, tuple):
            return f"tuple (len={len(obj)})"
        if isinstance(obj, set):
            return f"set (len={len(obj)})"
        return type(obj).__name__
    except Exception as e:  # pragma: no cover
        return f"{type(obj).__name__} (summary error: {e})"


def is_tabular(obj: Any) -> bool:
    """Heuristic: object has a table-like shape suitable for CSV export."""
    if pd is not None and isinstance(obj, pd.DataFrame):
        return True
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return True  # list of dicts
    if isinstance(obj, dict):
        values = list(obj.values())
        if values and all(hasattr(v, "__len__") for v in values):
            lengths = [len(v) for v in values if hasattr(v, "__len__")]
            if lengths and len(set(lengths)) == 1:
                return True  # dict of equally-long lists
    return False


def to_jsonable(obj: Any, *, max_items: int = 2000, _seen: set[int] | None = None) -> Any:
    """Convert arbitrary objects to a JSON-serializable preview, with size guard."""
    # Simple primitives bypass recursion tracking
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "<recursion>"
    _seen.add(oid)

    # numpy
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            if obj.size > max_items:
                flat = obj.ravel()[:max_items].tolist()
                return {
                    "_type": "ndarray",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "preview_count": len(flat),
                    "preview": flat,
                }
            return obj.tolist()

    # pandas
    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            data = obj.head(max_items).to_dict()
            return {
                "_type": "Series",
                "index_name": obj.index.name,
                "name": obj.name,
                "length": int(obj.shape[0]),
                "preview": data,
            }
        if isinstance(obj, pd.DataFrame):
            rows = min(len(obj), max_items)
            return {
                "_type": "DataFrame",
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "columns": obj.columns.tolist(),
                "preview_rows": int(rows),
                "preview": obj.head(rows).to_dict(orient="records"),
            }

    # Collections
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                out["_truncated"] = True
                break
            if not isinstance(k, (str, int, float, bool)):
                k = repr(k)
            out[str(k)] = to_jsonable(v, max_items=max_items, _seen=_seen)
        return out

    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        truncated = len(seq) > max_items
        if truncated:
            seq = seq[:max_items]
        converted = [to_jsonable(x, max_items=max_items, _seen=_seen) for x in seq]
        if truncated:
            converted.append({"_truncated": True, "_note": f"Only first {max_items} items included"})
        return converted if not isinstance(obj, tuple) else {"_type": "tuple", "items": converted}

    # Fallback
    try:
        return repr(obj)
    except Exception:  # pragma: no cover
        return f"<unserializable {type(obj).__name__}>"


# --- Export helpers ----------------------------------------------------------

def write_json(obj: Any, out_json: str) -> None:
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv_from_tabular(obj: Any, out_csv: str) -> None:
    if pd is not None and isinstance(obj, pd.DataFrame):
        obj.to_csv(out_csv, index=False)
        return
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        # List of dicts
        keys = sorted({k for row in obj for k in row.keys()})
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in obj:
                w.writerow(row)
        return
    if isinstance(obj, dict):
        # Dict of equal-length lists -> rows
        keys = list(obj.keys())
        values = list(obj.values())
        if values and all(hasattr(v, "__len__") for v in values):
            lengths = [len(v) for v in values if hasattr(v, "__len__")]
            if lengths and len(set(lengths)) == 1:
                L = lengths[0]
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    for i in range(L):
                        row = {}
                        for k in keys:
                            val = obj[k][i]
                            if np is not None and isinstance(val, np.generic):
                                val = val.item()
                            row[str(k)] = val
                        w.writerow(row)
                return
    # If we reach here, we couldn't serialize to CSV
    raise ValueError("Object is not in a CSV-friendly tabular format")


def process_pickle(path: str) -> None:
    print(f"\n=== Processing: {path} ===")
    data = load_pickle(path)
    print("Summary:", summarize(data))

    base, _ = os.path.splitext(path)
    json_path = base + ".json"
    csv_path = base + ".csv"

    # JSON preview export
    preview = to_jsonable(data, max_items=2000)
    write_json(preview, json_path)
    print(f"JSON preview -> {json_path}")

    # CSV export when applicable
    if is_tabular(data):
        try:
            write_csv_from_tabular(data, csv_path)
            print(f"Tabular data detected. CSV -> {csv_path}")
        except Exception as e:
            print(f"Tabular export skipped: {e}")
    else:
        print("Not tabular; CSV export skipped.")


# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    # Process the main pickle
    if PICKLE_PATH and os.path.exists(PICKLE_PATH):
        process_pickle(PICKLE_PATH)
    else:
        print("No valid PICKLE_PATH set or file does not exist. Set PICKLE_PATH at the top of this file.")

    # Optionally process more pickles
    if ADDITIONAL_PICKLES:
        for p in ADDITIONAL_PICKLES:
            if p and os.path.exists(p):
                process_pickle(p)
            else:
                print(f"Skipping missing file: {p}")
