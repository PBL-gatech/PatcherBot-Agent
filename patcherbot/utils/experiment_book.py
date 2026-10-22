"""Append-only logger for experiment book sessions."""

from collections.abc import Mapping
from datetime import datetime
import json
import logging
import math
import shutil
from uuid import uuid4
from numbers import Integral
import os
from pathlib import Path


class ExperimentBookLogger:
    """Write experiment details, notes, and snapshot events for one GUI session."""

    DEFAULT_FOLDER_PATH = "experiments/Data/experiment_book_data/"

    def __init__(self, folder_path=None, session_time=None):
        base_path = folder_path or self.DEFAULT_FOLDER_PATH
        self.time_truth = session_time or datetime.now().astimezone()
        self.folder_path = os.path.join(
            str(base_path),
            self.time_truth.strftime("%Y_%m_%d-%H_%M"),
        )
        self.filename = os.path.join(self.folder_path, "experiment_book.log")
        self.session_dir = Path(self.folder_path)
        self.log_path = Path(self.filename)
        self.folder_created = False
        logging.info(
            "ExperimentBookLogger initialized. Folder path set to: %s",
            self.folder_path,
        )

    def create_folder(self):
        """Create the timestamped session folder on the first write."""
        if self.folder_created:
            return
        try:
            os.makedirs(self.folder_path, exist_ok=True)
        except OSError:
            logging.getLogger(__name__).exception(
                "Error creating experiment book folder: %s",
                self.folder_path,
            )
            raise
        self.folder_created = True
        logging.info("Created experiment book folder at: %s", self.folder_path)

    def write_details(self, details, timestamp=None):
        normalized = {
            "experiment_name": self._detail_value(details.get("experiment_name")),
            "strain_culture": self._detail_value(details.get("strain_culture")),
            "gender": self._detail_value(details.get("gender")),
            "age": self._detail_value(details.get("age")),
        }
        entry = "\n".join([
            self._entry_header("DETAILS", timestamp),
            f"Experiment name: {normalized['experiment_name']}",
            f"Strain/Culture: {normalized['strain_culture']}",
            f"Gender: {normalized['gender']}",
            f"Age: {normalized['age']}",
        ])
        self._write_to_file(entry)
        return normalized

    def write_note(self, text, timestamp=None):
        cleaned = str(text).strip()
        if not cleaned:
            raise ValueError("Note text cannot be empty.")
        self._write_to_file(
            f"{self._entry_header('NOTE', timestamp)}\n{cleaned}"
        )
        return cleaned

    def write_snapshot(self, image_path, camera_role, timestamp=None):
        if image_path is None:
            raise ValueError("Snapshot path cannot be empty.")
        raw_path = str(image_path).strip()
        if not raw_path:
            raise ValueError("Snapshot path cannot be empty.")
        cleaned_path = Path(raw_path).as_posix()
        cleaned_role = str(camera_role).strip().lower()
        if cleaned_role not in ("main", "aux"):
            raise ValueError("Snapshot camera role must be 'main' or 'aux'.")
        entry = "\n".join([
            self._entry_header("SNAPSHOT", timestamp),
            f"Camera: {cleaned_role}",
            f"Saved: {cleaned_path}",
        ])
        self._write_to_file(entry)
        return cleaned_path

    def write_atlas(self, image_path, caption, details, timestamp=None):
        """Associate a saved atlas selection with the active experiment."""
        cleaned_path = Path(image_path).as_posix()
        entry = "\n".join([
            self._entry_header("ATLAS", timestamp),
            f"Experiment name: {details['experiment_name']}",
            str(caption),
            f"Saved: {cleaned_path}",
        ])
        self._write_to_file(entry)
        return cleaned_path

    def write_state_tally(self, counts, state_labels, timestamp=None):
        """Append one complete, ordered state-press tally snapshot."""
        if not isinstance(counts, Mapping) or not isinstance(state_labels, Mapping):
            raise TypeError("State tally counts and labels must be mappings.")
        if not state_labels or set(counts) != set(state_labels):
            raise ValueError("State tally counts must match the configured states.")

        normalized = {}
        lines = []
        for state_name, display_label in state_labels.items():
            label = str(display_label).strip()
            count = counts[state_name]
            if not label:
                raise ValueError("State tally labels cannot be empty.")
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise ValueError("State tally counts must be non-negative integers.")
            normalized[state_name] = int(count)
            lines.append(f"{label}: {int(count)}")

        entry = "\n".join([
            self._entry_header("STATE TALLY", timestamp),
            *lines,
        ])
        self._write_to_file(entry)
        return normalized

    def write_origin(self, record):
        """Log an origin whose snapshot was saved by the shared camera recorder."""
        saved = dict(record)
        axis = saved.get("axis")
        origin_id = str(saved.get("origin_id", ""))
        if axis not in ("x", "y"):
            raise ValueError("Origin axis must be x or y.")
        if len(origin_id) != 32 or any(c not in "0123456789abcdef" for c in origin_id):
            raise ValueError("Origin ID must be a UUID hex string.")
        timestamp = datetime.fromisoformat(saved["saved_at"])
        image_path = saved.get("image_path")
        if image_path is None or not str(image_path).strip():
            raise ValueError("Origin snapshot path cannot be empty.")
        saved["image_path"] = Path(str(image_path).strip()).as_posix()
        try:
            metadata = json.dumps(saved, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Origin metadata must be JSON serializable.") from error
        self._write_to_file(
            f"{self._entry_header('ORIGIN ' + axis.upper(), timestamp)}\n{metadata}"
        )
        return saved

    @staticmethod
    def _normalize_tares(tares):
        if not isinstance(tares, Mapping) or set(tares) != {"pipette", "stage"}:
            raise ValueError("Tares must contain pipette and stage coordinates.")
        normalized = {}
        for name in ("pipette", "stage"):
            values = list(tares[name])
            if len(values) != 3:
                raise ValueError("Each tare must contain three coordinates.")
            normalized[name] = []
            for value in values:
                if value is None:
                    normalized[name].append(None)
                elif isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError("Tare coordinates must be finite numbers or unset.")
                else:
                    normalized[name].append(float(value))
        return normalized

    def write_tares(self, tares):
        """Persist display tare coordinates, including individually unset axes."""
        normalized = self._normalize_tares(tares)
        self._write_to_file(self._entry_header("TARES") + "\n" +
                            json.dumps(normalized, allow_nan=False))
        return normalized

    @staticmethod
    def read_details(source_path):
        """Read reusable details without changing a source book or live hardware."""
        source = Path(source_path).expanduser().resolve()
        if source.is_dir():
            source = source / "experiment_book.log"
        sections = []
        kind, body, saved_at = None, [], None
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            header = None
            if line.startswith("[") and "] " in line:
                stamp, candidate = line[1:].split("] ", 1)
                try:
                    datetime.fromisoformat(stamp)
                except ValueError:
                    pass
                else:
                    header = (candidate, stamp)
            if header is not None:
                if kind is not None:
                    sections.append((kind, saved_at, body))
                kind, saved_at = header
                body = []
            elif kind is not None:
                body.append(line)
        if kind is not None:
            sections.append((kind, saved_at, body))
        prefixes = {"Experiment name: ": "experiment_name", "Strain/Culture: ": "strain_culture",
                    "Gender: ": "gender", "Age: ": "age"}
        details, tares = None, None
        details_saved_at = None
        source_started_at = sections[0][1] if sections else None
        origins = {"x": None, "y": None}
        notes = []
        atlases = []
        for kind, saved_at, body in sections:
            if kind == "IMPORT":
                origins, tares = {"x": None, "y": None}, None
            elif kind == "DETAILS":
                details_saved_at = saved_at
                details = {key: line[len(prefix):] for line in body
                           for prefix, key in prefixes.items() if line.startswith(prefix)}
            elif kind in ("ORIGIN X", "ORIGIN Y"):
                origins[kind[-1].lower()] = json.loads("\n".join(body))
            elif kind == "TARES":
                tares = json.loads("\n".join(body))
            elif kind == "ATLAS":
                lines = list(body)
                while lines and not lines[-1].strip():
                    lines.pop()
                if (len(lines) < 3 or not lines[0].startswith("Experiment name: ") or
                        not lines[-1].startswith("Saved: ")):
                    raise ValueError("Incomplete saved atlas entry.")
                atlases.append(dict(experiment_name=lines[0][len("Experiment name: "):],
                                    caption="\n".join(lines[1:-1]),
                                    image_path=lines[-1][len("Saved: "):].strip(), saved_at=saved_at))
            elif kind == "ATLAS IMPORT":
                atlases.append(json.loads("\n".join(body)))
            elif kind == "NOTE":
                text = "\n".join(body).strip()
                if text:
                    notes.append(dict(text=text, saved_at=saved_at))
        if details is None or set(details) != set(prefixes.values()) or not details["experiment_name"].strip():
            raise ValueError("The source book has no complete experiment details.")
        if tares is not None:
            tares = ExperimentBookLogger._normalize_tares(tares)
        for axis, record in origins.items():
            if record is None:
                continue
            if not isinstance(record, dict) or record.get("axis") != axis:
                raise ValueError("Invalid saved " + axis.upper() + " origin.")
            identifier = str(record.get("origin_id", ""))
            if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
                raise ValueError("Invalid saved origin ID.")
            datetime.fromisoformat(record["saved_at"])
            datetime.fromisoformat(record["captured_at"])
            xyz = record["stage_xyz_um"]
            value = record["origins_um"][axis]
            if (not isinstance(xyz, list) or len(xyz) != 3 or
                    any(isinstance(v, bool) or not math.isfinite(float(v)) for v in xyz) or
                    isinstance(value, bool) or not math.isfinite(float(value))):
                raise ValueError("Saved origin coordinates must be finite.")
            record["stage_xyz_um"] = [float(value) for value in xyz]
            normalized_origins = {}
            for name in ("x", "y"):
                coordinate = record["origins_um"].get(name)
                if coordinate is not None:
                    if isinstance(coordinate, bool) or not math.isfinite(float(coordinate)):
                        raise ValueError("Saved origin coordinates must be finite.")
                    coordinate = float(coordinate)
                normalized_origins[name] = coordinate
            record["origins_um"] = normalized_origins
            if (record.get("camera_role") != "main" or
                    not isinstance(record.get("experiment"), dict) or
                    not isinstance(record.get("session_id"), str) or
                    not isinstance(record.get("frame_number"), int) or
                    isinstance(record.get("frame_number"), bool)):
                raise ValueError("Invalid origin capture metadata.")
            raw_image = record.get("image_path")
            if not isinstance(raw_image, str) or not raw_image.strip():
                raise ValueError("Origin image path is missing.")
            image = Path(raw_image)
            candidates = [image] if image.is_absolute() else [source.parent / image, Path.cwd() / image]
            resolved = next((item.resolve() for item in candidates if item.is_file()), None)
            if resolved is None:
                raise ValueError("Origin image is missing: " + raw_image)
            record["image_path"] = str(resolved)
            json.dumps(record, allow_nan=False)
        for atlas in atlases:
            if (not isinstance(atlas, dict) or
                    any(not isinstance(atlas.get(key), str) for key in
                        ("image_path", "caption", "saved_at", "experiment_name")) or
                    not atlas["image_path"].strip()):
                raise ValueError("Invalid saved atlas metadata.")
            datetime.fromisoformat(atlas["saved_at"])
            image = Path(atlas["image_path"])
            candidates = [image] if image.is_absolute() else [source.parent / image, Path.cwd() / image]
            resolved = next((item.resolve() for item in candidates if item.is_file()), None)
            if resolved is None:
                raise ValueError("Atlas image is missing: " + atlas["image_path"])
            atlas.setdefault("source_book", str(source))
            atlas.setdefault("source_image_path", str(resolved))
            atlas["image_path"] = str(resolved)
        return dict(details=details, origins=origins, tares=tares, notes=notes,
                    atlases=atlases, source_path=str(source),
                    source_started_at=source_started_at, details_saved_at=details_saved_at)

    def import_details(self, source_path, tares=None):
        """Copy reusable evidence into this book while keeping its active destination."""
        payload = self.read_details(source_path)
        if tares is not None:
            payload["tares"] = self._normalize_tares(tares)
        source = Path(payload["source_path"])
        if source == self.log_path.resolve():
            raise ValueError("Choose a different experiment book to import.")
        self.create_folder()
        imported = self.session_dir.resolve() / "imports" / uuid4().hex
        imported.mkdir(parents=True)
        shutil.copy2(source, imported / "experiment_book.log")
        for axis, record in payload["origins"].items():
            if record is None:
                continue
            image = Path(record["image_path"])
            copied = imported / (axis + "_origin" + image.suffix)
            shutil.copy2(image, copied)
            record.setdefault("source_origin_id", record["origin_id"])
            record.setdefault("source_book", str(source))
            record.setdefault("source_image_path", str(image))
            record["image_path"] = str(copied)
        for index, atlas in enumerate(payload["atlases"]):
            image = Path(atlas["image_path"])
            copied = imported / ("atlas_" + str(index + 1) + image.suffix)
            shutil.copy2(image, copied)
            atlas["image_path"] = str(copied)
        reference = dict(source_book=str(source), source_copy=str(imported / "experiment_book.log"),
                         imported_at=self._timestamp(),
                         source_started_at=payload["source_started_at"],
                         details_saved_at=payload["details_saved_at"])
        entries = [self._entry_header("IMPORT") + "\n" + json.dumps(reference)]
        details = payload["details"]
        entries.append("\n".join([self._entry_header("DETAILS"),
            "Experiment name: " + details["experiment_name"],
            "Strain/Culture: " + details["strain_culture"],
            "Gender: " + details["gender"], "Age: " + details["age"]]))
        for axis, record in payload["origins"].items():
            if record is not None:
                entries.append(self._entry_header("ORIGIN " + axis.upper(),
                    datetime.fromisoformat(record["saved_at"])) + "\n" + json.dumps(record, allow_nan=False))
        for note in payload["notes"]:
            entries.append(self._entry_header("NOTE", datetime.fromisoformat(note["saved_at"])) +
                           "\n" + note["text"])
        for atlas in payload["atlases"]:
            entries.append(self._entry_header("ATLAS IMPORT", datetime.fromisoformat(atlas["saved_at"])) +
                           "\n" + json.dumps(atlas, allow_nan=False))
        entries.append(self._entry_header("TARES") + "\n" + json.dumps(payload["tares"], allow_nan=False))
        self._write_to_file("\n\n".join(entries))
        return payload

    def _write_to_file(self, entry):
        self.create_folder()
        with open(self.filename, "a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(entry.rstrip())
            log_file.write("\n\n")

    @staticmethod
    def _detail_value(value):
        cleaned = "" if value is None else str(value).strip()
        return cleaned or "N/A"

    @staticmethod
    def _timestamp(timestamp=None):
        value = timestamp if isinstance(timestamp, datetime) else datetime.now().astimezone()
        if value.tzinfo is None:
            value = value.astimezone()
        return value.isoformat(timespec="seconds")

    def _entry_header(self, entry_type, timestamp=None):
        return f"[{self._timestamp(timestamp)}] {entry_type}"
