"""Append-only logger for experiment book sessions."""

from datetime import datetime
import logging
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
