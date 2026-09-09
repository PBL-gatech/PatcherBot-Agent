from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from patcherbot.utils.FileLogger import FileLogger


class GraphRecorder:
    """
    Multi-pipette recorder for live electrophysiology summary data.

    Recording strategy
    ------------------
    During acquisition:
        - one queue-backed CSV is written per pipette
        - each CSV is independently owned by one FileLogger
        - rows are deduplicated by acquisition_id when provided

    When recording stops:
        - all CSV writers are flushed
        - one Excel workbook is generated
        - each pipette CSV becomes one workbook sheet

    The per-pipette CSV files are intentionally retained as the durable,
    crash-recoverable source data. The XLSX workbook is a convenient
    post-recording view/export.
    """

    HEADER = (
        "timestamp;"
        "acquisition_id;"
        "pressure;"
        "resistance;"
        "current;"
        "voltage"
    )

    def __init__(
        self,
        recording_state_manager,
        pipette_ids: Optional[Iterable] = None,
        folder_path="experiments/Data/rig_recorder_data/",
        workbook_filename="graph_recording",
        session_id: Optional[str] = None,
    ):
        self.recording_state_manager = recording_state_manager
        self.folder_path = folder_path
        self.workbook_filename = workbook_filename

        if session_id is None:
            session_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        self.session_id = str(session_id)

        # Mirror FileLogger's TEST_ path behavior so the workbook and
        # temporary CSVs end up in the same recording session.
        workbook_base = folder_path
        if bool(getattr(recording_state_manager, "testMode", False)):
            workbook_base = workbook_base.replace(
                "Data/",
                "Data/TEST_",
            )

        self.session_folder = Path(workbook_base) / self.session_id
        self.csv_folder = self.session_folder / "_graph_csv"
        self.workbook_path = (
            self.session_folder
            / f"{self.workbook_filename}.xlsx"
        )

        self._writers = {}
        self._csv_paths = {}
        self._last_acquisition = {}
        self._lock = threading.RLock()
        self._closed = False

        if pipette_ids is not None:
            for pipette_id in pipette_ids:
                self.register_pipette(pipette_id)

    # ------------------------------------------------------------------
    # Pipette registration
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_id(value) -> str:
        text = str(value).strip()
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in text
        )
        return safe or "pipette"

    def register_pipette(self, pipette_id):
        """
        Register one pipette stream.

        Registration is lazy-safe: calling this more than once for the same
        pipette simply returns the existing low-level writer.
        """
        with self._lock:
            self._ensure_open()

            if pipette_id in self._writers:
                return self._writers[pipette_id]

            safe_id = self._safe_id(pipette_id)

            writer = FileLogger(
                self.recording_state_manager,
                folder_path=self.folder_path,
                recorder_filename=safe_id,
                filetype="csv",
                header=self.HEADER,
                session_id=(
                    f"{self.session_id}/_graph_csv"
                ),
            )

            self._writers[pipette_id] = writer
            self._csv_paths[pipette_id] = Path(
                writer.filename
            )
            self._last_acquisition[pipette_id] = None

            logging.info(
                "Registered graph recorder stream for pipette %s",
                pipette_id,
            )

            return writer

    # ------------------------------------------------------------------
    # Sample writing
    # ------------------------------------------------------------------

    @staticmethod
    def _to_scalar(value):
        """
        Convert scalar/array-like graph data to one CSV-safe number.

        Live graph recording is a summary stream, not a raw waveform store.
        For arrays, the finite mean is recorded.
        """
        if value is None:
            return None

        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return value

        if arr.size == 0:
            return None

        if arr.size == 1:
            scalar = float(arr.reshape(-1)[0])
            return scalar if np.isfinite(scalar) else None

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None

        return float(np.mean(finite))

    def record_sample(
        self,
        pipette_id,
        timestamp,
        pressure,
        resistance,
        current,
        voltage,
        *,
        acquisition_id=None,
    ) -> bool:
        """
        Record one live electrophysiology summary sample.

        Returns True only when a new row was actually queued.

        acquisition_id should identify the DAQ acquisition that produced
        this sample. Supplying it prevents the GUI refresh timer from
        recording the same acquisition more than once.
        """
        if not self.recording_state_manager.is_recording_enabled():
            return False

        with self._lock:
            self._ensure_open()

            writer = self._writers.get(pipette_id)
            if writer is None:
                writer = self.register_pipette(pipette_id)

            dedupe_key = (
                acquisition_id
                if acquisition_id is not None
                else timestamp
            )

            if self._last_acquisition.get(pipette_id) == dedupe_key:
                return False

            self._last_acquisition[pipette_id] = dedupe_key

        writer.write_row(
            (
                timestamp,
                acquisition_id,
                self._to_scalar(pressure),
                self._to_scalar(resistance),
                self._to_scalar(current),
                self._to_scalar(voltage),
            )
        )

        return True

    # ------------------------------------------------------------------
    # Workbook export
    # ------------------------------------------------------------------

    @staticmethod
    def _sheet_name(value, used_names) -> str:
        """
        Convert a pipette ID to a legal, unique Excel sheet name.
        """
        name = str(value).strip() or "Pipette"
        name = re.sub(r"[\[\]:*?/\\]", "_", name)
        name = name[:31]

        base = name
        suffix = 1

        while name in used_names:
            suffix_text = f"_{suffix}"
            name = (
                base[: 31 - len(suffix_text)]
                + suffix_text
            )
            suffix += 1

        used_names.add(name)
        return name

    def flush(self):
        """Wait until every pipette CSV writer has finished queued writes."""
        with self._lock:
            writers = list(self._writers.values())

        for writer in writers:
            writer.flush()

    def export_workbook(self):
        """
        Rebuild graph_recording.xlsx from the per-pipette CSV files.

        Returns the workbook Path on success, or None if no graph data exists
        or if export fails. CSV source data is never removed by this method.
        """
        self.flush()

        with self._lock:
            csv_items = list(self._csv_paths.items())

        existing = [
            (pipette_id, path)
            for pipette_id, path in csv_items
            if path.exists() and path.stat().st_size > 0
        ]

        if not existing:
            return None

        try:
            self.session_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            temp_workbook = (
                self.session_folder
                / f".{self.workbook_filename}.tmp.xlsx"
            )

            used_names = set()

            with pd.ExcelWriter(
                temp_workbook,
                engine="openpyxl",
            ) as excel_writer:

                for pipette_id, csv_path in existing:
                    try:
                        frame = pd.read_csv(
                            csv_path,
                            sep=";",
                        )
                    except pd.errors.EmptyDataError:
                        continue

                    sheet_name = self._sheet_name(
                        pipette_id,
                        used_names,
                    )

                    frame.to_excel(
                        excel_writer,
                        sheet_name=sheet_name,
                        index=False,
                    )

            # If every CSV happened to be empty after parsing, the writer
            # may have no useful sheets. openpyxl still creates a workbook,
            # but keeping an empty export is not useful.
            if not used_names:
                try:
                    temp_workbook.unlink()
                except FileNotFoundError:
                    pass
                return None

            os.replace(
                temp_workbook,
                self.workbook_path,
            )

            logging.info(
                "Exported graph workbook: %s",
                self.workbook_path,
            )

            return self.workbook_path

        except PermissionError:
            logging.exception(
                "Could not replace graph workbook. "
                "It may currently be open in Excel: %s",
                self.workbook_path,
            )
            return None

        except Exception:
            logging.exception(
                "Failed exporting graph workbook: %s",
                self.workbook_path,
            )
            return None

    def handle_recording_stopped(self):
        """
        Finalize the current graph recording period.

        Low-level writers stay alive so recording may be started again
        during the same application session.
        """
        self.flush()
        return self.export_workbook()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError(
                "Cannot write to a closed GraphRecorder"
            )

    def close(self):
        """
        Export the final workbook and close all pipette writers exactly once.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            writers = list(self._writers.values())

        # Export while FileLogger workers are still available to flush.
        self.export_workbook()

        for writer in writers:
            writer.close()

        logging.info(
            "Closed GraphRecorder: %s",
            self.workbook_path,
        )
