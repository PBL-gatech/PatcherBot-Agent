"""Standalone directory-conversion UI. Run directly; no arguments required."""

import os
from pathlib import Path
import sys
from typing import NamedTuple, Optional

import h5py
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

if __package__:
    from .HDF5Converter import HDF5Converter
else:
    from HDF5Converter import HDF5Converter

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConversionJob(NamedTuple):
    source: Path
    destination: Path
    cell_id: Optional[int] = None


class ConversionCancelled(RuntimeError):
    pass


def discover_jobs(root, mode="auto", restore=False, cancelled=None):
    """Find recordings or cells grouped by exact stage XYZ within each session."""
    if root is None or not str(root).strip():
        raise ValueError("Choose a directory")
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Select a directory")
    if mode not in ("auto", "recordings", "protocols"):
        raise ValueError("Unknown directory layout")
    converter = HDF5Converter()
    if not restore and mode != "recordings" and converter.is_patch_session(root.parent):
        root = root.parent
    jobs = []
    for directory, subdirectories, filenames in os.walk(root):
        if cancelled is not None and cancelled():
            raise ConversionCancelled("Conversion cancelled")
        folder = Path(directory)
        subdirectories[:] = sorted(name for name in subdirectories
                                   if not name.startswith(".") and not (folder / name).is_symlink())
        if not restore and mode != "recordings" and converter.is_patch_session(folder):
            for cell_id in sorted(converter.cell_groups(folder)):
                jobs.append(ConversionJob(folder, folder / f"cell_{cell_id}.h5", cell_id))
            subdirectories.clear()
            continue
        extensions = {".h5", ".hdf5"} if restore else {".csv"}
        # Filter names before filesystem checks: rig camera folders may contain many images.
        files = sorted(folder / name for name in filenames
                       if Path(name).suffix.lower() in extensions and not (folder / name).is_symlink())
        if restore:
            for source in files:
                try:
                    with h5py.File(source, "r") as archive:
                        if archive.attrs.get("format_id") != converter.FORMAT_ID:
                            continue
                        kind = archive.attrs.get("kind")
                        version = archive.attrs.get("format_version")
                        expected = {"cell": converter.CELL_VERSION,
                                    "protocol": converter.PROTOCOL_VERSION}.get(kind, converter.FORMAT_VERSION)
                        if version != expected:
                            continue
                    destination = source.parent if kind in ("cell", "protocol") else source.with_suffix(".csv")
                    jobs.append(ConversionJob(source, destination))
                except (OSError, ValueError):
                    continue
        elif mode != "protocols":
            jobs.extend(ConversionJob(source, source.with_suffix(".h5")) for source in files)
    return sorted(jobs)


class ConversionWorker(QThread):
    discovered = pyqtSignal(int)
    item_status = pyqtSignal(int, str)
    current_progress = pyqtSignal(int, str)
    batch_progress = pyqtSignal(int)
    message = pyqtSignal(str)
    summary = pyqtSignal(str)

    def __init__(self, jobs=None, overwrite=False, parent=None, *, root=None, mode="auto", restore=False):
        super().__init__(parent)
        self.jobs = [ConversionJob(*job) for job in jobs] if jobs is not None else None
        self.overwrite = overwrite
        self.root, self.mode, self.restore = root, mode, restore

    def run(self):
        if self.jobs is None:
            try:
                self.jobs = discover_jobs(self.root, self.mode, self.restore,
                                          cancelled=self.isInterruptionRequested)
            except Exception as error:
                self.summary.emit(str(error))
                return
        self.discovered.emit(len(self.jobs))
        if not self.jobs:
            self.summary.emit("No matching files found. Check the directory, action, or layout.")
            return
        converter = HDF5Converter()
        counts = {"Converted": 0, "Skipped": 0, "Failed": 0}
        cancelled = False
        for index, job in enumerate(self.jobs):
            source, destination, cell_id = job
            if self.isInterruptionRequested():
                cancelled = True
                break
            self.item_status.emit(index, "Working")
            label = f"{source.name} / cell_{cell_id}" if cell_id is not None else source.name
            self.message.emit(f"[{index + 1}/{len(self.jobs)}] {source}"
                              + (f" / cell_{cell_id}" if cell_id is not None else ""))
            last_progress = [None]

            def update(stage, completed, total):
                if self.isInterruptionRequested():
                    raise ConversionCancelled("Conversion cancelled")
                fraction = completed / total if total else 1
                percent = int(100 * fraction)
                if stage == "compress":
                    percent = int(50 * fraction)
                elif stage == "verify":
                    percent = 50 + int(50 * fraction)
                marker = stage, percent
                if marker != last_progress[0]:
                    self.current_progress.emit(percent, f"{stage.capitalize()}: {label}")
                    last_progress[0] = marker

            try:
                converter.convert(source, destination, cell_id=cell_id,
                                  overwrite=self.overwrite, progress=update)
                status = "Converted"
                self.message.emit(f"  Saved: {destination}")
            except ConversionCancelled:
                self.item_status.emit(index, "Cancelled")
                cancelled = True
                break
            except FileExistsError as error:
                status = "Skipped"
                self.message.emit(f"  Skipped: {error}")
            except Exception as error:
                status = "Failed"
                self.message.emit(f"  Failed: {error}")
            counts[status] += 1
            self.item_status.emit(index, status)
            self.batch_progress.emit(index + 1)
        prefix = "Cancelled" if cancelled else "Finished"
        self.summary.emit(f"{prefix}: {counts['Converted']} converted, {counts['Skipped']} skipped, "
                          f"{counts['Failed']} failed. Original files retained.")


class DataConverter(QWidget):
    """Choose a directory and convert it in a cancellable background worker."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Converter")
        self.resize(980, 620)
        self.worker = None
        self.close_when_idle = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.settings = QGroupBox("Data to convert")
        form = QFormLayout(self.settings)
        path_row = QHBoxLayout()
        self.directory = QLineEdit(str(REPO_ROOT / "experiments" / "Data"))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.directory, 1)
        path_row.addWidget(browse)
        form.addRow("Directory", path_row)
        self.action = QComboBox()
        self.action.addItems(["Compress recordings and patch-clamp cells to HDF5", "Restore HDF5 archives to original files"])
        form.addRow("Action", self.action)
        self.mode = QComboBox()
        self.mode.addItem("Automatic - recognize rig and patch-clamp layouts", "auto")
        self.mode.addItem("Standalone recordings - one HDF5 per CSV", "recordings")
        self.mode.addItem("Patch-clamp cells - group by exact stage XYZ per session", "protocols")
        form.addRow("Layout", self.mode)
        self.action.currentIndexChanged.connect(lambda index: self.mode.setEnabled(index == 0))
        explanation = QLabel("Includes subfolders. Exact stage X, Y, and Z matches within a session form one cell archive, "
                             "including all recording indices, protocols, images, and metadata. "
                             "Archives use the lowest recording index: cell_<index>.h5. "
                             "Rig-recorder CSVs keep their names and locations. Original files are retained.")
        explanation.setWordWrap(True)
        form.addRow(explanation)
        self.overwrite = QCheckBox("Replace existing destination files (otherwise skip them)")
        form.addRow(self.overwrite)
        layout.addWidget(self.settings)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Convert")
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.status = QLabel("Choose a directory and press Convert.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.current = QProgressBar()
        self.current.setFormat("Current conversion: %p%")
        self.current.setValue(0)
        self.overall = QProgressBar()
        self.overall.setRange(0, 1)
        self.overall.setValue(0)
        self.overall.setFormat("Batch: %v / %m")
        layout.addWidget(self.current)
        layout.addWidget(self.overall)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log, 1)

    def _browse(self):
        selected = QFileDialog.getExistingDirectory(self, "Select data directory", self.directory.text())
        if selected:
            self.directory.setText(selected)

    def _start(self):
        self.log.clear()
        self.current.setValue(0)
        self.overall.setRange(0, 0)
        self.status.setText("Preparing conversions...")
        self.settings.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = ConversionWorker(root=self.directory.text(), mode=self.mode.currentData(),
                                       restore=self.action.currentIndex() == 1,
                                       overwrite=self.overwrite.isChecked(), parent=self)
        self.worker.discovered.connect(self._discovered)
        self.worker.current_progress.connect(self._progress)
        self.worker.batch_progress.connect(self.overall.setValue)
        self.worker.message.connect(self.log.appendPlainText)
        self.worker.summary.connect(self.status.setText)
        self.worker.finished.connect(self._worker_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _discovered(self, count):
        self.overall.setRange(0, max(1, count))
        self.overall.setValue(0)
        self.log.appendPlainText(f"{count} conversion(s) to process.")

    def _progress(self, percent, label):
        self.current.setValue(percent)
        self.status.setText(label)

    def _cancel(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling... completed outputs will be kept.")

    def _worker_finished(self):
        self.worker = None
        self.settings.setEnabled(True)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.overall.maximum() == 0:
            self.overall.setRange(0, 1)
            self.overall.setValue(0)
        if self.close_when_idle:
            self.close()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.close_when_idle = True
            self._cancel()
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    window = DataConverter()
    window.show()
    raise SystemExit(application.exec_())
