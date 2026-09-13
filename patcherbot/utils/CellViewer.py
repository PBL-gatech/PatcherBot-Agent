"""Read-only viewer for cell HDF5 archives. Run this file and choose Open cell."""

import csv
import io
import json
from pathlib import Path, PurePosixPath
import sys

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

if __package__:
    from .HDF5Converter import HDF5Converter
else:
    from HDF5Converter import HDF5Converter


class CellArchive:
    """Read the catalogue first, then decompress only the selected member."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        with h5py.File(self.path, "r") as archive:
            if (archive.attrs.get("format_id") != HDF5Converter.FORMAT_ID
                    or archive.attrs.get("kind") != "cell"
                    or archive.attrs.get("format_version") != HDF5Converter.CELL_VERSION):
                raise ValueError("Choose a cell archive created by HDF5Converter")
            self.cell_id = int(archive.attrs["cell_id"])
            self.recording_indices = [int(index) for index in archive.attrs.get(
                "recording_indices", [self.cell_id])]
            self.stage_coordinates = (tuple(float(value) for value in archive.attrs[
                "stage_coordinates"]) if archive.attrs.get("cell_identity") == "stage_xyz" else None)
            self.session = str(archive.attrs["original_filename"])
            self.metadata = json.loads(archive.attrs["cell_metadata"])
            self.members = []
            archive["files"].visititems(lambda name, item: self.members.append(name)
                                       if isinstance(item, h5py.Dataset) else None)
            self.members.sort()
            if len(self.members) != archive.attrs.get("file_count"):
                raise ValueError("Cell archive member count mismatch")

    def load(self, member, cancelled=None):
        """Return (trace, Nx3 array), (image, bytes), or (table, rows)."""
        if member not in self.members:
            raise ValueError("File is not in this cell archive")

        def check(stage=None, completed=0, total=0):
            if cancelled is not None and cancelled():
                raise RuntimeError("Loading cancelled")

        check()
        with io.BytesIO() as output, h5py.File(self.path, "r") as archive:
            data = archive["files"][member]
            HDF5Converter()._read_file(data, data.attrs, check, "read", 0, len(data), output)
            output.seek(0)
            if member in self.metadata:
                rows = self.metadata[member]
                columns = list(dict.fromkeys(key for row in rows for key in row))
                return "table", [columns] + [[row.get(key, "") for key in columns] for row in rows]
            if Path(member).suffix.lower() != ".csv":
                return "image", output.read()
            if Path(member).stem.endswith("_stim"):
                return "table", list(csv.reader(io.StringIO(output.read().decode("utf-8-sig"))))
            # Keep the three recorded columns unchanged; the UI labels protocols.
            values = np.loadtxt(output, dtype=np.float64, ndmin=2)
        check()
        if values.size == 0 or values.shape[1] != 3:
            raise ValueError("Expected three numeric columns: time, command, response")
        return "trace", values


class MemberLoader(QThread):
    loaded = pyqtSignal(str, str, object)
    failed = pyqtSignal(str)

    def __init__(self, archive, member, parent=None):
        super().__init__(parent)
        self.archive, self.member = archive, member

    def run(self):
        try:
            kind, data = self.archive.load(self.member, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.loaded.emit(self.member, kind, data)
        except Exception as error:
            if not self.isInterruptionRequested():
                self.failed.emit(str(error))


class CellViewer(QWidget):
    """Browse a cell's protocols, traces, images, and metadata without extracting."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Viewer")
        self.resize(1200, 780)
        self.archive = None
        self.worker = None
        self.close_when_idle = False
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.open_button = QPushButton("Open cell...")
        self.open_button.clicked.connect(self._choose_file)
        self.path = QLineEdit()
        self.path.setReadOnly(True)
        self.path.setPlaceholderText("Choose a cell archive (.h5)")
        top.addWidget(self.open_button)
        top.addWidget(self.path, 1)
        layout.addLayout(top)
        self.title = QLabel("Open a cell archive to browse its recordings, images, and metadata.")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        split = QSplitter()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Protocols and files")
        self.tree.currentItemChanged.connect(self._select_member)
        split.addWidget(self.tree)
        self.tabs = QTabWidget()
        self.graphs = pg.GraphicsLayoutWidget()
        self.graphs.setBackground("w")
        self.command = self.graphs.addPlot(row=0, col=0, title="Command")
        self.response = self.graphs.addPlot(row=1, col=0, title="Response")
        self.response.setXLink(self.command)
        for plot in (self.command, self.response):
            plot.setLabel("bottom", "Time", units="s")
            plot.showGrid(x=True, y=True, alpha=0.2)
            for axis in ("left", "bottom"):
                plot.getAxis(axis).setTextPen("k")
                plot.getAxis(axis).setPen("k")
        self.tabs.addTab(self.graphs, "Trace")
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image_area = QScrollArea()
        self.image_area.setWidget(self.image)
        self.tabs.addTab(self.image_area, "Image")
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.tabs.addTab(self.metadata, "Cell metadata")
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.table, "Table")
        split.addWidget(self.tabs)
        split.setSizes([340, 860])
        layout.addWidget(split, 1)
        self.status = QLabel("No archive open.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _choose_file(self):
        initial = str(self.archive.path.parent) if self.archive else str(
            Path(__file__).resolve().parents[2] / "experiments" / "Data" / "patch_clamp_data")
        path, _ = QFileDialog.getOpenFileName(self, "Open cell archive", initial,
                                             "Cell archives (*.h5 *.hdf5)")
        if path:
            try:
                self.open_file(path)
            except Exception as error:
                self.status.setText(f"Could not open archive: {error}")

    def open_file(self, path):
        if self.worker is not None:
            raise RuntimeError("Wait for the selected recording to finish loading")
        archive = CellArchive(path)
        self.archive = archive
        self.path.setText(str(archive.path))
        if archive.stage_coordinates is None:
            identity = f"Recording {archive.cell_id} (legacy index grouping)"
        else:
            x, y, z = archive.stage_coordinates
            identity = f"Stage XYZ ({x}, {y}, {z})"
        recordings = ", ".join(str(index) for index in archive.recording_indices)
        self.title.setText(f"{identity} | Session {archive.session} | Recordings {recordings}")
        self.tree.clear()
        parents = {"": self.tree.invisibleRootItem()}
        for member in archive.members:
            parts = PurePosixPath(member).parts
            for depth, name in enumerate(parts, 1):
                key = "/".join(parts[:depth])
                if key not in parents:
                    item = QTreeWidgetItem(parents["/".join(parts[:depth - 1])], [name])
                    parents[key] = item
                if depth == len(parts):
                    parents[key].setData(0, Qt.UserRole, member)
                    parents[key].setToolTip(0, member)
        self.tree.expandAll()
        sections = [name + "\n" + "\n\n".join(
            "\n".join(f"{key}: {value}" for key, value in row.items()) for row in rows)
                    for name, rows in archive.metadata.items() if rows]
        self.metadata.setPlainText("\n\n".join(sections) or "No metadata rows for this cell.")
        self.command.clear()
        self.response.clear()
        self.image.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.tabs.setCurrentWidget(self.metadata)
        self.status.setText(f"{len(archive.members)} files. Select a recording or image on the left.")

    def _select_member(self, item, previous=None):
        member = item.data(0, Qt.UserRole) if item is not None else None
        if not member or self.worker is not None:
            return
        self.status.setText(f"Loading {member}...")
        self.tree.setEnabled(False)
        self.open_button.setEnabled(False)
        self.worker = MemberLoader(self.archive, member, self)
        self.worker.loaded.connect(self._show_member)
        self.worker.failed.connect(lambda message: self.status.setText(f"Could not load file: {message}"))
        self.worker.finished.connect(self._load_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _show_member(self, member, kind, data):
        if self.close_when_idle:
            return
        if kind == "trace":
            self.command.clear()
            self.response.clear()
            protocol = PurePosixPath(member).parts[0]
            # Holding data is written [time, response, command] by graph.py.
            command_column, response_column = (2, 1) if protocol == "HoldingProtocol" else (1, 2)
            known = protocol in HDF5Converter.PROTOCOL_NAMES
            current_clamp = protocol == "CurrentProtocol"
            self.command.setLabel("left", "Command monitor" if current_clamp else "Command",
                                  units="V" if known else None)
            self.response.setLabel("left", "Voltage response" if current_clamp else "Current response",
                                   units=("V" if current_clamp else "A") if known else None)
            for plot, column, color in ((self.command, command_column, "#1766a3"),
                                        (self.response, response_column, "#bd342b")):
                curve = plot.plot(data[:, 0], data[:, column], pen=color,
                                  autoDownsample=True, downsampleMethod="peak")
                curve.setClipToView(True)
                plot.enableAutoRange()
            self.tabs.setCurrentWidget(self.graphs)
            detail = f"{len(data):,} samples; drag to pan, scroll to zoom."
        elif kind == "image":
            pixmap = QPixmap()
            if not pixmap.loadFromData(data):
                self.status.setText(f"Cannot decode image: {member}")
                return
            self.image.setPixmap(pixmap)
            self.image.adjustSize()
            self.tabs.setCurrentWidget(self.image_area)
            detail = f"{pixmap.width()} x {pixmap.height()} pixels."
        else:
            columns = data[0] if data else []
            rows = data[1:]
            self.table.clear()
            self.table.setColumnCount(len(columns))
            self.table.setHorizontalHeaderLabels(columns)
            self.table.setRowCount(min(500, len(rows)))
            for row_index, row in enumerate(rows[:500]):
                for column, value in enumerate(row[:len(columns)]):
                    self.table.setItem(row_index, column, QTableWidgetItem(str(value)))
            self.table.resizeColumnsToContents()
            self.tabs.setCurrentWidget(self.table)
            detail = f"Showing {min(500, len(rows))} of {len(rows)} rows."
        self.status.setText(f"{member} | {detail}")

    def _load_finished(self):
        self.worker = None
        self.tree.setEnabled(True)
        self.open_button.setEnabled(True)
        if self.close_when_idle:
            self.close()

    def closeEvent(self, event):
        if self.worker is not None:
            self.close_when_idle = True
            self.worker.requestInterruption()
            self.status.setText("Finishing the current read before closing...")
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    viewer = CellViewer()
    viewer.show()
    raise SystemExit(application.exec_())
