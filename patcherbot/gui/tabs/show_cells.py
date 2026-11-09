"""Detected-cell list window."""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


class CellListWindow(QtWidgets.QDialog):
    closed = QtCore.pyqtSignal()
    action_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, thumbnail_size=96):
        super().__init__(parent=parent)
        self.setWindowTitle("Show Cells")
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.thumbnail_size = thumbnail_size
        self._cells = {"selected": [], "proposed": []}
        self.table = self._make_table()
        self.candidate_table = self._make_table()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.table, "Cells to Patch (0/20)")
        self.tabs.addTab(self.candidate_table, "Proposed Candidates (0)")
        self.tabs.currentChanged.connect(self._update_actions)
        self.table.itemSelectionChanged.connect(self._update_actions)
        self.candidate_table.itemSelectionChanged.connect(self._update_actions)

        self.buttons = {}
        actions = QtWidgets.QHBoxLayout()
        for action, label in (("promote", "Promote"), ("demote", "Demote"),
                              ("up", "Move Up"), ("down", "Move Down"),
                              ("delete", "Delete")):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, action=action: self._request_action(action))
            self.buttons[action] = button
            actions.addWidget(button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addLayout(actions)
        self.setLayout(layout)
        self.resize(800, 600)
        self._update_actions()

    def _make_table(self):
        table = QtWidgets.QTableWidget(0, 5)
        table.setHorizontalHeaderLabels([
            "Image",
            "Fluo Image",
            "Cell",
            "Stage (px)",
            "Stage (um)",
        ])
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        table.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 12)
        return table

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def update_cells(self, cells, stage_reference=None, full_refresh=True):
        self._update_table(self.table, "selected", cells, full_refresh)
        self.tabs.setTabText(0, f"Cells to Patch ({len(cells)}/20)")
        self._update_actions()

    def update_candidates(self, cells, full_refresh=True):
        self._update_table(self.candidate_table, "proposed", cells, full_refresh)
        self.tabs.setTabText(1, f"Proposed Candidates ({len(cells)})")
        self._update_actions()

    def show_candidates(self):
        self.tabs.setCurrentIndex(1)

    def _update_table(self, table, source, cells, full_refresh):
        previous = self._cells[source]
        selected_ids = {id(previous[index.row()]) for index in table.selectionModel().selectedRows()
                        if index.row() < len(previous)}
        full_refresh = full_refresh or tuple(map(id, previous)) != tuple(map(id, cells))
        self._cells[source] = list(cells)
        if not full_refresh:
            return
        table.blockSignals(True)
        try:
            table.clearSelection()
            table.setRowCount(len(cells))
            for row, cell in enumerate(cells):
                stage_px, img, stage_um, img_fluo = self._unpack_cell(cell)
                self._set_image_cell(row, 0, img, table=table)
                self._set_image_cell(row, 1, img_fluo, empty_text="N/A", table=table)
                self._set_item(row, 2, str(row + 1), table=table)
                self._set_item(row, 3, self._format_vec(stage_px), table=table)
                self._set_item(row, 4, self._format_vec(stage_um), table=table)
                if id(cell) in selected_ids:
                    table.selectionModel().select(
                        table.model().index(row, 0),
                        QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
        finally:
            table.blockSignals(False)

    def _active_list(self):
        if self.tabs.currentIndex() == 0:
            return self.table, "selected"
        return self.candidate_table, "proposed"

    def _request_action(self, action):
        table, source = self._active_list()
        rows = sorted(index.row() for index in table.selectionModel().selectedRows())
        cells = [self._cells[source][row] for row in rows if row < len(self._cells[source])]
        if cells:
            self.action_requested.emit({"action": action, "source": source, "cells": cells})

    def _update_actions(self, *_args):
        table, source = self._active_list()
        rows = {index.row() for index in table.selectionModel().selectedRows()}
        selected = bool(rows)
        capacity = max(0, 20 - len(self._cells["selected"]))
        self.buttons["promote"].setVisible(source == "proposed")
        self.buttons["demote"].setVisible(source == "selected")
        self.buttons["promote"].setEnabled(selected and capacity > 0)
        self.buttons["promote"].setToolTip(f"Add to Cells to Patch ({capacity} spaces available)")
        self.buttons["demote"].setEnabled(selected)
        self.buttons["delete"].setEnabled(selected)
        self.buttons["up"].setEnabled(any(row > 0 and row - 1 not in rows for row in rows))
        self.buttons["down"].setEnabled(any(row < table.rowCount() - 1 and row + 1 not in rows
                                            for row in rows))

    def _unpack_cell(self, cell):
        if cell is None:
            return None, None, None, None
        if len(cell) >= 4:
            return cell[0], cell[1], cell[2], cell[3]
        if len(cell) == 3:
            return cell[0], cell[1], cell[2], None
        return None, None, None, None

    def _set_item(self, row, col, text, table=None):
        table = self.table if table is None else table
        item = table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            table.setItem(row, col, item)
        item.setText(text)

    def _set_image_cell(self, row, col, image, empty_text="", table=None):
        table = self.table if table is None else table
        if image is None:
            table.removeCellWidget(row, col)
            item = QtWidgets.QTableWidgetItem(empty_text)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            table.setItem(row, col, item)
            return

        pixmap = self._image_to_pixmap(image)
        label = QtWidgets.QLabel()
        label.setAlignment(Qt.AlignCenter)
        if pixmap is not None:
            label.setPixmap(
                pixmap.scaled(
                    self.thumbnail_size,
                    self.thumbnail_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        table.setCellWidget(row, col, label)

    def _image_to_pixmap(self, image):
        if image is None:
            return None
        img = np.array(image)
        if img.ndim == 2:
            img8 = self._normalize_to_uint8(img)
            q_image = QtGui.QImage(
                img8.data,
                img8.shape[1],
                img8.shape[0],
                img8.strides[0],
                QtGui.QImage.Format_Grayscale8,
            ).copy()
        else:
            img8 = self._normalize_to_uint8(img[..., 0])
            q_image = QtGui.QImage(
                img8.data,
                img8.shape[1],
                img8.shape[0],
                img8.strides[0],
                QtGui.QImage.Format_Grayscale8,
            ).copy()
        return QtGui.QPixmap.fromImage(q_image)

    def _normalize_to_uint8(self, img):
        img = img.astype(np.float32)
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val) * 255.0
        else:
            img = np.zeros_like(img, dtype=np.float32)
        return img.astype(np.uint8)

    def _format_vec(self, vec):
        if vec is None:
            return "N/A"
        arr = np.array(vec).astype(float).ravel()
        return ", ".join(f"{v:.1f}" for v in arr)

