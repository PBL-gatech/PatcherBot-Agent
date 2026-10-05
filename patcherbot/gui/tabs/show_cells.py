"""Detected-cell list window."""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


class CellListWindow(QtWidgets.QDialog):
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None, thumbnail_size=96):
        super().__init__(parent=parent)
        self.setWindowTitle("Selected Cells")
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.thumbnail_size = thumbnail_size
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Image",
            "Fluo Image",
            "Cell",
            "Stage (px)",
            "Stage (um)",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 12)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def update_cells(self, cells, stage_reference=None, full_refresh=True):
        if self.table.rowCount() != len(cells):
            self.table.setRowCount(len(cells))
            full_refresh = True

        for row, cell in enumerate(cells):
            stage_px, img, stage_um, img_fluo = self._unpack_cell(cell)

            if full_refresh:
                self._set_image_cell(row, 0, img)
                self._set_image_cell(row, 1, img_fluo, empty_text="N/A")
                self._set_item(row, 2, str(row + 1))
                self._set_item(row, 3, self._format_vec(stage_px))
                self._set_item(row, 4, self._format_vec(stage_um))

    def _unpack_cell(self, cell):
        if cell is None:
            return None, None, None, None
        if len(cell) >= 4:
            return cell[0], cell[1], cell[2], cell[3]
        if len(cell) == 3:
            return cell[0], cell[1], cell[2], None
        return None, None, None, None

    def _set_item(self, row, col, text):
        item = self.table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
        item.setText(text)

    def _set_image_cell(self, row, col, image, empty_text=""):
        if image is None:
            self.table.removeCellWidget(row, col)
            item = QtWidgets.QTableWidgetItem(empty_text)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
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
        self.table.setCellWidget(row, col, label)

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

