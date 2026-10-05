"""Read-only movement/graph snapshot viewer. Run directly; no arguments needed.

Adapted from rig_replay3D.DataManager's parsers, nearest-movement matching,
coordinate transform, and update_graphs' next-snapshot/100-update history.
"""

import csv
import io
import itertools
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTabWidget, QVBoxLayout, QWidget,
)

if __package__:
    from .HDF5Converter import HDF5Converter
else:
    from HDF5Converter import HDF5Converter


class RecordingData:
    FIELDS = {
        "movement": "timestamp st_x st_y st_z pi_x pi_y pi_z".split(),
        "graph": "timestamp pressure resistance current voltage".split(),
    }

    def __init__(self):
        self.rows = {}
        self.times = {}
        self.paths = []
        self.skipped = 0

    @staticmethod
    def numeric_list(value):
        # Same safe parser as rig_replay3D; never evaluate recorded strings.
        value = re.sub(r'np\.float64\((.*?)\)', r'\1', value.strip().strip('[]'))
        return np.asarray([float(item) for item in value.split(',') if item.strip()])

    def load(self, paths, cancelled=lambda: False, progress=lambda message: None):
        csv.field_size_limit(2**31 - 1)
        for source in paths:
            source = Path(source).resolve(strict=True)
            progress(f"Reading {source.name}...")
            if source.suffix.lower() in ('.h5', '.hdf5'):
                buffer = io.BytesIO()
                with h5py.File(source, 'r') as archive:
                    if (archive.attrs.get('format_id') != HDF5Converter.FORMAT_ID
                            or archive.attrs.get('format_version') != HDF5Converter.FORMAT_VERSION):
                        raise ValueError('Choose a standalone movement or graph recording archive')
                    name = archive.attrs['original_filename']
                    def check(stage, done, total):
                        if cancelled():
                            raise RuntimeError('Loading cancelled')
                    try:
                        HDF5Converter()._read_file(archive['csv_bytes'], archive.attrs, check,
                                                   'read', 0, archive.attrs['source_size'], buffer)
                    except Exception:
                        buffer.close()
                        raise
                buffer.seek(0)
                stream = io.TextIOWrapper(buffer, encoding='utf-8-sig', newline='')
            elif source.suffix.lower() == '.csv':
                name = source.name
                stream = source.open(encoding='utf-8-sig', newline='')
            else:
                raise ValueError('Choose CSV or HDF5 recordings')
            with stream:
                kind = next((key for key in self.FIELDS if key + '_recording' in name.lower()), None)
                if kind is None:
                    raise ValueError(f'Unrecognized recording name: {name}')
                if kind in self.rows:
                    raise ValueError(f'Choose only one {kind} recording (CSV or HDF5)')
                rows = self._parse(stream, kind, cancelled)
            if not rows:
                raise ValueError(f'No valid snapshots in {source.name}')
            rows.sort(key=lambda row: row[0])
            self.rows[kind] = rows
            self.times[kind] = np.asarray([row[0] for row in rows])
            self.paths.append(source)
        if not self.rows:
            raise ValueError('Select at least one recording')
        self.start = min(times[0] for times in self.times.values())
        return self

    def _parse(self, stream, kind, cancelled):
        first = next((line for line in stream if line.strip()), '')
        fields = self.FIELDS[kind]
        lines = itertools.chain([first], stream)
        if ';' in first:
            reader = csv.reader(lines, delimiter=';')
            header = next(reader)
            normalized = [value.strip().lower() for value in header]
            if 'timestamp' in normalized:
                if not set(fields).issubset(normalized):
                    raise ValueError(f'{kind} header is missing required columns')
                positions = [normalized.index(field) for field in fields]
            else:
                positions = list(range(len(fields)))
                reader = itertools.chain([header], reader)
            records = ([row[index] for index in positions] if len(row) > max(positions) else [] for row in reader)
        else:
            # Legacy key:value lines, including bracketed current/voltage arrays.
            keys = '|'.join(fields)
            pattern = re.compile(rf'({keys})\s*:\s*(.*?)(?=\s*/?\s*(?:{keys})\s*:|$)')
            records = (list(dict(pattern.findall(line)).get(field, '') for field in fields) for line in lines)
        rows = []
        for values in records:
            if cancelled():
                raise RuntimeError('Loading cancelled')
            try:
                if kind == 'movement':
                    row = [float(value) for value in values]
                    if len(row) != 7 or not np.isfinite(row).all():
                        raise ValueError('Invalid movement snapshot')
                else:
                    row = [float(value) for value in values[:3]] + [self.numeric_list(value) for value in values[3:]]
                    if len(row) != 5 or not np.isfinite(row[0]):
                        raise ValueError('Invalid graph timestamp')
                rows.append(row)
            except (ValueError, IndexError):
                self.skipped += 1
        return rows

    def indices_at(self, timestamp):
        """Replay uses the next graph snapshot and nearest movement (earlier wins ties)."""
        result = {}
        for kind, times in self.times.items():
            index = int(np.searchsorted(times, timestamp))
            if kind == 'movement':
                if index == len(times) or (index > 0 and timestamp - times[index - 1] <= times[index] - timestamp):
                    index -= 1
            result[kind] = index if index < len(times) else None
        return result

    @staticmethod
    def replay_position(row):
        # rig_replay3D: micrometers -> mm, inverted axes, pipette Z offset 1.365 mm.
        position = -np.asarray(row[1:7], dtype=float) / 1000
        position[5] += 1.365
        return position


class RecordingLoader(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = paths

    def run(self):
        try:
            data = RecordingData().load(self.paths, self.isInterruptionRequested, self.progress.emit)
            if not self.isInterruptionRequested():
                self.loaded.emit(data)
        except Exception as error:
            self.failed.emit(str(error))


class RecordingViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Recording Viewer')
        self.resize(1250, 850)
        self.data = None
        self.worker = None
        self.closing = False
        self.playing = False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.open_button = QPushButton('Open recordings...')
        self.open_button.clicked.connect(self._choose)
        row.addWidget(self.open_button)
        row.addWidget(QLabel('Select movement and/or graph CSV/HDF5 files together.'))
        row.addStretch()
        layout.addLayout(row)
        self.sources = QLabel('No recordings open.')
        self.sources.setWordWrap(True)
        layout.addWidget(self.sources)
        self.tabs = QTabWidget()
        self.plots, self.curves = {}, {'stage': [], 'pipette': []}
        self.movement_cursors = {}
        for tab, specs in (
            ('Graph snapshots', [('voltage', 'Voltage', 'V'), ('current', 'Current', 'A'),
                                 ('pressure', 'Pressure', 'mbar'), ('resistance', 'Resistance (recorded units)', None)]),
            ('Movement', [(f'{group}_{axis}', f'{label} {axis.upper()}', None)
                          for axis in 'xyz' for group, label in
                          (('stage', 'Stage / microscope'), ('pipette', 'Pipette'))]),
        ):
            canvas = pg.GraphicsLayoutWidget()
            canvas.setBackground('w')
            for index, (key, label, units) in enumerate(specs):
                plot = canvas.addPlot(row=index // 2, col=index % 2, title=label)
                plot.setLabel('left', label, units=units)
                plot.setLabel('bottom', 'Sample within snapshot' if key in ('voltage', 'current') else 'Elapsed time',
                              units=None if key in ('voltage', 'current') else 's')
                plot.showGrid(x=True, y=True, alpha=0.2)
                for axis in ('left', 'bottom'):
                    plot.getAxis(axis).setTextPen('k')
                    plot.getAxis(axis).setPen('k')
                self.plots[key] = plot
                if tab == 'Movement':
                    group, axis = key.split('_')
                    color = dict(zip('xyz', ('#d33', '#298b35', '#2867bd')))[axis]
                    curve = plot.plot(pen=color)
                    curve.setDownsampling(auto=True, method='peak')
                    self.curves[group].append(curve)
                    cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen('#222', style=Qt.DashLine))
                    plot.addItem(cursor, ignoreBounds=True)
                    self.movement_cursors[key] = cursor
                else:
                    self.curves[key] = plot.plot(pen='#2867bd')
            self.tabs.addTab(canvas, tab)
        layout.addWidget(self.tabs, 1)
        movement_controls = QHBoxLayout()
        movement_controls.addWidget(QLabel('Movement display'))
        self.movement_mode = QComboBox()
        self.movement_mode.addItems(['Displacement from start (um)', 'Absolute replay coordinates (mm)'])
        self.movement_history = QComboBox()
        self.movement_history.addItems(['Whole recording + selected timestamp', 'Last 100 snapshots'])
        for control in (self.movement_mode, self.movement_history):
            control.currentIndexChanged.connect(self._movement_display_changed)
            movement_controls.addWidget(control)
        movement_controls.addStretch()
        layout.addLayout(movement_controls)
        self.position = QLabel('Movement coordinates use the RigReplay3D display transform (mm).')
        self.position.setWordWrap(True)
        layout.addWidget(self.position)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self.pause)
        self.slider.valueChanged.connect(self.show_snapshot)
        layout.addWidget(self.slider)
        controls = QHBoxLayout()
        self.clock = QComboBox()
        self.clock.currentIndexChanged.connect(self._clock_changed)
        controls.addWidget(QLabel('Timeline'))
        controls.addWidget(self.clock)
        for text, delta in (('Previous', -1), ('Next', 1)):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, step=delta: self.step(step))
            controls.addWidget(button)
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)
        self.speed = QComboBox()
        for speed in (0.25, 0.5, 1, 2, 4, 10):
            self.speed.addItem(f'{speed}x', speed)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._schedule)
        controls.addWidget(self.speed)
        controls.addStretch()
        layout.addLayout(controls)
        self.status = QLabel('Open recordings, then drag the slider or press Play.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self.slider.setValue(self.slider.value() + 1))

    def _choose(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Open movement and/or graph recordings',
            str(Path(__file__).resolve().parents[2] / 'experiments/Data/rig_recorder_data'),
            'Recordings (*.h5 *.hdf5 *.csv)')
        if paths:
            self.open_files(paths)

    def open_files(self, paths):
        if self.worker is not None:
            return
        self.pause()
        self.open_button.setEnabled(False)
        self.worker = RecordingLoader(paths, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.failed.connect(lambda error: self.status.setText(f'Could not load: {error}'))
        self.worker.loaded.connect(self.set_data)
        self.worker.finished.connect(self._finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def set_data(self, data):
        self.data = data
        if 'movement' in data.rows:
            raw = np.asarray(data.rows['movement'])[:, 1:7]
            self.movement_mm = -raw / 1000
            self.movement_mm[:, 5] += 1.365
            self.movement_displacement = -(raw - raw[0])
        self._movement_cache_key = None
        self.sources.setText(' | '.join(str(path) for path in data.paths))
        self.clock.blockSignals(True)
        self.clock.clear()
        for kind in ('graph', 'movement'):
            if kind in data.rows:
                self.clock.addItem(kind.capitalize() + ' timestamps', kind)
        self.clock.blockSignals(False)
        self.tabs.setTabEnabled(0, 'graph' in data.rows)
        self.tabs.setTabEnabled(1, 'movement' in data.rows)
        self.tabs.setCurrentIndex(0 if 'graph' in data.rows else 1)
        self._clock_changed()

    def _clock_changed(self):
        self.pause()
        if self.data is not None:
            self.slider.blockSignals(True)
            self.slider.setRange(0, len(self.data.times[self.clock.currentData()]) - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.show_snapshot(0)

    def show_snapshot(self, index):
        if self.data is None:
            return
        times = self.data.times[self.clock.currentData()]
        timestamp = times[index]
        selected = self.data.indices_at(timestamp)
        selected[self.clock.currentData()] = index  # Preserve distinct snapshots sharing a timestamp.
        graph_index = selected.get('graph')
        detail = []
        if graph_index is not None:
            rows = self.data.rows['graph']
            snapshot = rows[graph_index]
            for key, column in (('current', 3), ('voltage', 4)):
                self.curves[key].setData(snapshot[column])
            history = rows[max(0, graph_index - 99):graph_index + 1]
            for key, column in (('pressure', 1), ('resistance', 2)):
                self.curves[key].setData([row[0] - self.data.start for row in history], [row[column] for row in history])
            detail.append(f'Graph timestamp {snapshot[0]:.6f}')
        else:
            for key in ('current', 'voltage', 'pressure', 'resistance'):
                self.curves[key].setData([], [])
        movement_index = selected.get('movement')
        if movement_index is not None:
            rows = self.data.rows['movement']
            position = self.movement_mm[movement_index]
            self._draw_movement(movement_index)
            self.position.setText('Replay coordinates (mm) | Stage / microscope XYZ: '
                + ', '.join(f'{value:.4f}' for value in position[:3]) + ' | Pipette XYZ: '
                + ', '.join(f'{value:.4f}' for value in position[3:]))
            detail.append(f'Movement timestamp {rows[movement_index][0]:.6f}')
        else:
            self.position.setText('No movement recording loaded.')
            for key in ('stage', 'pipette'):
                for curve in self.curves[key]:
                    curve.setData([], [])
            for cursor in self.movement_cursors.values():
                cursor.hide()
        self.status.setText(f'Snapshot {index + 1}/{len(times)} | Elapsed {timestamp-self.data.start:.3f} s | '
                            + ' | '.join(detail) + (f' | {self.data.skipped} malformed rows skipped' if self.data.skipped else ''))
        if self.playing:
            self._schedule()

    def _movement_display_changed(self):
        if self.data is not None:
            self._movement_cache_key = None
            self.show_snapshot(self.slider.value())
            for key in self.movement_cursors:
                self.plots[key].enableAutoRange()
                self.plots[key].autoRange()

    def _draw_movement(self, index):
        relative = self.movement_mode.currentIndex() == 0
        full = self.movement_history.currentIndex() == 0
        times = self.data.times['movement'] - self.data.start
        values = self.movement_displacement if relative else self.movement_mm
        key = (relative, full, None if full else index)
        if key != self._movement_cache_key:
            visible = slice(None) if full else slice(max(0, index - 99), index + 1)
            for group, offset in (('stage', 0), ('pipette', 3)):
                for axis, curve in enumerate(self.curves[group]):
                    # A downsampling factor from the full trace can hide a short window.
                    curve.setDownsampling(ds=1, auto=full, method='peak')
                    curve.setData(times[visible], values[visible, offset + axis])
                    plot = self.plots[f"{group}_{'xyz'[axis]}"]
                    plot.setLabel('left', 'Displacement (um)' if relative else 'Position (mm)', units='')
            self._movement_cache_key = key
        for cursor in self.movement_cursors.values():
            cursor.show()
            cursor.setValue(times[index])

    def step(self, delta):
        self.pause()
        self.slider.setValue(self.slider.value() + delta)

    def pause(self):
        self.playing = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.play_button.setText('Play')

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.data is not None and self.slider.maximum() > 0:
            if self.slider.value() == self.slider.maximum():
                self.slider.setValue(0)
            self.playing = True
            self.play_button.setText('Pause')
            self._schedule()

    def _schedule(self):
        if not self.playing:
            return
        index = self.slider.value()
        if index >= self.slider.maximum():
            self.pause()
            return
        times = self.data.times[self.clock.currentData()]
        delay = (times[index + 1] - times[index]) * 1000 / self.speed.currentData()
        self.timer.start(max(1, min(2**31 - 1, int(delay))))

    def _finished(self):
        self.worker = None
        self.open_button.setEnabled(True)
        if self.closing:
            self.close()

    def closeEvent(self, event):
        self.pause()
        if self.worker is not None:
            self.closing = True
            self.worker.requestInterruption()
            event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = RecordingViewer()
    viewer.show()
    raise SystemExit(app.exec_())
