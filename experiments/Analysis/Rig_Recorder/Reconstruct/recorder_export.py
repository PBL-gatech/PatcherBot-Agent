"""Compose timestamp-synchronized, per-state rig recording videos."""

import bisect
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets


class _Encoder:
    """Stream RGB frames to ffmpeg, with an optional OpenCV fallback."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.process = None
        self.writer = None
        self.errors = None
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg and Path("C:/ffmpeg/bin/ffmpeg.exe").is_file():
            ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"
        if ffmpeg:
            self.errors = tempfile.TemporaryFile()
            try:
                self.process = subprocess.Popen(
                    [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "rawvideo", "-pixel_format", "rgb24",
                     "-video_size", f"{width}x{height}", "-framerate", str(fps),
                     "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     str(self.path)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                self.errors.close()
                raise
        else:
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("Video export requires ffmpeg or OpenCV. Install ffmpeg and add it to PATH.") from exc
            self.cv2, self.np = cv2, np
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self.writer.isOpened():
                self.writer.release()
                raise RuntimeError("Unable to open the MP4 encoder.")

    def write(self, image):
        image = image.convertToFormat(QtGui.QImage.Format_RGB888)
        pointer = image.bits()
        pointer.setsize(image.byteCount())
        data = bytes(pointer)
        row_bytes = image.width() * 3
        if image.bytesPerLine() != row_bytes:
            data = b"".join(data[row * image.bytesPerLine():row * image.bytesPerLine() + row_bytes]
                            for row in range(image.height()))
        if self.process is not None:
            self.process.stdin.write(data)
        else:
            rgb = self.np.frombuffer(data, dtype=self.np.uint8).reshape(image.height(), image.width(), 3)
            self.writer.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))

    def close(self, cancelled=False):
        error = None
        if self.process is not None:
            try:
                self.process.stdin.close()
                if cancelled:
                    self.process.terminate()
                code = self.process.wait(timeout=30)
                if code and not cancelled:
                    self.errors.seek(0)
                    error = self.errors.read().decode("utf-8", errors="replace")
            except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
                self.process.kill()
                self.process.wait()
                if not cancelled:
                    error = str(exc)
            finally:
                self.errors.close()
        elif self.writer is not None:
            self.writer.release()
        if error:
            raise RuntimeError(f"Video encoding failed: {error.strip()}")


def state_segments(manager, tracking_states=True, fps=30):
    """Return state time ranges and camera-index bounds, including unlabeled gaps.

    Recorded state boundaries split clips even between sparse camera frames.
    Overlapping state windows use the same precedence as the replay manager.
    """
    if not manager.timestamps:
        return []
    steps = [b - a for a, b in zip(manager.timestamps, manager.timestamps[1:]) if b > a]
    final_step = statistics.median(steps) if steps else 1 / fps
    recording_start = manager.timestamps[0]
    recording_end = manager.timestamps[-1] + final_step
    boundaries = {recording_start, recording_end}
    if tracking_states:
        for start, end, _, _ in manager.state_windows:
            if recording_start < start < recording_end:
                boundaries.add(start)
            if recording_start < end < recording_end:
                boundaries.add(end)
    boundaries = sorted(boundaries)
    segments = []
    previous_key = object()
    for start_time, finish_time in zip(boundaries, boundaries[1:]):
        midpoint = start_time + (finish_time - start_time) / 2
        state, outcome = manager.get_active_state_for_timestamp(midpoint) if tracking_states else (None, None)
        window_key = None
        if state:
            for start, end, name, code in reversed(manager.state_windows):
                if start <= midpoint <= end and name == state and code == outcome:
                    window_key = (start, end)
                    break
        key = (state, outcome, window_key)
        start_index = max(0, bisect.bisect_right(manager.timestamps, start_time) - 1)
        end_index = max(start_index, bisect.bisect_left(manager.timestamps, finish_time) - 1)
        if key == previous_key:
            segments[-1][1] = end_index
            segments[-1][4] = finish_time
        else:
            segments.append([start_index, end_index, state, start_time, finish_time])
            previous_key = key
    return segments


def _fit_rect(image, area):
    scale = min(area.width() / image.width(), area.height() / image.height())
    width, height = image.width() * scale, image.height() * scale
    return QtCore.QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)


def _draw_image(painter, image, area):
    if image is None or image.isNull():
        return None
    rect = _fit_rect(image, area)
    painter.drawImage(rect, image)
    return rect


def _plot(painter, area, title, values, color, x_values=None):
    """Draw an independently scaled trace; unknown sample timing stays labeled Samples."""
    painter.fillRect(area, QtGui.QColor(22, 26, 32))
    painter.setPen(QtGui.QColor("white"))
    painter.drawText(area.adjusted(9, 5, -8, 0), QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft, title)
    trace = area.adjusted(56, 34, -14, -32)
    painter.setPen(QtGui.QColor(100, 108, 118))
    painter.drawRect(trace)
    pairs = []
    for i, value in enumerate(values):
        try:
            value = float(value)
            x = float(x_values[i]) if x_values is not None else float(i)
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(value) and math.isfinite(x):
            pairs.append((x, value))
    if not pairs:
        painter.drawText(trace, QtCore.Qt.AlignCenter, "No data")
        return
    xs, ys = zip(*pairs)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        padding = max(abs(ymin) * .05, 1e-12)
        ymin, ymax = ymin - padding, ymax + padding
    painter.setPen(QtGui.QColor(210, 215, 224))
    painter.drawText(QtCore.QRectF(area.left() + 2, trace.top() - 8, 49, 20), QtCore.Qt.AlignRight, f"{ymax:.3g}")
    painter.drawText(QtCore.QRectF(area.left() + 2, trace.bottom() - 12, 49, 20), QtCore.Qt.AlignRight, f"{ymin:.3g}")
    painter.drawText(QtCore.QRectF(trace.left(), trace.bottom() + 3, trace.width(), 24),
                     QtCore.Qt.AlignCenter, f"{xmin:.3g} - {max(xs):.3g}  {'s' if x_values is not None else 'samples'}")
    path = QtGui.QPainterPath()
    for i, (x, y) in enumerate(pairs):
        point = QtCore.QPointF(trace.left() + (x - xmin) / (xmax - xmin) * trace.width(),
                              trace.bottom() - (y - ymin) / (ymax - ymin) * trace.height())
        path.moveTo(point) if i == 0 else path.lineTo(point)
    painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.8))
    if len(pairs) == 1:
        painter.drawEllipse(path.currentPosition(), 2, 2)
    else:
        painter.drawPath(path)


def _export_font():
    # Qt's offscreen platform may not discover native fonts automatically.
    if not QtGui.QFontDatabase().families():
        for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if Path(path).is_file() and QtGui.QFontDatabase.addApplicationFont(path) >= 0:
                break
        if not QtGui.QFontDatabase().families():
            raise RuntimeError("No readable font is available for export state and scale labels.")
    font = QtGui.QFont()
    if not QtGui.QFontInfo(font).family():
        font.setFamily(QtGui.QFontDatabase().families()[0])
    return font


def compose_frame(main_image, three_d_image, aux_image, manager, timestamp, state, size, graph_times=None):
    """Compose microscope/insets and a full-height electrophysiology column."""
    width, height = size
    frame = QtGui.QImage(width, height, QtGui.QImage.Format_RGB888)
    frame.fill(QtGui.QColor(10, 12, 16))
    painter = QtGui.QPainter(frame)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    font = _export_font()
    font.setPixelSize(max(14, int(height / 48)))
    painter.setFont(font)
    left_width = int(width * .68)
    microscope_rect = _draw_image(painter, main_image, QtCore.QRectF(0, 0, left_width, height))
    inset_width = int(left_width * .28)
    inset_height = int(height * .26)
    inset_3d = QtCore.QRectF(12, 12, inset_width, inset_height)
    painter.fillRect(inset_3d, QtGui.QColor("black"))
    _draw_image(painter, three_d_image, inset_3d)
    if aux_image is not None and not aux_image.isNull():
        inset_aux = QtCore.QRectF(left_width - inset_width - 12, 12, inset_width, inset_height)
        painter.fillRect(inset_aux, QtGui.QColor("black"))
        _draw_image(painter, aux_image, inset_aux)
    # Calibration refers to the original microscope pixels, before letterboxing/resizing.
    if microscope_rect is not None:
        bar_width = 5 * 6 * microscope_rect.width() / main_image.width()
        x_end, y = microscope_rect.right() - 24, microscope_rect.bottom() - 32
        painter.setPen(QtGui.QPen(QtGui.QColor("black"), 7))
        painter.drawLine(QtCore.QPointF(x_end - bar_width, y), QtCore.QPointF(x_end, y))
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 4))
        painter.drawLine(QtCore.QPointF(x_end - bar_width, y), QtCore.QPointF(x_end, y))
        label_rect = QtCore.QRectF(x_end - max(bar_width, 70), y - 32, max(bar_width, 70), 28)
        painter.fillRect(label_rect, QtGui.QColor(0, 0, 0, 160))
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(label_rect, QtCore.Qt.AlignCenter, "5 \u00b5m")
    if state:
        label = str(state).replace("_", " ")
        label_width = painter.fontMetrics().horizontalAdvance(label) + 24
        label_rect = QtCore.QRectF(16, height - 48, label_width, 32)
        painter.fillRect(label_rect, QtGui.QColor(0, 0, 0, 185))
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(label_rect, QtCore.Qt.AlignCenter, label)
    graph_times = graph_times if graph_times is not None else [entry['time'] for entry in manager.graph_data]
    graph_index = bisect.bisect_right(graph_times, timestamp) - 1
    graph_entry = manager.graph_data[graph_index] if graph_index >= 0 else {}
    history = manager.graph_data[max(0, graph_index - 99):graph_index + 1] if graph_index >= 0 else []
    origin = manager.timestamps[0]
    history_times = [entry['time'] - origin for entry in history]
    traces = [("Voltage (V)", graph_entry.get('voltage', []), "#63a8ff", None),
              ("Current (A)", graph_entry.get('current', []), "#ff8383", None),
              ("Pressure (mbar)", [entry['pressure'] for entry in history], "#70d59d", history_times),
              ("Resistance (M\u03a9)", [entry['resistance'] for entry in history], "#d792eb", history_times)]
    for index, (title, values, color, xs) in enumerate(traces):
        area = QtCore.QRectF(left_width + 6, index * height / 4 + 3, width - left_width - 12, height / 4 - 6)
        _plot(painter, area, title, values, color, xs)
    painter.end()
    return frame


def _capture_3d(window):
    view = window.three_d_view
    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
    if hasattr(view, "readQImage"):
        image = view.readQImage()
    elif hasattr(view, "grabFramebuffer"):
        image = view.grabFramebuffer()
    else:
        image = view.grab().toImage()
    if image.isNull():
        raise RuntimeError("The 3D view could not be captured. Open the 3D view before exporting.")
    return image


def export_recording(window, settings, output_dir):
    """Export each contiguous state run; return completed MP4 paths even when cancelled.

    Camera timestamps determine duration. Sparse frames are held, and faster frames
    are sampled at export_fps (30 by default). An unfinished clip is removed.
    Playback position, selected view, and enabled controls are restored on exit.
    """
    manager = window.data_manager
    if not manager.image_paths:
        raise ValueError("Load a recording before exporting videos.")
    fps = float(settings.get('export_fps', 30))
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Export FPS must be between 1 and 120.")
    size = (1280, 960) if settings.get('export_aspect_ratio', '16:9') == '4:3' else (1280, 720)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = state_segments(manager, settings.get('tracking_states', True), fps)
    graph_times = [entry['time'] for entry in manager.graph_data]
    saved_index = window.current_index
    saved_widget = window.graph_stack.currentWidget()
    saved_view_key = window.current_view_key
    saved_inference = getattr(window, 'enable_replay_inference', None)
    saved_overlays = getattr(window, '_frame_overlay_results', None)
    was_playing = window.timer.isActive()
    window.timer.stop()
    controls = window.findChildren(QtWidgets.QPushButton) + window.findChildren(QtWidgets.QComboBox) + window.findChildren(QtWidgets.QSlider) + window.findChildren(QtWidgets.QShortcut)
    control_states = [(control, control.isEnabled()) for control in controls]
    for control, _ in control_states:
        control.setEnabled(False)
    progress = QtWidgets.QProgressDialog("Preparing video export...", "Cancel", 0, len(manager.timestamps), window)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    completed = []
    window._export_cancelled = False
    encoder = None
    partial_path = None
    try:
        if saved_inference is not None:
            window.enable_replay_inference = bool(settings.get('export_inference', True))
            window._frame_overlay_results = None
            if window.enable_replay_inference:
                for flag, ensure, label in (
                    ('enable_pipette_detector', '_ensure_pipette_detector', 'pipette detector'),
                    ('enable_cell_detector', '_ensure_cell_detector', 'cell detector'),
                    ('enable_pipette_focuser', '_ensure_pipette_focuser', 'pipette focuser'),
                ):
                    if getattr(window, flag, False):
                        progress.setLabelText(f"Preparing {label} for export...")
                        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                        if getattr(window, ensure)() is None:
                            raise RuntimeError(f"Cannot export inference: selected {label} could not be loaded. Check the model and its dependencies, or disable inference in export settings.")
        window.graph_stack.setCurrentWidget(window.three_d_view)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        for number, (start, end, state, start_time, finish_time) in enumerate(segments, 1):
            if progress.wasCanceled():
                window._export_cancelled = True
                break
            name = re.sub(r'[^\w-]+', '_', state or 'unlabeled').strip('_') or 'unlabeled'
            path = output_dir / f"{number:03d}_{name}.mp4"
            suffix = 1
            while path.exists() or path.with_suffix('.partial.mp4').exists():
                path = output_dir / f"{number:03d}_{name}_{suffix}.mp4"
                suffix += 1
            partial_path = path.with_suffix('.partial.mp4')
            encoder = _Encoder(partial_path, *size, fps)
            count = max(1, int(math.ceil((finish_time - start_time) * fps - 1e-5)))
            previous_index = None
            for frame_number in range(count):
                QtWidgets.QApplication.processEvents()
                if progress.wasCanceled():
                    window._export_cancelled = True
                    break
                timestamp = start_time + frame_number / fps
                index = max(start, min(end, bisect.bisect_right(manager.timestamps, timestamp) - 1))
                if index != previous_index:
                    window.current_index = index
                    window.update_view()
                    camera_state, _ = manager.get_active_state_for_timestamp(manager.timestamps[index])
                    if camera_state != state and hasattr(window, '_run_frame_inference'):
                        window._run_frame_inference(manager.image_paths[index], state)
                    main_pixmap = QtGui.QPixmap(manager.image_paths[index])
                    if main_pixmap.isNull():
                        raise RuntimeError(f"Unable to load camera frame: {manager.image_paths[index]}")
                    main_pixmap = window._apply_frame_overlays(main_pixmap, manager.image_paths[index])
                    main_image = main_pixmap.toImage()
                    previous_index = index
                window.update_3d_view(timestamp)
                three_d_image = _capture_3d(window)
                aux_path = manager.get_aux_image_for_timestamp(timestamp)
                aux_image = QtGui.QImage(aux_path) if aux_path else None
                frame = compose_frame(main_image, three_d_image, aux_image, manager, timestamp, state, size, graph_times)
                encoder.write(frame)
                progress.setLabelText(f"Exporting {number}/{len(segments)}: {state or 'unlabeled'}")
                progress.setValue(index)
            encoder.close(cancelled=window._export_cancelled)
            encoder = None
            if window._export_cancelled:
                partial_path.unlink(missing_ok=True)
                partial_path = None
                break
            partial_path.replace(path)
            partial_path = None
            completed.append(path)
        return completed
    finally:
        try:
            if encoder is not None:
                encoder.close(cancelled=True)
        finally:
            if partial_path is not None:
                partial_path.unlink(missing_ok=True)
            progress.close()
            window.current_index = saved_index
            window.current_view_key = saved_view_key
            window.graph_stack.setCurrentWidget(saved_widget)
            if saved_inference is not None:
                window.enable_replay_inference = saved_inference
                window._frame_overlay_results = saved_overlays
            try:
                window.update_view()
            finally:
                for control, enabled in control_states:
                    control.setEnabled(enabled)
                if was_playing:
                    window.timer.start()
