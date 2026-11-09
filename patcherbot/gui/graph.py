import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QToolButton, QComboBox, QSizePolicy
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from matplotlib.colors import LinearSegmentedColormap, to_hex


from pyqtgraph import PlotWidget, PlotDataItem, mkPen




import threading

import numpy as np
from collections import deque
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.utils import FileLogger
from patcherbot.utils import EPhysLogger
import time

from datetime import datetime

# new import

from patcherbot.interface.graph import GraphInterface

__all__ = [
    "EPhysGraph",
    "PipetteEphysPanel",
    "CurrentProtocolGraph",
    "VoltageProtocolGraph",
    "LeakSubtractionGraph",
    "HoldingProtocolGraph",
    "OptogeneticStimProtocolGraph",
    "OptogeneticWavelengthProtocolGraph",
    "NoiseGraph",
]


# Colour family per pipette, by position in the window: pipette 0 keeps the
# original Prussian-blue gradient, pipette 1 is the same gradient in orange.
PIPETTE_GRADIENTS = [
    ("#003153", "#ffffff"),   # pipette 0: blue
    ("#b34700", "#ffffff"),   # pipette 1: orange
]


def _pipette_shades(position, fractions):
    """Hex colours along pipette ``position``'s gradient (0 = darkest, 1 = white)."""
    start, end = PIPETTE_GRADIENTS[position % len(PIPETTE_GRADIENTS)]
    cmap = LinearSegmentedColormap.from_list("", [start, end])
    return [to_hex(cmap(float(fraction))) for fraction in fractions]


class _PipetteTraces:
    """One pipette's data source, logger and plotted items in a protocol window."""

    def __init__(self, position, label, graph_interface, ephys_filename, recording_state_manager):
        self.position = position
        self.label = label
        self.graph_interface = graph_interface
        self.pipette_slug = "" if label is None else "_" + "_".join(label.strip().lower().split())
        self.file_prefix = self.pipette_slug.lstrip("_") + "_" if self.pipette_slug else ""
        # Files stay per pipette, named as before: <protocol>_pipette_<n>.
        self.ephys_logger = EPhysLogger(ephys_filename=f"{ephys_filename}{self.pipette_slug}",
                                        recording_state_manager=recording_state_manager)
        self.items = []
        self.latest = None

    @property
    def base_color(self):
        return PIPETTE_GRADIENTS[self.position % len(PIPETTE_GRADIENTS)][0]


class ProtocolGraph(QWidget):
    def __init__(self, graph_interface: "GraphInterface | dict[str, GraphInterface]",
                 recording_state_manager: RecordingStateManager,
                 window_title: str, y_label: str, y_unit: str,
                 x_label: str, x_unit: str, ephys_filename: str,
                 pipette_name: str | None = None):
        """
        :param graph_interface: one GraphInterface, or an ordered
            ``{label: GraphInterface}`` dict to draw several pipettes in this
            window (first entry blue, second orange, with a legend).
        """
        super().__init__()
        self.recording_state_manager = recording_state_manager
        if isinstance(graph_interface, dict):
            if not graph_interface:
                raise ValueError("ProtocolGraph needs at least one GraphInterface")
            interfaces = dict(graph_interface)
        else:
            interfaces = {pipette_name: graph_interface}
        self.pipettes = [
            _PipetteTraces(position, label, interface, ephys_filename, recording_state_manager)
            for position, (label, interface) in enumerate(interfaces.items())
        ]
        self.multi_pipette = len(self.pipettes) > 1

        # Single-pipette attributes (used by the optogenetic windows).
        first = self.pipettes[0]
        self.graph_interface = first.graph_interface
        self.ephys_logger = first.ephys_logger
        self.pipette_name = first.label
        self.pipette_slug = first.pipette_slug
        self.file_prefix = first.file_prefix

        # Set window title and layout
        self.window_title = window_title
        if not self.multi_pipette and first.label:
            window_title = f"{window_title} - {first.label}"
        self.setWindowTitle(window_title)
        layout = QVBoxLayout()

        # Create and configure the PlotWidget
        self.plotWidget = PlotWidget()
        self.plotWidget.setBackground("w")
        self.plotWidget.getAxis("left").setPen("k")
        self.plotWidget.getAxis("bottom").setPen("k")
        self.plotWidget.setLabel("left", y_label, units=y_unit)
        self.plotWidget.setLabel("bottom", x_label, units=x_unit)
        layout.addWidget(self.plotWidget)

        # Colour key: one entry per pipette (not tied to any plotted trace).
        self.legend = None
        self._legend_samples = {}
        if self.multi_pipette:
            self.legend = self.plotWidget.addLegend()
            for pipette in self.pipettes:
                self._set_legend_label(pipette, pipette.label)

        self.latestDisplayedData = None
        self._drawn_sample = None

        self.setLayout(layout)
        self.raise_()
        self.show()

        # Hide window initially and remap close event to simply hide it
        self.setHidden(True)
        self.closeEvent = lambda event: self.setHidden(True)

        # Set up timer to call update_plot() every updateDt ms
        self.updateTimer = QtCore.QTimer()
        self.updateDt = 1500  # milliseconds
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)
        if not self.updateTimer.isActive():
            logging.info(f"{window_title} Timer not active")

    # ----- helpers shared by the protocol windows -----
    def _set_legend_label(self, pipette, text):
        if self.legend is None:
            return
        old = self._legend_samples.pop(pipette.position, None)
        if old is not None:
            self.legend.removeItem(old)
        sample = PlotDataItem(pen=mkPen(pipette.base_color, width=3))
        self.legend.addItem(sample, text)
        self._legend_samples[pipette.position] = sample

    def _start_pipette_update(self, pipette):
        """Show the window, clear the previous run, and drop this pipette's old traces."""
        if self.isHidden():
            self.setHidden(False)
        index = self.recording_state_manager.sample_number
        if index != self._drawn_sample:
            # A new protocol run: start from an empty plot.
            self.plotWidget.clear()
            for other in self.pipettes:
                other.items = []
            self._drawn_sample = index
        for item in pipette.items:
            self.plotWidget.removeItem(item)
        pipette.items = []
        return index

    def _plot(self, pipette, x, y, pen):
        item = self.plotWidget.plot(x, y, pen=pen)
        pipette.items.append(item)
        return item

    def update_plot(self):
        """This method should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement update_plot()")


class CurrentProtocolGraph(ProtocolGraph):
    def __init__(self, graph_interface, recording_state_manager: RecordingStateManager,
                 *, pipette_name: str | None = None):
        super().__init__(graph_interface, recording_state_manager,
                         window_title="Current Protocol",
                         y_label="Voltage", y_unit="V",
                         x_label="Time", x_unit="s",
                         ephys_filename="CurrentProtocol", pipette_name=pipette_name)

    def update_plot(self):
        finished = []
        for pipette in self.pipettes:
            daq = pipette.graph_interface.daq
            data = daq.current_protocol_data
            # Nothing new, or the same list we already drew.
            if data is None or data is pipette.latest:
                continue
            index = self._start_pipette_update(pipette)

            # One shade of this pipette's colour per current step.
            color_range = daq.pulseRange
            color_range = 1 if color_range is None else int(color_range)
            logging.debug(f"color range: {color_range}")
            colors = _pipette_shades(pipette.position,
                                     [i / color_range for i in range(max(color_range, len(data)))])
            pulses = daq.pulses

            for i, graph in enumerate(data):
                timeData, respData, readData = graph[0], graph[1], graph[2]
                self._plot(pipette, timeData, respData, colors[i])
                pulse = str(pulses[i]) if pulses is not None else str(i)
                pipette.ephys_logger.write_ephys_data(index, timeData, readData, respData,
                                                      colors[i] + "_" + pulse)
            pipette.latest = data
            if len(data) >= color_range:
                finished.append(pipette)

        # Save after every pipette is drawn, so each saved plot shows them all.
        for pipette in finished:
            pipette.ephys_logger.save_ephys_plot(self.recording_state_manager.sample_number,
                                                 self.plotWidget)
            pipette.graph_interface.daq.current_protocol_data = None  # Reset after saving
            pipette.latest = None
        self.latestDisplayedData = [pipette.latest for pipette in self.pipettes]


class VoltageProtocolGraph(ProtocolGraph):
    def __init__(self, graph_interface, recording_state_manager: RecordingStateManager,
                 *, pipette_name: str | None = None):
        super().__init__(graph_interface, recording_state_manager,
                         window_title="Voltage Protocol (Membrane Test)",
                         y_label="PicoAmps", y_unit="A",
                         x_label="Time", x_unit="s",
                         ephys_filename="VoltageProtocol", pipette_name=pipette_name)

    def update_plot(self):
        drawn = []
        for pipette in self.pipettes:
            daq = pipette.graph_interface.daq
            sweeps_raw = daq.voltage_protocol_data
            membrane_test = daq.voltage_membrane_test
            if isinstance(sweeps_raw, np.ndarray) or sweeps_raw is None:
                sweeps = []
            else:
                sweeps = list(sweeps_raw)
            if membrane_test is None and not sweeps:
                continue
            index = self._start_pipette_update(pipette)

            # Render sweep traces with this pipette's gradient
            if sweeps:
                sweep_count = len(sweeps)
                colors = _pipette_shades(pipette.position,
                                         [i / max(sweep_count - 1, 1) for i in range(sweep_count)])
                steps = getattr(daq, "vclamp_steps", None)
                for i, trace in enumerate(sweeps):
                    timeData, respData, commandData = trace
                    self._plot(pipette, timeData, respData, colors[i])
                    if steps is not None and i < len(steps):
                        step_label = f"{int(round(steps[i] * 1e3))}mV"
                    else:
                        step_label = f"step{i}"
                    pipette.ephys_logger.write_ephys_data(
                        index, timeData, commandData, respData, f"{colors[i]}_{step_label}")
                drawn.append((pipette, None))

            # Plot and export membrane-test data
            if membrane_test is not None:
                # Single pipette: the original colour (black); several: the
                # pipette's own darkest shade so traces can be told apart.
                mem_color = (pipette.base_color if self.multi_pipette
                             else membrane_test.get("color", "#000000"))
                timeData = membrane_test.get("time")
                respData = membrane_test.get("response")
                commandData = membrane_test.get("command")
                step_mV = membrane_test.get("step_mV")
                sweep_hold_mV = membrane_test.get("sweep_hold_mV", membrane_test.get("hold_mV"))
                fit_ok = membrane_test.get("fit_ok", True)

                if timeData is not None and respData is not None:
                    self._plot(pipette, timeData, respData, mem_color)
                self._show_fit_state(pipette, fit_ok)

                step_label = f"{int(round(step_mV))}mV" if step_mV is not None else "step"
                hold_label = f"{int(round(sweep_hold_mV))}mV" if sweep_hold_mV is not None else "hold"
                if timeData is not None and respData is not None:
                    pipette.ephys_logger.write_ephys_data(
                        index, timeData, commandData, respData, mem_color,
                        filename_override=(f"{pipette.file_prefix}MembraneTest_{index}_"
                                           f"{mem_color}_{hold_label}_{step_label}"))
                    drawn.append((pipette, f"{pipette.file_prefix}MembraneTest_{index}"))

            # Reset DAQ buffers after plotting
            daq.voltage_protocol_data = None
            daq.voltage_membrane_test = None
            daq.vclamp_steps = None
            daq.vclamp_hold_value = None

        # Save after every pipette is drawn, so each saved plot shows them all.
        index = self.recording_state_manager.sample_number
        for pipette, filename_override in drawn:
            if filename_override is None:
                pipette.ephys_logger.save_ephys_plot(index, self.plotWidget)
            else:
                pipette.ephys_logger.save_ephys_plot(index, self.plotWidget,
                                                     filename_override=filename_override)
        if drawn:
            self.latestDisplayedData = len(drawn)

    def _show_fit_state(self, pipette, fit_ok):
        """Mark a membrane test whose capacitance fit failed (the trace shown
        is then the first, unfit one)."""
        suffix = "" if fit_ok else " (no Cm fit)"
        if self.multi_pipette:
            self._set_legend_label(pipette, f"{pipette.label}{suffix}")
        else:
            base = self.window_title + (f" - {pipette.label}" if pipette.label else "")
            self.setWindowTitle(base + suffix)


class LeakSubtractionGraph(ProtocolGraph):
    def __init__(self, graph_interface, recording_state_manager: RecordingStateManager,
                 *, pipette_name: str | None = None):
        super().__init__(
            graph_interface,
            recording_state_manager,
            window_title="Leak Subtraction (P/4)",
            y_label="PicoAmps",
            y_unit="A",
            x_label="Time",
            x_unit="s",
            ephys_filename="LeakSubtraction",
            pipette_name=pipette_name,
        )

    def update_plot(self):
        drawn = []
        for pipette in self.pipettes:
            daq = pipette.graph_interface.daq
            leak_data = getattr(daq, "leak_subtraction_data", None)
            if leak_data is None or len(leak_data) == 0:
                continue
            index = self._start_pipette_update(pipette)

            sweep_count = len(leak_data)
            colors = _pipette_shades(pipette.position,
                                     [i / max(sweep_count - 1, 1) for i in range(sweep_count)])
            for i, entry in enumerate(leak_data):
                timeData = entry["time"]
                respData = entry["response"]
                commandData = entry["command"]
                self._plot(pipette, timeData, respData, colors[i])

                target_voltage = entry.get("target_voltage")
                if target_voltage is not None:
                    step_label = f"{int(round(target_voltage * 1e3))}mV"
                else:
                    step_label = f"step{i}"
                pipette.ephys_logger.write_ephys_data(
                    index, timeData, commandData, respData, f"{colors[i]}_{step_label}")

            drawn.append(pipette)
            daq.leak_subtraction_data = None
            daq.leak_subtraction_meta = None

        index = self.recording_state_manager.sample_number
        for pipette in drawn:
            pipette.ephys_logger.save_ephys_plot(
                index, self.plotWidget,
                filename_override=f"{pipette.file_prefix}LeakSubtraction_{index}")
        if drawn:
            self.latestDisplayedData = len(drawn)


class HoldingProtocolGraph(ProtocolGraph):
    def __init__(self, graph_interface, recording_state_manager: RecordingStateManager,
                 *, pipette_name: str | None = None):
        super().__init__(graph_interface, recording_state_manager,
                         window_title="Holding Protocol (E/I PSC Test)",
                         y_label="PicoAmps", y_unit="A",
                         x_label="Time", x_unit="s",
                         ephys_filename="HoldingProtocol", pipette_name=pipette_name)

    def update_plot(self):
        drawn = []
        for pipette in self.pipettes:
            daq = pipette.graph_interface.daq
            data = daq.holding_protocol_data
            if data is None:
                continue
            index = self._start_pipette_update(pipette)
            # Single pipette: black as before; several: the pipette's colour.
            color = pipette.base_color if self.multi_pipette else "k"
            self._plot(pipette, data[0, :], data[1, :], color)
            pipette.ephys_logger.write_ephys_data(index, data[0, :], data[1, :], data[2, :], color)
            drawn.append(pipette)
            self.latestDisplayedData = data.copy()
            daq.holding_protocol_data = None  # Reset after plotting

        index = self.recording_state_manager.sample_number
        for pipette in drawn:
            pipette.ephys_logger.save_ephys_plot(index, self.plotWidget)


class OptogeneticBaseGraph(ProtocolGraph):
    def __init__(
        self,
        graph_interface: GraphInterface,
        recording_state_manager: RecordingStateManager,
        *,
        window_title: str,
        protocol_key: str,
        pipette_name: str | None = None,
    ):
        super().__init__(
            graph_interface,
            recording_state_manager,
            window_title=window_title,
            y_label="PicoAmps",
            y_unit="A",
            x_label="Time",
            x_unit="s",
            ephys_filename="OptogeneticProtocol",
            pipette_name=pipette_name,
        )
        self.protocol_key = protocol_key

    def update_plot(self):
        daq = self.graph_interface.daq
        entry = daq.pop_optogenetic_entry(self.protocol_key)
        if entry is None:
            return

        index = self.recording_state_manager.sample_number

        if self.isHidden():
            self.setHidden(False)

        self.plotWidget.clear()
        data = entry.get("data") or []
        pulses = entry.get("pulses") or []
        pulse_range = entry.get("pulse_range")
        stim_data = entry.get("stim_data")
        protocol_type = entry.get("protocol_type")
        if protocol_type is None and stim_data:
            protocol_type = stim_data[0].get("protocol_type")

        if pulse_range is None:
            pulse_range = len(data)

        if not data:
            return

        protocol_label = "unknown"
        if protocol_type is not None:
            name = getattr(protocol_type, "name", None)
            if isinstance(name, str) and name.strip():
                protocol_label = name.strip().lower()
            else:
                text = str(protocol_type).strip().lower()
                if text:
                    protocol_label = text

        rep_indices = []
        power_values = []
        for pulse in pulses:
            try:
                rep_indices.append(int(pulse.get("replicate", 0)))
            except (TypeError, ValueError, AttributeError):
                rep_indices.append(0)
            try:
                power = float(pulse.get("power_percent"))
            except (TypeError, ValueError, AttributeError):
                power = None
            if power is not None and np.isfinite(power):
                power_values.append(power)

        rep_count = max(rep_indices) + 1 if rep_indices else 1
        power_keys = []
        power_rank = {}
        segment = 1.0
        if protocol_label == "power":
            power_keys = sorted({round(p, 6) for p in power_values}, reverse=True)
            power_rank = {value: idx for idx, value in enumerate(power_keys)}
            segment = 1.0 / max(len(power_keys), 1)

        color_map = EPhysGraph.laserColorMap
        for idx, trace in enumerate(data):
            if trace is None or len(trace) < 3:
                continue
            timeData, respData, readData = trace
            pulse = pulses[idx] if idx < len(pulses) else {}
            wavelength = pulse.get("wavelength") if isinstance(pulse, dict) else None
            try:
                power = float(pulse.get("power_percent")) if isinstance(pulse, dict) else None
            except (TypeError, ValueError):
                power = None
            rep_index = rep_indices[idx] if idx < len(rep_indices) else 0

            color_label = "unknown"
            base_hex = "#e0e0e0"
            if wavelength is not None:
                if isinstance(wavelength, str):
                    cleaned = wavelength.strip()
                    if cleaned.isdigit():
                        wavelength = int(cleaned)
                    else:
                        key = cleaned.lower()
                        if key in color_map:
                            color_label, base_hex = key, color_map[key][1]
                if isinstance(wavelength, int):
                    options = self.graph_interface.get_laser_wavelength_options()
                    if options:
                        index = max(1, min(len(options), wavelength))
                        wavelength = options[index - 1]
                    else:
                        default_keys = list(color_map.keys())
                        if default_keys:
                            key = default_keys[max(1, min(len(default_keys), wavelength)) - 1]
                            color_label, base_hex = key, color_map[key][1]
                            wavelength = None
                if wavelength is not None and not isinstance(wavelength, int):
                    name = getattr(wavelength, "name", str(wavelength))
                    key = name.strip().lower()
                    if key in color_map:
                        color_label, base_hex = key, color_map[key][1]
                    elif key:
                        color_label = key

            if protocol_label == "power":
                power_key = round(power, 6) if power is not None and np.isfinite(power) else None
                rank = power_rank.get(power_key, len(power_keys) - 1 if power_keys else 0)
                ratio = rank * segment
                if rep_count > 0:
                    ratio += (rep_index / rep_count) * segment
            else:
                ratio = rep_index / max(rep_count - 1, 1)

            max_whiten_ratio = 0.7
            ratio = max(0.0, min(max_whiten_ratio, float(ratio)))
            cmap = LinearSegmentedColormap.from_list("", [base_hex, "#ffffff"])
            color_hex = to_hex(cmap(ratio))
            self.plotWidget.plot(timeData, respData, pen=color_hex)

            safe_label = str(color_label).strip().lower().replace(" ", "")
            rep_tag = f"rep{rep_index}"
            marker = f"{protocol_label}_{safe_label}_{rep_tag}_{color_hex}"
            self.ephys_logger.write_ephys_data(
                index,
                timeData,
                readData,
                respData,
                marker
            )

            if idx == pulse_range - 1:
                self.ephys_logger.save_optogenetic_plot(index, self.plotWidget, protocol_type)

        if stim_data:
            self.ephys_logger.write_optogenetic_stim_data(index, stim_data, protocol_type)

        self.latestDisplayedData = entry
        daq.optogenetic_protocol_data = None
        daq.optogenetic_stim_data = None
        daq.optogenetic_protocol_type = None


class OptogeneticStimProtocolGraph(OptogeneticBaseGraph):
    def __init__(self, graph_interface: GraphInterface, recording_state_manager: RecordingStateManager,
                 *, pipette_name: str | None = None):
        super().__init__(
            graph_interface,
            recording_state_manager,
            window_title="Optogenetic Stim Protocol",
            protocol_key="power",
            pipette_name=pipette_name,
        )


class OptogeneticWavelengthProtocolGraph(OptogeneticBaseGraph):
    def __init__(self, graph_interface: GraphInterface, recording_state_manager: RecordingStateManager,
                 *, pipette_name: str | None = None):
        super().__init__(
            graph_interface,
            recording_state_manager,
            window_title="Optogenetic Wavelength Protocol",
            protocol_key="wavelength",
            pipette_name=pipette_name,
        )

class NoiseGraph(QWidget):
    noise_state_changed = pyqtSignal(bool)

    def __init__(self, graph_interface: GraphInterface):
        super().__init__()
        self.graph_interface = graph_interface
        self.setWindowTitle("Noise Graph (4-10 ms)")

        main_layout = QVBoxLayout()
        plot_layout = QVBoxLayout()
        stats_layout = QHBoxLayout()

        self.zoomPlot = PlotWidget()
        self.fftPlot = PlotWidget()
        for plot in [self.zoomPlot, self.fftPlot]:
            plot.setBackground("w")
            plot.getAxis("left").setPen("k")
            plot.getAxis("bottom").setPen("k")

        self.zoomPlot.setLabel("left", "Current", units="A")
        self.zoomPlot.setLabel("bottom", "Time", units="ms")
        self.fftPlot.setLabel("left", "FFT Magnitude", units="A")
        self.fftPlot.setLabel("bottom", "Frequency", units="Hz")

        plot_layout.addWidget(self.zoomPlot)
        plot_layout.addWidget(self.fftPlot)

        self.p2pLabel = QLabel("Peak-to-peak: N/A")
        self.stdLabel = QLabel("Std dev: N/A")
        self.avgP2pLabel = QLabel("Avg P2P (1 ms): N/A")
        stats_layout.addWidget(self.p2pLabel)
        stats_layout.addWidget(self.stdLabel)
        stats_layout.addWidget(self.avgP2pLabel)
        stats_layout.addStretch(1)

        main_layout.addLayout(plot_layout)
        main_layout.addLayout(stats_layout)
        self.setLayout(main_layout)

        self.updateDt = 42  # ms
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.update_plot)

        self.setHidden(True)
        self.closeEvent = lambda event: (event.ignore(), self.stop())

    def is_active(self):
        return self.updateTimer.isActive()

    def start(self):
        if not self.updateTimer.isActive():
            self.updateTimer.start(self.updateDt)
        self.setHidden(False)
        self.raise_()
        self.noise_state_changed.emit(True)

    def stop(self):
        if self.updateTimer.isActive():
            self.updateTimer.stop()
        self.setHidden(True)
        self.noise_state_changed.emit(False)

    def update_plot(self):
        metrics = self.graph_interface.get_noise_metrics()
        if not metrics:
            return

        window_time = metrics["window_time"]
        window_resp = metrics["window_resp"]
        window_time_ms = window_time * 1000.0
        self.zoomPlot.clear()
        self.zoomPlot.plot(window_time_ms, window_resp, pen="k")
        self.zoomPlot.setXRange(4.0, 10.0, padding=0.0)

        self.p2pLabel.setText(f"Peak-to-peak: {metrics['p2p']:.3e} A")
        self.stdLabel.setText(f"Std dev: {metrics['std']:.3e} A")
        self.avgP2pLabel.setText(f"Avg P2P (1 ms): {metrics['avg_p2p']:.3e} A")

        freqs = metrics.get("freqs")
        fft_magnitude = metrics.get("fft_magnitude")
        self.fftPlot.clear()
        if freqs is not None and fft_magnitude is not None:
            self.fftPlot.plot(freqs, fft_magnitude, pen="k")

class PipetteEphysPanel(QWidget):
    """One pipette's column in the Electrophysiology window.

    Shows the command, response, pressure and resistance traces for one
    GraphInterface and records them to its own rig-recorder file.
    """

    def __init__(self, label, graph_interface: GraphInterface,
                 recording_state_manager: RecordingStateManager,
                 recorder_filename="graph_recording", update_dt_ms=42,
                 show_title=True, show_axis_titles=True):
        super().__init__()
        self.label = label
        self.graph_interface = graph_interface
        self.recording_state_manager = recording_state_manager
        self.update_dt_ms = update_dt_ms
        self.selected = False

        self.titleLabel = QLabel(label)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setVisible(show_title)

        self.cmdPlot = PlotWidget()
        self.respPlot = PlotWidget()
        self.pressurePlot = PlotWidget()
        self.resistancePlot = PlotWidget()
        self.plots = (self.cmdPlot, self.respPlot, self.pressurePlot, self.resistancePlot)
        for plot in self.plots:
            plot.setBackground("w")
            plot.getAxis("left").setPen("k")
            plot.getAxis("bottom").setPen("k")

        # Only the leftmost column spells out the y-axis names; the others
        # show units alone to leave more room for the traces.
        def left(text):
            return text if show_axis_titles else None
        self.cmdPlot.setLabel("left", left("Command Voltage"), units="V")
        self.cmdPlot.setLabel("bottom", "Time", units="s")
        self.respPlot.setLabel("left", left("Current (resp)"), units="A")
        self.respPlot.setLabel("bottom", "Time", units="s")
        self.pressurePlot.setLabel("left", left("Pressure"), units="mbar")
        self.pressurePlot.setLabel("bottom", "Time", units="s")
        self.resistancePlot.setLabel("left", left("Resistance"), units="Ohms")
        self.resistancePlot.setLabel("bottom", "Samples", units="")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.titleLabel)
        for plot in self.plots:
            layout.addWidget(plot)
        self.setLayout(layout)
        # Columns share the window's width rather than widening it, so more
        # pipettes means a narrower (time-compressed) view of each.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.pressureData = deque(maxlen=100)
        self.resistanceDeque = deque(maxlen=100)
        self.lastReadData = None
        self.lastrespData = None
        self.last_pressure = None
        self.last_packet = None

        self.recorder = FileLogger(
            recording_state_manager,
            folder_path="experiments/Data/rig_recorder_data/",
            recorder_filename=recorder_filename,
        )
        self._update_title()

    def set_selected(self, selected):
        self.selected = bool(selected)
        self._update_title()

    def _update_title(self, resistance=None):
        parts = [self.label]
        if resistance is not None:
            parts.append(f"R: {resistance:.2f} MΩ")
        if self.graph_interface.pressure_is_simulated():
            parts.append("sim. pressure")
        self.titleLabel.setText(" · ".join(parts))
        self.titleLabel.setStyleSheet(
            "font-weight: bold; background-color: #cfe3ff;" if self.selected else "")

    def refresh(self):
        """Fetch this pipette's latest pressure and DAQ packet, plot and record them."""
        pressure = self.graph_interface.get_last_pressure()
        # No measurement yet (e.g. right after start-up): show 0 mbar.
        pressure = int(pressure) if pressure is not None else 0
        self.last_pressure = pressure
        self.pressureData.append(pressure)
        pressureX = [i * self.update_dt_ms / 1000 for i in range(len(self.pressureData))]
        self.pressurePlot.clear()
        self.pressurePlot.plot(pressureX, list(self.pressureData))

        daq_data = self.graph_interface.get_last_data()
        self.last_packet = daq_data
        if daq_data is None:
            return
        timeData = daq_data.get("timeData")
        respData = daq_data.get("respData")
        readData = daq_data.get("readData")
        totalResistance = daq_data.get("totalResistance")

        if timeData is not None and readData is not None:
            self.cmdPlot.clear()
            self.cmdPlot.plot(timeData, readData, pen="k")
            self.lastReadData = readData

        if timeData is not None and respData is not None:
            self.respPlot.clear()
            self.respPlot.plot(timeData, respData, pen="k")
            self.lastrespData = respData

        if totalResistance is not None:
            self.resistanceDeque.append(totalResistance)
            x_vals = list(range(len(self.resistanceDeque)))
            self.resistancePlot.clear()
            self.resistancePlot.plot(x_vals, list(self.resistanceDeque), pen="k")
        self._update_title(totalResistance)

        if self.recording_state_manager.is_recording_enabled():
            try:
                self.recorder.write_graph_data(
                    datetime.now().timestamp(),
                    pressure,
                    totalResistance,
                    list(self.lastrespData) if self.lastrespData is not None else [],
                    list(self.lastReadData) if self.lastReadData is not None else []
                )
            except Exception as e:
                logging.error(f"Error writing graph data for {self.label}: {e}")


class EPhysGraph(QWidget):
    pressureLowerBound = -450
    pressureUpperBound = 730
    laserPowerLowerBound = 0
    laserPowerUpperBound = 100
    laserColorMap = {
        "red": ("Red", "#ff3b30"),
        "green": ("Green", "#34c759"),
        "cyan": ("Cyan", "#00bcd4"),
        "uv": ("UV", "#6a5acd"),
        "blue": ("Blue", "#007aff"),
        "infrared": ("Infrared", "#000000"),
    }

    def __init__(self, graph_interface: "GraphInterface | dict[str, GraphInterface]",
                 recording_state_manager: RecordingStateManager):
        """
        Initialize the electrophysiology GUI.
        :param graph_interface: A GraphInterface, or an ordered ``{label: GraphInterface}``
            dict with one entry per pipette. Every pipette gets its own column of
            plots (first entry leftmost); the bottom bar controls the pipette
            chosen in its "Control" picker.
        :param recording_state_manager: The recording state manager used for data logging.
        """
        super().__init__()
        self.setWindowTitle("Electrophysiology")
        self.recording_state_manager = recording_state_manager
        if isinstance(graph_interface, dict):
            if not graph_interface:
                raise ValueError("EPhysGraph needs at least one GraphInterface")
            self.graph_interfaces = dict(graph_interface)
        else:
            self.graph_interfaces = {"Pipette 0": graph_interface}
        self.graph_interface = next(iter(self.graph_interfaces.values()))  # ONLY import GraphInterface!
        self.updateDt = 42   # ms

        # One column of plots per pipette. The first pipette keeps the
        # original recorder file name so existing analysis scripts still work.
        several = len(self.graph_interfaces) > 1
        self.panels = {}
        for index, (label, interface) in enumerate(self.graph_interfaces.items()):
            suffix = "" if index == 0 else "_" + "_".join(label.strip().lower().split())
            self.panels[label] = PipetteEphysPanel(
                label, interface, recording_state_manager,
                recorder_filename="graph_recording" + suffix,
                update_dt_ms=self.updateDt,
                show_title=several,
                show_axis_titles=index == 0,
            )
        self.selected_panel = next(iter(self.panels.values()))
        self.selected_panel.set_selected(several)

        # Build bottom control bar.
        self.bottomBar = QWidget()
        bottomBarLayout = QHBoxLayout()
        bottomBarLayout.setContentsMargins(0, 0, 0, 0)  # Preserve zero margins as in old UI.
        self.bottomBar.setLayout(bottomBarLayout)

        # Which pipette the bar controls; only shown with several pipettes.
        self.pipettePicker = None
        if several:
            self.pipettePickerLabel = QLabel("Control:")
            bottomBarLayout.addWidget(self.pipettePickerLabel)
            self.pipettePicker = QComboBox()
            self.pipettePicker.addItems(list(self.graph_interfaces))
            self.pipettePicker.currentTextChanged.connect(self._select_pipette)
            bottomBarLayout.addWidget(self.pipettePicker)

        self.resistanceLabel = QLabel("Resistance:")
        bottomBarLayout.addWidget(self.resistanceLabel)

        # Toggle button for cell vs. bath mode.
        self.modelType = QPushButton("Bath Mode")
        self.modelType.setStyleSheet("background-color: blue; color: white; border-radius: 5px; padding: 5px;")
        bottomBarLayout.addWidget(self.modelType)

        self.accessResistanceLabel = QLabel("Access Resistance: N/A")
        bottomBarLayout.addWidget(self.accessResistanceLabel)
        self.membraneResistanceLabel = QLabel("Membrane Resistance: N/A")
        bottomBarLayout.addWidget(self.membraneResistanceLabel)
        self.membraneCapacitanceLabel = QLabel("Membrane Capacitance: N/A")
        bottomBarLayout.addWidget(self.membraneCapacitanceLabel)

        self.pressureLabel = QLabel("Pressure:")
        bottomBarLayout.addWidget(self.pressureLabel)

        # Pressure command text box.
        self.pressureCommandBox = QLineEdit()
        self.pressureCommandBox.setMaxLength(5)
        self.pressureCommandBox.setFixedWidth(100)
        initial_pressure = self.graph_interface.get_last_pressure() or 0
        self.pressureCommandBox.setPlaceholderText(f"{initial_pressure} mbar")
        self.pressureCommandBox.setValidator(QtGui.QIntValidator(self.pressureLowerBound, self.pressureUpperBound))
        self.pressureCommandBox.returnPressed.connect(self.pressureCommandBoxReturnPressed)
        bottomBarLayout.addWidget(self.pressureCommandBox)

        # Pressure slider.
        self.pressureCommandSlider = QSlider(QtCore.Qt.Horizontal)
        self.pressureCommandSlider.setMinimum(self.pressureLowerBound)
        self.pressureCommandSlider.setMaximum(self.pressureUpperBound)
        self.pressureCommandSlider.setValue(initial_pressure)
        self.pressureCommandSlider.setTickInterval(100)
        self.pressureCommandSlider.setTickPosition(QSlider.TicksBelow)
        self.pressureCommandSlider.sliderReleased.connect(self.pressureCommandSliderChanged)
        bottomBarLayout.addWidget(self.pressureCommandSlider)

        # Up and down buttons for fine pressure control.
        self.upButton = QToolButton()
        self.upButton.setArrowType(QtCore.Qt.UpArrow)
        self.upButton.setFixedWidth(50)
        self.upButton.clicked.connect(self.incrementPressure)
        bottomBarLayout.addWidget(self.upButton)

        self.downButton = QToolButton()
        self.downButton.setArrowType(QtCore.Qt.DownArrow)
        self.downButton.setFixedWidth(50)
        self.downButton.clicked.connect(self.decrementPressure)
        bottomBarLayout.addWidget(self.downButton)

        # Atmospheric pressure toggle.
        self.atmosphericPressureButton = QPushButton("ATM Pressure OFF")
        bottomBarLayout.addWidget(self.atmosphericPressureButton)
        self.atmosphericPressureButton.clicked.connect(self.togglePressure)
        self.atmtoggle = True

        # Noise check toggle.
        self.noiseButton = QPushButton("Check Noise")
        bottomBarLayout.addWidget(self.noiseButton)
        self.noiseButton.clicked.connect(self.toggleNoise)

        # Zap controls.
        self.zapLabel = QLabel("Zap Duration:")
        bottomBarLayout.addWidget(self.zapLabel)
        self.zapDurationDropdown = QComboBox()
        self.zapDurationDropdown.setFixedWidth(100)
        zap_options = ["25 µs", "50 µs", "100 µs", "200 µs", "500 µs", "1 ms", "10 ms", "20 ms", "50 ms"]
        for option in zap_options:
            self.zapDurationDropdown.addItem(option)
        self.zapDurationDropdown.currentIndexChanged.connect(self.handle_zap_duration_change)
        bottomBarLayout.addWidget(self.zapDurationDropdown)
        self.zapButton = QPushButton("Zap")
        self.zapButton.setFixedWidth(50)
        self.zapButton.clicked.connect(self.handle_zap_button_press)
        bottomBarLayout.addWidget(self.zapButton)
        for interface in self.graph_interfaces.values():
            interface.set_zap_duration(25e-6)  # Default zap duration in seconds

        # Laser controls.
        initial_power = self.graph_interface.get_laser_power()
        if initial_power is None:
            initial_power = 0
        initial_power = int(round(initial_power))

        self.laserPowerLabel = QLabel(f"Power: {initial_power} %")
        bottomBarLayout.addWidget(self.laserPowerLabel)

        self.laserPowerBox = QLineEdit()
        self.laserPowerBox.setMaxLength(3)
        self.laserPowerBox.setFixedWidth(80)
        self.laserPowerBox.setValidator(QtGui.QIntValidator(self.laserPowerLowerBound, self.laserPowerUpperBound))
        self.laserPowerBox.setPlaceholderText(f"Set to: {initial_power} %")
        self.laserPowerBox.returnPressed.connect(self.laserPowerBoxReturnPressed)
        bottomBarLayout.addWidget(self.laserPowerBox)

        self.laserLeftButton = QToolButton()
        self.laserLeftButton.setArrowType(QtCore.Qt.LeftArrow)
        self.laserLeftButton.setFixedWidth(30)
        self.laserLeftButton.clicked.connect(self.handle_laser_left)
        bottomBarLayout.addWidget(self.laserLeftButton)

        self.laserToggleButton = QPushButton("Off")
        self.laserToggleButton.setFixedWidth(70)
        self.laserToggleButton.clicked.connect(self.handle_laser_toggle)
        bottomBarLayout.addWidget(self.laserToggleButton)

        self.laserRightButton = QToolButton()
        self.laserRightButton.setArrowType(QtCore.Qt.RightArrow)
        self.laserRightButton.setFixedWidth(30)
        self.laserRightButton.clicked.connect(self.handle_laser_right)
        bottomBarLayout.addWidget(self.laserRightButton)

        bottomBarLayout.addStretch(1)
        self.bottomBar.setMaximumHeight(20)
        self.bottomBar.setMinimumHeight(20)

        # Compose main layout: pipette columns side by side, bar underneath.
        panelLayout = QHBoxLayout()
        panelLayout.setContentsMargins(0, 0, 0, 0)
        for panel in self.panels.values():
            panelLayout.addWidget(panel, 1)
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(panelLayout)
        mainLayout.addWidget(self.bottomBar)
        self.setLayout(mainLayout)

        # cellMode type switch
        self.modelType.clicked.connect(self.toggleModeType)

        # QTimer for periodic GUI updates
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        self.noiseGraph = NoiseGraph(self.graph_interface)
        self.noiseGraph.noise_state_changed.connect(self.updateNoiseButton)

        self.show()
        self.raise_()

    def _select_pipette(self, label):
        """Point the bottom bar at another pipette.

        Every pipette keeps plotting in its own column; this only changes which
        pipette the bar's controls and readouts (and the noise check) act on.
        The laser is shared, so its cached settings carry over.
        """
        new_panel = self.panels.get(label)
        if new_panel is None or new_panel is self.selected_panel:
            return
        new_interface = new_panel.graph_interface
        new_interface.adopt_laser_state(self.graph_interface)
        self.selected_panel.set_selected(False)
        new_panel.set_selected(True)
        self.selected_panel = new_panel
        self.graph_interface = new_interface
        self.noiseGraph.graph_interface = new_interface

        self.resistanceLabel.setText("Resistance:")
        self.accessResistanceLabel.setText("Access Resistance: N/A")
        self.membraneResistanceLabel.setText("Membrane Resistance: N/A")
        self.membraneCapacitanceLabel.setText("Membrane Capacitance: N/A")

        self.handle_zap_duration_change()   # apply the selected zap duration to this amplifier
        self.updateModeType()
        self.updatePressureATM()
        self._update_bar()

    def update_plot(self):
        """
        Periodically refresh every pipette's column, then the bottom bar from
        the pipette it controls.
        """
        for panel in self.panels.values():
            panel.refresh()
        self._update_bar()
        self.update_laser_controls()

    def _update_bar(self):
        """Show the controlled pipette's latest pressure and DAQ readouts in the bar."""
        panel = self.selected_panel
        pressure = panel.last_pressure
        if pressure is not None:
            pressure_set = int(self.graph_interface.get_pressure())
            # Update the slider only if the user is not interacting with it.
            if not self.pressureCommandSlider.isSliderDown():
                self.pressureCommandSlider.setValue(pressure)
            self.pressureCommandBox.setPlaceholderText(f"Set to: {pressure_set} mbar")
            simulated = " (simulated)" if self.graph_interface.pressure_is_simulated() else ""
            self.pressureLabel.setText(f"Pressure{simulated}: {pressure:.2f} mbar")
            self.updatePressureATM()

        daq_data = panel.last_packet
        if daq_data is None:
            return
        totalResistance = daq_data.get("totalResistance")
        accessResistance = daq_data.get("accessResistance")
        membraneResistance = daq_data.get("membraneResistance")
        membraneCapacitance = daq_data.get("membraneCapacitance")
        if totalResistance is not None:
            self.resistanceLabel.setText(f"Resistance: {totalResistance:.2f} MΩ")
        if accessResistance is not None:
            self.accessResistanceLabel.setText(f"Access Resistance: {accessResistance:.2f} MΩ")
        if membraneResistance is not None:
            self.membraneResistanceLabel.setText(f"Membrane Resistance: {membraneResistance:.2f} MΩ")
        if membraneCapacitance is not None:
            self.membraneCapacitanceLabel.setText(f"Membrane Capacitance: {membraneCapacitance:.2f} pF")
        self.updateModeType()

    def pressureCommandSliderChanged(self):
        """
        On slider release, set the pressure via GraphInterface.
        """
        pressure = self.pressureCommandSlider.value()
        self.graph_interface.set_pressure(pressure)
        # Update the text with the new set pressure.
 

    def pressureCommandBoxReturnPressed(self):
        """
        When a pressure value is entered in the text box, update the setpoint.
        it triggers the slider to change which updates the pressure.

        """
        try:
            text = self.pressureCommandBox.text().replace("Set to:", "").replace("mbar", "").strip()
            self.pressureCommandBox.clear()
            pressure = float(text)
            pressure = max(self.pressureLowerBound, min(self.pressureUpperBound, pressure))
            self.pressureCommandSlider.setValue(int(pressure))
            self.pressureCommandSliderChanged()

        except ValueError:
            logging.warning("Invalid pressure input.")
        except Exception as e:
            logging.error(f"Error in pressureCommandBoxReturnPressed: {e}")

    def incrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value + 5
        if new_value <= self.pressureUpperBound:
            self.pressureCommandSlider.setValue(new_value)
            self.pressureCommandSlider.sliderReleased.emit()

    def decrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value - 5
        if new_value >= self.pressureLowerBound:
            self.pressureCommandSlider.setValue(new_value)
            self.pressureCommandSlider.sliderReleased.emit()

    def togglePressure(self):
        """
        Toggle atmospheric pressure mode using GraphInterface.
        """
        if self.atmtoggle:
            self.atmosphericPressureButton.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
            self.atmosphericPressureButton.setText("ATM Pressure ON")
            self.graph_interface.set_ATM(True)
        else:
            self.atmosphericPressureButton.setStyleSheet("")
            self.atmosphericPressureButton.setText("ATM Pressure OFF")
            self.graph_interface.set_ATM(False)
        self.atmtoggle = not self.atmtoggle

    def updatePressureATM(self):
        """
        Update the atmospheric pressure button based on the current state.
        """
        if self.graph_interface.get_ATM():
            self.atmosphericPressureButton.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
            self.atmosphericPressureButton.setText("ATM Pressure ON")
        else:
            self.atmosphericPressureButton.setStyleSheet("")
            self.atmosphericPressureButton.setText("ATM Pressure OFF")

    def toggleModeType(self):
        """
        Toggle between cell and bath modes using GraphInterface.
        This requires GraphInterface to provide toggle_cell_mode() and get_cell_mode() methods.
        """
        mode = self.graph_interface.getCellMode()
        if  not mode:
            self.modelType.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
            self.modelType.setText("Cell Mode")
        else:
            self.modelType.setStyleSheet("background-color: blue; color: white; border-radius: 5px; padding: 5px;")
            self.modelType.setText("Bath Mode")

        self.graph_interface.setCellMode(not mode)

    def updateModeType(self):
        if self.graph_interface.getCellMode():
            self.modelType.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
            self.modelType.setText("Cell Mode")
        else:
            self.modelType.setStyleSheet("background-color: blue; color: white; border-radius: 5px; padding: 5px;")
            self.modelType.setText("Bath Mode")

    def handle_zap_button_press(self):
        """
        Provide visual feedback and execute the zap command via GraphInterface.
        """
        self.zapButton.setStyleSheet("background-color: yellow; color: black; border-radius: 5px; padding: 5px;")
        logging.info("Zapping...")
        self.graph_interface.zap()
        QtCore.QTimer.singleShot(250, self.reset_zap_button)

    def reset_zap_button(self):
        self.zapButton.setStyleSheet("")

    def handle_zap_duration_change(self):
        """
        Convert the selected zap duration to seconds and update via GraphInterface.
        """
        text = self.zapDurationDropdown.currentText()
        if "µs" in text or "us" in text:
            value = float(text.replace("µs", "").replace("us", "").strip())
            zap_duration = value * 1e-6
        elif "ms" in text:
            value = float(text.replace("ms", "").strip())
            zap_duration = value * 1e-3
        else:
            zap_duration = float(text)
        logging.info(f"Setting zap duration to {zap_duration} seconds")
        self.graph_interface.set_zap_duration(zap_duration)

    def laserPowerBoxReturnPressed(self):
        """
        When a laser power value is entered, update the setpoint.
        """
        try:
            text = self.laserPowerBox.text().replace("Set to:", "").replace("%", "").strip()
            self.laserPowerBox.clear()
            power = float(text)
            power = max(self.laserPowerLowerBound, min(self.laserPowerUpperBound, power))
            applied = self.graph_interface.set_laser_power(power)
            if applied is None:
                applied = power
            self.laserPowerBox.setPlaceholderText(f"Set to: {int(round(applied))} %")
        except ValueError:
            logging.warning("Invalid laser power input.")
        except Exception as e:
            logging.error(f"Error in laserPowerBoxReturnPressed: {e}")

    def _resolve_laser_label(self, wavelength):
        if wavelength is None:
            return "Unknown", "#e0e0e0"
        if isinstance(wavelength, str):
            cleaned = wavelength.strip()
            if cleaned.isdigit():
                wavelength = int(cleaned)
            else:
                key = cleaned.lower()
                if key in self.laserColorMap:
                    return self.laserColorMap[key]
        if isinstance(wavelength, int):
            options = self.graph_interface.get_laser_wavelength_options()
            if options:
                index = max(1, min(len(options), wavelength))
                return self._resolve_laser_label(options[index - 1])
            default_keys = list(self.laserColorMap.keys())
            if default_keys:
                index = max(1, min(len(default_keys), wavelength))
                return self.laserColorMap[default_keys[index - 1]]
            return f"Ch {wavelength}", "#e0e0e0"
        name = getattr(wavelength, "name", str(wavelength))
        key = name.strip().lower()
        if key in self.laserColorMap:
            return self.laserColorMap[key]
        return name, "#e0e0e0"

    def update_laser_controls(self):
        power_state = self.graph_interface.get_laser_power_state()
        wavelength = self.graph_interface.get_laser_wavelength()
        power = self.graph_interface.get_laser_power()
        if power is None:
            power = 0
        power_value = int(round(power))
        self.laserPowerBox.setPlaceholderText(f"Set to: {power_value} %")
        self.laserPowerLabel.setText(f"Power: {power_value} %")
        for widget in (self.laserPowerBox, self.laserLeftButton, self.laserToggleButton, self.laserRightButton):
            widget.setEnabled(True)

        if power_state != "on":
            self.laserToggleButton.setText("Off")
            self.laserToggleButton.setStyleSheet(
                "background-color: white; color: black; border-radius: 5px; padding: 5px;"
            )
            return

        label, color = self._resolve_laser_label(wavelength)
        text_color = "black" if color in ("#ffffff", "#e0e0e0") else "white"
        self.laserToggleButton.setText(label)
        self.laserToggleButton.setStyleSheet(
            f"background-color: {color}; color: {text_color}; border-radius: 5px; padding: 5px;"
        )

    def updateNoiseButton(self, active):
        if active:
            self.noiseButton.setText("Stop Noise Check")
        else:
            self.noiseButton.setText("Check Noise")

    def toggleNoise(self):
        if self.noiseGraph.is_active():
            self.noiseGraph.stop()
        else:
            self.noiseGraph.start()

    def handle_laser_left(self):
        self.graph_interface.wavelength_down()
        self.update_laser_controls()

    def handle_laser_right(self):
        self.graph_interface.wavelength_up()
        self.update_laser_controls()

    def handle_laser_toggle(self):
        self.graph_interface.toggle_laser_output()
        self.update_laser_controls()
