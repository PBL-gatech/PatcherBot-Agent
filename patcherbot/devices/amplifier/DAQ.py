import collections
import contextlib
import csv
import functools
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize

# Import your hardware libraries as needed
import nidaqmx
import nidaqmx.constants
import serial

from patcherbot.controller.base import TaskController
from patcherbot.utils.pipette_keys import pipette_key_for

__all__ = ['NiDAQ', 'ArduinoDAQ', 'FakeDAQ', 'PipetteChannels', 'PipetteDAQView',
           'build_channel_map', 'enumerated_channels']


# ========================================================
#  Channel model
# ========================================================
@dataclass(frozen=True)
class PipetteChannels:
    """Channel names used by one pipette.

    read: analog input carrying the amplifier's command monitor
    resp: analog input carrying the amplifier's response
    cmd:  analog output that drives the amplifier's command input
    """
    read: str
    resp: str
    cmd: str

    @classmethod
    def from_spec(cls, spec):
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, dict):
            raise TypeError(f"Channel spec must be a mapping with read/resp/cmd, got {spec!r}")
        missing = {"read", "resp", "cmd"} - set(spec)
        if missing:
            raise ValueError(f"Channel spec {spec!r} is missing {sorted(missing)}")
        return cls(read=str(spec["read"]), resp=str(spec["resp"]), cmd=str(spec["cmd"]))


def build_channel_map(pipette_keys=None, channels=None, default_channels=None):
    """Return an ordered ``{pipette_key: PipetteChannels}`` mapping.

    pipette_keys: pipette names in rig order (defaults to ``["pipette_0"]``).
    channels: optional list of ``{"read", "resp", "cmd"}`` specs, one per pipette.
    default_channels: ``f(index) -> PipetteChannels`` used when ``channels`` is None.
    """
    keys = [str(key) for key in pipette_keys] if pipette_keys else [pipette_key_for(0)]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate pipette keys: {keys}")
    if channels is not None:
        specs = list(channels)
        if len(specs) < len(keys):
            raise ValueError(
                f"{len(keys)} pipettes selected but only {len(specs)} channel sets configured"
            )
        specs = [PipetteChannels.from_spec(spec) for spec in specs[:len(keys)]]
    elif default_channels is not None:
        specs = [default_channels(index) for index in range(len(keys))]
    else:
        raise ValueError("Either channels or default_channels must be provided")

    channel_map = collections.OrderedDict(zip(keys, specs))
    names = [name for spec in specs for name in (spec.read, spec.resp, spec.cmd)]
    duplicates = sorted(name for name, count in collections.Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Channels assigned to more than one role/pipette: {duplicates}")
    return channel_map


def enumerated_channels(index):
    """Default channel names for pipette ``index``: ai{2i} (read), ai{2i+1} (resp), ao{i} (cmd)."""
    return PipetteChannels(f"ai{2 * index}", f"ai{2 * index + 1}", f"ao{index}")


# State the DAQ keeps separately for each pipette. Reading or writing one of
# these attributes on a DAQ goes to the "active" pipette (see DAQ.for_pipette),
# so protocol code written for one pipette works unchanged for each pipette.
_PER_PIPETTE_FIELDS = {
    "cellMode": False,
    "totalResistance": None,
    "latestAccessResistance": None,
    "latestMembraneResistance": None,
    "latestMembraneCapacitance": None,
    "holding_current": None,
    "holding_voltage": None,
    "pulses": None,
    "pulseRange": None,
    "current_protocol_data": None,
    "voltage_protocol_data": None,
    "voltage_command_data": None,
    "voltage_membrane_test": None,
    "voltageTotalResistance": None,
    "voltageMembraneResistance": None,
    "voltageAccessResistance": None,
    "voltageMembraneCapacitance": None,
    "vclamp_steps": None,
    "vclamp_hold_value": None,
    "leak_subtraction_data": None,
    "leak_subtraction_meta": None,
    "holding_protocol_data": None,
    "optogenetic_protocol_data": None,
    "optogenetic_stim_data": None,
    "optogenetic_protocol_type": None,
    "optogenetic_pulses": None,
    "optogenetic_pulseRange": None,
    "optogenetic_protocol_queue": collections.deque,  # factory: one queue per pipette
}


def _new_pipette_state():
    return {
        name: (default() if callable(default) else default)
        for name, default in _PER_PIPETTE_FIELDS.items()
    }


def multi_pipette_protocol(method):
    """Protocol that acquires every requested pipette in the same reads.

    The decorated method is written as ``method(self, keys, **kwargs)`` and
    returns ``{pipette_key: result}``. Callers pass ``pipettes=[...]``
    (default: the active pipette only). Protocols use the shared hardware one
    at a time.
    """
    @functools.wraps(method)
    def wrapper(self, *args, pipettes=None, **kwargs):
        keys = self._resolve_pipettes(pipettes)
        with self._protocol_lock:
            return method(self, keys, *args, **kwargs)
    return wrapper


class _PerPipetteAttribute:
    """Descriptor that stores a DAQ attribute per pipette."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._pipette_state()[self.name]

    def __set__(self, obj, value):
        obj._pipette_state()[self.name] = value


class DAQAcquisitionThread(threading.Thread):
    """
    A thread that continuously sends a square wave through the DAQ,
    acquires the response, calculates resistance and capacitance,
    and stores only the most recent measurement in a queue (maxlen=1).

    Parameters:
      daq: An instance of a DAQ subclass (e.g. NiDAQ, ArduinoDAQ, or FakeDAQ)
      wave_freq: Frequency of the square wave (Hz)
      samplesPerSec: Sampling rate in samples per second
      dutyCycle: Duty cycle for the square wave (between 0 and 1)
      amplitude: Amplitude of the square wave (in DAQ output units)
      recordingTime: Duration of each acquisition (seconds)
      callback: Optional function called once per pipette per cycle with:
                  pipette_key, totalResistance, accessResistance,
                  membraneResistance, membraneCapacitance, respData, readData.
      interval: Time (seconds) to wait between acquisitions (default is recordingTime).
    """
    def __init__(self, daq, wave_freq=40, samplesPerSec=100000, dutyCycle=0.5,
                 amplitude=0.5, recordingTime=0.025, callback=None, interval=None):
        super().__init__(daemon=True)
        self.daq = daq
        self._pause_evt = daq._pause_evt     # set == allowed to run
        self._idle_evt  = daq._idle_evt      # set == thread is parked
        self.wave_freq = wave_freq
        self.samplesPerSec = samplesPerSec
        self.dutyCycle = dutyCycle
        self.amplitude = amplitude
        self.recordingTime = recordingTime
        self.callback = callback
        self.interval = interval if interval is not None else recordingTime
        self.running = True
        self._last_data_queue = collections.deque(maxlen=1)
        self._fail_streak = 0           # consecutive bad-fit counter
        self._fail_limit  = 3           # restart after 3 misses

    def run(self):
        """
        Continuous acquisition loop that

        • honours pause / resume,
        • never feeds the GUI None values (uses NaN on bad fits),
        • restarts the DAQ after three consecutive bad fits.
        """
        self._fail_streak = 0            # counts consecutive bad fits
        self._fail_limit  = 3            # restart threshold

        # REVIEW(pause): _idle_evt is clear only while this thread may touch
        # the hardware (one square-wave cycle). DAQ._park_live_thread waits
        # for it, so protocols never close/rebuild tasks mid-read.
        self._idle_evt.set()
        while self.running:
            # ---------- wait until resumed ---------------------------------
            self._pause_evt.wait()
            if not self.running:
                break
            self._idle_evt.clear()
            if not self._pause_evt.is_set():
                # REVIEW(pause): a pause arrived between wait() and clear();
                # step back without touching the hardware.
                self._idle_evt.set()
                continue

            cycle_start = time.perf_counter()

            try:
                # ---------- one square-wave cycle, all pipettes at once ----
                results = self.daq.getDataFromSquareWave(
                    self.wave_freq, self.samplesPerSec,
                    self.dutyCycle, self.amplitude,
                    self.recordingTime,
                    pipettes=self.daq.pipette_keys,
                )
                if not results:              # empty FIFO read → retry
                    continue

                packets = {}
                any_fit_ok = False
                for pipette_key, data in results.items():
                    (t_r, respData), (_, readData), totalR, memR, accR, memC = data
                    if totalR is None:
                        totalR = accR = memR = memC = np.nan   # GUI-safe NaNs
                    else:
                        any_fit_ok = True

                    packets[pipette_key] = {
                        "timeData":            t_r,
                        "respData":            respData,
                        "readData":            readData,
                        "totalResistance":     totalR,
                        "membraneResistance":  memR,
                        "accessResistance":    accR,
                        "membraneCapacitance": memC
                    }

                    # update live attributes even if NaN
                    state = self.daq._pipette_state(pipette_key)
                    state["totalResistance"]           = totalR
                    state["latestMembraneResistance"]  = memR
                    state["latestAccessResistance"]    = accR
                    state["latestMembraneCapacitance"] = memC

                    if self.callback:
                        self.callback(pipette_key, totalR, accR, memR, memC,
                                      respData, readData)

                self._last_data_queue.append(packets)

                # ---------- restart after repeated failed fits -------------
                # Only when every pipette failed: one empty pipette (no cell,
                # no amplifier connected) must not keep resetting the others.
                if any_fit_ok:
                    self._fail_streak = 0
                else:
                    self._fail_streak += 1
                    if self._fail_streak >= self._fail_limit:
                        with self.daq._deviceLock:
                            self.daq._restartAcquisition()
                        self._fail_streak = 0

            except nidaqmx.errors.DaqError as e:
                # common FIFO/buffer overruns
                self.daq.warning(f"Acquisition error {e.error_code}: {e}")
                if e.error_code in (-200279, -200284, -200286):
                    with self.daq._deviceLock:
                        self.daq._restartAcquisition()
                time.sleep(0.05)            # brief cooldown

            except Exception as e:
                logging.warning(f"Error in DAQAcquisitionThread: {e}")

            finally:
                self._idle_evt.set()        # REVIEW(pause): hardware released

            # ---------- honour requested interval --------------------------
            elapsed       = time.perf_counter() - cycle_start
            time.sleep(max(0.0, self.interval - elapsed))

    def get_last_data(self, pipette_key=None):
        """Return the most recent packet for one pipette, or ``{pipette_key: packet}``
        for every pipette when ``pipette_key`` is None. Returns None if no data yet."""
        if not self._last_data_queue:
            return None
        packets = self._last_data_queue[-1]
        if pipette_key is None:
            return {key: dict(packet) for key, packet in packets.items()}
        packet = packets.get(pipette_key)
        return dict(packet) if packet is not None else None

    def stop(self):
        """Stop the acquisition thread."""
        self.running = False

class DAQ(TaskController):
    """
    Base DAQ class with common methods for patch-clamp protocols.
    Subclasses must override the _readAnalogInput() and _sendSquareWave()
    methods to handle device-specific operations.
    """
    # Class-level constants (may be overwritten by subclasses)
    C_CLAMP_AMP_PER_VOLT = None
    C_CLAMP_VOLT_PER_VOLT = None
    V_CLAMP_VOLT_PER_VOLT = None
    V_CLAMP_VOLT_PER_AMP = None

    def __init__(self, channel_map=None):
        """
        channel_map: ordered ``{pipette_key: PipetteChannels}`` from
        build_channel_map (which validates it). One DAQ serves every pipette
        on the rig; defaults to a single pipette on ai0/ai1/ao0.
        """
        super().__init__()
        # Per-pipette state must exist before any per-pipette attribute is
        # touched (see _PER_PIPETTE_FIELDS). cellMode, resistances, protocol
        # results etc. all live here, one dict per pipette.
        self._pipette_states = {}
        self._active = threading.local()
        if channel_map is None:
            channel_map = build_channel_map(default_channels=enumerated_channels)
        self.channel_map = collections.OrderedDict(channel_map)
        for key in self.channel_map:
            self._pipette_state(key)

        self._optogenetic_queue_lock = threading.Lock()
        self._deviceLock = threading.Lock()
        # Protocols use the shared hardware one at a time.
        self._protocol_lock = threading.RLock()
        self._pause_evt = threading.Event(); self._pause_evt.set()   # allow run
        self._idle_evt  = threading.Event(); self._idle_evt.set()    # initially idle
        self.sweep_clip = None

    # --------------------------
    # Channels and pipettes
    # --------------------------
    @property
    def pipette_keys(self):
        return list(self.channel_map)

    @property
    def primary_pipette_key(self):
        return next(iter(self.channel_map))

    @property
    def active_pipette_key(self):
        """Pipette that per-pipette attributes refer to in the calling thread."""
        return getattr(self._active, "pipette_key", None) or self.primary_pipette_key

    @property
    def ai_channels(self):
        """Every analog input name, in acquisition order (read, resp per pipette)."""
        return [name for spec in self.channel_map.values() for name in (spec.read, spec.resp)]

    @property
    def ao_channels(self):
        """Every analog output name, one command channel per pipette."""
        return [spec.cmd for spec in self.channel_map.values()]

    def channels_for(self, pipette_key=None):
        """Return the PipetteChannels of one pipette (default: the active one)."""
        key = self.active_pipette_key if pipette_key is None else pipette_key
        try:
            return self.channel_map[key]
        except KeyError:
            raise KeyError(f"No DAQ channels configured for pipette {key!r}; "
                           f"known pipettes: {self.pipette_keys}") from None

    @contextlib.contextmanager
    def for_pipette(self, pipette_key):
        """Make ``pipette_key`` the active pipette for this thread.

        Per-pipette attributes (cellMode, totalResistance, protocol results…)
        read and write that pipette's state inside the block.
        """
        self.channels_for(pipette_key)              # validate
        previous = getattr(self._active, "pipette_key", None)
        self._active.pipette_key = pipette_key
        try:
            yield pipette_key
        finally:
            self._active.pipette_key = previous

    def _pipette_state(self, pipette_key=None):
        key = self.active_pipette_key if pipette_key is None else pipette_key
        state = self._pipette_states.get(key)
        if state is None:
            state = self._pipette_states.setdefault(key, _new_pipette_state())
        return state

    def _resolve_channels(self, channels=None):
        """Normalize a ``channels`` argument to a list of AI channel names.

        None means every configured AI channel.
        """
        available = self.ai_channels
        if channels is None:
            return available
        names = [channels] if isinstance(channels, str) else list(channels)
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(f"Unknown analog input channel(s) {unknown}; configured: {available}")
        return names

    def _resolve_pipettes(self, pipettes=None):
        """Normalize a ``pipettes`` argument to a validated list of pipette keys
        (duplicates dropped, order kept). None means the active pipette only."""
        if pipettes is None:
            return [self.active_pipette_key]
        keys = [pipettes] if isinstance(pipettes, str) else list(dict.fromkeys(pipettes))
        for key in keys:
            self.channels_for(key)
        return keys

    @staticmethod
    def _rows_to_channel_dict(raw, channel_names):
        """Turn a (channels x samples) block, rows in task order, into a keyed dict."""
        rows = np.asarray(raw, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[0] != len(channel_names):
            raise ValueError(f"Got {rows.shape[0]} rows of data for {len(channel_names)} channels")
        return {name: rows[index].copy() for index, name in enumerate(channel_names)}

    def _read_pipettes(self, keys, samplesPerSec, recordingTime):
        """ONE read covering every listed pipette -> ``{key: (read, resp)}``.

        All pipettes come from the same acquisition (one hardware task
        holding all their channels), so they share one sample clock.
        """
        specs = [self.channels_for(key) for key in keys]
        data = self._readAnalogInput(
            samplesPerSec, recordingTime,
            channels=[name for spec in specs for name in (spec.read, spec.resp)])
        return {key: (data[spec.read], data[spec.resp]) for key, spec in zip(keys, specs)}

    def setCellMode(self, mode: bool) -> None:
        self.cellMode = mode
        self.info(f"Setting cell mode to {mode} ({self.active_pipette_key})")

    def getCellMode(self) -> bool:
        return self.cellMode

    def _normalize_optogenetic_protocol_key(self, protocol_type):
        if protocol_type is None:
            return "unknown"
        name = getattr(protocol_type, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip().lower()
        text = str(protocol_type).strip().lower()
        return text if text else "unknown"

    def pop_optogenetic_entry(self, protocol_key):
        key = self._normalize_optogenetic_protocol_key(protocol_key)
        with self._optogenetic_queue_lock:
            for entry in self.optogenetic_protocol_queue:
                if entry.get("protocol_key") == key:
                    self.optogenetic_protocol_queue.remove(entry)
                    return entry
        return None
    # --------------------------
    # Acquisition Methods
    # --------------------------
    def start_acquisition(self, wave_freq=80, samplesPerSec=10000, dutyCycle=0.5,
                          amplitude=0.5, recordingTime=0.012, interval=None, callback=None):
        """
        Start the asynchronous acquisition thread for continuous DAQ measurements.
        Parameters:
          wave_freq: Frequency of the square wave in Hz.
          samplesPerSec: Sampling rate (samples per second).
          dutyCycle: Duty cycle (0 to 1) of the square wave.
          amplitude: Amplitude for the square wave.
          recordingTime: Duration for each acquisition (seconds).
          interval: Time (seconds) between acquisitions (defaults to recordingTime).
          callback: Optional function to call with new data.
        """
        if not hasattr(self, "_daq_acq_thread") or self._daq_acq_thread is None:
            self._daq_acq_thread = DAQAcquisitionThread(
                daq=self,
                wave_freq=wave_freq,
                samplesPerSec=samplesPerSec,
                dutyCycle=dutyCycle,
                amplitude=amplitude,
                recordingTime=recordingTime,
                callback=callback,
                interval=interval
            )
            self._daq_acq_thread.start()

    def get_last_acquisition(self, pipette_key=None):
        """
        Retrieve the most recent live test-pulse packet for one pipette
        (default: the active pipette). The packet has keys:
          "timeData", "respData", "readData", "totalResistance",
          "membraneResistance", "accessResistance", "membraneCapacitance"
        Returns None if no data is available yet.
        """
        thread = getattr(self, "_daq_acq_thread", None)
        if not thread:
            return None
        key = self.active_pipette_key if pipette_key is None else pipette_key
        return thread.get_last_data(key)

    def get_last_acquisition_all(self):
        """Return ``{pipette_key: packet}`` from the latest live cycle (or None)."""
        thread = getattr(self, "_daq_acq_thread", None)
        return thread.get_last_data() if thread else None

    def compute_noise_metrics(self, timeData=None, respData=None,
                              window_start=0.004, window_end=0.010,
                              avg_p2p_window=0.001):
        """
        Compute noise metrics from response data within a time window.
        Returns a dict with windowed data, p2p, std, avg_p2p, and FFT results,
        or None if inputs are insufficient.
        """
        if timeData is None or respData is None:
            last = self.get_last_acquisition()
            if last is None:
                return None
            if timeData is None:
                timeData = last.get("timeData")
            if respData is None:
                respData = last.get("respData")

        if timeData is None or respData is None:
            return None

        time_array = np.asarray(timeData)
        resp_array = np.asarray(respData)
        if time_array.size < 2 or resp_array.size < 2:
            return None

        window_mask = (time_array >= window_start) & (time_array <= window_end)
        if not np.any(window_mask):
            return None

        window_time = time_array[window_mask]
        window_resp = resp_array[window_mask]
        if window_time.size < 2:
            return None

        dt = float(np.mean(np.diff(window_time)))
        if not np.isfinite(dt) or dt <= 0:
            dt = None

        p2p = float(np.max(window_resp) - np.min(window_resp))
        std = float(np.std(window_resp))

        avg_p2p = p2p
        if dt is not None:
            window_samples = int(round(avg_p2p_window / dt))
            if window_samples < 2:
                window_samples = 2
            if window_resp.size >= window_samples:
                p2p_values = []
                for start in range(0, window_resp.size - window_samples + 1):
                    segment = window_resp[start:start + window_samples]
                    p2p_values.append(float(np.max(segment) - np.min(segment)))
                if p2p_values:
                    avg_p2p = float(np.mean(p2p_values))

        freqs = None
        fft_magnitude = None
        if dt is not None and window_resp.size >= 2:
            detrended = window_resp - np.mean(window_resp)
            fft_vals = np.fft.rfft(detrended)
            freqs = np.fft.rfftfreq(detrended.size, d=dt)
            fft_magnitude = np.abs(fft_vals)

        return {
            "window_time": window_time,
            "window_resp": window_resp,
            "dt": dt,
            "p2p": p2p,
            "std": std,
            "avg_p2p": avg_p2p,
            "freqs": freqs,
            "fft_magnitude": fft_magnitude
        }

    def stop_acquisition(self):
        """
        Stop the asynchronous DAQ acquisition thread.
        """
        if hasattr(self, "_daq_acq_thread") and self._daq_acq_thread:
            self._daq_acq_thread.stop()
            self._daq_acq_thread.join()
            self._daq_acq_thread = None
            
    def _park_live_thread(self, timeout: float = 2.0):
        """Stop the live test pulse thread between cycles, leaving the tasks running.

        REVIEW(pause): blocks until the thread has finished any cycle in
        progress (see DAQAcquisitionThread.run). Undo with ``_pause_evt.set()``.
        Returns False if the thread was already parked.
        """
        if not self._pause_evt.is_set():
            return False
        self._pause_evt.clear()                     # ask thread to park
        if not self._idle_evt.wait(timeout):
            raise TimeoutError("DAQ thread failed to become idle")
        return True

    def pause_acquisition(self, timeout: float = 2.0):
        """
        Block until the acquisition thread is parked **and** both DAQmx tasks
        have been stopped, giving the caller sole ownership of the hardware.
        """
        if self._park_live_thread(timeout):
            # --- stop the streaming tasks ---------------------------------
            try:
                if hasattr(self, "ai_task"): self.ai_task.stop()
                if hasattr(self, "ao_task"): self.ao_task.stop()
            except Exception:
                pass                                # tasks may already be stopped

    def resume_acquisition(self):
        """
        Restart the continuous AI+AO pair (freshly synced) and wake the thread.
        """
        if not self._pause_evt.is_set():
            # ensure tasks are running and synchronized
            self._restartAcquisition()              # subclass implementation
            self._pause_evt.set()                   # let the thread continue
    
    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime,
                              pipettes=None):
        """
        Single-cycle acquisition for several pipettes at once: fire the AO
        buffer, do ONE read of every requested pipette's AI channels, then
        process each pipette.

        Returns ``{pipette_key: (time/resp, time/read, totalR, memR, accR, memC)}``.
        ``pipettes`` defaults to the active pipette only.
        """
        keys = self._resolve_pipettes(pipettes)
        with self._deviceLock:
            # Device-specific wrappers (implemented in subclasses)
            self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
            traces = self._read_pipettes(keys, samplesPerSec, recordingTime)

        results = {}
        for key, (read, resp) in traces.items():
            # cellMode and conversions are per pipette
            with self.for_pipette(key):
                results[key] = self._squareWaveProcessor(
                    np.vstack([read, resp]), samplesPerSec, amplitude, sweep_clip=None)
        return results
      
    # --------------------------
    # ABSTRACT (DEVICE-SPECIFIC)
    # --------------------------
    
    def _setupAcquisition(self):
        """
        Abstract: set up and start ai_task & ao_task.
        """
        raise NotImplementedError("Subclasses must implement _setupAcquisition()")
    
    def _restartAcquisition(self):
        """
        Abstract: restart the acquisition, if it hangs or needs to be reset.
        This is used to reset the DAQ device when needed.
        """
        raise NotImplementedError("Subclasses must implement _restartAcquisition()")

    def _readAnalogInput(self, samplesPerSec, recordingTime, channels=None):
        """
        Read one buffer and return ``{channel_name: samples}``.

        ``channels`` is any subset of ``self.ai_channels`` (None = all). All
        returned channels come from the same acquisition, so they share one
        sample clock. Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _readAnalogInput()")

    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Send a square wave command to the device.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _sendSquareWave()")

    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Default implementation for sending a current square wave.
        Subclasses may override if needed.
        """
        return self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)

    def getDataFromCurrentProtocol(self, *args, **kwargs):
        """Run the current-step protocol (I-clamp).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    def getDataFromHoldingProtocol(self, *args, **kwargs):
        """Acquire baseline holding data (no command output).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    def getDataFromOptogeneticProtocol(self, *args, **kwargs):
        """Acquire holding data while running a laser protocol.

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    @multi_pipette_protocol
    def getDataFromVoltageProtocol(
            self,
            keys,
            *,                       # keyword-only
            wave_freq: int = 40,
            samplesPerSec: int = 50_000,   # 50 kS/s
            dutyCycle: float = 0.5,
            amplitude: float = 0.5,
            recordingTime: float = 0.025,  # 25 ms period
            max_attempts: int = 15,
            membrane_hold: float | None = None
    ):
        """Membrane test on the live stream, every requested pipette at once.

        Runs square-wave cycles until each pipette has 5 acceptable capacitance
        fits (or ``max_attempts``). Each cycle is ONE read covering every
        pipette still collecting. The live thread is parked meanwhile but the
        tasks keep streaming. Returns ``{key: mean capacitance (pF) or None}``.
        """
        for key in keys:
            self._pipette_state(key)["voltage_membrane_test"] = None
        capacitances = {key: [] for key in keys}
        first_good = {}
        first_any = {}
        attempts = 0
        parked = self._park_live_thread()
        time.sleep(recordingTime * 2)        # let the stream move past the last cycle
        try:
            while attempts < max_attempts:
                collecting = [key for key in keys if len(capacitances[key]) < 5]
                if not collecting:
                    break
                attempts += 1
                cycle = self.getDataFromSquareWave(
                    wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime,
                    pipettes=collecting,
                )
                for key, result in cycle.items():
                    first_any.setdefault(key, result)
                    membrane_capacitance = result[5]
                    if (membrane_capacitance is not None
                            and np.isfinite(membrane_capacitance)
                            and 0 < membrane_capacitance < 250):   # pF, arbitrary upper limit
                        first_good.setdefault(key, result)
                        capacitances[key].append(membrane_capacitance)
                    else:
                        self.debug(f"Voltage step attempt {attempts} ({key}): unacceptable fit -> skipped")
                time.sleep(recordingTime)    # so consecutive reads don't overlap in the FIFO
        finally:
            if parked:
                self._pause_evt.set()

        results = {}
        for key in keys:
            if len(capacitances[key]) < 5:
                self.warning(f"Voltage protocol ({key}): {len(capacitances[key])} acceptable "
                             f"traces (expected 5) within {attempts} attempts")
            with self.for_pipette(key):
                # Show the first acceptable trace; fall back to the first one.
                results[key] = self._store_membrane_test(
                    first_good.get(key, first_any.get(key)), capacitances[key],
                    membrane_hold, amplitude)
        return results

    def _membrane_step_mV(self, command, amplitude):
        """Size of the membrane-test command step, in mV."""
        return float(amplitude * self.V_CLAMP_VOLT_PER_VOLT * 1e3)

    def _store_membrane_test(self, first_result, capacitances, membrane_hold, amplitude):
        """Store one pipette's membrane-test results (active pipette)."""
        if first_result is None:
            self.voltageMembraneCapacitance = None
            return None

        (self.voltage_protocol_data,
         self.voltage_command_data,
         self.voltageTotalResistance,
         self.voltageMembraneResistance,
         self.voltageAccessResistance,
         _) = first_result
        protocol_data, command_data = first_result[:2]
        self.voltage_membrane_test = {
            "time": protocol_data[0].copy(),
            "response": protocol_data[1].copy(),
            "command": command_data[1].copy(),
            "step_mV": self._membrane_step_mV(command_data[1], amplitude),
            "hold_mV": membrane_hold * 1e3 if membrane_hold is not None else None,
            "color": "#000000",
            # False when no trace gave an acceptable Cm fit and this is the
            # first (unfit) trace, shown so the window isn't empty.
            "fit_ok": bool(capacitances),
        }
        self.voltageMembraneCapacitance = (
            float(np.mean(capacitances)) if capacitances else None
        )
        self.latestMembraneCapacitance = self.voltageMembraneCapacitance
        if self.voltageTotalResistance is not None:
            self.totalResistance = self.voltageTotalResistance
        if self.voltageMembraneResistance is not None:
            self.latestMembraneResistance = self.voltageMembraneResistance
        if self.voltageAccessResistance is not None:
            self.latestAccessResistance = self.voltageAccessResistance
        return self.voltageMembraneCapacitance

    def getVoltageClampSweep(self, *args, **kwargs):
        """Run the voltage-step protocol (V-clamp).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    # --------------------------
    # COMMON CALCULATION METHODS
    # --------------------------


    def createSquareWave(self, wave_freq, samplesPerSec,
                        dutyCycle, amplitude, recordingTime,
                        pre_pad_ms: float = 0.5,*, store: bool = True): 
        
        """
        Build a square-wave buffer.

        Returns
        -------
        (wave: np.ndarray, rate: int, numSamples: int)

        Setting *store=False* makes the routine **side-effect–free**,
        letting protocol methods generate *temporary* buffers without
        clobbering the continuous-acquisition one.

        If *store=True* (default) the buffer is cached exactly as before
        on the instance (`self.wave`, `_wave_rate`, `_wave_samples`) so
        all existing callers continue to work unchanged.
        """     
        numSamples = int(samplesPerSec * recordingTime)
        period     = int(samplesPerSec / wave_freq)
        onTime     = int(period * dutyCycle)

        wave = np.zeros(numSamples, dtype=float)
        pad  = int(pre_pad_ms * 1e-3 * samplesPerSec)

        for i in range(0, numSamples - pad, period):
            wave[i + pad : i + pad + onTime] = amplitude

        if store:
            self.wave          = wave
            self._wave_rate    = samplesPerSec
            self._wave_samples = numSamples

        return wave, samplesPerSec, numSamples

    def createSquareWaveVoltage(self,
                                wave_freq,
                                samplesPerSec,
                                dutyCycle,
                                amplitude,
                                recordingTime,
                                *,
                                store: bool = False):
        """
        Build the V-clamp buffer with baseline-test-baseline segments.

        recordingTime represents the total duration (baseline + test + baseline).
        Each segment occupies one third of the total time.
        """
        segment_time = float(recordingTime) / 3.0

        baseline_wave, *_ = self.createSquareWave(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=0.0,
            recordingTime=segment_time,
            pre_pad_ms=0,
            store=False,
        )
        test_wave, rate, _ = self.createSquareWave(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=amplitude,
            recordingTime=segment_time,
            pre_pad_ms=0,
            store=False,
        )

        wave = np.concatenate((baseline_wave, test_wave, baseline_wave))

        if store:
            self.wave = wave
            self._wave_rate = samplesPerSec
            self._wave_samples = wave.size

        return wave, rate, wave.size

    def createSquareWaveCurrent(self,
                                wave_freq,          # test-pulse frequency (Hz)
                                samplesPerSec,
                                dutyCycle,
                                amplitude,          # test-pulse amplitude (Volts)
                                recordingTime):
        """
        Build the composite I-clamp buffer required by the legacy protocol.

        Layout per pulse
        ----------------
            • baseline   :  @ 0 pA,   1 Hz square-wave
            • calibrator :  @ –20 pA, freq = wave_freq
            • test pulse :  @ ±pA,    freq = wave_freq

        All segments share the same sampling rate and 50 % duty-cycle.

        Returns
        -------
        wave : np.ndarray  (Volts)
        rate : int         (samples / sec)
        numSamples : int   (wave.size)
        """
        import numpy as np

        # ───── 1. baseline : 0 pA, 1 Hz ──────────────────────────────────
        base_wave, *_ = self.createSquareWave(
            wave_freq=1, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=0.0,
            recordingTime=recordingTime, pre_pad_ms=0, store=False)

        # ───── 2. calibrator : –20 pA, double test-freq ──────────────────
        calib_wave, *_ = self.createSquareWave(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=(-20e-12) / self.C_CLAMP_AMP_PER_VOLT,  # pA → Volts
            recordingTime=recordingTime, pre_pad_ms=0, store=False)

        # ───── 3. test pulse : ±pA, wave_freq ────────────────────────────
        test_wave, rate, _ = self.createSquareWave(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=amplitude,
            recordingTime=recordingTime, pre_pad_ms=0, store=False)
        
        #  ───────────────── 4. pad the base wave ───────────────────────────
        # add the base wave at the end

        # ───── concatenate and return ────────────────────────────────────
        wave = np.concatenate((base_wave, calib_wave, test_wave,base_wave))
        return wave, rate, wave.size


    def _squareWaveProcessor(self, raw_data, samplesPerSec, amplitude,sweep_clip=None):
        """
        Cut out the pulse, convert units, compute R & C.
        Identical to our previous implementation, unchanged.
        """
        read = raw_data[0]
        resp = raw_data[1]
        N = read.size
        timeData = np.linspace(0, N/samplesPerSec, N, dtype=float)

        grad = np.gradient(read, timeData)
        max_i = np.argmax(grad)
        min_i = np.argmin(grad[max_i:]) + max_i
        
        if sweep_clip is not None:
            left,right = sweep_clip
        else:
            left, right = 100,300


        idx0 = max(0, max_i-left)
        idx1 = min(N, min_i+right)
        td = timeData[idx0:idx1] - timeData[idx0]
        rd = resp[idx0:idx1] * self.V_CLAMP_VOLT_PER_AMP
        cd = read[idx0:idx1] * self.V_CLAMP_VOLT_PER_VOLT


        if self.getCellMode():
            accR, memR, memC = self._getParamsfromCurrent(
                cd, rd, td, amplitude * self.V_CLAMP_VOLT_PER_VOLT
            )
            totalR = None
            if accR is not None and memR is not None:
                totalR = accR + memR
        else:
            totalR = self._getResistancefromCurrent(rd, amplitude * self.V_CLAMP_VOLT_PER_VOLT)
            if totalR is not None:
                totalR *= 1e-6

            accR, memR, memC = None, None, None

        return (np.array([td, rd]),
                np.array([td, cd]),
                totalR,
                memR,
                accR,
                memC)

    def capacitance(self):
        """
        Returns the latest membrane capacitance measurement.
        If no measurement is available, returns None.
        """
        return self.latestMembraneCapacitance

    def resistance(self):
        """
        Returns the latest total resistance measurement.
        If no measurement is available, returns None.
        """
        return self.totalResistance
    
    def accessResistance(self):
        """
        Returns the latest access resistance measurement.
        If no measurement is available, returns None.
        """
        return self.latestAccessResistance

    def _getResistancefromCurrent(self, data, cmdVoltage) -> float | None:
        try:
            mean = np.mean(data)
            lowAvg = np.mean(data[data < mean])
            highAvg = np.mean(data[data > mean])
            resistance = cmdVoltage / (highAvg - lowAvg)
            return resistance
        except Exception as e:
            return None
        
    

    def filter_data(self, T_ms, X_mV, Y_pA):
        """
        Process the acquired data to extract the relevant segment for curve-fitting,
        using NumPy arrays instead of a pandas DataFrame.
        
        Parameters:
          T_ms: np.array of time in milliseconds (shifted so first element is 0)
          X_mV: np.array of command voltages in millivolts
          Y_pA: np.array of responses in picoamps
          
        Returns:
          sub_data: portion of Y_pA between the peak and negative peak indices
          sub_time: corresponding time points from T_ms
          sub_command: corresponding X_mV values
          plot_params: list of [peak_time, peak_current_index, negative_peak_time, negative_peak_index]
          mean_pre_peak: mean of Y_pA from start until the positive peak in dX/dT
          mean_post_peak: mean of Y_pA from the first near-zero gradient point to the negative peak
        """
        # Compute gradient of the command signal
        X_dT = np.gradient(X_mV, T_ms)
        positive_peak_index = np.argmax(X_dT)
        negative_peak_index = np.argmin(X_dT)
        peak_current_index = np.argmax(Y_pA)
        peak_time = T_ms[peak_current_index]
        negative_peak_time = T_ms[negative_peak_index]
        pre_peak_current = Y_pA[:positive_peak_index+1]
        sub_data = Y_pA[peak_current_index:negative_peak_index+1]
        sub_time = T_ms[peak_current_index:negative_peak_index+1]
        sub_command = X_mV[peak_current_index:negative_peak_index+1]
        mean_pre_peak = pre_peak_current.mean()
        sub_gradient = np.gradient(sub_data, sub_time)
        close_to_zero_index = np.where(np.isclose(sub_gradient, 0, atol=1e-2))[0]
        zero_gradient_time = None
        if close_to_zero_index.size > 0:
            zero_gradient_index = close_to_zero_index[0]
            zero_gradient_time = sub_time[zero_gradient_index]
        if zero_gradient_time is not None:
            mask = (T_ms >= zero_gradient_time) & (T_ms <= T_ms[negative_peak_index])
            mean_post_peak = Y_pA[mask].mean() if np.any(mask) else None
        else:
            mean_post_peak = None
        plot_params = [peak_time, peak_current_index, negative_peak_time, negative_peak_index]
        return sub_data, sub_time, sub_command, plot_params, mean_pre_peak, mean_post_peak

    def monoExp(self, x, m, t, b):
        return m * np.exp(-t * x) + b


    def optimizer(self, fit_data, I_peak_pA, I_peak_time_ms, I_ss_pA):
        """
        Robust mono-exponential fit:

            I(t_ms) = m · exp(−t · t_ms) + b

        with parameter
            t = 1 / τ      [1 / ms]

        Returns
        -------
        (m, t, b) in pA, 1/ms, pA   – or (None, None, None) on failure
        """
        # -------- data -----------------------------------------------------
        x_ms = fit_data['T_ms']
        y_pA = fit_data['Y_pA']

        if not (np.isfinite(x_ms).all() and np.isfinite(y_pA).all()):
            return None, None, None

        # scale only the current to nA  (better conditioning)
        y_nA = y_pA * 1e-3

        # -------- initial guess -------------------------------------------
        m0 = I_peak_pA * 1e-3           # nA
        b0 = I_ss_pA   * 1e-3           # nA
        t0 = max(1.0 / max(I_peak_time_ms, 0.1), 0.01)   # 1/ms

        p0 = [m0, t0, b0]

        # -------- bounds ---------------------------------------------------
        # Current:   ±10 nA   (±10 000 pA)
        # Time const: t = 1/τ  → τ between 0.05 ms and 100 ms
        bounds = ([-10.0,     1/100.0, -10.0],       # lower  (nA, 1/ms, nA)
                [ 10.0,     1/0.05,  10.0])        # upper

        # -------- residual & Jacobian -------------------------------------
        def residual(p, x, y):
            return self.monoExp(x, *p) - y

        def jacobian(p, x, y):
            m, t, b = p
            e = np.exp(-t * x)
            return np.vstack((e,               # ∂/∂m
                            -m * x * e,      # ∂/∂t
                            np.ones_like(x)  # ∂/∂b
                            )).T

        # -------- bounded robust least-squares -----------------------------
        try:
            res = scipy.optimize.least_squares(
                residual, p0, jac=jacobian, args=(x_ms, y_nA),
                bounds=bounds,
                loss="soft_l1",
                f_scale=1.0,          # keeps internal (f / f_scale)**2 in range
                max_nfev=400
            )
        except Exception:
            return None, None, None

        if not (res.success and np.all(np.isfinite(res.x))):
            return None, None, None

        m_fit_nA, t_fit, b_fit_nA = res.x
        return m_fit_nA * 1e3, t_fit, b_fit_nA * 1e3   # back to pA


    def _getParamsfromCurrent(self, readData, respData, timeData, amplitude) -> tuple:
        """
        Calculate access resistance, membrane resistance, and membrane capacitance
        from the response of a voltage protocol using NumPy arrays instead of pandas.
        """
        R_a_MOhms, R_m_MOhms, C_m_pF = None, None, None
        if len(readData) and len(respData) and len(timeData):
            try:
                # Shift time to start at zero and convert units:
                T = timeData - timeData[0]           # seconds (shifted)
                T_ms = T * 1000                      # convert to milliseconds
                X_mV = readData * 1000               # volts -> millivolts
                Y_pA = respData * 1e12               # amps -> picoamps

                # Use the numpy-based filter_data function.
                filtered_data, filtered_time, filtered_command, plot_params, I_prev_pA, I_post_pA = \
                    self.filter_data(T_ms, X_mV, Y_pA)
                self.holding_current = I_prev_pA
                peak_time, peak_index, negative_peak_time, negative_peak_index = plot_params
                if peak_index + 1 < len(Y_pA):
                    I_peak_pA = Y_pA[peak_index + 1]
                    I_peak_time = T_ms[peak_index + 1]
                else:
                    I_peak_pA = Y_pA[peak_index]
                    I_peak_time = T_ms[peak_index]
                # Prepare the data for curve fitting. Time axis zeroed.
                fit_data = {
                    'T_ms': filtered_time - filtered_time[0],
                    'X_mV': filtered_command,
                    'Y_pA': filtered_data
                }
                mean_voltage = filtered_command.mean()
                m, t, b = self.optimizer(fit_data, I_peak_pA, I_peak_time, I_post_pA) # feed in previous initial conditions if exists
                if m is not None and t is not None and b is not None:
                    tau = 1 / t
                    R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, mean_voltage, I_peak_pA, I_prev_pA, I_post_pA)
            except Exception as e:
                return None, None, None
        else:
            return 0, 0, 0
        return R_a_MOhms, R_m_MOhms, C_m_pF

    def calc_param(self, tau, mean_voltage, I_peak, I_prev, I_ss):
        """
        Calculate access resistance (R_a), membrane resistance (R_m) and membrane capacitance (C_m)
        from the measured currents.
        """
        I_d = I_peak - I_prev   # in pA
        I_dss = I_ss - I_prev   # in pA
        R_a_MOhms = ((mean_voltage * 1e-3) / (I_d * 1e-12)) * 1e-6
        R_m_MOhms = (((mean_voltage * 1e-3) - R_a_MOhms * 1e6 * I_dss * 1e-12) / (I_dss * 1e-12)) * 1e-6
        C_m_pF = (tau * 1e-3) / (1 / (1 / (R_a_MOhms * 1e6) + 1 / (R_m_MOhms * 1e6))) * 1e12
        return R_a_MOhms, R_m_MOhms, C_m_pF

# Per-pipette attributes are descriptors on DAQ so every subclass (NiDAQ,
# FakeDAQ, ArduinoDAQ) stores them per pipette without code changes.
for _field_name in _PER_PIPETTE_FIELDS:
    setattr(DAQ, _field_name, _PerPipetteAttribute(_field_name))
del _field_name


class PipetteDAQView:
    """One pipette's handle on the single shared DAQ.

    Holds a reference (never a copy). Every call made through the view runs
    with this pipette active, so ``view.resistance()``,
    ``view.current_protocol_data``, ``view.setCellMode(...)`` and protocol
    calls all use this pipette's channels and state. A protocol called through
    a view covers only this pipette; call the DAQ itself with
    ``pipettes=[...]`` to acquire several pipettes together.
    """

    def __init__(self, daq, pipette_key):
        daq.channels_for(pipette_key)       # validate
        object.__setattr__(self, "daq", daq)
        object.__setattr__(self, "pipette_key", pipette_key)

    def __getattr__(self, name):
        daq = self.daq
        if name in _PER_PIPETTE_FIELDS:
            return daq._pipette_state(self.pipette_key)[name]
        value = getattr(daq, name)
        if not callable(value):
            return value
        key = self.pipette_key

        def call_for_pipette(*args, **kwargs):
            with daq.for_pipette(key):
                return value(*args, **kwargs)

        call_for_pipette.__name__ = name
        # TaskInterface.execute() runs only methods of a TaskController and
        # finds the owner through __self__, so expose the shared DAQ.
        call_for_pipette.__self__ = daq
        return call_for_pipette

    def __setattr__(self, name, value):
        if name in _PER_PIPETTE_FIELDS:
            self.daq._pipette_state(self.pipette_key)[name] = value
        else:
            setattr(self.daq, name, value)

    def __repr__(self):
        return f"PipetteDAQView({type(self.daq).__name__}, {self.pipette_key!r})"


# ========================================================
#  NI-DAQ Subclass
# ========================================================
class NiDAQ(DAQ):
    """NI-DAQmx DAQ serving every pipette on the rig at once.

    All pipettes' AI channels (read + resp) live in ONE AI task and all their
    command channels in ONE AO task, armed on the AI start trigger. Every
    pipette is therefore sampled on the same clock and stimulated in the same
    samples; each read returns every pipette's data together.
    """
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA per V (DAQ output)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3)  # 10 mV per V (DAQ input)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3)  # 20 mV per V (DAQ output)
    V_CLAMP_VOLT_PER_AMP = (2 * 1e-9)   # 2 mV per pA (DAQ input)

    def __init__(self, readDev=None, readChannel=None, cmdDev=None, cmdChannel=None,
                 respDev=None, respChannel=None, pipette_keys=None, channels=None):
        """
        channels: list of ``{"read": "Dev1/ai17", "resp": "Dev1/ai19",
            "cmd": "Dev1/ao0"}`` (full physical channel names), one per
            pipette in rig order, so entry n belongs to pipette_<n>.
        readDev ... respChannel: legacy single-pipette form. Gives one channel
            set; any further pipettes share it (with a warning).
        """
        keys = list(pipette_keys) if pipette_keys else [pipette_key_for(0)]
        legacy = (readDev, readChannel, cmdDev, cmdChannel, respDev, respChannel)
        if channels is not None:
            if any(value is not None for value in legacy):
                raise ValueError("Give NiDAQ either 'channels' or the single-pipette "
                                 "readDev/readChannel/... params, not both")
            channel_map = build_channel_map(keys, channels=channels)
        else:
            if any(value is None for value in legacy):
                raise ValueError("NiDAQ needs a 'channels' list or all of readDev, readChannel, "
                                 "cmdDev, cmdChannel, respDev, respChannel")
            channel_map = build_channel_map(
                keys[:1],
                channels=[{
                    "read": f"{readDev}/{readChannel}",
                    "resp": f"{respDev}/{respChannel}",
                    "cmd": f"{cmdDev}/{cmdChannel}",
                }],
            )
        super().__init__(channel_map=channel_map)
        if channels is None and len(keys) > 1:
            self.warning(
                f"NiDAQ has channels for 1 pipette but {len(keys)} were selected; "
                f"{keys[1:]} will share {keys[0]}'s channels."
            )
        self._check_single_task_devices()
        for key, spec in self.channel_map.items():
            self.info(f"{key}: read {spec.read}, response {spec.resp}, command {spec.cmd}")

        # Channels (and pipettes) held by the current AI / AO tasks, in task order.
        self._ai_task_keys = []
        self._ai_task_channels = []
        self._ao_task_keys = []

        # 1) Pre-generate the square wave buffer
        self.createSquareWave(
            wave_freq=40, samplesPerSec=50000,
            dutyCycle=0.5, amplitude=0.5,
            recordingTime=0.025
        )

        # 2) Configure & start persistent AO/AI tasks
        self._setupAcquisition()

        # 3) Kick off the acquisition thread
        self.start_acquisition(
            wave_freq=40, samplesPerSec=50000,
            dutyCycle=0.5, amplitude=0.5,
            recordingTime=0.025,
            interval= 0.000  # 0 ms per cycle
        )

    # --------------------------------------------------------------
    # Task building: every task holds the channels of several pipettes
    # --------------------------------------------------------------
    @staticmethod
    def _device_of(physical_channel):
        return physical_channel.split("/")[0]

    @staticmethod
    def _routing_device(device):
        # On cDAQ the routable trigger lines live on the CHASSIS, not the module.
        return device.split("Mod")[0] if "Mod" in device else device

    def _check_single_task_devices(self):
        """Warn if the AI channels can't all go into one task."""
        devices = {self._device_of(name) for name in self.ai_channels}
        chassis = {self._routing_device(device).lower() for device in devices}
        if len(chassis) > 1:
            self.warning(f"AI channels span several devices {sorted(devices)}; one AI task "
                         "can only hold them if the devices are synchronized for "
                         "multi-device tasks.")

    def _start_trigger_terminal(self):
        """AI start-trigger terminal the AO task is armed on (LabVIEW
        "Get Terminal with Device Prefix" -> "/<device>/ai/StartTrigger")."""
        device = self._device_of(self.ai_channels[0])
        return f"/{self._routing_device(device)}/ai/StartTrigger"

    def _add_ai_channels(self, task, keys=None):
        """Add each pipette's read + resp channel to ``task``; return the channel order."""
        keys = self.pipette_keys if keys is None else list(keys)
        names = [name for key in keys
                 for name in (self.channel_map[key].read, self.channel_map[key].resp)]
        for name in names:
            task.ai_channels.add_ai_voltage_chan(
                name,
                terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF,
                min_val=-10.0, max_val=10.0,
            )
        return names

    def _add_ao_channels(self, task, keys=None):
        """Add each pipette's command channel to ``task``."""
        keys = self.pipette_keys if keys is None else list(keys)
        for key in keys:
            task.ao_channels.add_ao_voltage_chan(self.channel_map[key].cmd)
        return keys

    def _new_ai_task(self, keys=None):
        """Create ``self.ai_task`` holding ``keys``' channels and remember its layout."""
        self.ai_task = nidaqmx.Task()
        self._ai_task_keys = self.pipette_keys if keys is None else list(keys)
        self._ai_task_channels = self._add_ai_channels(self.ai_task, self._ai_task_keys)
        return self.ai_task

    def _new_ao_task(self, keys=None):
        """Create ``self.ao_task`` holding ``keys``' command channels."""
        self.ao_task = nidaqmx.Task()
        self._ao_task_keys = self._add_ao_channels(self.ao_task, keys)
        return self.ao_task

    def _ao_block(self, waves, keys=None):
        """AO write data for ``keys`` (default: the AO task's pipettes).

        ``waves`` is one wave sent to every pipette, or ``{key: wave}`` where
        a missing/None entry means 0 V on that pipette. Returns a 1-D array
        for one channel and a (channels x samples) array for several.
        """
        keys = self._ao_task_keys if keys is None else list(keys)
        if isinstance(waves, dict):
            rows = [waves.get(key) for key in keys]
        else:
            rows = [waves] * len(keys)
        length = max((len(row) for row in rows if row is not None), default=0)
        rows = [np.zeros(length) if row is None else np.asarray(row, dtype=float) for row in rows]
        return rows[0] if len(rows) == 1 else np.vstack(rows)

    def _split_rows(self, raw, keys=None):
        """Split one AI read into ``{key: [read, resp]}``.

        ``keys`` are the pipettes the task was built with (default: the
        running AI task's); rows are in _add_ai_channels order.
        """
        keys = self._ai_task_keys if keys is None else list(keys)
        names = [name for key in keys
                 for name in (self.channel_map[key].read, self.channel_map[key].resp)]
        by_channel = self._rows_to_channel_dict(raw, names)
        return {
            key: np.vstack([by_channel[self.channel_map[key].read],
                            by_channel[self.channel_map[key].resp]])
            for key in keys
        }

    def _read_train(self, exp_samples, timeout=2.0):
        """Wait for one full train and read it for every pipette in the AI task."""
        while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
            self.sleep(0.002)
        return self._split_rows(self.ai_task.read(exp_samples, timeout=timeout))

    def _close_tasks(self):
        """Stop and close the current AI/AO tasks (errors ignored)."""
        for task_name in ("ai_task", "ao_task"):
            task = getattr(self, task_name, None)
            if task is None:
                continue
            try:
                task.stop()
            except Exception:
                pass
            try:
                task.close()
            except Exception:
                pass

    def _restartAcquisition(self):
        try:
            self.ai_task.stop(); self.ao_task.stop()
        except Exception:
            pass                           # tasks may already be stopped
        try:
            self.ai_task.close(); self.ao_task.close()
        except Exception:
            pass
        # Re-create tasks exactly as at start-up
        self._setupAcquisition()
        try:
            # read and discard one buffer length
            _ = self.ai_task.read(
                number_of_samples_per_channel=self._wave_samples, timeout=1.0)
        except nidaqmx.errors.DaqError:
            pass

    def _setupAcquisition(self):
        """
        Configure the continuous AI + AO pair so they start exactly like
        INIT_memtest.vi in LabVIEW:

        ▸ AI owns its own sample clock (CONTINUOUS) and holds every pipette's
            read + resp channels.
        ▸ AO runs at the same rate, in regeneration mode, one command channel
            per pipette (all get the same test pulse).
        ▸ AO is armed with a digital-edge start-trigger that listens to
            the AI StartTrigger ("/<Dev>/ai/StartTrigger").
        ▸ Tasks are started in LabVIEW order:  AO first (armed) → AI starts
            (emits trigger) → AO begins automatically.
        """
        c = nidaqmx.constants

        # ------------------------------------------------------------------
        # 1.  ANALOG INPUT  (read + response channels of every pipette)
        # ------------------------------------------------------------------
        self._new_ai_task()
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=self._wave_rate,
            sample_mode=c.AcquisitionType.CONTINUOUS,
        )

        # Allow the device to overwrite unread data so the fifo keeps running
        self.ai_task.in_stream.overwrite_mode = c.OverwriteMode.OVERWRITE_UNREAD_SAMPLES

        # When we call task.read(), start relative to the newest sample
        self.ai_task.in_stream.read_relative_to = c.ReadRelativeTo.MOST_RECENT_SAMPLE
        # …and back-up exactly one buffer-length (per channel) so we still get a full cycle
        self.ai_task.in_stream.read_offset = -self._wave_samples

        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size,
            self._wave_rate * 10)   # keep 10 s of slack (~200 k samples)

        # ------------------------------------------------------------------
        # 2.  ANALOG OUTPUT  (one command channel per pipette)
        # ------------------------------------------------------------------
        self._new_ao_task()
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=self._wave_rate,
            sample_mode=c.AcquisitionType.CONTINUOUS,
            samps_per_chan=self._wave_samples,
        )
        # Regeneration so a single-period buffer loops forever
        self.ao_task.out_stream.regen_mode = c.RegenerationMode.ALLOW_REGENERATION

        # Arm AO with the AI StartTrigger
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            self._start_trigger_terminal(), trigger_edge=c.Edge.RISING
        )

        # ------------------------------------------------------------------
        # 3.  PRIME AO FIFO  (avoid -200462)
        # ------------------------------------------------------------------
        self.ao_task.write(self._ao_block(self.wave), auto_start=False)

        # ------------------------------------------------------------------
        # 4.  START TASKS  (AO waits → AI fires trigger)
        # ------------------------------------------------------------------
        self.ao_task.start()   # armed / waiting for trigger
        self.ai_task.start()   # emits /ai/StartTrigger → AO begins

        # ------------------------------------------------------------------
        # 5.  Cache the current-wave signature
        # ------------------------------------------------------------------
        self._cur_wave_freq  = 40            # matches the pre-built buffer
        self._cur_wave_rate  = self._wave_rate
        self._cur_wave_duty  = 0.5
        self._cur_wave_amp   = 0.5
        self._cur_wave_rtime = 0.025

    def _setupAcquisitionCurrent(self, recordingTime=500, dutyCycle=0.5, keys=None):
        """
        Configure a dedicated continuous AI + AO pair for the current-step
        protocol, holding ``keys``' channels (default: every pipette).
        The background stream must be paused first.
        """
        c = nidaqmx.constants

        samplesPerSec = 100_000
        dutyCycle     = float(dutyCycle)
        recordingTime = (float(recordingTime) * 1e-3)  # convert ms to seconds
        wave_freq     = 1.0 / (recordingTime)  # 1 Hz square wave

        # ------------------------------------------------------------------
        # 1. prime buffer (0 pA test pulse)
        # ------------------------------------------------------------------
        wave, rate, numSamples = self.createSquareWaveCurrent(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=0.0,
            recordingTime=recordingTime)
        # ------------------------------------------------------------------
        # 2. tasks (AI owns clock; AO waits on AI start trigger)
        # ------------------------------------------------------------------
        self._new_ai_task(keys)
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=c.AcquisitionType.CONTINUOUS)
        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size, rate * 10)

        self._new_ao_task(keys)
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=c.AcquisitionType.CONTINUOUS,
            samps_per_chan=numSamples)
        self.ao_task.out_stream.regen_mode = c.RegenerationMode.ALLOW_REGENERATION

        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            self._start_trigger_terminal(), trigger_edge=c.Edge.RISING)

        # prime AO FIFO, then start in LabVIEW order (AO-armed → AI-fires)
        self.ao_task.write(self._ao_block(wave), auto_start=False)
        self.ao_task.start()
        self.ai_task.start()

    def _setupAcquisitionVoltage(self,
                                 *,
                                 wave_freq: float,
                                 samplesPerSec: int,
                                 dutyCycle: float,
                                 recordingTime: float,
                                 exp_samples: int,
                                 buffer_trains: int,
                                 keys=None):
        """
        Configure the dedicated AI/AO pair for the V-clamp step protocol,
        holding ``keys``' channels (default: every pipette).
        Mirrors the current-clamp setup while preserving voltage-specific buffer sizing.
        """
        c = nidaqmx.constants

        # ------------------------------------------------------------------
        # 1. prime buffer (0 V step)
        # ------------------------------------------------------------------
        zero_wave, _, num_samples = self.createSquareWaveVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=0.0,
            recordingTime=recordingTime,
            store=False,
        )

        # ------------------------------------------------------------------
        # 2. tasks (AI clock; AO waits on AI start trigger)
        # ------------------------------------------------------------------
        self._new_ai_task(keys)
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=samplesPerSec,
            sample_mode=c.AcquisitionType.CONTINUOUS,
        )
        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size,
            exp_samples * buffer_trains,
        )

        self._new_ao_task(keys)
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=samplesPerSec,
            sample_mode=c.AcquisitionType.CONTINUOUS,
            samps_per_chan=num_samples,
        )
        self.ao_task.out_stream.regen_mode = c.RegenerationMode.DONT_ALLOW_REGENERATION
        self.ao_task.out_stream.output_buf_size = max(
            self.ao_task.out_stream.output_buf_size,
            num_samples * buffer_trains,
        )

        # ------------------------------------------------------------------
        # 3. trigger routing (AO armed by AI start trigger)
        # ------------------------------------------------------------------
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            self._start_trigger_terminal(),
            trigger_edge=c.Edge.RISING,
        )

        # ------------------------------------------------------------------
        # 4. prime AO FIFO and start in LabVIEW order
        # ------------------------------------------------------------------
        self.ao_task.write(self._ao_block(zero_wave), auto_start=False)
        self.ao_task.start()
        self.ai_task.start()

    def _readAnalogInput(self, samplesPerSec, recordingTime, channels=None):
        """
        Wrapper for the AI task: read exactly one bufferful of every channel
        in the task, then return the requested ones as ``{channel: samples}``.
        """
        names = self._resolve_channels(channels)
        missing = [name for name in names if name not in self._ai_task_channels]
        if missing:
            raise ValueError(f"Channels {missing} are not in the running AI task "
                             f"({self._ai_task_channels})")
        numSamples = int(samplesPerSec * recordingTime)
        data = self.ai_task.read(
            number_of_samples_per_channel=numSamples,
            timeout=10.0
        )
        by_channel = self._rows_to_channel_dict(data, self._ai_task_channels)
        return {name: by_channel[name] for name in names}

    def _sendSquareWave(self, wave_freq, samplesPerSec,
                        dutyCycle, amplitude, recordingTime):
        """
        Update the AO buffer *without* disturbing the trigger relationship.
        Every pipette's command channel gets the same wave.

        • If the requested wave is identical to the one already streaming,
        do nothing (prevents DAQmx -200015).
        • If the length / rate is unchanged but the shape is different,
        rewrite the buffer **in-place** while the task is running
        (regen-mode allows this).
        • If rate or length must change, stop *both* tasks, re-configure,
        and restart them **in the same trigger order** (AO-first, AI-second).

        """
        c = nidaqmx.constants

        same_shape = (wave_freq     == getattr(self, "_cur_wave_freq",  None) and
                    samplesPerSec == getattr(self, "_cur_wave_rate",  None) and
                    dutyCycle     == getattr(self, "_cur_wave_duty",  None) and
                    amplitude     == getattr(self, "_cur_wave_amp",   None) and
                    recordingTime == getattr(self, "_cur_wave_rtime", None))

        if same_shape:
            return  # nothing to do

        # ---------- build the new buffer ---------------------------------------
        self.createSquareWave(wave_freq, samplesPerSec,
                            dutyCycle, amplitude, recordingTime)

        # ---------- CASE 1: rate & length unchanged  ---------------------------
        if samplesPerSec == self._cur_wave_rate and self._wave_samples == len(self.wave):
            # Task is running; regen_mode lets us replace data in FIFO
            self.ao_task.write(self._ao_block(self.wave), auto_start=False)

        # ---------- CASE 2: rate OR length changed  ----------------------------
        else:
            # 1) stop both tasks
            self.ai_task.stop()
            self.ao_task.stop()

            # 2) re-configure AO timing for new rate / buffer size
            self.ao_task.timing.cfg_samp_clk_timing(
                rate=self._wave_rate,
                sample_mode=c.AcquisitionType.CONTINUOUS,
                samps_per_chan=self._wave_samples)

            # 3) load new buffer
            self.ao_task.write(self._ao_block(self.wave), auto_start=False)

            # 4) restart in trigger order
            self.ao_task.start()
            self.ai_task.start()

        # ---------- remember the new signature ---------------------------------
        self._cur_wave_freq  = wave_freq
        self._cur_wave_rate  = samplesPerSec
        self._cur_wave_duty  = dutyCycle
        self._cur_wave_amp   = amplitude
        self._cur_wave_rtime = recordingTime

    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Queue the next current-step train on the running AO task.

        ``amplitude`` is one value for every pipette, or ``{key: amplitude}``
        where None means 0 V for that pipette (it has no step this train).
        """
        def wave_for(value):
            wave, _, _ = self.createSquareWaveCurrent(
                wave_freq=wave_freq, samplesPerSec=samplesPerSec,
                dutyCycle=dutyCycle, amplitude=value,
                recordingTime=recordingTime
            )
            return wave

        if isinstance(amplitude, dict):
            waves = {key: None if value is None else wave_for(value)
                     for key, value in amplitude.items()}
        else:
            waves = wave_for(amplitude)
        self.ao_task.write(self._ao_block(waves), auto_start=False)

    def _sendSquareWaveVoltage(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Generate the next voltage-step wave and queue it on the running AO
        task for every pipette in it.
        """
        wave, _, _ = self.createSquareWaveVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=amplitude,
            recordingTime=recordingTime,
            store=False,
        )
        self.ao_task.write(self._ao_block(wave), auto_start=False)

    def _build_optogenetic_timeline(self, protocol_steps: list[dict]):
        stim_timeline: list[dict] = []
        cursor = 0.0
        for step in protocol_steps:
            duration = float(step.get("duration_s", 0.0))
            if duration <= 0:
                continue
            entry = dict(step)
            entry["start_s"] = cursor
            cursor += duration
            entry["end_s"] = cursor
            stim_timeline.append(entry)

        if not stim_timeline:
            raise ValueError("protocol_steps must contain positive durations")

        protocol_type = stim_timeline[0].get("protocol_type")
        return stim_timeline, cursor, protocol_type

    def _wait_until(self, deadline_s: float, *, tight_timing: bool = True) -> None:
        """
        Sleep until deadline_s; optionally spin for the last few ms to reduce jitter.
        """
        while True:
            remaining = deadline_s - time.perf_counter()
            if remaining <= 0:
                return
            if not tight_timing or remaining > 0.005:
                time.sleep(min(0.001, max(0.0, remaining - 0.002)))
                continue
            while time.perf_counter() < deadline_s:
                pass

    def _setup_optogenetic_ai_task(self, rate_hz: int, num_samples: int, keys=None):
        """Finite AI task holding ``keys``' channels (default: every pipette)."""
        ai = nidaqmx.Task()
        self._add_ai_channels(ai, keys)
        ai.timing.cfg_samp_clk_timing(
            rate=rate_hz,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=num_samples)
        return ai

    def _read_optogenetic_ai(self, ai_task, num_samples: int, duration_s: float, keys=None):
        """Read the finite task, close it, and return ``{key: [read, resp]}``."""
        raw = ai_task.read(
            number_of_samples_per_channel=num_samples,
            timeout=duration_s + 2.0)
        ai_task.stop()
        ai_task.close()
        return self._split_rows(raw, self.pipette_keys if keys is None else keys)

    # --------------------------------------------------------------
    # Protocols: every requested pipette is acquired in the same tasks
    # and reads. Called as daq.<protocol>(pipettes=[...], ...) and
    # returning {pipette_key: result}. Pipettes not requested get no
    # channels in the protocol tasks (their command stays at 0 V); the
    # live test pulse is paused for every pipette meanwhile.
    # --------------------------------------------------------------
    @multi_pipette_protocol
    def getLeakSubtraction(
            self,
            keys,
            *,
            start_voltage: float,
            end_voltage: float,
            holding_voltage: float,
            step_voltage: float = 20e-3,
            wave_freq: float = 5.0,
            samplesPerSec: int = 50_000,
            dutyCycle: float = 0.5,
            pulse_recording_time: float = 0.600,
            repeats: int = 4,
    ):
        """
        Execute a P/4 leak subtraction sequence prior to a voltage sweep,
        on every pipette in ``keys`` at once.

        For each sweep target, this generates ``repeats`` square pulses whose
        amplitude is ``-(target - holding) / 4``.  Each pulse uses a 50 % duty
        cycle square wave lasting ``pulse_recording_time / 3`` seconds inside a
        baseline–pulse–baseline layout that mirrors ``getVoltageClampSweep``.
        Averaged traces are stored per pipette on ``leak_subtraction_data``
        and the raw repeats are discarded after averaging.
        """
        if step_voltage == 0:
            raise ValueError("step_voltage must be non-zero")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.V_CLAMP_VOLT_PER_VOLT in (None, 0):
            raise RuntimeError("V_CLAMP_VOLT_PER_VOLT must be configured before leak subtraction.")
        if self.V_CLAMP_VOLT_PER_AMP in (None, 0):
            raise RuntimeError("V_CLAMP_VOLT_PER_AMP must be configured before leak subtraction.")

        samplesPerSec = int(samplesPerSec)
        dutyCycle = float(dutyCycle)
        pulse_recording_time = float(pulse_recording_time)
        wave_freq = float(wave_freq)

        span = end_voltage - start_voltage
        step = abs(step_voltage)

        if span == 0:
            targets = np.array([start_voltage], dtype=float)
        else:
            step_signed = step if span > 0 else -step
            targets = np.arange(start_voltage, end_voltage, step_signed, dtype=float)
            if targets.size == 0 or abs(targets[-1] - end_voltage) > 1e-9:
                targets = np.append(targets, end_voltage)

        leak_plan = []
        leak_step_values = []
        leak_target_voltages = []

        for idx, target in enumerate(targets):
            delta = target - holding_voltage
            leak_step = -delta / 4.0
            leak_target_voltage = holding_voltage + leak_step
            ao_amplitude = leak_step / self.V_CLAMP_VOLT_PER_VOLT

            leak_step_values.append(leak_step)
            leak_target_voltages.append(leak_target_voltage)

            for repeat_idx in range(repeats):
                leak_plan.append({
                    "target_index": idx,
                    "target_voltage": target,
                    "leak_target_voltage": leak_target_voltage,
                    "ao_amplitude": ao_amplitude,
                    "repeat_index": repeat_idx,
                })

        meta = {
            "holding_voltage": holding_voltage,
            "targets": targets,
            "leak_target_voltages": np.array(leak_target_voltages, dtype=float),
            "leak_step_values": np.array(leak_step_values, dtype=float),
            "repeats": repeats,
            "wave_freq": wave_freq,
            "pulse_recording_time": pulse_recording_time,
        }
        if not leak_plan:
            for key in keys:
                with self.for_pipette(key):
                    self.leak_subtraction_data = []
                    self.leak_subtraction_meta = dict(meta)
            return {key: [] for key in keys}

        self.pause_acquisition()
        self._close_tasks()

        buffer_trains = max(len(leak_plan) + 2, 4)
        _, _, num_samples = self.createSquareWaveVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=0.0,
            recordingTime=pulse_recording_time,
            store=False,
        )
        exp_samples = num_samples
        total_duration = exp_samples / samplesPerSec

        pulses_by_target = {key: [[] for _ in range(targets.size)] for key in keys}

        try:
            # Task setup is inside the try so a rejected setup still restores
            # the live stream (finally below).
            self._setupAcquisitionVoltage(
                wave_freq=wave_freq,
                samplesPerSec=samplesPerSec,
                dutyCycle=dutyCycle,
                recordingTime=pulse_recording_time,
                exp_samples=exp_samples,
                buffer_trains=buffer_trains,
                keys=keys,
            )

            first_amp = leak_plan[0]["ao_amplitude"]
            self._sendSquareWaveVoltage(
                wave_freq=wave_freq,
                samplesPerSec=samplesPerSec,
                dutyCycle=dutyCycle,
                amplitude=first_amp,
                recordingTime=pulse_recording_time,
            )

            self._read_train(exp_samples)        # discard the 0 V prime train

            for idx, pulse in enumerate(leak_plan):
                if idx + 1 < len(leak_plan):
                    next_amp = leak_plan[idx + 1]["ao_amplitude"]
                    self._sendSquareWaveVoltage(
                        wave_freq=wave_freq,
                        samplesPerSec=samplesPerSec,
                        dutyCycle=dutyCycle,
                        amplitude=next_amp,
                        recordingTime=pulse_recording_time,
                    )

                trains = self._read_train(exp_samples)   # every pipette, one read
                time_vec = np.linspace(0, total_duration, exp_samples, dtype=float)
                for key, raw in trains.items():
                    with self.for_pipette(key):
                        self._squareWaveProcessor(raw, samplesPerSec, pulse["ao_amplitude"])
                    pulses_by_target[key][pulse["target_index"]].append({
                        "time": time_vec,
                        "response": raw[1] * self.V_CLAMP_VOLT_PER_AMP,
                        "command": raw[0] * self.V_CLAMP_VOLT_PER_VOLT,
                        "repeat_index": pulse["repeat_index"],
                    })

                self.info(
                    f"Captured P/4 leak pulse {pulse['repeat_index'] + 1}/{repeats} "
                    f"for {pulse['target_voltage'] * 1e3:.0f} mV sweep target."
                )
        finally:
            self._close_tasks()
            self.resume_acquisition()

        results = {}
        for key in keys:
            leak_results: list[dict] = []
            for idx, target in enumerate(targets):
                pulses = pulses_by_target[key][idx]
                if not pulses:
                    continue
                leak_results.append({
                    "time": pulses[0]["time"].copy(),
                    "response": np.vstack([p["response"] for p in pulses]).mean(axis=0),
                    "command": np.vstack([p["command"] for p in pulses]).mean(axis=0),
                    "target_voltage": target,
                    "leak_target_voltage": leak_target_voltages[idx],
                    "leak_step_voltage": leak_step_values[idx],
                    "repeat_count": len(pulses),
                })
            with self.for_pipette(key):
                self.leak_subtraction_data = leak_results
                self.leak_subtraction_meta = dict(meta)
            results[key] = leak_results
        return results

    @multi_pipette_protocol
    def getVoltageClampSweep(
            self,
            keys,
            *,
            start_voltage: float,
            end_voltage: float,
            holding_voltage: float,
            step_voltage: float = 20e-3,
            wave_freq: int = 4,
            samplesPerSec: int = 50_000,
            dutyCycle: float = 0.5,
            recordingTime: float = 0.750
    ):
        """
        Run a finite V-clamp sweep on every pipette in ``keys`` at once.
        Returns ``{key: [[time, response, command], ...]}`` processed
        identically to the membrane-test traces.
        """
        if step_voltage == 0:
            raise ValueError("step_voltage must be non-zero")

        samplesPerSec = int(samplesPerSec)
        dutyCycle = float(dutyCycle)
        recordingTime = float(recordingTime)
        wave_freq = float(wave_freq)

        span = end_voltage - start_voltage
        step = abs(step_voltage)

        if span == 0:
            targets = np.array([start_voltage], dtype=float)
        else:
            step_signed = step if span > 0 else -step
            targets = np.arange(start_voltage, end_voltage, step_signed, dtype=float)
            if targets.size == 0 or abs(targets[-1] - end_voltage) > 1e-9:
                targets = np.append(targets, end_voltage)

        self.info(f"V-clamp steps (V): {targets}")
        sweeps = {key: [] for key in keys}

        self.pause_acquisition()
        self._close_tasks()
        buffer_trains = max(len(targets) + 2, 4)
        _, _, num_samples = self.createSquareWaveVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=0.0,
            recordingTime=recordingTime,
            store=False,
        )
        exp_samples = num_samples
        total_duration = num_samples / samplesPerSec

        try:
            # Inside the try so a rejected setup still restores the live stream.
            self._setupAcquisitionVoltage(
                wave_freq=wave_freq,
                samplesPerSec=samplesPerSec,
                dutyCycle=dutyCycle,
                recordingTime=recordingTime,
                exp_samples=exp_samples,
                buffer_trains=buffer_trains,
                keys=keys,
            )

            if targets.size:
                first_amp = (targets[0] - holding_voltage) / self.V_CLAMP_VOLT_PER_VOLT
                self._sendSquareWaveVoltage(
                    wave_freq, samplesPerSec, dutyCycle, first_amp, recordingTime
                )

            self._read_train(exp_samples)        # discard the 0 V prime train

            for idx, target in enumerate(targets):
                if idx + 1 < len(targets):
                    next_amp = (targets[idx + 1] - holding_voltage) / self.V_CLAMP_VOLT_PER_VOLT
                    self._sendSquareWaveVoltage(
                        wave_freq, samplesPerSec, dutyCycle, next_amp, recordingTime
                    )
                    self.info(f"Waiting for {target*1000:.0f} mV train…")

                trains = self._read_train(exp_samples)   # every pipette, one read
                step_amp_volts = (target - holding_voltage) / self.V_CLAMP_VOLT_PER_VOLT
                time_vec = np.linspace(0, total_duration, exp_samples, dtype=float)
                for key, raw in trains.items():
                    with self.for_pipette(key):
                        self._squareWaveProcessor(raw, samplesPerSec, step_amp_volts)
                    sweeps[key].append([time_vec,
                                        raw[1] * self.V_CLAMP_VOLT_PER_AMP,
                                        raw[0] * self.V_CLAMP_VOLT_PER_VOLT])
                self.info(f"Captured {target*1000:.0f} mV pulse")

        finally:
            self._close_tasks()
            self.resume_acquisition()

        for key in keys:
            with self.for_pipette(key):
                self.vclamp_steps = targets
                self.vclamp_hold_value = holding_voltage
                self.voltage_protocol_data = sweeps[key]
                if targets.size and self.voltage_membrane_test is not None:
                    self.voltage_membrane_test["sweep_hold_mV"] = holding_voltage * 1e3
        return sweeps

    def _current_pulses(self, custom, startCurrentPicoAmp, endCurrentPicoAmp, stepCurrentPicoAmp):
        """Current steps (pA) for the active pipette.

        The default (non-custom) range is −2 … +10 pA/pF in 1 pA/pF steps of
        that pipette's own membrane capacitance; raises if it has none.
        """
        if not custom:
            capacitance = self.voltageMembraneCapacitance
            if capacitance is None or not np.isfinite(capacitance) or capacitance <= 0:
                raise RuntimeError("Bad capacitance measurement")
            # Convert to absolute pF, round *down* to the nearest whole pF
            cap_pf = int(abs(capacitance))
            startCurrentPicoAmp = -2 * cap_pf
            endCurrentPicoAmp   = 10 * cap_pf
            stepCurrentPicoAmp  = cap_pf

        startCurrentPicoAmp = max(startCurrentPicoAmp, -20000)
        endCurrentPicoAmp   = min(endCurrentPicoAmp,   20000)

        pulses = np.arange(startCurrentPicoAmp,
                           endCurrentPicoAmp + stepCurrentPicoAmp,
                           stepCurrentPicoAmp, dtype=float)
        if 0.0 not in pulses:
            pulses = np.sort(np.append(pulses, 0.0))
        return pulses

    @multi_pipette_protocol
    def getDataFromCurrentProtocol(self,
                                   keys,
                                   custom: bool = False,
                                   factor: float | None = None,
                                   startCurrentPicoAmp: float | None = None,
                                   endCurrentPicoAmp: float | None = None,
                                   stepCurrentPicoAmp: float = 10,
                                   recordingTimeMs: float = 500,
                                   dutyCycle: float = 0.5):
        """
        Run an I-clamp step protocol on every pipette in ``keys`` at once,
        on one continuous stream.

        Each pipette gets its own step list (scaled by its own capacitance),
        so in train i every pipette receives its own step; pipettes that have
        run out of steps get 0 V. A pipette without a usable capacitance is
        skipped (no channels, no data).

        Returns ``{key: (list_of_traces, pulses_array, pulse_count)}``.
        """
        # ───────── constants ─────────
        samplesPerSec  = 100_000
        dutyCycle      = float(dutyCycle)
        recordingTime  = recordingTimeMs * 1e-3   # 0.25 s / segment
        wave_freq      = 1.0 / recordingTime      # 4 Hz
        fullRecTime    = 4 * recordingTime        # 1.0 s / train
        exp_samples    = int(samplesPerSec * fullRecTime)

        # ─ 1. build each pipette's pulse list ───────
        pulses_by_key = {}
        for key in keys:
            with self.for_pipette(key):
                try:
                    pulses_by_key[key] = self._current_pulses(
                        custom, startCurrentPicoAmp, endCurrentPicoAmp, stepCurrentPicoAmp)
                except RuntimeError as exc:
                    self.error(f"{key}: {exc} (run the voltage protocol first, or use "
                               "custom current steps); skipping its I-clamp protocol.")
                    continue
                self.info(f"{key} I-clamp pulses (pA): {pulses_by_key[key]}")
        active = [key for key in keys if key in pulses_by_key]
        traces = {key: [] for key in active}

        def amplitudes(train_index):
            """{key: AO volts for that train, or None once a pipette is done}."""
            return {key: ((pulses_by_key[key][train_index] * 1e-12) / self.C_CLAMP_AMP_PER_VOLT
                          if train_index < len(pulses_by_key[key]) else None)
                    for key in active}

        if not active:
            raise RuntimeError(
                "I-clamp protocol: no pipette has a usable membrane capacitance; "
                "run the voltage protocol first, or use custom current steps.")

        train_count = max(len(pulses_by_key[key]) for key in active)

        # ─ 2. park background stream ─
        self.pause_acquisition()
        self._close_tasks()

        try:
            # ─ 3. start dedicated tasks (prime buffer = 0 pA test) ──────
            # Inside the try so a rejected setup still restores the live stream.
            self._setupAcquisitionCurrent(recordingTime * 1e3, dutyCycle=dutyCycle,
                                          keys=active)   # expects ms

            # a) queue FIRST real pulses so they become train #2
            self._sendSquareWaveCurrent(wave_freq, samplesPerSec,
                                        dutyCycle, amplitudes(0), recordingTime)

            # b) wait 1 s and discard train #1 (0 pA prime)
            self._read_train(exp_samples)

            for idx in range(train_count):
                # ─── queue the NEXT pulses *before* waiting ──────────
                if idx + 1 < train_count:
                    self._sendSquareWaveCurrent(wave_freq, samplesPerSec,
                                                dutyCycle, amplitudes(idx + 1),
                                                recordingTime)

                self.info(f"Waiting for I-clamp train {idx + 1}/{train_count}…")

                # ─── read it: one read for every pipette ────────────
                trains = self._read_train(exp_samples)
                tvec = np.linspace(0, fullRecTime, exp_samples, dtype=float)
                for key, raw in trains.items():
                    if idx < len(pulses_by_key[key]):
                        traces[key].append([tvec, raw[1] / self.C_CLAMP_VOLT_PER_VOLT, raw[0]])

                self.info(f"Captured I-clamp train {idx + 1}/{train_count}")

        finally:
            # ─ 4. restore background stream ────────────────────────
            self._close_tasks()
            self.resume_acquisition()

        results = {}
        for key in keys:
            pulses = pulses_by_key.get(key, np.array([], dtype=float))
            with self.for_pipette(key):
                self.pulses = pulses
                self.pulseRange = len(pulses)
                self.current_protocol_data = traces.get(key, [])
                results[key] = (self.current_protocol_data, pulses, len(pulses))
        return results

    @multi_pipette_protocol
    def getDataFromHoldingProtocol(self, keys, *, rate_hz: int = 50_000, duration_s: float = 60.0):
        """
        OLD-STYLE baseline read-out (no command output), every pipette in
        ``keys`` in the same finite AI task.

        • Suspends the membrane-test thread.
        • Builds one finite AI-only task at 50 kS/s for 60 s
          (same numbers as the legacy code).
        • Does *not* touch the AO task, avoiding the –200547 write error.
        • Restarts the continuous stream afterwards.

        Returns
        -------
        {key: np.ndarray [time_s, resp_A, read_V]}
        """
        c = nidaqmx.constants

        # -------------------------------------------------- 1. park background
        self.pause_acquisition()

        try:
            num_samples = int(rate_hz * duration_s)

            # ---------------------------------------------- 2. finite AI task
            ai = nidaqmx.Task()
            self._add_ai_channels(ai, keys)
            ai.timing.cfg_samp_clk_timing(
                rate=rate_hz,
                sample_mode=c.AcquisitionType.FINITE,
                samps_per_chan=num_samples)

            ai.start()
            raw = ai.read(
                number_of_samples_per_channel=num_samples,
                timeout=duration_s + 2.0)
            ai.stop(); ai.close()

            t = np.linspace(0, duration_s, num_samples, dtype=float)
            results = {}
            for key, rows in self._split_rows(raw, keys).items():
                # Keep unit conversions consistent with other V-clamp protocol paths.
                with self.for_pipette(key):
                    self.holding_protocol_data = np.array([
                        t,
                        rows[1] * self.V_CLAMP_VOLT_PER_AMP,
                        rows[0] * self.V_CLAMP_VOLT_PER_VOLT,
                    ])
                    results[key] = self.holding_protocol_data
            return results

        finally:
            # ---------------------------------------------- 3. resume stream
            self.resume_acquisition()

    @multi_pipette_protocol
    def getDataFromOptogeneticProtocol(
        self,
        keys,
        *,
        laser,
        protocol_steps: list[dict],
        rate_hz: int = 50_000,
    ):
        """
        Modified holding protocol that records only during optogenetic "on"
        steps. Each "on" step is one finite AI task covering every pipette in
        ``keys``, so the laser fires once and every pipette records it.

        Returns
        -------
        {key: (list[list[np.ndarray]], list[dict])}
            Per pipette: one trace per on-step plus the stimulation timeline.
        """
        if laser is None:
            raise ValueError("laser is required")
        if protocol_steps is None or len(protocol_steps) == 0:
            raise ValueError("protocol_steps must contain at least one step")

        stim_timeline, _, protocol_type = self._build_optogenetic_timeline(protocol_steps)

        self.pause_acquisition()
        ai = None
        actual_timeline = []

        try:
            try:
                laser.power_off()
            except Exception:
                pass
            t0 = time.perf_counter()
            aligned_traces = {key: [] for key in keys}
            aligned_pulses: list[dict] = []

            for step in stim_timeline:
                duration_s = float(step.get("duration_s", 0.0))
                if duration_s <= 0:
                    continue

                planned_start = float(step.get("start_s", 0.0))
                planned_end = float(step.get("end_s", planned_start + duration_s))
                self._wait_until(t0 + planned_start, tight_timing=True)

                state = step.get("state", "on")
                wavelength = step.get("wavelength")
                power_percent = step.get("power_percent")
                if state == "on":
                    if isinstance(wavelength, str) and wavelength.strip().lower() == "off":
                        state = "off"
                    else:
                        name = getattr(wavelength, "name", None)
                        if isinstance(name, str) and name.upper() == "OFF":
                            state = "off"

                step_start = time.perf_counter() - t0
                if state == "on":
                    if wavelength is not None:
                        laser.set_wavelength(wavelength)
                    if power_percent is not None:
                        laser.set_power_level(power_percent, wavelength)
                    try:
                        laser.power_off()
                    except Exception:
                        pass

                    num_samples = int(rate_hz * duration_s)
                    if num_samples <= 0:
                        laser.power_off()
                        continue

                    ai = self._setup_optogenetic_ai_task(rate_hz, num_samples, keys)
                    ai.start()

                    # Laser on from 1/3 to 2/3 of the recording window.
                    record_start = time.perf_counter()
                    laser_on_at = record_start + (duration_s / 3.0)
                    laser_off_at = record_start + (2.0 * duration_s / 3.0)

                    self._wait_until(laser_on_at, tight_timing=True)
                    try:
                        laser.power_on()
                    except Exception:
                        pass
                    self._wait_until(laser_off_at, tight_timing=True)
                    try:
                        laser.power_off()
                    except Exception:
                        pass

                    rows_by_key = self._read_optogenetic_ai(ai, num_samples, duration_s, keys)
                    ai = None

                    t = np.linspace(0, duration_s, num_samples, dtype=float)
                    for key, raw in rows_by_key.items():
                        aligned_traces[key].append([
                            t,
                            raw[1] * self.V_CLAMP_VOLT_PER_AMP,
                            raw[0] * self.V_CLAMP_VOLT_PER_VOLT,
                        ])
                    aligned_pulses.append({
                        "wavelength": wavelength,
                        "power_percent": power_percent,
                        "replicate": step.get("replicate"),
                        "protocol_type": step.get("protocol_type"),
                    })
                    try:
                        laser.power_off()
                    except Exception:
                        pass
                else:
                    try:
                        laser.power_off()
                    except Exception:
                        pass

                self._wait_until(t0 + planned_end, tight_timing=True)
                step_end = time.perf_counter() - t0

                entry = dict(step)
                entry["state"] = state
                if power_percent is not None:
                    entry["power_percent"] = power_percent
                entry["start_s"] = step_start
                entry["end_s"] = step_end
                actual_timeline.append(entry)

            stim_used = actual_timeline or stim_timeline
            if protocol_type is None and stim_used:
                protocol_type = stim_used[0].get("protocol_type")
            protocol_key = self._normalize_optogenetic_protocol_key(protocol_type)

            results = {}
            for key in keys:
                with self.for_pipette(key):
                    self.optogenetic_protocol_data = aligned_traces[key]
                    self.optogenetic_stim_data = stim_used
                    self.optogenetic_protocol_type = protocol_type
                    self.optogenetic_pulses = list(aligned_pulses)
                    self.optogenetic_pulseRange = len(aligned_pulses)
                    with self._optogenetic_queue_lock:
                        self.optogenetic_protocol_queue.append({
                            "data": aligned_traces[key],
                            "pulses": list(aligned_pulses),
                            "pulse_range": len(aligned_pulses),
                            "stim_data": stim_used,
                            "protocol_type": protocol_type,
                            "protocol_key": protocol_key,
                        })
                results[key] = (aligned_traces[key], stim_used)
            return results

        finally:
            try:
                laser.power_off()
            except Exception:
                pass
            try:
                if ai is not None:
                    ai.stop(); ai.close()
            except Exception:
                pass
            self.resume_acquisition()

# ========================================================
#  ArduinoDAQ Subclass
# ========================================================
class ArduinoDAQ(DAQ):
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA per V (DAQ output)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3)  # 10 mV per V (DAQ input)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3)  # 20 mV per V (DAQ output)
    V_CLAMP_VOLT_PER_AMP = (2 * 1e-9)   # 2 V per A (DAQ input)

    def __init__(self, DAQSerial=None, pipette_keys=None):
        # Single pipette: the serial stream carries one (read, resp) pair.
        keys = list(pipette_keys) if pipette_keys else [pipette_key_for(0)]
        self._arduino_channel_map = build_channel_map(
            keys[:1], channels=[{"read": "ai0", "resp": "ai1", "cmd": "ao0"}])
        if DAQSerial is not None and isinstance(DAQSerial, serial.Serial):
            self.DAQSerial = DAQSerial
            self.info(f"ArduinoDAQ initialized with serial port: {self.DAQSerial.port} at {self.DAQSerial.baudrate} baud.")
        else:
            self.DAQSerial = None
            self.error("DAQSerial must be an instance of serial.Serial and already opened.")
            raise ValueError("DAQSerial must be an instance of serial.Serial and already opened.")
        super().__init__(channel_map=self._arduino_channel_map)
        self.info(f'Using {self.DAQSerial.port} for reading and writing.')

    def _readAnalogInput(self, samplesPerSec, recordingTime, channels=None):
        names = self._resolve_channels(channels)
        if self.DAQSerial is None:
            self.error("Serial port not initialized.")
            raise RuntimeError("Serial port not initialized.")
        numSamplesExpected = int(recordingTime * samplesPerSec)
        collecting_data = False
        command_vals = []
        response_vals = []
        ADC_scale = 3.3 / 4096  # ADC scaling factor

        while True:
            if self.DAQSerial.in_waiting > 0:
                line = self.DAQSerial.readline().decode('utf-8').strip()
                if line == "start":
                    collecting_data = True
                elif line == "end":
                    break
                elif collecting_data:
                    values = line.split(',')
                    if len(values) == 2:
                        cmd_val, resp_val = values
                        command_vals.append(float(cmd_val) * ADC_scale)
                        response_vals.append(float(resp_val) * ADC_scale)
                    else:
                        self.error(f"Unexpected data format: {line}")

        readData = np.array(command_vals, dtype=float)
        respData = np.array(response_vals, dtype=float)
        by_channel = self._rows_to_channel_dict([readData, respData], self.ai_channels)
        return {name: by_channel[name] for name in names}

    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        scaling = 1024  # 10-bit DAC scaling
        if self.DAQSerial is None:
            self.error("Serial port not initialized.")
            raise RuntimeError("Serial port not initialized.")

        signalDurationMicros = int(recordingTime * 1e6)  # Duration in microseconds
        waveFrequencyMicros = int(1e6 / wave_freq)  # Period in microseconds
        waveAmplitude = int(amplitude * scaling)  # Scale amplitude
        sampleIntervalMicros = int(1e6 / samplesPerSec)  # Sampling interval
        dutyCyclePercent = int(dutyCycle * 100)
        command_str = f"a {signalDurationMicros} {waveFrequencyMicros} {waveAmplitude} {sampleIntervalMicros} {dutyCyclePercent}\n"
        self.DAQSerial.reset_input_buffer()
        self.DAQSerial.reset_output_buffer()
        self.DAQSerial.write(command_str.encode())
        # For Arduino, we do not return a task object; simply return None.
        return None

# ========================================================
#  FakeDAQ Subclass 
# ========================================================
class FakeDAQ(DAQ):
    """DAQ implementation using recorded traces for development or testing without hardware."""

    default_recorded_data = 'FakDAQ_graph_recording.csv'

    # Use unity conversions so that the base class computations work out of the
    # box.  These constants map DAQ units directly to physical units.
    C_CLAMP_AMP_PER_VOLT = 1.0
    C_CLAMP_VOLT_PER_VOLT = 1.0
    V_CLAMP_VOLT_PER_VOLT = 1.0
    V_CLAMP_VOLT_PER_AMP = 1.0

    def __init__(self, pipette_keys=None, channels=None):
        """
        pipette_keys: selected pipettes in rig order (the rig builder passes
            these). Defaults to one pipette.
        channels: optional list of {"read", "resp", "cmd"} names, one per
            pipette. By default pipette i uses ai{2i} (read), ai{2i+1} (resp)
            and ao{i} (cmd).
        """
        super().__init__(channel_map=build_channel_map(
            pipette_keys,
            channels=channels,
            default_channels=enumerated_channels,
        ))
        for key in self.pipette_keys:
            self._pipette_state(key)["totalResistance"] = 6 * 10 ** 6  # baseline fake resistance
        self._last_wave_params = None
        self._last_command = None
        self._recorded_traces = self._load_recorded_data(self.default_recorded_data)
        # One replay position shared by every channel: each read is one
        # "clock tick" for all channels, like a single hardware task.
        self._recorded_trace_index = 0
        self._replay_lock = threading.Lock()
        # (pipette index, row in the recorded trace) for each AI channel
        self._fake_channel_sources = {}
        for index, spec in enumerate(self.channel_map.values()):
            self._fake_channel_sources[spec.read] = (index, 0)   # command row
            self._fake_channel_sources[spec.resp] = (index, 1)   # response row
        self.start_acquisition(wave_freq=40, samplesPerSec=100000, dutyCycle=0.5,
                                amplitude=0.5, recordingTime=0.025, interval=None)



    def _restartAcquisition(self):
        time.sleep(0.01)
        self._setupAcquisition()
        return None

    def _setupAcquisition(self):
        '''
        Cache the signature of the fake continuous-acquisition waveform.
        '''
        self.createSquareWave(40, 100_000, 0.5, 0.5, 0.025)
        self._cur_wave_freq  = 40            # matches the pre-built buffer
        self._cur_wave_rate  = self._wave_rate
        self._cur_wave_duty  = 0.5
        self._cur_wave_amp   = 0.5
        self._cur_wave_rtime = 0.025

        return None

    def _setupAcquisitionCurrent(self, recordingTime=500, dutyCycle=0.5):
        """Initialize the fake current-step waveform."""
        samplesPerSec = 100_000
        dutyCycle     = float(dutyCycle)
        recordingTime = (float(recordingTime) * 1e-3)  # convert ms to seconds
        wave_freq     = 1.0 / (recordingTime)  # 1 Hz square wave

        # ------------------------------------------------------------------
        # 1. prime buffer (0 pA test pulse)
        # ------------------------------------------------------------------
        self.createSquareWaveCurrent(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=0.0,
            recordingTime=recordingTime)

        return None


    def _setupAcquisitionVoltage(self,
                                 *,
                                 wave_freq: float,
                                 samplesPerSec: int,
                                 dutyCycle: float,
                                 recordingTime: float,
                                 exp_samples: int,
                                 buffer_trains: int):
        """Initialize the fake voltage-step waveform."""

        # ------------------------------------------------------------------
        # 1. prime buffer (0 V step)
        # ------------------------------------------------------------------
        zero_wave, _, num_samples = self.createSquareWaveVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=0.0,
            recordingTime=recordingTime,
            store=False,
        )

        return None

    def _readAnalogInput(self, samplesPerSec, recordingTime, channels=None):
        """Return the next recorded tick as ``{channel: samples}``."""
        return self._read_recorded(channels)

    def _read_recorded(self, channels=None):
        """Advance the shared replay clock one tick and return every requested
        channel from that tick, trimmed to a common length.

        Pipette i replays recorded trace (tick + i), so pipettes get different
        but simultaneous data. The read channel carries the recorded command
        row and the resp channel the recorded response row.
        """
        names = self._resolve_channels(channels)
        with self._replay_lock:
            tick = self._recorded_trace_index
            self._recorded_trace_index = (tick + 1) % len(self._recorded_traces)
        rows = {}
        for name in names:
            pipette_index, row = self._fake_channel_sources[name]
            trace = self._recorded_traces[(tick + pipette_index) % len(self._recorded_traces)]
            rows[name] = trace[row]
        length = min((values.size for values in rows.values()), default=0)
        return {name: values[:length].copy() for name, values in rows.items()}

    def _squareWaveProcessor(self, raw_data, samplesPerSec, amplitude, sweep_clip=None):
        """Measure the recorded command amplitude before processing the trace."""
        command = np.asarray(raw_data[0], dtype=float)
        if command.size:
            low, high = np.percentile(command, [5, 95])
            amplitude = (high - low) / self.V_CLAMP_VOLT_PER_VOLT
        return super()._squareWaveProcessor(raw_data, samplesPerSec, amplitude, sweep_clip)


    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """Generate and store a square wave command."""
        command, rate, _ = self.createSquareWave(
            wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime
        )
        self._last_wave_params = (wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
        self._last_command = command
        self._wave_rate = rate
        # Base class expects a task-like object; returning None is acceptable.
        return None

    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        '''
        Generate the next current-step wave
        '''
        wave, rate, numSamples = self.createSquareWaveCurrent(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=amplitude,
            recordingTime=recordingTime
        )
        self._last_command = wave
        self._last_wave_params = (wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
        return None

    def _sendSquareWaveVoltage(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Generate the next voltage-step wave
        """
        wave, _, _ = self.createSquareWaveVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=amplitude,
            recordingTime=recordingTime,
            store=False,
        )
        self._last_command = wave
        self._last_wave_params = (wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
        return None

    @staticmethod
    def _parse_recorded_array(value):
        text = (value or "").strip().strip("[]")
        text = text.replace("np.float64(", "").replace(")", "")
        return np.fromstring(text, sep=",")

    @classmethod
    def _load_recorded_data(cls, data_path):
        # Resolve relative to this .py file (not the working directory).
        # If data_path is absolute, pathlib ignores the left side and uses it as-is.
        graph_file = Path(__file__).parent / data_path
        if not graph_file.is_file():
            raise FileNotFoundError(f"Recorded DAQ data not found: {graph_file}")

        traces = []
        with graph_file.open("r", newline="", encoding="utf-8-sig") as data_file:
            reader = csv.DictReader(data_file, delimiter=";")
            if reader.fieldnames is None:
                raise ValueError(f"Missing header in recorded DAQ data: {graph_file}")
            columns = {name.strip().lower(): name for name in reader.fieldnames}
            if not {"current", "voltage"}.issubset(columns):
                raise ValueError(f"{graph_file} must contain current and voltage columns")
            for row in reader:
                current = cls._parse_recorded_array(row.get(columns["current"]))
                voltage = cls._parse_recorded_array(row.get(columns["voltage"]))
                sample_count = min(current.size, voltage.size)
                if sample_count:
                    traces.append((voltage[:sample_count], current[:sample_count]))

        if not traces:
            raise ValueError(f"No paired current/voltage traces found in {graph_file}")
        return traces

    @staticmethod
    def _recorded_time(sample_count, rate_hz):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        return np.arange(sample_count, dtype=float) / float(rate_hz)

    @staticmethod
    def _trace_targets(start, end, step):
        if step == 0:
            raise ValueError("step must be non-zero")
        span = end - start
        if span == 0:
            return np.array([start], dtype=float)
        step_signed = abs(step) if span > 0 else -abs(step)
        targets = np.arange(start, end, step_signed, dtype=float)
        if targets.size == 0 or abs(targets[-1] - end) > 1e-9:
            targets = np.append(targets, end)
        return targets

    # --------------------------------------------------------------
    # Protocols: each step is one read covering every requested pipette.
    # Called as daq.<protocol>(pipettes=[...], ...) -> {pipette_key: result}.
    # --------------------------------------------------------------
    @multi_pipette_protocol
    def getLeakSubtraction(
            self,
            keys,
            *,
            start_voltage: float,
            end_voltage: float,
            holding_voltage: float,
            step_voltage: float = 20e-3,
            wave_freq: float = 5.0,
            samplesPerSec: int = 50_000,
            dutyCycle: float = 0.5,
            pulse_recording_time: float = 0.600,
            repeats: int = 4,
    ):
        """Average recorded traces for each target in a P/4 leak subtraction sequence."""
        if step_voltage == 0:
            raise ValueError("step_voltage must be non-zero")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        targets = self._trace_targets(start_voltage, end_voltage, step_voltage)
        data_by_key = {key: [] for key in keys}
        leak_steps = []
        leak_targets = []

        for target in targets:
            leak_step = -(target - holding_voltage) / 4.0
            leak_target = holding_voltage + leak_step
            repeat_reads = [self._read_pipettes(keys, samplesPerSec, pulse_recording_time)
                            for _ in range(repeats)]
            for key in keys:
                traces = [read[key] for read in repeat_reads]
                # repeats come from separate reads, which can differ in length
                sample_count = min(command.size for command, _ in traces)
                commands = np.vstack([command[:sample_count] for command, _ in traces])
                responses = np.vstack([response[:sample_count] for _, response in traces])
                data_by_key[key].append({
                    "time": self._recorded_time(sample_count, samplesPerSec),
                    "response": responses.mean(axis=0),
                    "command": commands.mean(axis=0),
                    "target_voltage": target,
                    "leak_target_voltage": leak_target,
                    "leak_step_voltage": leak_step,
                    "repeat_count": repeats,
                })
            leak_steps.append(leak_step)
            leak_targets.append(leak_target)

        results = {}
        for key in keys:
            with self.for_pipette(key):
                self.leak_subtraction_data = data_by_key[key]
                self.leak_subtraction_meta = {
                    "holding_voltage": holding_voltage,
                    "targets": targets,
                    "leak_target_voltages": np.asarray(leak_targets, dtype=float),
                    "leak_step_values": np.asarray(leak_steps, dtype=float),
                    "repeats": repeats,
                    "wave_freq": wave_freq,
                    "pulse_recording_time": pulse_recording_time,
                }
            results[key] = data_by_key[key]
        return results

    @multi_pipette_protocol
    def getVoltageClampSweep(
            self,
            keys,
            *,
            start_voltage: float,
            end_voltage: float,
            holding_voltage: float,
            step_voltage: float = 20e-3,
            wave_freq: int = 4,
            samplesPerSec: int = 50_000,
            dutyCycle: float = 0.5,
            recordingTime: float = 0.750
    ):
        targets = self._trace_targets(start_voltage, end_voltage, step_voltage)
        sweeps = {key: [] for key in keys}
        for _ in targets:
            for key, (command, response) in self._read_pipettes(keys, samplesPerSec, recordingTime).items():
                sweeps[key].append([
                    self._recorded_time(command.size, samplesPerSec),
                    response,
                    command,
                ])

        results = {}
        for key in keys:
            with self.for_pipette(key):
                self.vclamp_steps = targets
                self.vclamp_hold_value = holding_voltage
                self.voltage_protocol_data = sweeps[key]
                if targets.size and self.voltage_membrane_test is not None:
                    self.voltage_membrane_test["sweep_hold_mV"] = holding_voltage * 1e3
            results[key] = sweeps[key]
        return results

    def _current_pulses(self, custom, startCurrentPicoAmp, endCurrentPicoAmp, stepCurrentPicoAmp):
        """Current steps (pA) for the active pipette. The default range scales
        with that pipette's own membrane capacitance."""
        if not custom:
            capacitance = self.latestMembraneCapacitance
            if capacitance is None or not np.isfinite(capacitance) or capacitance <= 0:
                cap_pf = max(1, int(abs(stepCurrentPicoAmp)))
            else:
                cap_pf = max(1, int(abs(capacitance)))
            startCurrentPicoAmp = -2 * cap_pf
            endCurrentPicoAmp = 10 * cap_pf
            stepCurrentPicoAmp = cap_pf
        if startCurrentPicoAmp is None or endCurrentPicoAmp is None:
            raise ValueError("Current protocol start and end currents are required")
        if stepCurrentPicoAmp <= 0:
            raise ValueError("stepCurrentPicoAmp must be positive")

        startCurrentPicoAmp = max(float(startCurrentPicoAmp), -20_000)
        endCurrentPicoAmp = min(float(endCurrentPicoAmp), 20_000)
        pulses = np.arange(
            startCurrentPicoAmp,
            endCurrentPicoAmp + stepCurrentPicoAmp,
            stepCurrentPicoAmp,
            dtype=float,
        )
        if 0.0 not in pulses:
            pulses = np.sort(np.append(pulses, 0.0))
        return pulses

    @multi_pipette_protocol
    def getDataFromCurrentProtocol(self,
                                   keys,
                                   custom: bool = False,
                                   factor: float | None = None,
                                   startCurrentPicoAmp: float | None = None,
                                   endCurrentPicoAmp: float | None = None,
                                   stepCurrentPicoAmp: float = 10,
                                   recordingTimeMs: float = 500,
                                   dutyCycle: float = 0.5):
        """Current-step protocol for several pipettes; sweep i is one read.

        Pipettes can have different numbers of pulses (the default range
        depends on each pipette's capacitance). Sweep i reads only pipettes
        that still have a pulse i; on hardware the others would get a 0 command.
        Returns ``{key: (sweeps, pulses, pulseRange)}``.
        """
        recording_time = float(recordingTimeMs) * 1e-3
        if recording_time <= 0:
            raise ValueError("recordingTimeMs must be positive")
        pulses_by_key = {}
        for key in keys:
            with self.for_pipette(key):
                pulses_by_key[key] = self._current_pulses(
                    custom, startCurrentPicoAmp, endCurrentPicoAmp, stepCurrentPicoAmp)

        sweeps = {key: [] for key in keys}
        sweep_count = max(len(pulses) for pulses in pulses_by_key.values())
        for sweep_index in range(sweep_count):
            stepping = [key for key in keys if sweep_index < len(pulses_by_key[key])]
            traces = self._read_pipettes(stepping, 100_000, recording_time)
            for key, (command_voltage, response_current) in traces.items():
                command_current = response_current
                response_voltage = command_voltage
                if self.C_CLAMP_AMP_PER_VOLT not in (None, 0):
                    command_current = command_current / self.C_CLAMP_AMP_PER_VOLT
                if self.C_CLAMP_VOLT_PER_VOLT not in (None, 0):
                    response_voltage = response_voltage / self.C_CLAMP_VOLT_PER_VOLT
                sweeps[key].append([
                    self._recorded_time(command_voltage.size, 100_000),
                    response_voltage,
                    command_current,
                ])

        results = {}
        for key in keys:
            pulses = pulses_by_key[key]
            with self.for_pipette(key):
                self.pulses = pulses
                self.pulseRange = len(pulses)
                self.current_protocol_data = sweeps[key]
            results[key] = (sweeps[key], pulses, len(pulses))
        return results

    @multi_pipette_protocol
    def getDataFromHoldingProtocol(self, keys, *, rate_hz: int = 50_000, duration_s: float = 60.0):
        results = {}
        for key, (command, response) in self._read_pipettes(keys, rate_hz, duration_s).items():
            sample_count = command.size
            with self.for_pipette(key):
                hold_voltage = self.holding_voltage
                if hold_voltage is None:
                    hold_voltage = float(np.mean(command)) if sample_count else 0.0
                command = np.full(sample_count, hold_voltage, dtype=float)
                self.holding_protocol_data = np.array([
                    self._recorded_time(sample_count, rate_hz),
                    response,
                    command,
                ])
                results[key] = self.holding_protocol_data
        return results

    def _membrane_step_mV(self, command, amplitude):
        """Measure the step from the recorded command trace."""
        return float((np.percentile(command, 95) - np.percentile(command, 5)) * 1e3)
