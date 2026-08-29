import threading
import time
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal as signal

# Import your hardware libraries as needed
import nidaqmx
import nidaqmx.constants
import serial

from patcherbot.controller.base import TaskController
from patcherbot.devices.pressurecontroller.BasePressureController import PressureController

__all__ = ['NiDAQ', 'ArduinoDAQ', 'FakeDAQ']

import threading
import numpy as np
import scipy.optimize
import collections
import logging


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
      callback: Optional function to call with the acquired data.
                The callback receives:
                  totalResistance, accessResistance, membraneResistance,
                  membraneCapacitance, respData, readData.
      interval: Time (seconds) to wait between acquisitions (default is recordingTime).
    """
    def __init__(self, daq, wave_freq=40, samplesPerSec=100000, dutyCycle=0.5,
                 amplitude=0.5, recordingTime=0.025, callback=None, interval=None):
        """
        Initialize the acquisition thread and configure acquisition parameters.

        args:
            daq: DAQ interface instance.
            wave_freq (float): Square wave frequency in Hz.
            samplesPerSec (int): Sampling rate.
            dutyCycle (float): Duty cycle (0–1).
            amplitude (float): Signal amplitude.
            recordingTime (float): Acquisition duration per cycle.
            callback (callable | None): Optional callback for processed data.
            interval (float | None): Time between acquisitions.       
        """
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

        while self.running:
            # ---------- wait until resumed ---------------------------------
            self._pause_evt.wait()
            if not self.running:
                break

            cycle_start = time.perf_counter()

            try:
                # ---------- one square-wave cycle --------------------------
                data = self.daq.getDataFromSquareWave(
                    self.wave_freq, self.samplesPerSec,
                    self.dutyCycle, self.amplitude,
                    self.recordingTime
                )
                if not data:                 # empty FIFO read → retry
                    continue

                (t_r, respData), (_, readData), totalR, memR, accR, memC = data
                fit_ok = (totalR is not None)

                # ---------- handle failed fit ------------------------------
                if not fit_ok:
                    self._fail_streak += 1
                    totalR = accR = memR = memC = np.nan   # GUI-safe NaNs

                    if self._fail_streak >= self._fail_limit:
                        with self.daq._deviceLock:
                            self.daq._restartAcquisition()
                        self._fail_streak = 0
                else:
                    self._fail_streak = 0     # reset on success

                # ---------- enqueue for GUI / callback ---------------------
                packet = {
                    "timeData":            t_r,
                    "respData":            respData,
                    "readData":            readData,
                    "totalResistance":     totalR,
                    "membraneResistance":  memR,
                    "accessResistance":    accR,
                    "membraneCapacitance": memC
                }
                self._last_data_queue.append(packet)

                # update live attributes even if NaN
                self.daq.totalResistance           = totalR
                self.daq.latestMembraneResistance  = memR
                self.daq.latestAccessResistance    = accR
                self.daq.latestMembraneCapacitance = memC

                if self.callback:
                    self.callback(totalR, accR, memR, memC,
                                respData, readData)

            except nidaqmx.errors.DaqError as e:
                # common FIFO/buffer overruns
                self.daq.warning(f"Acquisition error {e.error_code}: {e}")
                if e.error_code in (-200279, -200284, -200286):
                    with self.daq._deviceLock:
                        self.daq._restartAcquisition()
                time.sleep(0.05)            # brief cooldown

            except Exception as e:
                logging.warning(f"Error in DAQAcquisitionThread: {e}")

            # ---------- honour requested interval --------------------------
            elapsed       = time.perf_counter() - cycle_start
            time.sleep(max(0.0, self.interval - elapsed))

    def get_last_data(self):
        """
        Return the most recent measurement or None if not available.
        
        returns:
            dict | None: Latest data packet containing time series and computed
            electrical properties, or None if no data is available.
        """
        return dict(self._last_data_queue[-1]) if self._last_data_queue else None

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

    def __init__(self):
        """
        Initialize the DAQ base class and internal state.

        Sets up protocol storage, acquisition state variables, synchronization
        primitives, and measurement tracking fields.
        """
        super().__init__()
        self.pulses = None
        self.pulseRange = None
        self.latest_protocol_data = None
        self.current_protocol_data = None
        self.voltage_protocol_data = None
        self.voltage_membrane_test = None
        self.vclamp_steps = None
        self.vclamp_hold_value = None
        self.leak_subtraction_data = None
        self.leak_subtraction_meta = None
        self.holding_protocol_data = None
        self.optogenetic_protocol_data = None
        self.optogenetic_stim_data = None
        self.optogenetic_protocol_type = None
        self.optogenetic_pulses = None
        self.optogenetic_pulseRange = None
        self.optogenetic_protocol_queue = collections.deque()
        self._optogenetic_queue_lock = threading.Lock()
        self._deviceLock = threading.Lock()
        self.isRunningProtocol = False
        self.totalResistance = None
        self.latestAccessResistance = None
        self.latestMembraneResistance = None
        self.latestMembraneCapacitance = None
        self.equalizer = False
        self.holding_current = None
        self.holding_voltage = None
        # False is for BATH mode, True for CELL mode
        self.cellMode = False
        self._pause_evt = threading.Event(); self._pause_evt.set()   # allow run
        self._idle_evt  = threading.Event(); self._idle_evt.set()    # initially idle
        self.sweep_clip = None


    def setCellMode(self, mode: bool) -> None:
        """
        Set the DAQ operating mode.

        Args:
            mode: True for cell mode, False for bath mode.
        """
        self.cellMode = mode
        self.info(f"Setting cell mode to {mode}")

    def getCellMode(self) -> bool:
        """
        Get the current DAQ operating mode.

        Returns:
            bool: True if in cell mode, False if in bath mode.
        """
        return self.cellMode

    def _normalize_optogenetic_protocol_key(self, protocol_type):
        """
        Normalize an optogenetic protocol identifier into a lowercase string key.

        Args:
            protocol_type: Protocol identifier, enum, or string.

        Returns:
            str: Normalized protocol key.
        """
        if protocol_type is None:
            return "unknown"
        name = getattr(protocol_type, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip().lower()
        text = str(protocol_type).strip().lower()
        return text if text else "unknown"

    def pop_optogenetic_entry(self, protocol_key):
        """
        Remove and return a matching optogenetic protocol entry from the queue.

        Args:
            protocol_key: Identifier for the protocol to retrieve.

        Returns:
            Optional[dict]: Matching protocol entry if found, otherwise None.
        """
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

    def get_last_acquisition(self):
        """
        Retrieve the most recent measurement from the acquisition thread.
        Returns a dictionary with keys:
          "timeData", "respData", "readData", "totalResistance",
          "membraneResistance", "accessResistance", "membraneCapacitance"
        or None if no data is available.

        Returns:
            Optional[dict]: Dictionary containing measurement data, or None if unavailable.
        """
        if hasattr(self, "_daq_acq_thread") and self._daq_acq_thread:
            return self._daq_acq_thread.get_last_data()
        return None

    def compute_noise_metrics(self, timeData=None, respData=None,
                              window_start=0.004, window_end=0.010,
                              avg_p2p_window=0.001):
        """
        Compute noise metrics from response data within a time window.
        Returns a dict with windowed data, p2p, std, avg_p2p, and FFT results,
        or None if inputs are insufficient.

        Args:
            timeData: Time array corresponding to the signal.
            respData: Response signal array.
            window_start: Start time of analysis window.
            window_end: End time of analysis window.
            avg_p2p_window: Window size for averaging peak-to-peak values.

        Returns:
            Optional[dict]: Dictionary containing noise metrics including p2p,
            standard deviation, average p2p, FFT frequencies, and magnitudes.
            Returns None if computation cannot be performed.
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
            
    def pause_acquisition(self, timeout: float = 2.0):
        """
        Block until the acquisition thread is parked **and** both DAQmx tasks
        have been stopped, giving the caller sole ownership of the hardware.
        
        Args:
            timeout: Maximum time in seconds to wait for the thread to idle.

        Raises:
            TimeoutError: If the acquisition thread does not become idle in time.
        """
        if self._pause_evt.is_set():
            self._pause_evt.clear()                 # ask thread to park
            if not self._idle_evt.wait(timeout):
                raise TimeoutError("DAQ thread failed to become idle")

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
    
    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Single-cycle acquisition: fire the pre-made AO buffer, read back AI,
        then process into (time/resp, time/read, totalR, memR, accR, memC).
        
        Args:
            wave_freq: Frequency of the square wave in Hz.
            samplesPerSec: Sampling rate in samples per second.
            dutyCycle: Duty cycle of the square wave (0 to 1).
            amplitude: Output amplitude of the square wave.
            recordingTime: Duration of the acquisition in seconds.

        Returns:
            tuple: Processed data including time/response arrays and computed
            resistance and capacitance values.
        """
        with self._deviceLock:
            # Device-specific wrappers (implemented in NiDAQ)
            self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
            raw = self._readAnalogInput(samplesPerSec, recordingTime)

        # Process raw 2×N array -> six-tuple
        return self._squareWaveProcessor(raw, samplesPerSec, amplitude,sweep_clip=None)
      
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

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Read analog input data from the device.
        Subclasses must implement this method.
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

    def getDataFromVoltageProtocol(self, *args, **kwargs):
        """Run the voltage-step protocol (V-clamp).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    def getVoltageClampSweep(self, *args, **kwargs):
        """Run the voltage-step protocol (V-clamp).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    def getLeakSubtraction(self, *args, **kwargs):
        """Run a P/4 leak subtraction sequence.

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

        Setting *store=False* makes the routine **side-effect–free**,
        letting protocol methods generate *temporary* buffers without
        clobbering the continuous-acquisition one.

        If *store=True* (default) the buffer is cached exactly as before
        on the instance (`self.wave`, `_wave_rate`, `_wave_samples`) so
        all existing callers continue to work unchanged.

        Args:
            wave_freq: Frequency of the square wave in Hz.
            samplesPerSec: Sampling rate in samples per second.
            dutyCycle: Fraction of each period the signal is "on" (0 to 1).
            amplitude: Amplitude of the square wave.
            recordingTime: Total duration of the waveform in seconds.
            pre_pad_ms: Initial delay before waveform onset in milliseconds.
            store: If True, cache the generated waveform on the instance.

        Returns:
            Tuple[np.ndarray, int, int]: Generated waveform, sampling rate,
            and number of samples.
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
        
        Args:
            wave_freq: Frequency of the square wave in Hz.
            samplesPerSec: Sampling rate in samples per second.
            dutyCycle: Duty cycle of the waveform (0 to 1).
            amplitude: Amplitude of the test segment.
            recordingTime: Total duration of the waveform in seconds.
            store: If True, cache the generated waveform on the instance.

        Returns:
            Tuple[np.ndarray, int, int]: Concatenated waveform, sampling rate,
            and total number of samples.
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

        Args:
            wave_freq: Frequency of test and calibration pulses in Hz.
            samplesPerSec: Sampling rate in samples per second.
            dutyCycle: Duty cycle of each segment (0 to 1).
            amplitude: Amplitude of the test pulse in volts.
            recordingTime: Duration of each segment in seconds.
        
        Returns:
            tuple[np.ndarray, int, int]:
                wave (Volts)
                rate (samples / sec)
                numSamples (wave.size)
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

        Args:
            raw_data: Raw input data array containing read and response channels.
            samplesPerSec: Sampling rate in samples per second.
            amplitude: Command amplitude used for the waveform.
            sweep_clip: Optional tuple specifying indices for clipping the sweep.

        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float],
                Optional[float], Optional[float]]:
            Processed response data, command data, total resistance,
            membrane resistance, access resistance, and membrane capacitance.
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

        Returns:
            Optional[float]: Latest membrane capacitance, or None if unavailable.
        """
        return self.latestMembraneCapacitance

    def resistance(self):
        """
        Returns the latest total resistance measurement.
        If no measurement is available, returns None.

        Returns:
            Optional[float]: Latest total resistance, or None if unavailable.
        """
        return self.totalResistance
    
    def accessResistance(self):
        """
        Returns the latest access resistance measurement.
        If no measurement is available, returns None.

        Returns:
            Optional[float]: Latest access resistance, or None if unavailable.
        """
        return self.latestAccessResistance

    def _getResistancefromCurrent(self, data, cmdVoltage) -> float | None:
        """
        Estimate resistance from current response data.

        Args:
            data: Current response data array.
            cmdVoltage: Applied command voltage.

        Returns:
            Optional[float]: Estimated resistance, or None if computation fails.
        """
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
        
        Args:
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
        """
        Evaluate a mono-exponential decay function.

        Args:
            x: Independent variable (typically time).
            m: Amplitude coefficient.
            t: Decay rate (inverse time constant, 1/τ).
            b: Baseline offset.

        Returns:
            np.ndarray: Computed exponential values.
        """
        return m * np.exp(-t * x) + b


    def optimizer(self, fit_data, I_peak_pA, I_peak_time_ms, I_ss_pA):
        """
        Robust mono-exponential fit:

            I(t_ms) = m · exp(−t · t_ms) + b

        with parameter
            t = 1 / τ      [1 / ms]

        Args:
            fit_data: Dictionary containing:
                - 'T_ms': Time values in milliseconds.
                - 'Y_pA': Current values in picoamps.
            I_peak_pA: Peak current value in picoamps.
            I_peak_time_ms: Time at peak current in milliseconds.
            I_ss_pA: Steady-state current in picoamps.

    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]:
            (m, t, b) in pA, 1/ms, pA 
            or (None, None, None) on failure
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

        Args:
            readData: Command signal array (voltage).
            respData: Response signal array (current).
            timeData: Time array in seconds.
            amplitude: Command amplitude in volts.

        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]:
            Access resistance (MΩ), membrane resistance (MΩ),
            and membrane capacitance (pF). Returns None values if computation fails.

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
        
        Args:
            tau: Time constant in milliseconds.
            mean_voltage: Mean command voltage in millivolts.
            I_peak: Peak current in picoamps.
            I_prev: Baseline current before stimulus in picoamps.
            I_ss: Steady-state current in picoamps.

        Returns:
            Tuple[float, float, float]:
            Access resistance (MΩ), membrane resistance (MΩ),
            and membrane capacitance (pF).
        """
        I_d = I_peak - I_prev   # in pA
        I_dss = I_ss - I_prev   # in pA
        R_a_MOhms = ((mean_voltage * 1e-3) / (I_d * 1e-12)) * 1e-6
        R_m_MOhms = (((mean_voltage * 1e-3) - R_a_MOhms * 1e6 * I_dss * 1e-12) / (I_dss * 1e-12)) * 1e-6
        C_m_pF = (tau * 1e-3) / (1 / (1 / (R_a_MOhms * 1e6) + 1 / (R_m_MOhms * 1e6))) * 1e12
        return R_a_MOhms, R_m_MOhms, C_m_pF

# ========================================================
#  NI-DAQ Subclass
# ========================================================
class NiDAQ(DAQ):
    """
    National Instruments DAQ implementation for patch-clamp acquisition.

    Provides hardware-specific implementations for analog input/output,
    continuous acquisition, and protocol execution using NI-DAQmx.
    """
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA per V (DAQ output)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3)  # 10 mV per V (DAQ input)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3)  # 20 mV per V (DAQ output)
    V_CLAMP_VOLT_PER_AMP = (2 * 1e-9)   # 2 mV per pA (DAQ input)

    def __init__(self, readDev, readChannel, cmdDev, cmdChannel, respDev, respChannel):
        """
        Initialize the NI-DAQ device, configure acquisition tasks, and start background acquisition.

        args:
            readDev (str): Device name for reading channel (e.g., "Dev1").
            readChannel (str): Analog input channel for command readback.
            cmdDev (str): Device name for command output.
            cmdChannel (str): Analog output channel for command signal.
            respDev (str): Device name for response input.
            respChannel (str): Analog input channel for measured response.
        """
        super().__init__()
        self.readDev = readDev
        self.cmdDev = cmdDev
        self.respDev = respDev
        self.readChannel = readChannel
        self.cmdChannel = cmdChannel
        self.respChannel = respChannel
        self.latestAccessResistance = None
        self.totalResistance = None
        self.latestMembraneResistance = None
        self.latestMembraneCapacitance = None
        self.cellMode = False
        self.info(f'Using {self.readDev}/{self.readChannel} for reading; '
                  f'{self.cmdDev}/{self.cmdChannel} for command; '
                  f'and {self.respDev}/{self.respChannel} for response.')
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

    def _restartAcquisition(self):
        """
        Restart analog input/output tasks after an error or buffer overrun.

        Recreates tasks and reinitializes synchronization between AI and AO.
        """
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

        ▸ AI owns its own sample clock (CONTINUOUS).
        ▸ AO runs at the same rate, in regeneration mode.
        ▸ AO is armed with a digital-edge start-trigger that listens to
            the AI StartTrigger ("/<Dev>/ai/StartTrigger").
        ▸ Tasks are started in LabVIEW order:  AO first (armed) → AI starts
            (emits trigger) → AO begins automatically.
        """
        import nidaqmx
        import nidaqmx.constants as c

        # ------------------------------------------------------------------
        # 1.  ANALOG INPUT  (read + response channels)
        # ------------------------------------------------------------------
        self.ai_task = nidaqmx.Task()
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.readDev}/{self.readChannel}",
            terminal_config=c.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0,
        )
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.respDev}/{self.respChannel}",
            terminal_config=c.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0,
        )
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=self._wave_rate,
            sample_mode=c.AcquisitionType.CONTINUOUS,
        )
        import nidaqmx.constants as c

        # Allow the device to overwrite unread data so the fifo keeps running
        self.ai_task.in_stream.overwrite_mode = c.OverwriteMode.OVERWRITE_UNREAD_SAMPLES

        # When we call task.read(), start relative to the newest sample
        self.ai_task.in_stream.read_relative_to = c.ReadRelativeTo.MOST_RECENT_SAMPLE
        # …and back-up exactly one buffer-length so we still get a full cycle
        self.ai_task.in_stream.read_offset = -self._wave_samples

        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size,
            self._wave_rate * 10)   # keep 10 s of slack (~200 k samples)
                # LabVIEW “Get Terminal with Device Prefix”  →  "/<device>/ai/StartTrigger"
        # On cDAQ the routable line lives on the CHASSIS, not the module.
        if "Mod" in self.readDev:                 # e.g.  "cDAQ1Mod1"
            chassis_name = self.readDev.split("Mod")[0]   # "cDAQ1"
            start_trig_term = f"/{chassis_name}/ai/StartTrigger"
        else:                                     # PCIe/X-series, USB -- use the device itself
            start_trig_term = f"/{self.readDev}/ai/StartTrigger"


        # ------------------------------------------------------------------
        # 2.  ANALOG OUTPUT  (command channel)
        # ------------------------------------------------------------------
        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(f"{self.cmdDev}/{self.cmdChannel}")
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=self._wave_rate,
            sample_mode=c.AcquisitionType.CONTINUOUS,
            samps_per_chan=self._wave_samples,
        )
        # Regeneration so a single-period buffer loops forever
        self.ao_task.out_stream.regen_mode = c.RegenerationMode.ALLOW_REGENERATION

        # Arm AO with the AI StartTrigger
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            start_trig_term, trigger_edge=c.Edge.RISING
        )

        # ------------------------------------------------------------------
        # 3.  PRIME AO FIFO  (avoid -200462)
        # ------------------------------------------------------------------
        self.ao_task.write(self.wave, auto_start=False)

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

    def _setupAcquisitionCurrent(self, recordingTime=500, dutyCycle=0.5):
        """
        Configure a dedicated continuous AI + AO pair for the
        current-step protocol.  The background stream must be paused first.
        
        args:
            recordingTime (float): Duration of each segment in milliseconds.
            dutyCycle (float): Duty cycle of generated waveform.
        """
        import nidaqmx, nidaqmx.constants as c

        samplesPerSec = 100_000
        dutyCycle     = float(dutyCycle)
        recordingTime = (float(recordingTime) * 1e-3)  # convert ms to seconds
        wave_freq     = 1.0 / (recordingTime)  # one period per segment

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
        self.ai_task = nidaqmx.Task()
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.readDev}/{self.readChannel}",
            terminal_config=c.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.respDev}/{self.respChannel}",
            terminal_config=c.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=c.AcquisitionType.CONTINUOUS)
        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size, rate * 10)

        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(f"{self.cmdDev}/{self.cmdChannel}")
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=c.AcquisitionType.CONTINUOUS,
            samps_per_chan=numSamples)
        self.ao_task.out_stream.regen_mode = c.RegenerationMode.ALLOW_REGENERATION

        start_trig_term = (f"/{self.readDev.split('Mod')[0] if 'Mod' in self.readDev else self.readDev}"
                           "/ai/StartTrigger")
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            start_trig_term, trigger_edge=c.Edge.RISING)

        # prime AO FIFO, then start in LabVIEW order (AO-armed → AI-fires)
        self.ao_task.write(wave, auto_start=False)
        self.ao_task.start()
        self.ai_task.start()

    def _build_optogenetic_timeline(self, protocol_steps: list[dict]):
        """
        Convert protocol step definitions into a time-aligned stimulation timeline.

        args:
            protocol_steps (list[dict]): List of step dictionaries containing duration and parameters.

        returns:
            tuple:
                - list[dict]: Timeline with start and end times for each step.
                - float: Total protocol duration in seconds.
                - Any: Protocol type identifier.

        raises:
            ValueError: If no valid steps with positive duration are provided.
        """
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
        
        args:
            deadline_s (float): Absolute time (perf_counter) to wait until.
            tight_timing (bool): If True, uses spin-waiting for final milliseconds
                to reduce timing jitter.
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

    def _run_optogenetic_timeline(
        self,
        laser,
        stim_timeline: list[dict],
        *,
        t0: float | None = None,
        tight_timing: bool = True,
    ) -> list[dict]:
        """
        Execute an optogenetic stimulation timeline using a laser device.

        args:
            laser: Laser control object with wavelength, power, and on/off methods.
            stim_timeline (list[dict]): Timeline entries with timing and stimulation parameters.
            t0 (float | None): Reference start time. If None, current time is used.
            tight_timing (bool): Whether to use high-precision timing.

        returns:
            list[dict]: Actual executed timeline with recorded start and end times.

        raises:
            ValueError: If no laser device is provided.
        """
        if laser is None:
            raise ValueError("laser is required")
        if t0 is None:
            t0 = time.perf_counter()
        actual = []
        for step in stim_timeline:
            start_s = float(step.get("start_s", 0.0))
            end_s = float(step.get("end_s", start_s))
            if end_s <= start_s:
                continue
            self._wait_until(t0 + start_s, tight_timing=tight_timing)
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
                laser.power_on()
            else:
                laser.power_off()
            self._wait_until(t0 + end_s, tight_timing=tight_timing)
            step_end = time.perf_counter() - t0

            entry = dict(step)
            entry["state"] = state
            if power_percent is not None:
                entry["power_percent"] = power_percent
            entry["start_s"] = step_start
            entry["end_s"] = step_end
            actual.append(entry)
        return actual

    def _setup_optogenetic_ai_task(self, rate_hz: int, num_samples: int):
        """
        args:
            rate_hz (int): Sampling rate in Hz.
            num_samples (int): Number of samples to acquire per channel.

        returns:
            nidaqmx.Task: Configured analog input task.
        """
        ai = nidaqmx.Task()
        ai.ai_channels.add_ai_voltage_chan(
            f"{self.readDev}/{self.readChannel}",
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0)
        ai.ai_channels.add_ai_voltage_chan(
            f"{self.respDev}/{self.respChannel}",
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0)
        ai.timing.cfg_samp_clk_timing(
            rate=rate_hz,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=num_samples)
        return ai

    def _read_optogenetic_ai(self, ai_task, num_samples: int, duration_s: float):
        """
        Read data from an optogenetic AI task and close it.

        args:
            ai_task: Configured NI-DAQmx analog input task.
            num_samples (int): Number of samples to read per channel.
            duration_s (float): Expected acquisition duration in seconds.

        returns:
            np.ndarray: Acquired data as a NumPy array.

        raises:
            nidaqmx.errors.DaqError: If the read operation fails.
        """
        raw = ai_task.read(
            number_of_samples_per_channel=num_samples,
            timeout=duration_s + 2.0)
        ai_task.stop()
        ai_task.close()
        return np.asarray(raw, dtype=float)

    def _setupAcquisitionVoltage(self,
                                 *,
                                 wave_freq: float,
                                 samplesPerSec: int,
                                 dutyCycle: float,
                                 recordingTime: float,
                                 exp_samples: int,
                                 buffer_trains: int):
        """
        Configure the dedicated AI/AO pair for the V-clamp step protocol.
        Mirrors the current-clamp setup while preserving voltage-specific buffer sizing.
        
        args:
            wave_freq (float): Frequency of the square wave.
            samplesPerSec (int): Sampling rate in samples per second.
            dutyCycle (float): Duty cycle of the waveform.
            recordingTime (float): Duration of each waveform segment in seconds.
            exp_samples (int): Expected number of samples per acquisition.
            buffer_trains (int): Number of waveform buffers to allocate.
        """
        import nidaqmx
        import nidaqmx.constants as c

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
        self.ai_task = nidaqmx.Task()
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.readDev}/{self.readChannel}",
            terminal_config=c.TerminalConfiguration.DIFF,
            min_val=-10.0,
            max_val=10.0,
        )
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.respDev}/{self.respChannel}",
            terminal_config=c.TerminalConfiguration.DIFF,
            min_val=-10.0,
            max_val=10.0,
        )
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=samplesPerSec,
            sample_mode=c.AcquisitionType.CONTINUOUS,
        )
        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size,
            exp_samples * buffer_trains,
        )

        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(f"{self.cmdDev}/{self.cmdChannel}")
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
        start_trig_term = (
            f"/{self.readDev.split('Mod')[0] if 'Mod' in self.readDev else self.readDev}"
            "/ai/StartTrigger"
        )
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            start_trig_term,
            trigger_edge=c.Edge.RISING,
        )

        # ------------------------------------------------------------------
        # 4. prime AO FIFO and start in LabVIEW order
        # ------------------------------------------------------------------
        self.ao_task.write(zero_wave, auto_start=False)
        self.ao_task.start()
        self.ai_task.start()

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Wrapper for the AI task: read exactly one bufferful.
        
        args:
            samplesPerSec (int): Sampling rate in samples per second.
            recordingTime (float): Duration of acquisition in seconds.

        returns:
            np.ndarray: Acquired analog input data.
        """
        numSamples = int(samplesPerSec * recordingTime)
        data = self.ai_task.read(
            number_of_samples_per_channel=numSamples,
            timeout=10.0
        )
        return np.array(data, dtype=float)

    def _sendSquareWave(self, wave_freq, samplesPerSec,
                        dutyCycle, amplitude, recordingTime):
        """
        Update the AO buffer *without* disturbing the trigger relationship.

        • If the requested wave is identical to the one already streaming,
        do nothing (prevents DAQmx -200015).
        • If the length / rate is unchanged but the shape is different,
        rewrite the buffer **in-place** while the task is running
        (regen-mode allows this).
        • If rate or length must change, stop *both* tasks, re-configure,
        and restart them **in the same trigger order** (AO-first, AI-second).
        
        args:
            wave_freq (float): Frequency of the square wave.
            samplesPerSec (int): Sampling rate.
            dutyCycle (float): Duty cycle of the waveform.
            amplitude (float): Amplitude of the waveform.
            recordingTime (float): Duration of waveform.
        """
        import nidaqmx.constants as c

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
            self.ao_task.write(self.wave, auto_start=False)

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
            self.ao_task.write(self.wave, auto_start=False)

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
        Generate a new current-square-wave buffer on the fly,
        write it to the always-running AO task, then restore the
        original membrane-test wave.
        
        args:
            wave_freq (float): Frequency of the waveform.
            samplesPerSec (int): Sampling rate.
            dutyCycle (float): Duty cycle.
            amplitude (float): Current amplitude (converted internally).
            recordingTime (float): Duration of waveform.
        """

        import nidaqmx.constants as c
        # 1) create the current square wave
        wave, rate, numSamples = self.createSquareWaveCurrent(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=amplitude,
            recordingTime=recordingTime
        )   

        # 2) write the new wave to the AO task
        self.ao_task.write(wave, auto_start=False)

    def _sendSquareWaveVoltage(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Generate the next voltage-step wave and queue it on the running AO task.
        
        args:
            wave_freq (float): Frequency of the waveform.
            samplesPerSec (int): Sampling rate.
            dutyCycle (float): Duty cycle.
            amplitude (float): Voltage amplitude.
            recordingTime (float): Duration of waveform.
        """
        wave, _, _ = self.createSquareWaveVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=amplitude,
            recordingTime=recordingTime,
            store=False,
        )
        self.ao_task.write(wave, auto_start=False)

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

    def _setup_optogenetic_ai_task(self, rate_hz: int, num_samples: int):
        ai = nidaqmx.Task()
        ai.ai_channels.add_ai_voltage_chan(
            f"{self.readDev}/{self.readChannel}",
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0)
        ai.ai_channels.add_ai_voltage_chan(
            f"{self.respDev}/{self.respChannel}",
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0)
        ai.timing.cfg_samp_clk_timing(
            rate=rate_hz,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=num_samples)
        return ai

    def _read_optogenetic_ai(self, ai_task, num_samples: int, duration_s: float):
        raw = ai_task.read(
            number_of_samples_per_channel=num_samples,
            timeout=duration_s + 2.0)
        ai_task.stop()
        ai_task.close()
        return np.asarray(raw, dtype=float)

    def getLeakSubtraction(
            self,
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
        Execute a P/4 leak subtraction sequence prior to a voltage sweep.

        For each sweep target, this generates ``repeats`` square pulses whose
        amplitude is ``-(target - holding) / 4``.  Each pulse uses a 50 % duty
        cycle square wave lasting ``pulse_recording_time / 3`` seconds inside a
        baseline–pulse–baseline layout that mirrors ``getVoltageClampSweep``.
        Averaged traces are stored on ``self.leak_subtraction_data`` for GUI /
        logging consumers and the raw repeats are discarded after averaging.
        
        args:
            start_voltage (float): Starting voltage of sweep.
            end_voltage (float): Ending voltage of sweep.
            holding_voltage (float): Holding voltage.
            step_voltage (float): Voltage increment between targets.
            wave_freq (float): Square wave frequency.
            samplesPerSec (int): Sampling rate.
            dutyCycle (float): Duty cycle of waveform.
            pulse_recording_time (float): Duration of each pulse segment.
            repeats (int): Number of repeats per voltage step.

        returns:
            list[dict]: Averaged leak-subtracted responses for each target voltage.

        raises:
            ValueError: If step_voltage is zero or repeats is non-positive.
            RuntimeError: If voltage/current conversion constants are not configured.

        """
        import numpy as np

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

        if not leak_plan:
            self.leak_subtraction_data = []
            self.leak_subtraction_meta = {
                "holding_voltage": holding_voltage,
                "targets": targets,
                "leak_step_values": np.array(leak_step_values, dtype=float),
                "repeats": repeats,
                "wave_freq": wave_freq,
                "pulse_recording_time": pulse_recording_time,
            }
            return self.leak_subtraction_data

        self.pause_acquisition()
        for task_name in ("ai_task", "ao_task"):
            task = getattr(self, task_name, None)
            if task is not None:
                try:
                    task.stop()
                except Exception:
                    pass
                try:
                    task.close()
                except Exception:
                    pass

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

        self._setupAcquisitionVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            recordingTime=pulse_recording_time,
            exp_samples=exp_samples,
            buffer_trains=buffer_trains,
        )

        if leak_plan:
            first_amp = leak_plan[0]["ao_amplitude"]
            self._sendSquareWaveVoltage(
                wave_freq=wave_freq,
                samplesPerSec=samplesPerSec,
                dutyCycle=dutyCycle,
                amplitude=first_amp,
                recordingTime=pulse_recording_time,
            )

        while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
            self.sleep(0.002)
        _ = self.ai_task.read(exp_samples, timeout=2.0)

        pulses_by_target: list[list[dict]] = [[] for _ in range(targets.size)]

        try:
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

                while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
                    self.sleep(0.002)

                raw = np.asarray(self.ai_task.read(exp_samples, timeout=2.0), dtype=float)
                self._squareWaveProcessor(raw, samplesPerSec, pulse["ao_amplitude"])

                time_vec = np.linspace(0, total_duration, exp_samples, dtype=float)
                resp_vec = raw[1] * self.V_CLAMP_VOLT_PER_AMP
                command_vec = raw[0] * self.V_CLAMP_VOLT_PER_VOLT

                pulses_by_target[pulse["target_index"]].append({
                    "time": time_vec,
                    "response": resp_vec,
                    "command": command_vec,
                    "repeat_index": pulse["repeat_index"],
                })

                self.info(
                    f"Captured P/4 leak pulse {pulse['repeat_index'] + 1}/{repeats} "
                    f"for {pulse['target_voltage'] * 1e3:.0f} mV sweep target."
                )
        finally:
            try:
                self.ai_task.stop(); self.ao_task.stop()
                self.ai_task.close(); self.ao_task.close()
            except Exception:
                pass
            self.resume_acquisition()

        leak_results: list[dict] = []
        for idx, target in enumerate(targets):
            pulses = pulses_by_target[idx]
            if not pulses:
                continue

            time_vec = pulses[0]["time"].copy()
            resp_stack = np.vstack([p["response"] for p in pulses])
            cmd_stack = np.vstack([p["command"] for p in pulses])

            avg_resp = resp_stack.mean(axis=0)
            avg_cmd = cmd_stack.mean(axis=0)

            leak_results.append({
                "time": time_vec,
                "response": avg_resp,
                "command": avg_cmd,
                "target_voltage": target,
                "leak_target_voltage": leak_target_voltages[idx],
                "leak_step_voltage": leak_step_values[idx],
                "repeat_count": len(pulses),
            })

        self.leak_subtraction_data = leak_results
        self.leak_subtraction_meta = {
            "holding_voltage": holding_voltage,
            "targets": targets,
            "leak_target_voltages": np.array(leak_target_voltages, dtype=float),
            "leak_step_values": np.array(leak_step_values, dtype=float),
            "repeats": repeats,
            "wave_freq": wave_freq,
            "pulse_recording_time": pulse_recording_time,
        }

        return self.leak_subtraction_data

    def getVoltageClampSweep(
            self,
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
        Run a finite V-clamp sweep that mirrors the I-clamp protocol lifecycle.
        Returns a list of [time, response, command] arrays processed identically
        to the membrane-test traces.
        
        args:
            start_voltage (float): Starting voltage of the sweep.
            end_voltage (float): Ending voltage of the sweep.
            holding_voltage (float): Holding voltage.
            step_voltage (float): Increment between voltage steps.
            wave_freq (int): Frequency of square wave.
            samplesPerSec (int): Sampling rate.
            dutyCycle (float): Duty cycle of waveform.
            recordingTime (float): Duration per step.

        returns:
            list[list[np.ndarray]]: List of sweeps, each containing
                [time_array, response_array, command_array].

        raises:
            ValueError: If step_voltage is zero.
        """
        import numpy as np

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

        self.vclamp_steps = targets
        self.info(f"V-clamp pulses (pA): {self.vclamp_steps}")
        self.vclamp_hold_value = holding_voltage
        self.voltage_protocol_data = []

        self.pause_acquisition()
        for task_name in ("ai_task", "ao_task"):
            task = getattr(self, task_name, None)
            if task is not None:
                try:
                    task.stop()
                except Exception:
                    pass
                try:
                    task.close()
                except Exception:
                    pass
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
        self._setupAcquisitionVoltage(
            wave_freq=wave_freq,
            samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            recordingTime=recordingTime,
            exp_samples=exp_samples,
            buffer_trains=buffer_trains,
        )

        if targets.size:
            first_delta = targets[0] - holding_voltage
            first_amp = first_delta / self.V_CLAMP_VOLT_PER_VOLT
            self._sendSquareWaveVoltage(
                wave_freq, samplesPerSec, dutyCycle, first_amp, recordingTime
            )


        while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
            self.sleep(0.002)
        _ = self.ai_task.read(exp_samples, timeout=2.0)

        sweeps: list[list[np.ndarray]] = []

        try:
            for idx, target in enumerate(targets):
                if idx + 1 < len(targets):
                    next_delta = targets[idx + 1] - holding_voltage
                    next_amp = next_delta / self.V_CLAMP_VOLT_PER_VOLT
                    self._sendSquareWaveVoltage(
                        wave_freq, samplesPerSec, dutyCycle, next_amp, recordingTime
                    )
                    self.info(f"Waiting for {target*1000:.0f} mV train…")

                while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
                    self.sleep(0.002)

                raw = np.asarray(self.ai_task.read(exp_samples, timeout=2.0), dtype=float)
                step_amp_volts = (target - holding_voltage) / self.V_CLAMP_VOLT_PER_VOLT
                self._squareWaveProcessor(raw, samplesPerSec, step_amp_volts)

                time_vec = np.linspace(0, total_duration, exp_samples, dtype=float)
                resp_vec = raw[1] * self.V_CLAMP_VOLT_PER_AMP
                command_vec = raw[0] * self.V_CLAMP_VOLT_PER_VOLT
                sweeps.append([time_vec, resp_vec, command_vec])
                self.info(f"Captured {target*1000:.0f} mV pulse")

        finally:
            try:
                self.ai_task.stop(); self.ao_task.stop()
                self.ai_task.close(); self.ao_task.close()
            except Exception:
                pass
            self.resume_acquisition()

        self.voltage_protocol_data = sweeps
        if targets.size and self.voltage_membrane_test is not None:
            self.voltage_membrane_test["sweep_hold_mV"] = holding_voltage * 1e3

        return self.voltage_protocol_data

    def getDataFromCurrentProtocol(self,
                                   custom: bool = False,
                                   factor: float | None = None,
                                   startCurrentPicoAmp: float | None = None,
                                   endCurrentPicoAmp: float | None = None,
                                   stepCurrentPicoAmp: float = 10,
                                   recordingTimeMs: float = 500,
                                   dutyCycle: float = 0.5):
        """
        Run an entire I-clamp step protocol on one continuous stream.
            
        args:
            custom (bool):
                If False, generates pulses normalized by membrane capacitance.
                If True, uses manually provided current range parameters.
            factor (float | None):
                Reserved for future scaling logic; currently unused.
            startCurrentPicoAmp (float | None):
                Starting current (pA) for custom mode.
            endCurrentPicoAmp (float | None):
                Ending current (pA) for custom mode.
            stepCurrentPicoAmp (float):
                Step size (pA) between pulses.
            recordingTimeMs (float):
                Duration of one pulse segment (ms).
            dutyCycle (float):
                Fraction of the waveform period spent in the high state.

        returns:
            tuple:
                current_protocol_data (list[list[np.ndarray]]):
                    List of traces, each containing [time, response, read].
                pulses (np.ndarray):
                    Array of applied current steps (pA).
                pulseRange (int):
                    Number of pulses applied.

        raises:
            RuntimeError:
                If capacitance-based normalization is required but unavailable or invalid.
        """
        # ───────── constants ─────────
        samplesPerSec  = 100_000
        dutyCycle      = float(dutyCycle)
        recordingTime  = recordingTimeMs * 1e-3   # 0.25 s / segment
        wave_freq      = 1.0 / recordingTime      # 4 Hz
        fullRecTime    = 4 * recordingTime        # 1.0 s / train
        exp_samples    = int(samplesPerSec * fullRecTime)

        # ─ 1. build pulse list ───────
        if not custom:
            # Ensure we have a sensible capacitance measurement
            if (self.voltageMembraneCapacitance is None or
                not np.isfinite(self.voltageMembraneCapacitance) or
                self.voltageMembraneCapacitance <= 0):
                raise RuntimeError("Bad capacitance measurement; aborting I-clamp protocol.")

            # Convert to absolute pF, round *down* to the nearest whole pF
            cap_pf = int(abs(self.voltageMembraneCapacitance))

            # Normalised stimulus: −2 pA/pF … +10 pA/pF with 1 pA/pF increments
            startCurrentPicoAmp = -2 * cap_pf
            endCurrentPicoAmp   = 10 * cap_pf
            stepCurrentPicoAmp  = cap_pf

        # Keep legacy manual mode untouched when custom=True

        startCurrentPicoAmp = max(startCurrentPicoAmp, -20000)
        endCurrentPicoAmp   = min(endCurrentPicoAmp,   20000)

        pulses = np.arange(startCurrentPicoAmp,
                           endCurrentPicoAmp + stepCurrentPicoAmp,
                           stepCurrentPicoAmp, dtype=float)
        if 0.0 not in pulses:
            pulses = np.sort(np.append(pulses, 0.0))

        self.pulses       = pulses
        self.pulseRange   = len(pulses)
        self.current_protocol_data = []
        self.info(f"I-clamp pulses (pA): {pulses}")

        # ─ 2. park background stream ─
        self.pause_acquisition()
        for t in ("ai_task", "ao_task"):
            try:
                getattr(self, t).stop(); getattr(self, t).close()
            except Exception:
                pass

        # ─ 3. start dedicated tasks (prime buffer = 0 pA test) ──────────
        self._setupAcquisitionCurrent(recordingTime * 1e3, dutyCycle=dutyCycle)   # expects ms

        # a) queue FIRST real pulse so it will become train #2
        next_amp_V = (pulses[0] * 1e-12) / self.C_CLAMP_AMP_PER_VOLT
        self._sendSquareWaveCurrent(wave_freq, samplesPerSec,
                                    dutyCycle, next_amp_V, recordingTime)

        # b) wait 1 s and discard train #1 (0 pA prime)
        while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
            self.sleep(0.002)
        _ = self.ai_task.read(exp_samples, timeout=2.0)

        try:
            for idx, pulse in enumerate(pulses):
                # ─── queue the NEXT pulse *before* waiting ───────────────
                if idx + 1 < len(pulses):
                    future_amp_V = (pulses[idx + 1] * 1e-12) \
                                    / self.C_CLAMP_AMP_PER_VOLT
                    self._sendSquareWaveCurrent(wave_freq, samplesPerSec,
                                                dutyCycle, future_amp_V,
                                                recordingTime)

                self.info(f"Waiting for {pulse:.0f} pA train…")

                # ─── wait until exactly one full train is ready ─────────
                while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
                    self.sleep(0.002)

                # ─── read it (matches current pulse) ────────────────────
                raw = np.asarray(self.ai_task.read(exp_samples, timeout=2.0),
                                 dtype=float)
                resp = raw[1] / self.C_CLAMP_VOLT_PER_VOLT
                read = raw[0]
                tvec = np.linspace(0, fullRecTime, exp_samples, dtype=float)
                self.current_protocol_data.append([tvec, resp, read])

                self.info(f"Captured {pulse:.0f} pA pulse")

        finally:
            # ─ 4. restore background stream ────────────────────────────
            try:
                self.ai_task.stop();  self.ao_task.stop()
                self.ai_task.close(); self.ao_task.close()
            except Exception:
                pass
            self.resume_acquisition()

        return self.current_protocol_data, self.pulses, self.pulseRange

    def getDataFromHoldingProtocol(self, *, rate_hz: int = 50_000, duration_s: float = 60.0):
        """
        OLD-STYLE baseline read-out (no command output).

        • Suspends the membrane-test thread.
        • Builds one finite AI-only task at 50 kS/s for 60 s
          (same numbers as the legacy code).
        • Does *not* touch the AO task, avoiding the –200547 write error.
        • Restarts the continuous stream afterwards.

        args:
            rate_hz (int):
                Sampling rate in Hz.
            duration_s (float):
                Duration of acquisition in seconds.
        
        returns:
            np.ndarray:
                [time_s, resp_V, read_V]  – identical to the legacy layout.
        """
        import nidaqmx
        import nidaqmx.constants as c
        import numpy as np

        # -------------------------------------------------- 1. park background
        self.pause_acquisition()

        try:
            num_samples = int(rate_hz * duration_s)

            # ---------------------------------------------- 2. finite AI task
            ai = nidaqmx.Task()
            ai.ai_channels.add_ai_voltage_chan(
                f"{self.readDev}/{self.readChannel}",
                terminal_config=c.TerminalConfiguration.DIFF,
                min_val=-10.0, max_val=10.0)
            ai.ai_channels.add_ai_voltage_chan(
                f"{self.respDev}/{self.respChannel}",
                terminal_config=c.TerminalConfiguration.DIFF,
                min_val=-10.0, max_val=10.0)
            ai.timing.cfg_samp_clk_timing(
                rate=rate_hz,
                sample_mode=c.AcquisitionType.FINITE,
                samps_per_chan=num_samples)

            ai.start()
            raw = ai.read(
                number_of_samples_per_channel=num_samples,
                timeout=duration_s + 2.0)
            ai.stop(); ai.close()

            raw = np.array(raw, dtype=float)
            # Keep unit conversions consistent with other V-clamp protocol paths.
            resp = raw[1] * self.V_CLAMP_VOLT_PER_AMP
            read = raw[0] * self.V_CLAMP_VOLT_PER_VOLT
            t = np.linspace(0, duration_s, num_samples, dtype=float)

            self.holding_protocol_data = np.array([t, resp, read])
            return self.holding_protocol_data

        finally:
            # ---------------------------------------------- 3. resume stream
            self.resume_acquisition()

    def getDataFromOptogeneticProtocol(
        self,
        *,
        laser,
        protocol_steps: list[dict],
        rate_hz: int = 50_000,
    ):
        """
        Modified holding protocol that records only during optogenetic "on" steps.

        Returns
        -------
        (list[list[np.ndarray]], list[dict])
            One trace per on-step plus the stimulation timeline.

        args:
            laser:
                Laser control object with wavelength, power, and on/off methods.
            protocol_steps (list[dict]):
                Sequence of stimulation steps with duration, state, wavelength,
                and power settings.
            rate_hz (int):
                Sampling rate in Hz for acquisition.

        returns:
            tuple:
                optogenetic_protocol_data (list[list[np.ndarray]]):
                    Recorded traces for each "on" step [time, response, read].
                optogenetic_stim_data (list[dict]):
                    Timeline of executed stimulation steps with actual timing.
            One trace per on-step plus the stimulation timeline.

        raises:
            ValueError:
                If laser is not provided or protocol_steps is empty.
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
            aligned_traces: list[list[np.ndarray]] = []
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

                    ai = self._setup_optogenetic_ai_task(rate_hz, num_samples)
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

                    raw = self._read_optogenetic_ai(ai, num_samples, duration_s)
                    ai = None

                    resp = raw[1] * self.V_CLAMP_VOLT_PER_AMP
                    read = raw[0] * self.V_CLAMP_VOLT_PER_VOLT
                    t = np.linspace(0, duration_s, num_samples, dtype=float)

                    aligned_traces.append([t, resp, read])
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
            if not aligned_traces:
                aligned_traces = []
                aligned_pulses = []

            self.optogenetic_protocol_data = aligned_traces
            self.optogenetic_stim_data = stim_used
            if protocol_type is None and self.optogenetic_stim_data:
                protocol_type = self.optogenetic_stim_data[0].get("protocol_type")
            self.optogenetic_protocol_type = protocol_type
            self.optogenetic_pulses = aligned_pulses
            self.optogenetic_pulseRange = len(aligned_pulses)

            protocol_key = self._normalize_optogenetic_protocol_key(protocol_type)
            entry = {
                "data": aligned_traces,
                "pulses": aligned_pulses,
                "pulse_range": len(aligned_pulses),
                "stim_data": stim_used,
                "protocol_type": protocol_type,
                "protocol_key": protocol_key,
            }
            with self._optogenetic_queue_lock:
                self.optogenetic_protocol_queue.append(entry)
            return self.optogenetic_protocol_data, self.optogenetic_stim_data

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

    def getDataFromVoltageProtocol(
            self,
            *,                       # keyword-only
            wave_freq: int = 40,
            samplesPerSec: int = 50_000,   # 50 kS/s
            dutyCycle: float = 0.5,
            amplitude: float = 0.5,
            recordingTime: float = 0.025,  # 25 ms period
            max_attempts: int = 15,
            membrane_hold: float | None = None
    ):
        """
        • Pauses the DAQAcquisitionThread (tasks keep streaming).
        • Collects **five acceptable** traces (≤ `max_attempts` cycles).
            – First acceptable trace populates the usual voltage-protocol
            attributes (`voltage_protocol_data`, Rₐ, Rₘ, …).
            – All five capacitances are averaged → `voltageMembraneCapacitance`.

        • Resumes the acquisition thread.

        args:
            wave_freq (int):
                Frequency of the square wave (Hz).
            samplesPerSec (int):
                Sampling rate in samples per second.
            dutyCycle (float):
                Fraction of period in the high state.
            amplitude (float):
                Command amplitude (V).
            recordingTime (float):
                Duration of one waveform period (s).
            max_attempts (int):
                Maximum number of acquisition attempts.
            membrane_hold (float | None):
                Holding voltage (V) for metadata annotation.

        returns:
            float | None:
                Averaged membrane capacitance (pF), or None if no valid traces found.
        """
        import math, time, numpy as np
        self.voltage_membrane_test = None


        # ─── 1. Pause the worker thread (leave AI / AO running) ─────────
        self._pause_evt.clear()
        time.sleep(recordingTime * 2)        # let current cycle pass

        cm_values   = []      # acceptable capacitances
        first_saved = False
        attempts    = 0

        try:
            while attempts < max_attempts and len(cm_values) < 5:
                attempts += 1

                # ─── 2. Grab one square-wave cycle on the live stream ──
                result = self.getDataFromSquareWave(
                    wave_freq, samplesPerSec, dutyCycle,
                    amplitude, recordingTime
                )
                (prot_data, cmd_data,
                totalR, memR, accR, memC) = result

                # ─── 3. Acceptability check ────────────────────────────
                good_cm = (memC is not None
                        and math.isfinite(memC)
                        and memC > 0
                        and memC < 250)  # pF, arbitrary upper limit
                self.info(f'cm: {memC:.2f} pF, ')

                if good_cm:
                    # store the first good trace exactly as before
                    if not first_saved:
                        (self.voltage_protocol_data,
                        self.voltage_command_data,
                        self.voltageTotalResistance,
                        self.voltageMembraneResistance,
                        self.voltageAccessResistance,
                        self.voltageMembraneCapacitance) = result
                        hold_mV = membrane_hold * 1e3 if membrane_hold is not None else None
                        self.voltage_membrane_test = {
                            "time": np.copy(prot_data[0]),
                            "response": np.copy(prot_data[1]),
                            "command": np.copy(cmd_data[1]),
                            "step_mV": float(amplitude * self.V_CLAMP_VOLT_PER_VOLT * 1e3),
                            "hold_mV": hold_mV,
                            "color": "#000000"
                        }
                        first_saved = True

                    cm_values.append(memC)
                else:
                    self.warning(f"Voltage step attempt {attempts}: "
                                "unacceptable fit -> skipped")

                # wait one extra period so traces do not overlap in FIFO
                time.sleep(recordingTime)

            if not cm_values:
                # no acceptable trace at all
                self.error("Voltage protocol: no acceptable trace found")
                self.voltageMembraneCapacitance = None
                return None

            # ─── 4. Average capacitance ──────────────────────────────
            self.voltageMembraneCapacitance = float(np.mean(cm_values))
            self.info(f"Averages: cm: {self.voltageMembraneCapacitance:.2f} pF, ")

            if len(cm_values) < 5:
                self.warning(f"Voltage protocol collected only {len(cm_values)} "
                            f"good traces (expected 5) within {attempts} attempts")

        finally:
            # ─── 5. Resume background acquisition ──────────────────────
            self._pause_evt.set()

        return self.voltageMembraneCapacitance

# ========================================================
#  ArduinoDAQ Subclass
# ========================================================
class ArduinoDAQ(DAQ):
    """
    Data acquisition interface for Arduino-based DAQ systems using serial communication.

    This implementation communicates with an Arduino over a serial port to send
    command waveforms and receive digitized analog input data. It supports basic
    square-wave stimulation and data acquisition but does not implement advanced
    protocols such as leak subtraction.
    """
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA per V (DAQ output)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3)  # 10 mV per V (DAQ input)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3)  # 20 mV per V (DAQ output)
    V_CLAMP_VOLT_PER_AMP = (2 * 1e-9)   # 2 V per A (DAQ input)

    def __init__(self, DAQSerial=None):
        """
        Initialize the ArduinoDAQ with an active serial connection.

        args:
            DAQSerial (serial.Serial | None):
                Open and configured serial connection to the Arduino device.

        raises:
            ValueError:
                If DAQSerial is not a valid, open serial.Serial instance.
        """
        if DAQSerial is not None and isinstance(DAQSerial, serial.Serial):
            self.DAQSerial = DAQSerial
            self.info(f"ArduinoDAQ initialized with serial port: {self.DAQSerial.port} at {self.DAQSerial.baudrate} baud.")
        else:
            self.DAQSerial = None
            self.error("DAQSerial must be an instance of serial.Serial and already opened.")
            raise ValueError("DAQSerial must be an instance of serial.Serial and already opened.")
        super().__init__()
        self.info(f'Using {self.DAQSerial.port} for reading and writing.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Read analog input data streamed from the Arduino over serial.

        Waits for "start" and "end" markers and parses incoming comma-separated
        values into command and response voltage arrays.

        args:
            samplesPerSec (int):
                Sampling rate in samples per second.
            recordingTime (float):
                Expected duration of the recording in seconds.

        returns:
            np.ndarray:
                Array of shape (2, N) containing:
                [command_voltage (V), response_voltage (V)].

        raises:
            RuntimeError:
                If the serial port is not initialized.
        """
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
        return np.array([readData, respData])

    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Send square-wave command parameters to the Arduino.

        Converts waveform parameters into a formatted command string and transmits
        it over serial. The Arduino is responsible for generating the waveform.

        args:
            wave_freq (float):
                Frequency of the square wave (Hz).
            samplesPerSec (int):
                Sampling rate in samples per second.
            dutyCycle (float):
                Fraction of the period spent in the high state.
            amplitude (float):
                Wave amplitude in DAQ voltage units.
            recordingTime (float):
                Duration of the waveform in seconds.

        returns:
            None

        raises:
            RuntimeError:
                If the serial port is not initialized.
        """
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

    def getLeakSubtraction(
            self,
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
        Placeholder implementation for leak subtraction on ArduinoDAQ.

        This method does not perform actual data acquisition. It computes and
        stores metadata describing the intended leak subtraction protocol.

        args:
            start_voltage (float):
                Starting voltage of the sweep (V).
            end_voltage (float):
                Ending voltage of the sweep (V).
            holding_voltage (float):
                Holding voltage (V).
            step_voltage (float):
                Voltage step size (V).
            wave_freq (float):
                Frequency of the square wave (Hz).
            samplesPerSec (int):
                Sampling rate in samples per second.
            dutyCycle (float):
                Fraction of the waveform period spent high.
            pulse_recording_time (float):
                Duration of each pulse (s).
            repeats (int):
                Number of repeats per step.

        returns:
            list:
                Empty list (no data collected).

        raises:
            ValueError:
                If step_voltage is zero or repeats is non-positive.
        """
        self.warning("Leak subtraction is not supported on ArduinoDAQ; returning empty result.")

        if step_voltage == 0:
            raise ValueError("step_voltage must be non-zero")
        if repeats <= 0:
            raise ValueError("repeats must be positive")

        span = end_voltage - start_voltage
        step = abs(step_voltage)
        if span == 0:
            targets = np.array([start_voltage], dtype=float)
        else:
            step_signed = step if span > 0 else -step
            targets = np.arange(start_voltage, end_voltage, step_signed, dtype=float)
            if targets.size == 0 or abs(targets[-1] - end_voltage) > 1e-9:
                targets = np.append(targets, end_voltage)

        leak_step_values = []
        leak_target_voltages = []
        for target in targets:
            delta = target - holding_voltage
            leak_step = -delta / 4.0
            leak_step_values.append(leak_step)
            leak_target_voltages.append(holding_voltage + leak_step)

        self.leak_subtraction_data = []
        self.leak_subtraction_meta = {
            "holding_voltage": holding_voltage,
            "targets": targets,
            "leak_target_voltages": np.array(leak_target_voltages, dtype=float),
            "leak_step_values": np.array(leak_step_values, dtype=float),
            "repeats": repeats,
            "wave_freq": float(wave_freq),
            "pulse_recording_time": float(pulse_recording_time),
        }
        return self.leak_subtraction_data

# ========================================================
#  FakeDAQ Subclass 
# ========================================================
class FakeDAQ(DAQ):
    """DAQ implementation used for development or testing without hardware."""

    # Use unity conversions so that the base class computations work out of the
    # box.  These constants map DAQ units directly to physical units.
    C_CLAMP_AMP_PER_VOLT = 1.0
    C_CLAMP_VOLT_PER_VOLT = 1.0
    V_CLAMP_VOLT_PER_VOLT = 1.0
    V_CLAMP_VOLT_PER_AMP = 1.0

    def __init__(self, pressureController: None | PressureController):
        """
        Initialize the FakeDAQ with default parameters and start acquisition.

        Sets a baseline resistance and begins a simulated acquisition loop.
        """
        super().__init__()
        self.pressureController = pressureController
        self.totalResistance = 1  # baseline fake resistance
        self.targetResistance = 1e3
        self._last_wave_params = None
        self._last_command = None
        self.start_acquisition(wave_freq=40, samplesPerSec=100000, dutyCycle=0.5,
                                amplitude=0.5, recordingTime=0.025, interval=None)
        self.active = False
        self._simulation_thread = None

    def resistance(self):
        """
        Return a simulated resistance value with noise.

        returns:
            float:
                Simulated resistance in ohms.
        """
        # if self.gigaseal_active:
        #     pressure = self.sim_pressure
        #     self.totalResistance += (np.random.normal(0, 0.1e6) ** abs(pressure))
        
        # if self.totalResistance >= 0.999 * self.gigaseal_target:
        #         self.totalResistance = self.gigaseal_target
        #         self.gigaseal_active = False
        return self.totalResistance + np.random.normal(0, 0.1)
    
    # def set_resistance(self, resistance):
    #     self.totalResistance = resistance

    # ------------------------------------------------------------------
    # Hardware emulation helpers
    # ------------------------------------------------------------------
    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Generate and store a square wave command.
        
        args:
            wave_freq (float):
                Frequency of the square wave (Hz).
            samplesPerSec (int):
                Sampling rate in samples per second.
            dutyCycle (float):
                Fraction of the period spent high.
            amplitude (float):
                Wave amplitude.
            recordingTime (float):
                Duration of the waveform (s).

        returns:
            None
        """
        numSamples = int(samplesPerSec * recordingTime)
        period = int(1 / wave_freq * samplesPerSec)
        onTime = int(period * dutyCycle)

        command = np.zeros(numSamples)
        for i in range(0, numSamples, period):
            command[i:i + onTime] = amplitude

        self._last_wave_params = (wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
        self._last_command = command
        # Base class expects a task-like object; returning None is acceptable.
        return None

    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Alias for square-wave generation in current-clamp mode.

        args:
            wave_freq (float):
                Frequency of the square wave (Hz).
            samplesPerSec (int):
                Sampling rate in samples per second.
            dutyCycle (float):
                Fraction of the period spent high.
            amplitude (float):
                Wave amplitude.
            recordingTime (float):
                Duration of the waveform (s).

        returns:
            None
        """
        return self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Return a simulated response to the previously generated command.
        
        Uses a simple ohmic model with additive Gaussian noise.

        args:
            samplesPerSec (int):
                Sampling rate in samples per second.
            recordingTime (float):
                Duration of the recording (s).

        returns:
            np.ndarray:
                Array of shape (2, N) containing:
                [command_signal, response_signal].
        """
        numSamples = int(samplesPerSec * recordingTime)
        if self._last_command is None or len(self._last_command) != numSamples:
            # No command was sent; return zeros
            command = np.zeros(numSamples)
        else:
            command = self._last_command

        # Simple ohmic response with noise
        response = command / max(self.totalResistance, 1.0)
        response += np.random.normal(0, 0.001, size=numSamples)

        return np.array([command, response], dtype=float)

    def getLeakSubtraction(
            self,
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
        Placeholder leak subtraction for FakeDAQ.

        Does not simulate actual acquisition but computes protocol metadata.

        args:
            start_voltage (float):
                Starting voltage of the sweep (V).
            end_voltage (float):
                Ending voltage of the sweep (V).
            holding_voltage (float):
                Holding voltage (V).
            step_voltage (float):
                Voltage step size (V).
            wave_freq (float):
                Frequency of the square wave (Hz).
            samplesPerSec (int):
                Sampling rate in samples per second.
            dutyCycle (float):
                Fraction of the waveform period spent high.
            pulse_recording_time (float):
                Duration of each pulse (s).
            repeats (int):
                Number of repeats per step.

        returns:
            list:
                Empty list (no data collected).

        raises:
            ValueError:
                If step_voltage is zero or repeats is non-positive.
        """
        self.warning("Leak subtraction is not simulated on FakeDAQ; returning empty result.")

        if step_voltage == 0:
            raise ValueError("step_voltage must be non-zero")
        if repeats <= 0:
            raise ValueError("repeats must be positive")

        span = end_voltage - start_voltage
        step = abs(step_voltage)
        if span == 0:
            targets = np.array([start_voltage], dtype=float)
        else:
            step_signed = step if span > 0 else -step
            targets = np.arange(start_voltage, end_voltage, step_signed, dtype=float)
            if targets.size == 0 or abs(targets[-1] - end_voltage) > 1e-9:
                targets = np.append(targets, end_voltage)

        leak_step_values = []
        leak_target_voltages = []
        for target in targets:
            delta = target - holding_voltage
            leak_step = -delta / 4.0
            leak_step_values.append(leak_step)
            leak_target_voltages.append(holding_voltage + leak_step)

        self.leak_subtraction_data = []
        self.leak_subtraction_meta = {
            "holding_voltage": holding_voltage,
            "targets": targets,
            "leak_target_voltages": np.array(leak_target_voltages, dtype=float),
            "leak_step_values": np.array(leak_step_values, dtype=float),
            "repeats": repeats,
            "wave_freq": float(wave_freq),
            "pulse_recording_time": float(pulse_recording_time),
        }
        return self.leak_subtraction_data
    
    # def start_gigaseal(self, pressure: PressureController):
    #     self.gigaseal_active = True
    #     self.sim_pressure = pressure

    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        resistance = self.resistance()

        t = np.linspace(
            0,
            recordingTime,
            int(samplesPerSec * recordingTime)
        )

        command = np.zeros_like(t)
        response = np.zeros_like(t)

        return (
            (t, response),
            (t, command),
            resistance,      # totalR
            self.targetResistance,   # memR
            resistance,      # accR
            20e-12           # memC
        )
    
    def update_pressure(self, pressure):
        self.pressureController.set_pressure(pressure)
        
    def tick(self, dt):
        # if self.active:
        pressure = self.pressureController.get_pressure()
        currResistance = self.totalResistance
        growth = np.exp(0.1 * abs(pressure))
        newResistance = currResistance + (
            growth * (1 - currResistance / self.targetResistance) * dt
            )
        # print(f"P={self.pressure}, growth={growth}")
        self.totalResistance = newResistance
        # else:
        #     return

    def _simulation_loop(self):
        dt = 5
        while self.active:
            self.tick(dt)
            time.sleep(0.1)
    
    def start(self):
        self.active = True
        self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._simulation_thread.start()
    
    def stop(self):
        self.active = False

        if (
            self._simulation_thread is not None
            and self._simulation_thread.is_alive()
            and self._simulation_thread is not threading.current_thread()
        ):
            self._simulation_thread.join(timeout=1.0)

        self._simulation_thread = None