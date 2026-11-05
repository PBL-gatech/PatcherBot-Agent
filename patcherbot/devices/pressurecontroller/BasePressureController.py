'''
A general pressure controller class with built-in acquisition methods.
'''
import collections
import math
import re
import threading
import time

from patcherbot.controller.base import TaskController

__all__ = ['PressureController', 'FakePressureController']


class _PressureSensorConnection:
    """Shared sensor transport, kept here for existing controller subclasses.

    One instance is attached to each injected readerSerial object. All controllers
    using that object share a transaction lock and an atomic raw-channel cache.
    Their existing acquisition threads can therefore reuse one response instead
    of independently consuming the same serial stream. Calibration stays in each
    controller. No additional polling thread or serial connection is created.

    MultiPressureRR replies S0:517.23,1:512.81E; PressureRR replies S517.23E.
    Legacy replies are assigned ONLY to sensor channel 0. Channel IDs in labelled
    replies are discovered from the frame, allowing more firmware inputs later.

    The caller owns opening, Arduino boot delay, and closing the serial object.
    This helper makes reads nonblocking and bounds writes with the configured
    timeout. Only participating controllers may read/write this sensor port.
    A timeout/transport error faults this connection: without request IDs a late
    reply cannot reliably be associated with a subsequent request. To recover,
    stop the controllers, reset/reconnect the board, and supply a NEW serial
    object after boot. No automatic retries or hardware resets are performed.
    """

    CACHE_INTERVAL = 0.05  # Share frames for one normal 20 Hz acquisition cycle.
    MAX_SAMPLE_AGE = 0.25  # Never expose an old sample as a current measurement.
    MAX_FRAME_BYTES = 4096

    def __init__(self, serial_port, timeout):
        self.serial = serial_port
        self.timeout = timeout
        self._transaction_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._values = {}
        self._received_at = None  # Host monotonic receive time, not ADC sample time.
        self._fault = None
        self._pin_bindings = {}  # Requested pin (or firmware default) per logical ID.
        serial_port.timeout = 0
        serial_port.write_timeout = timeout

    def configure_pin(self, channel, pin):
        """Apply a C<id>:A<n> command once per channel, sharing the I/O lock.

        None leaves firmware defaults untouched, including legacy firmware.
        Explicit pins require the updated MultiPressureRR acknowledgement; a
        timeout must not silently leave the sensor reading a different input.
        Two objects requesting the same logical ID must agree on its pin setting.
        Pin mappings are volatile on the board and reapplied on a new connection.
        """
        with self._transaction_lock:
            if self._fault is not None:
                raise RuntimeError('pressure sensor connection faulted; reconnect/reset board') from self._fault
            if channel in self._pin_bindings:
                if self._pin_bindings[channel] != pin:
                    raise ValueError('conflicting sensor_pin settings for sensor channel {}'.format(channel))
                return
            if pin is not None:
                # Do not let pre-remapping samples appear as measurements of the
                # new pin. Other polling threads wait on this transaction lock.
                with self._cache_lock:
                    self._values = {}
                    self._received_at = None
                try:
                    self._request_pin_configuration(channel, pin)
                except ValueError:
                    raise  # Firmware explicitly rejected it; mapping unchanged.
                except Exception as exc:
                    with self._cache_lock:
                        self._fault = exc
                    raise
            self._pin_bindings[channel] = pin

    def _request_pin_configuration(self, channel, pin):
        """Read a bounded, newline-terminated pin acknowledgement in fragments."""
        request = 'C{}:{}\n'.format(channel, pin).encode('ascii')
        expected = 'K{}:{}'.format(channel, pin).encode('ascii')
        self.serial.reset_input_buffer()
        if self.serial.write(request) != len(request):
            raise OSError('incomplete sensor pin configuration write')
        deadline = time.monotonic() + self.timeout
        line = bytearray()
        while True:
            expired = time.monotonic() >= deadline
            available = self.serial.in_waiting
            if not available:
                if expired:
                    break
                time.sleep(min(0.001, max(0, deadline - time.monotonic())))
                continue
            for byte in self.serial.read(min(available, 128)):
                if byte == ord('\n'):
                    reply = bytes(line).rstrip(b'\r')
                    line.clear()
                    if reply == expected:
                        return
                    if reply.startswith(b'F:'):
                        raise ValueError('sensor firmware rejected {}: {}'.format(
                            request.decode().strip(), reply.decode('ascii', errors='replace')))
                elif len(line) < 128:
                    line.append(byte)
            if expired:
                break
        raise TimeoutError('sensor pin configuration was not acknowledged; upload configurable MultiPressureRR firmware')

    @staticmethod
    def _parse_frame(frame):
        """Validate the entire frame before publishing any of its raw readings."""
        body = frame[1:-1].decode('ascii')
        if ':' not in body:
            fields = ['0:' + body]  # Original single-sensor firmware.
        else:
            fields = body.split(',')
        values = {}
        for field in fields:
            # Firmware emits nonnegative decimal ADC counts. Exponents would
            # conflict with the E terminator and are not part of this protocol.
            match = re.fullmatch(r'([0-9]+):([0-9]+(?:\.[0-9]+)?)', field)
            if match is None:
                raise ValueError('invalid pressure sensor field')
            channel, value = int(match[1]), float(match[2])
            if channel in values or not math.isfinite(value):
                raise ValueError('duplicate channel or nonfinite pressure reading')
            values[channel] = value
        return values

    def cached(self, channel, max_age=MAX_SAMPLE_AGE):
        """Read without waiting for serial I/O; raise if missing, faulted, or old."""
        with self._cache_lock:
            if self._fault is not None:
                raise RuntimeError('pressure sensor connection faulted; reconnect/reset board') from self._fault
            if self._received_at is None or time.monotonic() - self._received_at > max_age:
                raise TimeoutError('no fresh pressure sensor measurement available')
            if channel not in self._values:
                raise ValueError('sensor channel {} missing from response'.format(channel))
            return self._values[channel]

    def read(self, channel):
        """Serialize a complete R transaction, or reuse a recent shared frame."""
        with self._transaction_lock:
            try:
                return self.cached(channel, self.CACHE_INTERVAL)
            except TimeoutError:
                pass  # First sample or cache due for refresh; request all inputs.

            try:
                values, received_at = self._request_frame()
            except Exception as exc:
                with self._cache_lock:
                    self._fault = exc
                raise
            with self._cache_lock:
                self._values = values
                self._received_at = received_at
            # A missing selected channel is a configuration error, not a serial
            # fault. Other controllers can still use the valid returned inputs.
            return self.cached(channel)

    def _request_frame(self):
        """Wait up to the deadline, but check buffered replies before timing out.

        Applying a Qt theme can hold Python execution long enough that this
        thread resumes after its deadline, with the Arduino reply already in the
        USB receive buffer. Service that outstanding request before declaring a
        timeout. This is not a retry: no new R is sent and no input is discarded.
        After the deadline, consume at most one bounded chunk of available input
        so continuous malformed traffic cannot extend the wait indefinitely.
        """
        frame = bytearray()
        self.serial.reset_input_buffer()
        if self.serial.write(b'R') != 1:
            raise OSError('incomplete pressure sensor request write')
        # Start response timing after sending R. A GUI stall before the write
        # must not consume the Arduino's response window; writes have their own
        # configured pyserial write_timeout.
        deadline = time.monotonic() + self.timeout
        while True:
            expired = time.monotonic() >= deadline
            available = self.serial.in_waiting
            if not available:
                if expired:
                    break
                time.sleep(min(0.001, max(0, deadline - time.monotonic())))
                continue
            chunk = self.serial.read(min(available, self.MAX_FRAME_BYTES))
            for byte in chunk:
                if byte == ord('S'):
                    frame = bytearray(b'S')  # Resynchronize after noise.
                elif frame:
                    frame.append(byte)
                    if len(frame) > self.MAX_FRAME_BYTES:
                        frame.clear()
                    elif byte == ord('E'):
                        try:
                            values = self._parse_frame(bytes(frame))
                        except (ValueError, UnicodeError):
                            frame.clear()
                            continue  # Await a valid frame within deadline.
                        return values, time.monotonic()
            if expired:
                break
        raise TimeoutError('no valid pressure sensor frame before timeout')


class PressureAcquisitionThread(threading.Thread):
    """
    A thread that continuously measures pressure via the controller's measure() method.
    The latest measurement is stored (using a deque with maxlen=1), and an optional
    callback is invoked with each new measurement.
    """
    def __init__(self, controller, interval=0.016, callback=None):
        """
        Initializes the acquisition thread.

        Args:
            controller (PressureController): Controller to measure pressure from.
            interval (float, optional): Sampling interval in seconds. Defaults to 0.016.
            callback (callable, optional): Optional callback for each measurement.
        """
        super().__init__(daemon=True)
        self.controller = controller
        self.interval = interval
        self.callback = callback
        self.running = True
        self._last_data_queue = collections.deque(maxlen=1)
        self._last_error = None

    def run(self):
        """
        Runs the acquisition loop until stopped. Measures pressure and optionally
        invokes the callback.
        """
        while self.running:
            try:
                # Call the measure() method with no port parameter.
                pressure = self.controller.measure()
                # Keep fractional pressure precision; display code can format it.
                # Instead of creating a dictionary, just store the numeric measurement.
                self._last_data_queue.append(pressure)
                self._last_error = None
                if self.callback:
                    self.callback(pressure)
            except Exception as e:
                self._last_data_queue.append(None)  # Do not retain a failed reading.
                # A shared fault affects both controllers. Report transitions,
                # rather than flooding the GUI log with the same error at 20 Hz.
                message = str(e)
                if message != self._last_error:
                    self.controller.error("Error in PressureAcquisitionThread: " + message)
                    self._last_error = message
            time.sleep(self.interval)

    def get_last_data(self):
        """
        Returns the most recent measurement.

        Returns:
            float or None: Last pressure measurement, or None if unavailable.
        """
        return self._last_data_queue[-1] if self._last_data_queue else None

    def stop(self):
        self.running = False


class PressureController(TaskController):
    """
    Base class for pressure controllers with built-in acquisition capabilities.
    """
    _sensor_binding_lock = threading.Lock()

    def __init__(self):
        """
        Initializes the pressure controller.
        """
        super().__init__()
        self._pressure = collections.defaultdict(int)
        # Holder for the acquisition thread; child classes can start it.
        self._pressure_acq_thread = None
        self.state = False  # True if the pressure controller is in ATM mode, False otherwise

    def _configure_pressure_sensor(self, sensor_channel=0, sensor_pin=None):
        """Bind Moscow/Serial drivers to the same injected sensor connection.

        Call AFTER readerSerial, serialCmdTimeout, readerOffset, and readerScale
        are set and BEFORE starting acquisition. sensor_channel identifies an ADC
        input, independently of the existing pressure-output channel. Pass the
        SAME serial object to both controllers, not two objects for the same COM
        port. Each controller retains its own offset/scale. Connection ownership
        remains with the caller; stopping one controller must not close a shared
        port while the other still needs it.
        """
        if type(sensor_channel) is not int or sensor_channel < 0:
            raise ValueError('sensor_channel must be a nonnegative integer')
        if sensor_pin is not None:
            if not isinstance(sensor_pin, str) or not re.fullmatch(r'[Aa][0-9]{1,3}', sensor_pin.strip()):
                raise ValueError('sensor_pin must be an analog pin name such as A0 or A10')
            sensor_pin = 'A{}'.format(int(sensor_pin.strip()[1:]))
            if len('C{}:{}\n'.format(sensor_channel, sensor_pin)) > 32:
                raise ValueError('sensor pin configuration exceeds firmware command length')
        if not math.isfinite(self.serialCmdTimeout) or self.serialCmdTimeout <= 0:
            raise ValueError('serial command timeout must be finite and positive')
        if (not math.isfinite(self.readerOffset) or not math.isfinite(self.readerScale)
                or self.readerScale == 0):
            raise ValueError('sensor calibration must be finite with a nonzero scale')
        if self.readerSerial is None or not self.readerSerial.is_open:
            raise ValueError('an open readerSerial connection is required')
        with PressureController._sensor_binding_lock:
            connection = getattr(self.readerSerial, '_patcherbot_pressure_sensor', None)
            if connection is None:
                connection = _PressureSensorConnection(self.readerSerial, self.serialCmdTimeout)
                self.readerSerial._patcherbot_pressure_sensor = connection
            elif connection.timeout != self.serialCmdTimeout:
                raise ValueError('controllers sharing a sensor must use the same serial timeout')
        connection.configure_pin(sensor_channel, sensor_pin)
        self._sensor_connection = connection
        self.sensor_channel = sensor_channel
        self.sensor_pin = sensor_pin
        self.lastVal = None

    def _measure_serial_pressure(self):
        """Select and calibrate one input; errors never return an old lastVal."""
        try:
            raw = self._sensor_connection.read(self.sensor_channel)
            pressure = (raw - self.readerOffset) / self.readerScale
            if not math.isfinite(pressure):
                raise ValueError('sensor calibration produced a nonfinite pressure')
            self.lastVal = pressure
            return self.lastVal
        except Exception:
            self.lastVal = None
            raise

    def measure(self, port=0):
        """
        Measures the instantaneous pressure on the designated port.
        Child classes must override this method with device-specific behavior.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.
        
        Note: This method must be overriden in sublcass.
        """
        pass

    def set_pressure(self, pressure: int, port=0):
        """
        Sets the pressure on the designated port.

        Args:
            pressure (int): Desired pressure.
            port (int, optional): Pressure port number. Defaults to 0.
        """
        self._pressure[port] = pressure

    def get_pressure(self, port=0):
        """
        Gets the pressure value (as set via set_pressure) on the designated port.
        Note that this is not a measured value.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.

        Returns:
            int: Pressure value.
        """
        return self._pressure[port]

    def ramp(self, amplitude=-230., duration=1.5, port=0):
        """
        Creates a pressure ramp over the specified duration.

        Args:
            amplitude (float, optional): Maximum pressure. Defaults to -230.
            duration (float, optional): Ramp duration in seconds. Defaults to 1.5.
            port (int, optional): Pressure port number. Defaults to 0.
        """
        t0 = time.time()
        t = t0
        while t - t0 < duration:
            self.set_pressure(int(amplitude * (t - t0) / duration), port)
            t = time.time()
        self.set_pressure(0, port)

    def set_ATM(self, atm):
        """
        Sets the atmospheric pressure state.

        Args:
            atm (bool): True for atmospheric mode, False otherwise.
        """
        self.state = atm

    def get_ATM(self):
        """
        Gets the current atmospheric pressure state.

        Returns:
            bool: Atmospheric mode state.
        """
        return self.state

    # --- Pressure Acquisition System Methods ---
    def start_acquisition(self, interval=0.05, callback=None):
        """
        Starts continuous pressure acquisition in a separate thread.
        Child classes should call this method to begin measuring pressure.
        
        Args:
            interval (float, optional): Time between measurements (s). Defaults to 0.05.
            callback (callable, optional): Optional function called with each measurement.
        """
        self.info("Starting pressure acquisition.")
        if self._pressure_acq_thread is None:
            self._pressure_acq_thread = PressureAcquisitionThread(self, interval, callback)
            self._pressure_acq_thread.start()

    def get_last_acquisition(self):
        """
        Returns the most recent pressure measurement as a scalar.
        Returns None if no data is available.

        Returns:
            float or None: Last measurement, or None if unavailable/stale/faulted.
        """
        if self._pressure_acq_thread:
            if hasattr(self, '_sensor_connection'):
                # Inspect the shared frame directly, without waiting on a serial
                # transaction or serving a stale per-controller thread queue.
                try:
                    raw = self._sensor_connection.cached(self.sensor_channel)
                    pressure = (raw - self.readerOffset) / self.readerScale
                    return pressure if math.isfinite(pressure) else None
                except (RuntimeError, TimeoutError, ValueError):
                    return None
            return self._pressure_acq_thread.get_last_data()
        return None

    def stop_acquisition(self):
        """
        Stops the continuous pressure acquisition thread.
        """
        self.info("Stopping pressure acquisition.")
        if self._pressure_acq_thread:
            self._pressure_acq_thread.stop()
            self._pressure_acq_thread.join()
            self._pressure_acq_thread = None


class FakePressureController(PressureController):
    """
    Simulated pressure outputs, optionally measured by an actual sensor board.

    With no readerSerial this retains the original fully simulated behavior.
    Supplying a reader enables the base shared sensor transport; set_pressure,
    get_pressure and atmosphere controls still operate on simulated state only.
    Two instances can share one readerSerial and select sensor_channel 0 and 1.
    Measured pressure then changes only with physical sensor input, not setpoints.
    """
    def __init__(self, readerSerial=None, sensor_channel=0,
                 sensor_offset=516.72, sensor_scale=0.3923,
                 serial_cmd_timeout=1, sensor_pin=None):
        """
        Start simulated acquisition or optional real sensor acquisition at 20 Hz.

        sensor_offset/scale convert raw ADC counts to mbar and are independent
        per instance. Defaults are the existing Moscow nominal values, NOT a
        calibration of a newly connected sensor. Override with measured values.
        The caller opens the port, waits for boot, and closes it after stopping
        all users; the JSON loader supports a once-per-connection boot_delay.
        No controllerSerial is used and no actuator commands are transmitted.
        """
        super(FakePressureController, self).__init__()
        self.pressure = 0
        if readerSerial is not None:
            self.readerSerial = readerSerial
            self.readerOffset = self.sensor_offset = float(sensor_offset)
            self.readerScale = self.sensor_scale = float(sensor_scale)
            self.serialCmdTimeout = float(serial_cmd_timeout)
            self._configure_pressure_sensor(sensor_channel, sensor_pin)
        elif sensor_pin is not None:
            raise ValueError('sensor_pin requires a readerSerial connection')
        # Start the acquisition thread automatically so that
        # ``get_last_acquisition`` returns values when used in the
        # fake rig setup.  Without this the GUI raises an exception
        # because no pressure measurements are available.
        self.start_acquisition()

    def measure(self, port=0):
        """
        Return the selected hardware measurement, or the simulated setpoint.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.

        Returns:
            float: Pressure in mbar; hardware read errors propagate to acquisition.
        """
        if hasattr(self, '_sensor_connection'):
            return self._measure_serial_pressure()
        return self.pressure

    def get_last_acquisition(self):
        """Keep the fake rig's existing integer GUI interface; None stays missing.

        The current graph initializes an integer QSlider from this getter. Keep
        that contract until the graph is updated; measure() and the shared raw
        cache retain fractional precision. This does not convert missing data to
        zero (the existing GUI still does that itself).
        """
        pressure = super().get_last_acquisition()
        return int(pressure) if pressure is not None else None

    def set_pressure(self, pressure, port=0):
        """
        Sets a fake pressure value and logs the update.

        Args:
            pressure (int): Desired pressure.
            port (int, optional): Pressure port number. Defaults to 0.
        """
        self.debug('Pressure set to: {}'.format(pressure))
        self.pressure = pressure

    def get_pressure(self, port=0):
        """
        Returns the current simulated pressure.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.

        Returns:
            int: Simulated pressure.
        """
        return self.pressure
