"""Configurable pressure control for rigs sharing the text serial command protocol.

Rig-specific controllers remain available for hardware with different behavior.
This class uses set/pulse/switchAtm/switchP output commands and optionally reads
PressureRR or MultiPressureRR sensors. It does not assume that other protocols
are interchangeable merely because they use a serial connection.
"""
import math
import threading
from .BasePressureController import PressureController


__all__ = ['SerialPressureController']


class SerialPressureController(PressureController):
    """One output channel and optional independently calibrated sensor channel.

    Construct one instance per pressure output. Any number of instances may share
    the same sensor serial object; the base transport caches every input in each
    labelled frame. Output commands sharing a serial object are also serialized.
    The caller/rig loader owns opening, boot waits, and closing injected ports.
    """
    DEFAULT_VALID_PRODUCTS = ["USB Serial"]  # TODO: move to a constants or json file?
    DEFAULT_VALID_VIDS = [0x1a86, 0x403]
    DEFAULT_NATIVE_ZERO = 1962                   # The native units at a 0 pressure (y-intercept)
    DEFAULT_NATIVE_PER_MBAR = float(3062 / 1100) # Native pressure transducer units per mbar (-400 to 700)
    DEFAULT_SERIAL_CMD_TIMEOUT = 1               # (in sec) max time allowed between sending a serial command and expecting a response
    DEFAULT_READER_OFFSET = 516.72               # Reader-specific offset before scaling to raw units
    DEFAULT_READER_SCALE = 0.3923                # Reader-specific scale factor to convert to raw units

    def __init__(self, channel, controllerSerial=None, readerSerial=None,
                 validProducts=None, validVIDs=None,
                 nativeZero=None, nativePerMbar=None,
                 readerOffset=None, readerScale=None,
                 serialCmdTimeout=None,
                 valid_products=None, valid_vids=None,
                 native_zero=None, native_per_mbar=None,
                 sensor_offset=None, sensor_scale=None,
                 serial_cmd_timeout=None, sensor_channel=0,
                 native_min=0, native_max=None, startup_pressure=0,
                 startup_atm=False, initialize_output=True, sensor_pin=None):
        """Configure output control and one independently calibrated sensor input.

        sensor_channel selects an input in MultiPressureRR's labelled response;
        channel continues to select the pressure OUTPUT. The default sensor ID
        is 0, preserving the original single-sensor protocol. To share a board,
        pass the SAME open readerSerial object to each controller, with distinct
        sensor_channel values and per-sensor offsets/scales. The base class
        serializes requests and caches all returned inputs for one 20 Hz cycle.
        The caller waits for Arduino boot and owns shared-port closure.
        sensor_pin optionally assigns a physical analog pin (e.g. A2 or A10)
        to this logical ID using configurable MultiPressureRR firmware. None
        preserves firmware defaults and compatibility with older sketches. New
        logical IDs activate additional channels up to the board's input capacity.

        Rig-specific output calibration is native_zero/native_per_mbar. Raw
        commands are clamped to integer native_min/native_max bounds; omitting
        native_max retains the previous 2*native_zero upper limit. Initialization
        vents/selects startup_atm then sets startup_pressure (mbar). Set
        initialize_output=False to send no output commands during construction;
        get_pressure()/get_ATM() then remain None until explicitly commanded.

        readerSerial=None supports output-only hardware: no sensor acquisition
        thread starts, measure() raises, and get_last_acquisition() returns None.
        A commanded setpoint is never presented as an actual measurement. All
        configuration is validated before transmitting output commands. Existing
        camelCase calibration aliases remain accepted for older JSON files.
        """
        super().__init__()
        self.controllerSerial = controllerSerial
        self.readerSerial = readerSerial

        if valid_products is None:
            valid_products = validProducts
        if valid_vids is None:
            valid_vids = validVIDs
        if native_zero is None:
            native_zero = nativeZero
        if native_per_mbar is None:
            native_per_mbar = nativePerMbar
        if sensor_offset is None:
            sensor_offset = readerOffset
        if sensor_scale is None:
            sensor_scale = readerScale
        if serial_cmd_timeout is None:
            serial_cmd_timeout = serialCmdTimeout

        self.channel = channel
        self.state = None
        self.setpoint_raw = None
        self.lastVal = None

        # Rig-specific calibration constants (override via rig JSON params)
        self.validProducts = list(valid_products) if valid_products is not None else list(self.DEFAULT_VALID_PRODUCTS)
        self.validVIDs = list(valid_vids) if valid_vids is not None else list(self.DEFAULT_VALID_VIDS)
        self.nativeZero = float(native_zero) if native_zero is not None else self.DEFAULT_NATIVE_ZERO
        self.nativePerMbar = float(native_per_mbar) if native_per_mbar is not None else self.DEFAULT_NATIVE_PER_MBAR
        self.readerOffset = float(sensor_offset) if sensor_offset is not None else self.DEFAULT_READER_OFFSET
        self.readerScale = float(sensor_scale) if sensor_scale is not None else self.DEFAULT_READER_SCALE
        self.serialCmdTimeout = float(serial_cmd_timeout) if serial_cmd_timeout is not None else self.DEFAULT_SERIAL_CMD_TIMEOUT
        self.sensor_offset = self.readerOffset
        self.sensor_scale = self.readerScale

        if controllerSerial is None or not controllerSerial.is_open:
            raise ValueError('an open controllerSerial connection is required')
        if type(channel) is not int or channel < 0:
            raise ValueError('channel must be a nonnegative integer')
        if type(sensor_channel) is not int or sensor_channel < 0:
            raise ValueError('sensor_channel must be a nonnegative integer')
        if (not math.isfinite(self.nativeZero) or not math.isfinite(self.nativePerMbar)
                or self.nativePerMbar == 0):
            raise ValueError('output calibration must be finite with a nonzero scale')
        if not math.isfinite(self.serialCmdTimeout) or self.serialCmdTimeout <= 0:
            raise ValueError('serial command timeout must be finite and positive')
        if type(native_min) is not int or (native_max is not None and type(native_max) is not int):
            raise ValueError('native_min/native_max must be integer DAC limits')
        self.native_min = native_min
        if native_max is None and not math.isfinite(2 * self.nativeZero):
            raise ValueError('default native_max must be finite; configure an explicit limit')
        self.native_max = int(2 * self.nativeZero) if native_max is None else native_max
        if self.native_min > self.native_max:
            raise ValueError('native_min must not exceed native_max')
        if type(startup_atm) is not bool or type(initialize_output) is not bool:
            raise ValueError('startup_atm and initialize_output must be boolean')
        startup_raw = self.mbarToNative(startup_pressure)  # Validate before any I/O.
        if readerSerial is controllerSerial:
            raise ValueError('sensor and output firmware require separate serial connections')
        self.sensor_channel = sensor_channel
        if readerSerial is not None:
            self._configure_pressure_sensor(sensor_channel, sensor_pin)
        elif sensor_pin is not None:
            raise ValueError('sensor_pin requires a readerSerial connection')

        # One lock per injected output port, shared across generic instances.
        with PressureController._sensor_binding_lock:
            lock = getattr(controllerSerial, '_patcherbot_pressure_output_lock', None)
            if lock is None:
                lock = threading.RLock()
                controllerSerial._patcherbot_pressure_output_lock = lock
        self._output_lock = lock
        # Match the bound write timeout across users of the shared output port.
        with self._output_lock:
            timeout = getattr(controllerSerial, '_patcherbot_pressure_output_timeout', None)
            if timeout is not None and timeout != self.serialCmdTimeout:
                raise ValueError('controllers sharing an output port must use the same serial timeout')
            controllerSerial._patcherbot_pressure_output_timeout = self.serialCmdTimeout
            controllerSerial.write_timeout = self.serialCmdTimeout
        if initialize_output:
            self.set_ATM(startup_atm)
            self.set_pressure_raw(startup_raw)
        if readerSerial is not None:
            self.start_acquisition()  # 20 Hz, reusing the shared sensor cache.


    def set_pressure(self, pressure, port=0):
        '''
        Tell pressure controller to go to a given setpoint pressure in mbar
        '''
        if port != 0:
            raise ValueError('use a separate controller instance for another output channel')
        nativeUnits = self.mbarToNative(pressure)
        # self.info(f"Setting pressure to {nativeUnits} mbar")
        self.set_pressure_raw(nativeUnits)
    
    def mbarToNative(self, pressure):
        '''
        Convert from a pressure in mBar to native units
        '''
        pressure = float(pressure)
        raw_pressure = pressure * self.nativePerMbar + self.nativeZero
        if not math.isfinite(pressure) or not math.isfinite(raw_pressure):
            raise ValueError('pressure must produce a finite DAC setpoint')
        return min(max(int(raw_pressure), self.native_min), self.native_max)


    def nativeToMbar(self, raw_pressure) -> float:
        '''
        Convert from native units to a pressure in mBar
        '''
        raw_pressure = float(raw_pressure)
        pressure = (raw_pressure - self.nativeZero) / self.nativePerMbar
        if not math.isfinite(pressure):
            raise ValueError('native pressure must convert to a finite mbar value')
        return pressure

    def set_pressure_raw(self, raw_pressure: int):
        '''
        Tell pressure controller to go to a given setpoint pressure in native DAC units
        '''
        raw_pressure = float(raw_pressure)
        if not math.isfinite(raw_pressure) or not raw_pressure.is_integer():
            raise ValueError('raw pressure must be a finite integer DAC value')
        raw_pressure = min(max(int(raw_pressure), self.native_min), self.native_max)
        self.info(f"Setting pressure to {self.nativeToMbar(raw_pressure)} mbar (raw: {raw_pressure})")

        cmd = f"set {self.channel} {raw_pressure}\n"
        self.info(f"Sending command: {cmd}")
        with self._output_lock:
            self._write_command(cmd)
            self.setpoint_raw = raw_pressure  # Commit only after a successful send.
        self.info(f"Sent command: {cmd}")

    def get_pressure(self, port=0) -> float | None:
        '''
        Gets the current setpoint in millibar
        '''
        if port != 0:
            raise ValueError('use a separate controller instance for another output channel')
        return None if self.setpoint_raw is None else self.nativeToMbar(self.setpoint_raw)

    def measure(self, port=0) -> float:
        '''
        Return this sensor input in mbar using shared framing and calibration.

        Both old S<number>E and new labelled frames are supported. Missing or
        failed readings raise; lastVal is never used as a silent fallback.
        '''
        if port != 0:
            raise ValueError('select sensor_channel when constructing the controller')
        if self.readerSerial is None:
            raise RuntimeError('no sensor reader configured; get_pressure() returns the commanded setpoint')
        return self._measure_serial_pressure()
    
    def pulse(self, delayMs):
        '''Tell the onboard arduino to pulse pressure for a certain period of time
        '''
        duration = float(delayMs)
        if not math.isfinite(duration) or not duration.is_integer() or duration < 0:
            raise ValueError('pulse duration must be a nonnegative integer in milliseconds')
        cmd = f"pulse {self.channel} {int(duration)}\n"
        self.info(f"Pulsing pressure for {delayMs} ms")
        self._write_command(cmd)
        
    def set_ATM(self, atm):
        '''Send a serial command activating or deactivating the atmosphere solenoid valve
           atm = True -> pressure output is at atmospheric pressure 
           atm = False -> pressure output comes from pressure regulator
        '''
        if type(atm) is not bool:
            raise ValueError('atmosphere state must be boolean')
        if atm:
            cmd = f"switchAtm {self.channel}\n" # switch to ATM command
            self.info(f"Switching to ATM: {cmd}")
        else:
            cmd = f"switchP {self.channel}\n" # switch to Pressure command
            self.info(f"Switching to Pressure: {cmd}")
        with self._output_lock:
            self._write_command(cmd)
            self.state = atm

    def _write_command(self, command):
        """Keep whole output transactions together and surface incomplete writes.

        Subclasses for a different command protocol may replace output methods;
        shared sensor acquisition does not depend on these command strings.
        Ports remain owned by rig setup and are not closed by this instance.
        """
        payload = command.encode('ascii')
        with self._output_lock:
            if self.controllerSerial.write(payload) != len(payload):
                raise OSError('incomplete pressure output command write')
            self.controllerSerial.flush()
