
'''
Pressure Controller classes to communicate with the Pressure Controller Box made by the IBB.
Additionally, redesigning a closed loop pressure controller for the Moscow Rig
'''
import logging
from .BasePressureController import PressureController
import serial.tools.list_ports
import serial
import time
import collections


all = ['MoscowPressureController']

class MoscowPressureController(PressureController):
    '''A PressureController child class that handles serial communication between the PC and
       the Arduino controlling the Moscow Pressure box
    '''
    DEFAULT_VALID_PRODUCTS = ["USB Serial"]  # TODO: move to a constants or json file?
    DEFAULT_VALID_VIDS = [0x1a86, 0x403]
    DEFAULT_NATIVE_ZERO = 1962                   # The native units at a 0 pressure (y-intercept)
    DEFAULT_NATIVE_PER_MBAR = float(3062 / 1100) # Native pressure transducer units per mbar (-400 to 700)
    DEFAULT_SERIAL_CMD_TIMEOUT = 1               # (in sec) max time allowed between sending a serial command and expecting a response
    DEFAULT_READER_OFFSET = 516.72               # Reader-specific offset before scaling to raw units
    DEFAULT_READER_SCALE = 0.3923                # Reader-specific scale factor to convert to raw units

    def __init__(self, channel, controllerSerial = None, readerSerial = None,
                 validProducts=None, validVIDs=None,
                 nativeZero=None, nativePerMbar=None,
                 readerOffset=None, readerScale=None,
                 serialCmdTimeout=None,
                 valid_products=None, valid_vids=None,
                 native_zero=None, native_per_mbar=None,
                 sensor_offset=None, sensor_scale=None,
                 serial_cmd_timeout=None, sensor_channel=0, sensor_pin=None):
        """
        Initialize a MoscowPressureController instance.

        Args:
            channel (int): Pressure box channel number.
            controllerSerial (serial.Serial, optional): Serial port connected to the
                Arduino controlling the pressure output. Defaults to None.
            readerSerial (serial.Serial, optional): Serial port connected to the pressure
                sensor Arduino. Defaults to None.
            validProducts (list[str], optional): USB product names allowed for discovery.
            validVIDs (list[int], optional): USB Vendor IDs allowed for discovery.
            nativeZero (float, optional): Native DAC units corresponding to 0 mBar.
            nativePerMbar (float, optional): DAC units per mBar of pressure.
            readerOffset (float, optional): Offset for the pressure sensor before scaling.
            readerScale (float, optional): Scale factor for the pressure sensor.
            serialCmdTimeout (float, optional): Maximum allowed time for serial commands in seconds.
            valid_products (list[str], optional): Alias for `validProducts`.
            valid_vids (list[int], optional): Alias for `validVIDs`.
            native_zero (float, optional): Alias for `nativeZero`.
            native_per_mbar (float, optional): Alias for `nativePerMbar`.
            sensor_offset (float, optional): Alias for `readerOffset`.
            sensor_scale (float, optional): Alias for `readerScale`.
            serial_cmd_timeout (float, optional): Alias for `serialCmdTimeout`.
            sensor_channel (int, optional): Logical sensor ID in MultiPressureRR
                responses. Independent of
                the pressure-output `channel`. Defaults to 0, also the only
                supported input for legacy S<number>E sensor responses.
            sensor_pin (str, optional): Physical analog input such as A2 or A10.
                None leaves the firmware mapping untouched. Explicit pins require
                the updated configurable MultiPressureRR sketch and board support.

        Raises:
            Exception: If `controllerSerial` or `readerSerial` are not provided.
        
        Notes:
            - Sets initial pressure to 0 mBar and starts acquisition at 20 Hz.
            - Sets initial state to non-atmospheric mode (controlled pressure).
            - Any argument that is `None` will be replaced with the class default.
            - For two sensors on one board, inject the SAME open readerSerial
              object into both controllers and select sensor_channel=0 and 1.
              The base class shares complete frames, locks serial transactions,
              and applies each instance's readerOffset/readerScale separately.
              The caller handles Arduino startup time and shared-port closure.

        Example (reader and output are already open, Arduino boot complete)::

            first = MoscowPressureController(4, controllerSerial=output,
                                             readerSerial=reader, sensor_channel=0)
            second = MoscowPressureController(5, controllerSerial=output,
                                              readerSerial=reader, sensor_channel=1)

        These sensor IDs default to A0/A1; sensor_pin can remap or activate IDs.
        Supply sensor_offset
        and sensor_scale separately for each calibrated sensor. Stop acquisition
        on BOTH controllers before the caller closes reader. Acquisition getters
        return None after a sensor fault; the existing GUI may display None as
        zero until its unavailable-data handling is updated separately.
        """
        super().__init__()
        # time.sleep(2) # wait for arduino to boot up

        if controllerSerial is not None:
            # no port specified, we will use the user supplied serial port
            self.controllerSerial = controllerSerial
        else:
            self.controllerSerial = None
            self.error("No controller serial port available")
        
        if readerSerial is not None:
            # no port specified, we will use the user supplied serial port
            self.readerSerial = readerSerial
        else: 
            self.readerSerial = None
            self.error("No reader serial port available")

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
        self.validProducts = list(valid_products) if valid_products is not None else list(self.DEFAULT_VALID_PRODUCTS)
        self.validVIDs = list(valid_vids) if valid_vids is not None else list(self.DEFAULT_VALID_VIDS)
        self.nativeZero = float(native_zero) if native_zero is not None else self.DEFAULT_NATIVE_ZERO
        self.nativePerMbar = float(native_per_mbar) if native_per_mbar is not None else self.DEFAULT_NATIVE_PER_MBAR
        self.readerOffset = float(sensor_offset) if sensor_offset is not None else self.DEFAULT_READER_OFFSET
        self.readerScale = float(sensor_scale) if sensor_scale is not None else self.DEFAULT_READER_SCALE
        self.serialCmdTimeout = float(serial_cmd_timeout) if serial_cmd_timeout is not None else self.DEFAULT_SERIAL_CMD_TIMEOUT
        self.sensor_offset = self.readerOffset
        self.sensor_scale = self.readerScale
        self.state = None
        self.setpoint_raw = None
        self.lastVal = 0.0

        # Validate/bind the shared sensor before sending any output commands or
        # starting acquisition. Existing single-sensor setups use channel 0.
        self._configure_pressure_sensor(sensor_channel, sensor_pin)

        # set initial configuration of pressure controller
        self.set_ATM(False)
        self.set_pressure(0) # set initial pressure to 0 mbar
        self.start_acquisition() # 20 Hz; shared reader caches all sensor inputs


    def set_pressure(self, pressure):
        '''
        Tell pressure controller to go to a given setpoint pressure in mbar
        
        Args:
            pressure (float): Target pressure in mBar.
        '''
        nativeUnits = self.mbarToNative(pressure)
        # self.info(f"Setting pressure to {nativeUnits} mbar")
        self.set_pressure_raw(nativeUnits)
    
    def mbarToNative(self, pressure):
        '''
        Convert from a pressure in mBar to native units
        
        Args:
            pressure (float): Pressure in mBar.

        Returns:
            int: DAC units clamped to 0 – (2 * nativeZero).
        '''
        raw_pressure = int((pressure * self.nativePerMbar + self.nativeZero))
        return min(max(raw_pressure, 0), self.nativeZero * 2) # clamp native units to 0-3924


    def nativeToMbar(self, raw_pressure) -> float:
        '''
        Convert from native units to a pressure in mBar
        
        Args:
            raw_pressure (int): DAC units.

        Returns:
            float: Pressure in mBar.
        '''
        pressure = (raw_pressure - self.nativeZero) / self.nativePerMbar
        return pressure

    def set_pressure_raw(self, raw_pressure: int):
        '''
        Tell pressure controller to go to a given setpoint pressure in native DAC units
        
        Args:
            raw_pressure (int): Target pressure in native DAC units.
        '''
        self.setpoint_raw = raw_pressure
        self.info(f"Setting pressure to {self.nativeToMbar(raw_pressure)} mbar (raw: {raw_pressure})")

        cmd = f"set {self.channel} {raw_pressure}\n"
        self.info(f"Sending command: {cmd}")
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        self.controllerSerial.flush()
        self.info(f"Sent command: {cmd}")

    def get_pressure(self) -> float:
        '''
        Gets the current setpoint in millibar
        
        Returns:
            float: Setpoint pressure in mBar.
        '''
        return self.nativeToMbar(self.setpoint_raw)

    def measure(self) -> float:
        '''
        Read this controller's sensor channel, using the base shared transport.
        
        Returns:
            float: Measured pressure in mBar.

        Accepts both S<number>E and S0:<value>,1:<value>E firmware replies.
        Timeouts, malformed-only replies, and missing channels raise instead
        of returning an old lastVal. Acquisition exposes unavailable data as None.
        '''
        return self._measure_serial_pressure()
    
    def pulse(self, delayMs):
        '''Tell the onboard arduino to pulse pressure for a certain period of time
        
        Args:
            delayMs (int): Duration of pulse in milliseconds.
        '''
        cmd = f"pulse {self.channel} {delayMs}\n"
        self.info(f"Pulsing pressure for {delayMs} ms")
        self.controllerSerial.write(bytes(cmd, 'ascii')) #do serial writing in main thread for timing?
        self.controllerSerial.flush()
        
    def set_ATM(self, atm):
        '''Send a serial command activating or deactivating the atmosphere solenoid valve
           atm = True -> pressure output is at atmospheric pressure 
           atm = False -> pressure output comes from pressure regulator
        
        Args:
            atm (bool): True for atmospheric mode, False for regulated pressure.
        '''
        if atm:
            cmd = f"switchAtm {self.channel}\n" # switch to ATM command
            self.info(f"Switching to ATM: {cmd}")
        else:
            cmd = f"switchP {self.channel}\n" # switch to Pressure command
            self.info(f"Switching to Pressure: {cmd}")
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        self.controllerSerial.flush()
        self.state = atm

