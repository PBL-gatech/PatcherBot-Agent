Pressure control
================

Holy Pipette can control different pressure controllers. All classes inherit
from `.PressureController` and expose the same ``set_pressure`` and ``measure``
API.

Supported controllers include:

- `.IBBPressureController`
- `.MoscowPressureController`
- `.FakePressureController` (development)

Examples
--------

Using the :class:`IBBPressureController`::

    from patcherbot.devices.pressurecontroller import IBBPressureController
    import serial

    port = serial.Serial('COM5', 9600, timeout=0)
    controller = IBBPressureController(channel=1, arduinoSerial=port)
    controller.set_pressure(25)
    pressure = controller.measure()

The :class:`MoscowPressureController` uses separate ports for commands and
sensor readings::

    from patcherbot.devices.pressurecontroller import MoscowPressureController
    import serial

    ctrl_serial = serial.Serial('COM5', 9600, timeout=0)
    read_serial = serial.Serial('COM9', 9600, timeout=0)
    controller = MoscowPressureController(channel=1,
                                          controllerSerial=ctrl_serial,
                                          readerSerial=read_serial)
    controller.set_pressure(25)
    pressure = controller.measure()

Pressure values are expressed in mBar.

Generic serial pressure controller
----------------------------------
``SerialPressureController`` extends ``PressureController`` and provides a common
implementation for rigs using ``set``, ``pulse``, ``switchAtm`` and ``switchP``
serial commands. Existing rig-specific classes remain available for different
hardware protocols. Calibration and capability choices belong in the rig JSON;
changing the class does not translate another device's serial protocol.

Select ``SerialPressureController (generic)`` in the rig builder for output plus
sensor acquisition, or its output-only variant when no reader is attached. In
JSON the class path is
``patcherbot.devices.pressurecontroller.SerialPressureController.SerialPressureController``.

Configure ``channel`` and ``controllerSerial`` for the pressure output. Configure
``readerSerial`` and ``sensor_channel`` independently for a sensor input. Multiple
instances can share reader port settings and choose sensor channels 0, 1, etc.;
the loader opens one connection and the base transport shares complete frames.
MultiPressureRR defaults channels 0 and 1 to A0 and A1. Both legacy
single-sensor and labelled multi-sensor replies are accepted. Sensor and output
firmware require separate serial connections.

Configurable analog inputs and additional channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set ``sensor_pin`` to an Arduino analog input name such as ``A2`` or ``A10`` to
assign a physical pin to the integer ``sensor_channel`` ID. Pin names are
case-insensitive in Python and resolved by the board; an UNO R4 WiFi accepts
A0-A5, while a Mega accepts A0-A15. Additional IDs are activated by configuration,
up to the board's number of analog inputs. IDs can be sparse and are not limited
to 0 and 1. The firmware pin table covers up to A15; supporting a board with more
analog inputs requires extending that table.

For example, a sensor entry can include::

    "readerSerial": {"port": "COM4", "baudrate": 9600, "timeout": 0, "boot_delay": 2},
    "sensor_channel": 2,
    "sensor_pin": "A3",
    "sensor_offset": 516.72,
    "sensor_scale": 0.3923

The offset/scale above are nominal examples; use each sensor's calibration.
This adds logical sensor 2 on A3 without editing or recompiling pin assignments.
The same parameters are supported by the generic, Moscow, and hardware-sensing
fake controllers. In the rig builder, use **Sensor channel ID** and **Analog pin**.

Upload the updated MultiPressureRR sketch once before using explicit pin settings.
Initialization sends ``C2:A3\n`` and requires acknowledgement ``K2:A3\r\n``.
Unsupported pins/IDs receive an explicit ``F:RANGE`` rejection. Configurations
are serialized with acquisition, and changing a pin resets that channel's filter
and invalidates the host's old cache. Repeated instances using the same logical
ID must agree on ``sensor_pin``; conflicting mappings fail initialization.

Omit ``sensor_pin`` or leave it blank in the dialog to retain firmware defaults
and compatibility with older firmware. Explicit settings must be acknowledged;
older sketches will report a configuration timeout rather than silently use the
wrong pin. Pin mappings are held in RAM and reapplied from JSON after reset.
The firmware still defaults to two active inputs when no pin configuration is
sent; unselected inputs in those default replies are ignored by the controller.

``native_zero`` and ``native_per_mbar`` calibrate the output. ``native_min`` and
``native_max`` set integer DAC limits; the default range is 0 to twice the native
zero for compatibility. Raw and mbar commands both respect those limits.
``sensor_offset`` and ``sensor_scale`` independently calibrate each sensor.
Replace nominal template values with the rig's verified calibration and limits.

``startup_pressure`` defaults to 0 mbar and ``startup_atm`` defaults to false.
``initialize_output=false`` suppresses output commands during construction;
setpoint and atmosphere state remain unknown (None) until commanded explicitly.
Port dictionaries may specify ``boot_delay`` to allow firmware startup before
device initialization. Output writes from generic instances sharing one port
are serialized, and incomplete writes raise without updating the saved setpoint.

If ``readerSerial`` is omitted, no sensor acquisition thread starts.
``get_pressure()`` still returns the commanded setpoint, ``measure()`` raises
because no measurement is available, and ``get_last_acquisition()`` returns None.
The setpoint is never substituted for a sensor measurement. With a reader,
acquisition starts at 20 Hz. Stop all acquisition users before closing a shared
port; the application loader/shutdown path owns those connections.

Fake pressure controller
------------------------
For development purposes, a `.FakePressureController` is implemented.
It behaves as a pressure controller, except it is not connected to an
actual device.

