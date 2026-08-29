from enum import Enum
import serial
import time
from .laser import Laser


class WavelengthChannel(Enum):
    """Wavelength channels available on the Lumencor light engine.

    Values map to bit positions used by the controller command.
    """
    RED = 0
    GREEN = 1
    CYAN = 2
    UV = 3
    # 4th bit is for green/yellow filter (not a wavelength)
    BLUE = 5
    INFRARED = 6
    TEAL = INFRARED
    OFF = 7


class ExcitationFilter(Enum):
    """Excitation filter states (green/yellow) used by the controller."""
    YELLOW = 0
    GREEN = 1


class LumencorLaser(Laser):
    """Control the Lumencor Spectra X light engine.

    WavelengthChannel selects the wavelength channel and ExcitationFilter toggles the
    green/yellow filter bit used when applying output.
    Documentation: https://cms.lumencor.com/system/uploads/fae/file/asset/150/57-10035_Spectra_X_Command_Reference.pdf
    """
    def __init__(self, com: serial.Serial):
        # Order used for wavelength selection (1-based)
        self._wavelength_order = [
            WavelengthChannel.RED,
            WavelengthChannel.GREEN,
            WavelengthChannel.CYAN,
            WavelengthChannel.UV,
            WavelengthChannel.BLUE,
            WavelengthChannel.INFRARED,
        ]
        self._default_wavelength_index = 2  # Matches PatchConfig default ("2" -> GREEN)
        self.com = com
        self.current_wavelength = WavelengthChannel.OFF
        self.current_excitation_filter = ExcitationFilter.GREEN
        self.power_state = "off"
        self._power_levels = {}
        self._output_enabled_channel = None
        self._output_enabled_filter = None
        super().__init__()

    def _index_to_wavelength(self, index: int | None):
        """Map a 1-based wavelength index to a WavelengthChannel, cycling through the list."""
        if index is None:
            return None
        if index <= 0:
            index = 1
        idx = (index - 1) % len(self._wavelength_order)
        return self._wavelength_order[idx]

    def _resolve_wavelength(self, value):
        """Accept WavelengthChannel, color names, or int/str indices and return a WavelengthChannel."""
        if value is None:
            return None
        if isinstance(value, WavelengthChannel):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            color_map = {
                "red": WavelengthChannel.RED,
                "green": WavelengthChannel.GREEN,
                "cyan": WavelengthChannel.CYAN,
                "uv": WavelengthChannel.UV,
                "blue": WavelengthChannel.BLUE,
                "infrared": WavelengthChannel.INFRARED,
                "teal": WavelengthChannel.INFRARED,
                "off": WavelengthChannel.OFF,
            }
            if key in color_map:
                return color_map[key]
            if key.isdigit():
                value = int(key)
        if isinstance(value, int):
            return self._index_to_wavelength(value)
        self.warning(f"Lumencor: Unsupported wavelength value {value!r}")
        return None

    def _select_wavelength(
        self,
        wavelength: WavelengthChannel | int | str | None,
        excitation_filter: ExcitationFilter | None = None,
    ):
        """Internal helper to update the selected wavelength and filter."""
        channel = self._resolve_wavelength(wavelength)
        if channel is None:
            self.info("Lumencor: No wavelength specified, skipping selection.")
            return None
        self.current_wavelength = channel
        if excitation_filter is not None:
            self.current_excitation_filter = excitation_filter
        return channel

    def _initialize(self):
        """Send hardware init commands to the Lumencor controller."""
        self.com.write(bytearray([0x57, 0x02, 0xFF, 0x50]))  # init cmd 1
        self.com.write(bytearray([0x57, 0x03, 0xAB, 0x50]))  # init cmd 2
        self.info("Lumencor laser initialized")

    def _apply_wavelength_output(
        self,
        wavelength: WavelengthChannel,
        excitation_filter: ExcitationFilter = ExcitationFilter.GREEN,
        *,
        force: bool = False,
    ):
        """Apply output selection to the controller."""
        if (
            not force
            and self._output_enabled_channel == wavelength
            and self._output_enabled_filter == excitation_filter
        ):
            return
        if wavelength == WavelengthChannel.OFF:
            cmd = bytearray([0x4F, 0x7F, 0x50])
            self.com.write(cmd)
            self.info("Lumencor: Output disabled.")
            self._output_enabled_channel = wavelength
            self._output_enabled_filter = excitation_filter
            return

        channel_index = 0x00
        channel_index |= 1 << wavelength.value  # set the bit of the wavelength we want

        # invert the bits so that 0 is on and 1 is off
        channel_index = ~channel_index & 0x7F  # we only want the first 7 bits

        if excitation_filter == ExcitationFilter.YELLOW:
            channel_index &= 0xEF  # set the 4th bit to 0

        cmd = bytearray([0x4F, channel_index, 0x50])
        self.com.write(cmd)
        self.info(
            "Lumencor: Output {} enabled ({} filter).".format(
                wavelength.name, excitation_filter.name
            )
        )
        self._output_enabled_channel = wavelength
        self._output_enabled_filter = excitation_filter

    def power_on(self):
        """Enable output for the selected wavelength."""
        if self.current_wavelength in (None, WavelengthChannel.OFF):
            self._select_wavelength(self._default_wavelength_index)
        power = self._power_levels.get(self.current_wavelength, 100)
        self.set_power_level(power, self.current_wavelength)
        self._apply_wavelength_output(self.current_wavelength, self.current_excitation_filter)
        self.power_state = "on"

    def power_off(self):
        """Disable laser output."""
        self._apply_wavelength_output(WavelengthChannel.OFF, self.current_excitation_filter)
        self.power_state = "off"

    def get_power_state(self):
        """Return cached power state."""
        return self.power_state

    def set_wavelength(
        self,
        wavelength: WavelengthChannel | int | str | None = None,
        excitation_filter: ExcitationFilter | None = None,
    ):
        """Select the output wavelength channel."""
        channel = self._select_wavelength(wavelength, excitation_filter)
        if channel is None:
            return
        if channel == WavelengthChannel.OFF:
            self.power_off()
            return
        if self.power_state == "on":
            power = self._power_levels.get(channel, 100)
            self.set_power_level(power, channel)
            self._apply_wavelength_output(channel, self.current_excitation_filter)

    def get_wavelength(self):
        """Return the currently selected wavelength channel."""
        return self.current_wavelength

    def set_power_level(
        self,
        power_percent: float,
        wavelength: WavelengthChannel | int | str | None = None,
        *,
        force: bool = False,
    ):
        """Set output power for a wavelength channel (0-100)."""
        if wavelength is None:
            channel = self.current_wavelength
            if channel in (None, WavelengthChannel.OFF):
                channel = self._select_wavelength(self._default_wavelength_index)
        else:
            channel = self._select_wavelength(wavelength)
        if channel is None or channel == WavelengthChannel.OFF:
            self.warning("Lumencor: No wavelength selected, skipping set_power_level.")
            return

        clamped = int(max(0, min(100, round(power_percent))))
        if not force and self._power_levels.get(channel) == clamped:
            return

        # calculate address of the DAC given wavelength channel
        dac_address = None
        channel_address = 0x01
        addr18_channels = [
            WavelengthChannel.UV,
            WavelengthChannel.CYAN,
            WavelengthChannel.GREEN,
            WavelengthChannel.RED,
        ]
        addr1a_channels = [WavelengthChannel.BLUE, WavelengthChannel.INFRARED]
        if channel in addr18_channels:
            dac_address = 0x18
            channel_address = channel_address << addr18_channels.index(channel)
        else:
            dac_address = 0x1A
            channel_address = channel_address << addr1a_channels.index(channel)

        # convert 0-100 to 0-2^8
        power = int((clamped / 100) * 255) & 0xFF
        # invert power (0xFF is 0% power, 0x00 is 100% power)
        power = ~power
        power_highnibble = (power & 0xF0) >> 4
        power_lownibble = power & 0x0F

        cmd = bytearray(
            [
                0x53,
                dac_address,
                0x03,
                channel_address & 0x0F,
                power_highnibble | 0xF0,
                power_lownibble << 4,
                0x50,
            ]
        )

        self.com.write(cmd)
        self._power_levels[channel] = clamped
        self.info("Lumencor: Power for {} set to {}%".format(channel.name, clamped))

    def get_laser_temp(self):
        """Return the internal IIC temperature in degrees C."""
        cmd = bytearray([0x53, 0x91, 0x02, 0x50])
        self.com.write(cmd)
        time.sleep(0.1)
        temp = self.com.read(2)
        # we only want the Most Significant 11 bits
        temp = temp[1] << 3 | temp[0] >> 5
        temp = temp * 0.125
        return temp

if __name__ == "__main__":
    laser_com = serial.Serial(
        "COM6",
        9600,
        timeout=1,
        stopbits=serial.STOPBITS_ONE,
        parity=serial.PARITY_NONE,
        bytesize=serial.EIGHTBITS,
    )
    l = LumencorLaser(laser_com)

    l.set_wavelength(WavelengthChannel.BLUE)
    l.set_power_level(100, WavelengthChannel.BLUE)
    l.power_on()
    time.sleep(1)

    l.set_wavelength(WavelengthChannel.RED)
    l.set_power_level(100, WavelengthChannel.RED)
    l.power_on()
    time.sleep(1)

    l.power_off()
