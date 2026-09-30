from .Config import Config, NumberWithUnit, Boolean, Selector
import logging


class ProtocolConfig(Config):
    """
    Control of patch-clamp protocols.
    """
    voltage_protocol = Boolean(default=True, doc='Run the Voltage Protocol automatically')
    current_protocol = Boolean(default=True, doc='Run the Current Protocol automatically')
    holding_protocol = Boolean(default=False, doc='Run the Holding Protocol automatically')
    voltage_sweep_protocol = Boolean(default=False, doc='Run the Voltage Sweep Protocol automatically')
    opto_random_wavelength_protocol = Boolean(default=False, doc='Run randomized wavelength optogenetic protocol')
    opto_random_power_protocol = Boolean(default=False, doc='Run randomized power optogenetic protocol')
    enable_neutralization_capacitance = Boolean(default=True, doc='Enable pipette neutralization capacitance for current clamp')
    enable_bridge_balance = Boolean(default=True, doc='Enable auto bridge balance for current clamp')

    custom_cclamp_protocol = Boolean(
        default=False,
        doc='Customize current-step pulse amplitudes (start/end/step only)'
    )
    cclamp_step = NumberWithUnit(30, bounds=(0, 3000), doc='Step Current', unit='pA', magnitude=1)
    cclamp_start = NumberWithUnit(-500, bounds=(-30000, 0), doc='Start Current', unit='pA', magnitude=1)
    cclamp_end = NumberWithUnit(500, bounds=(0, 30000), doc='End Current', unit='pA', magnitude=1)
    cclamp_hold = NumberWithUnit(-20, bounds=(-400, 400), doc='Holding Current (always applied)', unit='pA', magnitude=1)
    cclamp_recording_time_ms = NumberWithUnit(500, bounds=(50, 5000), doc='Current protocol step duration', unit='ms', magnitude=1)
    cclamp_duty_cycle = NumberWithUnit(0.5, bounds=(0.05, 0.95), doc='Current protocol duty cycle', unit='%', magnitude=0.01)
    hclamp_duration = NumberWithUnit(30, bounds=(0, 600), doc='Holding Protocol Duration', unit='s')

    vclamp_start = NumberWithUnit(-110e-3, bounds=(-200e-3, 0), doc='Voltage-clamp sweep start', unit='mV', magnitude=1e-3)
    vclamp_step = NumberWithUnit(20e-3, bounds=(0, 100e-3), doc='Voltage-clamp sweep step', unit='mV', magnitude=1e-3)
    vclamp_end = NumberWithUnit(50e-3, bounds=(0, 200e-3), doc='Voltage-clamp sweep end', unit='mV', magnitude=1e-3)
    vclamp_hold = NumberWithUnit(-70e-3, bounds=(-200e-3, 0), doc='Voltage-clamp holding potential (gigaseal/voltage protocols)', unit='mV', magnitude=1e-3)
    vclamp_sweep_hold = NumberWithUnit(-110e-3, bounds=(-200e-3, 0), doc='Voltage-clamp sweep holding potential', unit='mV', magnitude=1e-3)
    opto_stabilize_time = NumberWithUnit(1.0, bounds=(0, 60), doc='Optogenetic stabilize time', unit='s')
    opto_off_time = NumberWithUnit(1, bounds=(0.001, 60), doc='Optogenetic off time', unit='s')
    opto_on_time = NumberWithUnit(0.005, bounds=(0.001, 2), doc='Optogenetic on time', unit='s')
    opto_replicates = NumberWithUnit(1, bounds=(1, 10), doc='Optogenetic replicates', unit='x')
    opto_wavelength_power = NumberWithUnit(50, bounds=(0, 100), doc='Power for randomized wavelength protocol', unit='%')
    opto_power_wavelength = Selector(
        default="green",
        objects=["red", "green", "cyan", "uv", "blue", "infrared"],
        doc='Wavelength for randomized power protocol',
    )

    categories = [
        ('Protocols', ['voltage_protocol', 'current_protocol', 'holding_protocol', 'voltage_sweep_protocol',
                       'opto_random_wavelength_protocol', 'opto_random_power_protocol']),
        ('Compensation', ['enable_neutralization_capacitance', 'enable_bridge_balance']),
        ('Current Clamp', ['custom_cclamp_protocol', 'cclamp_step', 'cclamp_start', 'cclamp_end', 'cclamp_hold',
                           'cclamp_recording_time_ms', 'cclamp_duty_cycle']),
        ('Voltage Sweep', ['vclamp_start', 'vclamp_step', 'vclamp_end', 'vclamp_sweep_hold', 'vclamp_hold']),
        ('Holding Protocol', ['hclamp_duration']),
        ('Optogenetic Protocol', ['opto_stabilize_time', 'opto_off_time', 'opto_on_time', 'opto_replicates',
                                  'opto_wavelength_power', 'opto_power_wavelength']),
    ]

    logging.info("ProtocolConfig initialized successfully.")
