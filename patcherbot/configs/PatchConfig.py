from .Config import Config, NumberWithUnit, Number, Boolean ,Selector
import logging

class PatchConfig(Config):
    '''
    Control of automatic patch clamp algorithm
    '''
    # Define parameters directly without try-except
    # --- Approach / resistance check ---
    pressure_near = NumberWithUnit(45, bounds=(0, 100), doc='Pressure during approach', unit='mbar')
    min_R = NumberWithUnit(2e6, bounds=(0, 1000e6), doc='Minimum normal resistance', unit='MΩ', magnitude=1e6)
    max_R = NumberWithUnit(25e6, bounds=(0, 1000e6), doc='Maximum normal resistance', unit='MΩ', magnitude=1e6)
    max_R_increase = NumberWithUnit(1e6, bounds=(0, 500e6), doc='Increase in resistance over time', unit='MΩ', magnitude=1e6)
    cell_distance = NumberWithUnit(20, bounds=(0, 100), doc='Initial distance above target cell', unit='μm') # 50 for Neurons, 20 for HEK cells
    max_locate_speed = NumberWithUnit(1000, bounds=(0, 10000), doc='Maximum speed for cell locating', unit='μm/s')
    approach_cell_max_shift = NumberWithUnit(256, bounds=(1, 10000), doc='Maximum cell-centering shift per axis during approach', unit='px')
    approach_pipette_speed = NumberWithUnit(20, bounds=(1, 10000), doc='Pipette-centering speed during approach', unit='μm/s')
    slice_start_distance = NumberWithUnit(75, bounds=(0, 100), doc='Initial distance above target cell in slice', unit='μm') # 20 um default
    max_distance = NumberWithUnit(30, bounds=(0, 100), doc='Maximum movement during hunt', unit='μm')
    hunt_limit_target_radius = Boolean(True, doc='Limit detection matching to the selected target radius')
    hunt_target_radius = NumberWithUnit(50, bounds=(1, 1000), unit='μm',
                                       doc='Maximum detection distance from the expected target when radius limiting is enabled')
    hunt_contact_distance = NumberWithUnit(5, bounds=(0.1, 100), unit='μm',
                                          doc='Maximum image-plane tip-to-cell distance for Adaptive contact success')
    hunt_search_margin = NumberWithUnit(5, bounds=(0, 100),
                                       doc='Extra microscope Search travel after preceding Spear descent', unit='\u03bcm')
    hunt_progress_cycles = Number(4, bounds=(1, 100),
                                  doc='Complete adaptive cycles between Z progress checks (whole number)')
    hunt_min_progress_percent = NumberWithUnit(25, bounds=(0, 100), unit='%',
                                               doc='Minimum pipette-to-cell Z gap reduction per cycle window')
    max_descent_speed = Number(-10,bounds=(-50,50),doc='Maximum descent speed for Neuron Hunting')
    max_clearing_speed = Number(-100,bounds=(-200,200),doc='Maximum speed for clearing Tissue during Neuron Hunting')
    use_centroid = Boolean(True, doc='Use centroid for pipette alignment during approach')
    track_cell = Boolean(False, doc='Track cell position during approach')
    tracking_mode = Selector(default='fast', objects=['fast', 'cellbody', 'hybrid'], doc='Cell tracking strategy during approach')
    track_max_fast_jump_px = Number(80, bounds=(1, 10000), doc='Maximum accepted fast-tracking jump in pixels')
    
    max_R_increase = NumberWithUnit(1e6, bounds=(0, 500e6), doc='Increase in resistance over time', unit='MΩ', magnitude=1e6)
    cell_R_increase = Number(0.300, bounds=(0, 1), doc='Cell detection resistance limit') # in MOhm
    
    gigaseal_R = Number(1000, bounds=(100, 20000), doc='Gigaseal resistance')  # in MOhm
    gigaseal_min_delta_R = Number(15, bounds=(0, 1000), doc='Minimum resistance increase to extend deadline') # in MOhm
    hold_switch = Number(12, bounds=(1, 1000), doc='Hold switch divisor: gigaseal_R / hold_switch')
    increase_slope_gate = Number(3000, bounds=(1, 100000), doc='Slope gate divisor to increase suction')
    constant_slope_gate = Number(10, bounds=(1, 100000), doc='Slope gate divisor to maintain suction')
    decrease_slope_gate = Number(5, bounds=(1, 100000), doc='Slope gate divisor to decrease suction')
    measurement_speed = NumberWithUnit(0.200, bounds=(0.001, 5), doc='Resistance sample interval for slope', unit='s')
    seal_min_time = NumberWithUnit(15, bounds=(0, 60), doc='Minimum time for seal', unit='s')
    seal_deadline = NumberWithUnit(150, bounds=(0, 300), doc='Maximum time for seal formation', unit='s')

    pressure_ramp_increment = NumberWithUnit(-5, bounds=(-100, 0), doc='Pressure ramp increment', unit='mbar')
    pressure_ramp_max = NumberWithUnit(-30, bounds=(-1000, 0), doc='Pressure ramp maximum', unit='mbar')

    Vramp_duration = NumberWithUnit(10, bounds=(0, 60), doc='Voltage ramp duration', unit='s')
    Vramp_amplitude = NumberWithUnit(-70e-3, bounds=(-200e-3, 0), doc='Holding Potential', unit='mV', magnitude=1e-3) # changed from -70 to -20 for HEK cells

    zap = Boolean(False, doc='Zap the cell to break the seal')
    pulse_pressure_break_in = NumberWithUnit(-345, bounds=(-1000, 0), doc='Pressure pulse for break-in', unit='mbar')
    pulse_pressure_duration = NumberWithUnit(1, bounds=(0, 5), doc='Duration of pressure pulse for break-in', unit='s')
    max_cell_R = NumberWithUnit(300e6, bounds=(0, 1000e6), doc='Maximum cell resistance', unit='MΩ', magnitude=1e6)
    max_access_R = NumberWithUnit(70, bounds=(0, 1000), doc='Maximum access resistance', unit='MΩ', magnitude=1)
    min_cell_C = NumberWithUnit(5e-12, bounds=(0, 1), doc='Minimum cell capacitance', unit='pF', magnitude=1e-12)

    cell_type_toggle = Boolean(default=False, doc='Toggle for automatic cell type protocol selection')
    cell_type = Selector(default='Plate',objects = ['Plate', 'Slice'], doc='Cell type for protocol selection')
    mode = Selector(default='Classic', objects=['Manual', 'Classic', 'Adaptive', 'Agent', 'Training'], doc='Mode for AutoPatch algorithm')
    auto_clean_pipette = Boolean(True, doc='Automatically clean pipette after attempt')
    constant_disturbance_speed = Number(100, bounds=(0.1, 1000), doc='Constant disturbance speed in -X (um/s)')
    lamp = Selector(default= '2', objects = ['1', '2', '3','4','5','6'], doc='default fluorescence cube slot')
    auto_capture_fluo = Boolean(False, doc='Capture fluorescence image on cell selection')
    categories = [
        ('Approach', ['min_R', 'max_R', 'pressure_near', 'cell_distance', 'slice_start_distance',
                      'max_locate_speed', 'approach_cell_max_shift', 'approach_pipette_speed',
                      'max_clearing_speed', 'use_centroid', 'track_cell',
                      'tracking_mode', 'track_max_fast_jump_px']),
        ('Hunt', ['max_distance', 'max_descent_speed', 'cell_R_increase', 'hunt_search_margin',
                  'hunt_progress_cycles', 'hunt_min_progress_percent', 'hunt_limit_target_radius',
                  'hunt_target_radius', 'hunt_contact_distance']),
        ('Sealing', ['pressure_sealing', 'gigaseal_R', 'gigaseal_min_delta_R', 'hold_switch', 'increase_slope_gate', 'constant_slope_gate', 'decrease_slope_gate', 'measurement_speed', 'Vramp_amplitude', 'seal_min_time', 'seal_deadline']),
        ('Break-in', ['zap', 'pressure_ramp_increment', 'pressure_ramp_max', 'pressure_ramp_duration','pulse_pressure_break_in','pulse_pressure_duration', 'max_cell_R','max_access_R','min_cell_C']),
        ('AutoPatching', ['cell_type_toggle','cell_type', 'mode','auto_clean_pipette']),
        ('Fluorescence', ['lamp', 'auto_capture_fluo'])  
    ]

    logging.getLogger().info("PatchConfig initialized successfully.")

