# Multi-pipette changes on `MultiNIDaq`: audit guide

**Scope.** This covers what is in the working tree now and differs from where `MultiNIDaq` branched, at `0cf69cd` ("more testing", 2026-10-30, branched from `big_update`). It covers multi-pipette work only. Code that was added and later removed within the branch isn't described.

- **Branch commits:** `d6adb86` (select pipettes at start), `84e403f` (multi protocol), `1a511da` (single-DAQ fix), `ec75795` (synchronous FakeDAQ), `d72c942` (cleanup + selector), `da5d5ce` (multi ephys + fake pressure).
- **Uncommitted:** the NiDAQ multi-channel rewrite (`DAQ.py`) and `Norwich_multiple_rig.json`.
- **Reproduce the diff:** `git diff 0cf69cd -- <file>`.

Line numbers are for the working tree as of this write-up.

---

## 1. The design in one pass

```
rig JSON (pipette_count, per-pipette device lists, one DAQ)
  → RigSelectorDialog: user picks how many pipettes (first N)
  → RigConfigManager.build_devices
       per-pipette slots → {"pipette_0": dev, "pipette_1": dev, ...}
       ONE DAQ, constructed with pipette_keys=[...]
  → PipetteInterface: one CalibratedUnit per pipette key
  → AutoPatchInterface: one AutoPatcher per pipette key, each given that
       pipette's amplifier/pressure and a PipetteDAQView(daq, key)
  → "Run protocols": _run_protocol_matrix
       prepare_<protocol>() on every selected pipette (parallel threads)
       ONE daq.<protocol>(pipettes=[keys], **kwargs) call per DAQ step
       finish_<protocol>() on every selected pipette (parallel threads)
  → DAQ stores results per pipette (per-pipette state, keyed pipette_<n>)
  → GUI: one GraphInterface per pipette → side-by-side ephys panels and
       per-pipette protocol windows, all reading through the pipette's view
```

**Two keys, two layers:**
- **Pipette key `pipette_<n>`.** This is the public identity everywhere: devices, DAQ API arguments and results, per-pipette state and GUI. It is defined in one place, [pipette_keys.py](../patcherbot/utils/pipette_keys.py).
- **Channel name** (`ai0` on fake, `Dev1/ai17` on NI). Only the DAQ's hardware layer (`_readAnalogInput`) uses it. `DAQ.channel_map` (`{pipette_key: PipetteChannels(read, resp, cmd)}`) translates between the two.

**`pipettes=` convention.** Every DAQ protocol and `getDataFromSquareWave` take `pipettes=[keys]` and return `{key: result}`. `pipettes=None` means the *active* pipette only (see `for_pipette`). The live thread passes all keys explicitly.

---

## 2. Suggested audit order

| # | File | What to look at |
|---|------|-----------------|
| 1 | [pipette_keys.py](../patcherbot/utils/pipette_keys.py) | key format (new file, 28 lines) |
| 2 | [rig_config.py](../rig_setup/rig_config.py) | slot split, building per-pipette devices, DAQ given keys, validation, pressure calibration |
| 3 | rig configs | `fake_rig.json`, `moscow_multiple_rig.json`, `Norwich_multiple_rig.json` |
| 4 | [DAQ.py](../patcherbot/devices/amplifier/DAQ.py) base | channel model, per-pipette state, decorator, live thread, pause, membrane test, view |
| 5 | [DAQ.py](../patcherbot/devices/amplifier/DAQ.py) `NiDAQ` | constructor, task helpers, each protocol |
| 6 | [DAQ.py](../patcherbot/devices/amplifier/DAQ.py) `FakeDAQ` | replay clock, protocols |
| 7 | [controller/patch.py](../patcherbot/controller/patch.py) | prepare / DAQ steps / finish split |
| 8 | [interface/patch.py](../patcherbot/interface/patch.py) | per-pipette AutoPatchers, protocol matrix |
| 9 | [interface/pipettes.py](../patcherbot/interface/pipettes.py) | per-pipette calibrated units |
| 10 | [patch_gui.py](../patch_gui.py), [gui/graph.py](../patcherbot/gui/graph.py), [gui/patch.py](../patcherbot/gui/patch.py), [interface/graph.py](../patcherbot/interface/graph.py) | GUI wiring, ephys panels, protocol windows, checkbox grid |
| 11 | [rig_selector.py](../rig_setup/rig_selector.py) | per-pipette device editor, choosing the pipette count |
| 12 | Pressure, calibration, fake camera, logger | smaller supporting changes |
| 13 | tests, `.gitignore` | coverage and tracking |

---

## 3. Changes by area

### 3.1 Pipette keys: [pipette_keys.py](../patcherbot/utils/pipette_keys.py) (new)
- `pipette_key_for(i)` → `"pipette_i"`, `pipette_index(key)` → `i` or `None`, and `pipette_label(key)` → `"Pipette i"` (0-based).
- **Check:** every producer of a pipette key goes through `pipette_key_for`. The exceptions are in §4 (#15).

### 3.2 Rig configuration and builder: [rig_config.py](../rig_setup/rig_config.py)
- **Slot lists.** [`SHARED_DEVICE_SLOTS`](../rig_setup/rig_config.py#L47) and [`PIPETTE_DEVICE_SLOTS`](../rig_setup/rig_config.py#L59) (`pipette_controller`, `pipette_camera`, `pressure`, `amplifier`).
  - `DEVICE_SLOTS` lists the pipette slots first, so the DAQ is built after them.
  - The DAQ is a shared slot: exactly one per rig.
- **[`build_devices`](../rig_setup/rig_config.py#L593)**
  - Takes `active_pipette_count`, validated against the config's `pipette_count`.
  - A per-pipette slot given as a list is cut to the first N ([L634](../rig_setup/rig_config.py#L634)) and stored as `{pipette_key_for(i): device}` ([L652](../rig_setup/rig_config.py#L652)). The slot-specific name `f"{slot}_{i}"` is only used in log and error messages.
  - A list in a shared slot raises an error (with a hint for `daq`).
  - The DAQ gets `pipette_keys=self._selected_pipette_keys(...)` ([L662](../rig_setup/rig_config.py#L662)). It's passed only if the constructor names that parameter and the config doesn't set it.
  - [`_selected_pipette_keys`](../rig_setup/rig_config.py#L723) returns the pipette controller dict's keys, or `[pipette_key_for(0)]` for single-device configs.
- **Indexed refs.** `$ref:slot[i]` ([L828](../rig_setup/rig_config.py#L828)) resolves positionally into a per-pipette dict. It relies on dict insertion order, which matches `i` because keys are created in order.
- **[`_validate_config`](../rig_setup/rig_config.py#L900)**
  - With `pipette_count > 1`, per-pipette slots must be lists of exactly `pipette_count`. The amplifier may instead be one mapping, sharing the same control handle across pipettes.
  - A shared slot may never be a list.
- **Pressure calibration, [`_apply_pressure_calibration`](../rig_setup/rig_config.py#L1107):**
  - `devices.pressure` may be a dict or a list (a dict counts as entry 0).
  - Every non-fake entry gets the top-level `native_zero` / `native_per_mbar` / `reader_offset` → `sensor_offset` / `reader_scale` → `sensor_scale`.
  - An optional `pressure_calibrations[i]` dict overrides those for entry *i*.
  - `FakePressureController` entries are skipped. IBB controllers get no reader keys (as before).
  - The inline override check now also counts `pressure_calibrations` ([L1040](../rig_setup/rig_config.py#L1040)).
- **[CalibrationConfig.py](../patcherbot/configs/CalibrationConfig.py#L80):** the new `pressure_calibrations = List(default=[])`. It's deliberately not in `categories`, so the calibration editor doesn't show it.
- **Configs:**
  - **`fake_rig.json`:** `pipette_count: 3`, per-pipette lists of fake devices, one `FakeDAQ`.
  - **`moscow_multiple_rig.json`:** `pipette_count: 2`.
    - `pipette_controller` is COM3/COM4.
    - `pressure` is `[MoscowPressureController, FakePressureController]`.
    - `amplifier` is `[MultiClampChannel 1, MultiClampChannel 2]`.
    - `pipette_camera` is `[FakeCamera ×2]`.
    - Its single NiDAQ uses the legacy params, so pipette 1 shares pipette 0's channels.
  - **`Norwich_multiple_rig.json`** (new): `pipette_count: 2`.
    - The second manipulator port is `"COM??"`.
    - `pressure` is `[SerialPressureController, FakePressureController]`; one `MultiClampChannel` with `channel: 1` is shared by both pipettes.
    - The NiDAQ `channels` list retains separate read, response, and command mappings for both pipettes. Sharing the amplifier handle sends amplifier commands to channel 1 only.
  - **`setup_test_rig.json`:** reformatted; still a single NiDAQ.
- **Check:**
  - per-pipette lists line up by index across slots (entry *i* of each slot belongs to the same physical pipette);
  - the DAQ `channels` list is in the same order;
  - nothing else depends on the old slot-named keys.

### 3.3 DAQ core (base class): [DAQ.py](../patcherbot/devices/amplifier/DAQ.py)
- **Channel model:**
  - [`PipetteChannels`](../patcherbot/devices/amplifier/DAQ.py#L30) is a frozen `(read, resp, cmd)`;
  - [`build_channel_map`](../patcherbot/devices/amplifier/DAQ.py#L53) builds an ordered `{key: PipetteChannels}`. It raises on duplicate keys, on fewer channel sets than pipettes, and on any channel used twice. Extra channel sets are ignored (first N);
  - [`enumerated_channels(i)`](../patcherbot/devices/amplifier/DAQ.py#L83) is the fake default `ai{2i}/ai{2i+1}/ao{i}`.
- **Per-pipette state:**
  - [`_PER_PIPETTE_FIELDS`](../patcherbot/devices/amplifier/DAQ.py#L91) lists every attribute stored per pipette: cell mode, resistances, all protocol results, the opto queue (factory, so one deque each).
  - [`_PerPipetteAttribute`](../patcherbot/devices/amplifier/DAQ.py#L146) descriptors route reads and writes to the *active* pipette's dict.
  - [`for_pipette(key)`](../patcherbot/devices/amplifier/DAQ.py#L388) sets the active pipette **thread-locally**. Concurrent threads (GUI, live thread, prepare/finish workers) therefore don't interfere. Default: the first pipette.
  - Check: every attribute the GUI, controllers and phases read from a DAQ is in this list. I grep-checked this and they are.
- **[`multi_pipette_protocol`](../patcherbot/devices/amplifier/DAQ.py#L130):** the decorated body is `method(self, keys, ...)`. The wrapper resolves `pipettes=` via [`_resolve_pipettes`](../patcherbot/devices/amplifier/DAQ.py#L423) (None means the active pipette; dedupes; validates) and holds `_protocol_lock` (an RLock) for the whole protocol.
- **Reads:**
  - [`_read_pipettes`](../patcherbot/devices/amplifier/DAQ.py#L443) makes **one** `_readAnalogInput` call covering every listed pipette and returns `{key: (read, resp)}`;
  - [`_rows_to_channel_dict`](../patcherbot/devices/amplifier/DAQ.py#L434) turns a rows-in-task-order block into `{channel: samples}`;
  - [`getDataFromSquareWave`](../patcherbot/devices/amplifier/DAQ.py#L645) sends the wave and does one read under `_deviceLock`, then processes each pipette under `for_pipette` (cell mode and fit are per pipette).
- **Live thread, [`DAQAcquisitionThread.run`](../patcherbot/devices/amplifier/DAQ.py#L197):**
  - one cycle covers all pipettes, and packets are stored as `{key: packet}`;
  - per-pipette live values are written into each pipette's state;
  - it restarts the DAQ only when **every** pipette's fit failed 3 times in a row ([L238](../patcherbot/devices/amplifier/DAQ.py#L238)), so an empty headstage can't keep resetting the others;
  - [`get_last_acquisition(key)`](../patcherbot/devices/amplifier/DAQ.py#L508) returns one pipette's packet (default: active).
- **Pause handshake** (marked `REVIEW(pause)`):
  - the thread clears `_idle_evt` only while it may touch hardware ([L208](../patcherbot/devices/amplifier/DAQ.py#L208));
  - [`_park_live_thread`](../patcherbot/devices/amplifier/DAQ.py#L609) clears `_pause_evt` and waits for idle;
  - [`pause_acquisition`](../patcherbot/devices/amplifier/DAQ.py#L623) parks the thread and then stops the tasks;
  - [`resume_acquisition`](../patcherbot/devices/amplifier/DAQ.py#L636) restarts the tasks and sets `_pause_evt`.
- **Membrane test, shared by NiDAQ and FakeDAQ, [`getDataFromVoltageProtocol`](../patcherbot/devices/amplifier/DAQ.py#L732):**
  - it parks the live thread **without stopping the tasks**, then cycles `getDataFromSquareWave` over the pipettes still collecting, until each has 5 fits in 0–250 pF or `max_attempts`;
  - per pipette it stores the first acceptable trace, falling back to the first trace;
  - the step size comes from the [`_membrane_step_mV`](../patcherbot/devices/amplifier/DAQ.py#L796) hook (base: `amplitude × V_CLAMP_VOLT_PER_VOLT`; FakeDAQ measures the recorded command);
  - [`_store_membrane_test`](../patcherbot/devices/amplifier/DAQ.py#L800) also updates `latest*` and `totalResistance`;
  - it returns `{key: mean Cm or None}`.
- **[`PipetteDAQView`](../patcherbot/devices/amplifier/DAQ.py#L1234):** one pipette's handle on the shared DAQ, a reference rather than a copy.
  - Per-pipette fields read and write that pipette's state directly.
  - Callables are wrapped to run under `for_pipette(key)`, so a protocol called through a view covers only that pipette (`pipettes=None` → active).
  - The wrapper gets `__self__ = daq` ([L1266](../patcherbot/devices/amplifier/DAQ.py#L1266)) because `TaskInterface.execute` locates the owning `TaskController` through it.
  - Non-field attributes are set on the shared DAQ.
- **Removed vs baseline:** base attributes `latest_protocol_data`, `isRunningProtocol` and `equalizer`, and plain `self.cellMode` (now per pipette).

### 3.4 NiDAQ: [DAQ.py](../patcherbot/devices/amplifier/DAQ.py#L1282)
**Principle:** every pipette's AI channels go in **one** AI task (one sample clock) and every command channel in **one** AO task at the same rate, armed on the AI start trigger. One read returns all pipettes in the same samples. Baseline NiDAQ was strictly single-pipette.

- **[Constructor](../patcherbot/devices/amplifier/DAQ.py#L1295):**
  - `channels=[{"read","resp","cmd"}, ...]` with full physical names, one per pipette in rig order;
  - **or** the legacy six params, giving one channel set (extra pipettes share it, with a warning);
  - both forms, or neither, raise `ValueError`;
  - [`_check_single_task_devices`](../patcherbot/devices/amplifier/DAQ.py#L1368) warns if the AI channels span unsynchronized devices.
- **Task helpers:**
  - [`_start_trigger_terminal`](../patcherbot/devices/amplifier/DAQ.py#L1377) uses the first AI channel's device, with the cDAQ chassis rule `cDAQ1Mod1` → `/cDAQ1/ai/StartTrigger`;
  - [`_add_ai_channels`](../patcherbot/devices/amplifier/DAQ.py#L1383) / [`_add_ao_channels`](../patcherbot/devices/amplifier/DAQ.py#L1396) add channels in `channel_map` order (DIFF, ±10 V);
  - [`_new_ai_task`](../patcherbot/devices/amplifier/DAQ.py#L1403) / `_new_ao_task` record `_ai_task_keys` / `_ai_task_channels` / `_ao_task_keys`;
  - [`_ao_block`](../patcherbot/devices/amplifier/DAQ.py#L1416) takes one wave (broadcast) or `{key: wave}` (None/missing → zeros) and returns 1-D for one channel or channels × samples;
  - [`_split_rows`](../patcherbot/devices/amplifier/DAQ.py#L1432) gives `{key: [read, resp]}`;
  - [`_read_train`](../patcherbot/devices/amplifier/DAQ.py#L1448) waits for one train and reads every pipette in one call;
  - [`_close_tasks`](../patcherbot/devices/amplifier/DAQ.py#L1454) stops and closes, ignoring errors.
- **Live stream:**
  - [`_setupAcquisition`](../patcherbot/devices/amplifier/DAQ.py#L1487) is LabVIEW-ordered (AO armed, then AI start), with all channels and the same test pulse on every AO channel;
  - [`_readAnalogInput`](../patcherbot/devices/amplifier/DAQ.py#L1672) maps rows by the running task's channel order and raises if a requested channel isn't in it;
  - [`_sendSquareWave`](../patcherbot/devices/amplifier/DAQ.py#L1690) has the same three cases as before, with every write going through `_ao_block`.
- **Protocol task setup:** `_setupAcquisitionCurrent` / `_setupAcquisitionVoltage` take `keys=`. `_sendSquareWaveCurrent` accepts a per-pipette `{key: amplitude or None}`.
- **Protocols** (all `@multi_pipette_protocol`, results stored per pipette under `for_pipette`, per-pipette return shapes unchanged):
  - **[`getLeakSubtraction`](../patcherbot/devices/amplifier/DAQ.py#L1848) / [`getVoltageClampSweep`](../patcherbot/devices/amplifier/DAQ.py#L2029):** one continuous AI + non-regenerating AO pair for all keys. Each queued train is the same wave on every pipette. Each `_read_train` is split per pipette, and `_squareWaveProcessor` still runs per pipette with its result discarded (as in baseline).
  - **[`getDataFromCurrentProtocol`](../patcherbot/devices/amplifier/DAQ.py#L2160):** a per-pipette step list from [`_current_pulses`](../patcherbot/devices/amplifier/DAQ.py#L2133), using each pipette's `voltageMembraneCapacitance`.
    - A pipette with no usable capacitance is **skipped**: error log, `([], [], 0)`, not in the tasks.
    - In train *i* each AO row gets its own step; finished pipettes get 0 V.
    - The number of trains equals the longest step list.
  - **[`getDataFromHoldingProtocol`](../patcherbot/devices/amplifier/DAQ.py#L2261):** one finite AI task for all keys and one read.
  - **[`getDataFromOptogeneticProtocol`](../patcherbot/devices/amplifier/DAQ.py#L2316):** one finite AI task per "on" step covering all keys, so the laser fires **once** per step for every pipette. Each pipette gets its own trace list and its own opto queue entry.
  - **Pipettes not in `keys`** get no channels in protocol tasks, so their command line is 0 V. The live stream is paused for all pipettes during any protocol.
- **Check:**
  - every AO write's row order equals `_ao_task_keys`;
  - every split uses the same key order the task was built with;
  - every protocol calls `resume_acquisition` on every exit path (see §4 #1).

### 3.5 FakeDAQ: [DAQ.py](../patcherbot/devices/amplifier/DAQ.py#L2560)
- **[`__init__`](../patcherbot/devices/amplifier/DAQ.py#L2572):** takes `pipette_keys` and optional `channels`, defaulting to `enumerated_channels`, and maps each AI channel to `(pipette index, row)`.
- **[`_read_recorded`](../patcherbot/devices/amplifier/DAQ.py#L2668):** each read advances **one** shared replay tick for every channel, like one hardware task. Pipette *i* replays trace (tick + i), and channels are trimmed to a common length.
- **Protocols:** leak, sweep, current, holding — each step is one `_read_pipettes` over all keys. The fake current protocol supports different step counts per pipette.
- **Data file:** `FakDAQ_graph_recording.csv` is a rig-recorder capture taken before a seal, so fake Ra/Rm/Cm are `None` and cell-mode R is NaN (known).
- `_membrane_step_mV` is overridden at [L2993](../patcherbot/devices/amplifier/DAQ.py#L2993).
- `ArduinoDAQ` takes `pipette_keys` but remains single-pipette, with no protocols.

### 3.6 Controller: [controller/patch.py](../patcherbot/controller/patch.py)
- **The split.** Each protocol is split into `prepare_<name>()` (amplifier setup; returns `[(daq_method, kwargs)]`) and `finish_<name>()` (cleanup). Protocols: voltage [L992](../patcherbot/controller/patch.py#L992), sweep [L1029](../patcherbot/controller/patch.py#L1029), current [L1061](../patcherbot/controller/patch.py#L1061), holding [L1142](../patcherbot/controller/patch.py#L1142), opto [L1173](../patcherbot/controller/patch.py#L1173).
  - `run_<name>()` = [`_run_split_protocol`](../patcherbot/controller/patch.py#L982) (prepare → [`_run_daq_steps`](../patcherbot/controller/patch.py#L972) → finish in `finally`).
  - `_run_daq_steps` calls the DAQ through this pipette's view and picks `[pipette_key]` from the result.
- **Cell mode** is now set by each prepare/finish (voltage, sweep and holding: True; current: False, then True in finish). `run_protocols` no longer toggles it.
- **Opto:** `prepare_optogenetic_protocol` returns one DAQ step per enabled opto run (laser object, steps and rate in kwargs).
- **[`run_protocol_matrix`](../patcherbot/controller/patch.py#L890):** a thin TaskController wrapper so the interface's matrix runs as a controller task (abort/success hooks).

### 3.7 Interface: [interface/patch.py](../patcherbot/interface/patch.py)
- **Per-pipette devices ([L55](../patcherbot/interface/patch.py#L55)):**
  - `amplifiers` / `pressures` are dicts keyed `pipette_<n>`; a single device becomes `{pipette_0: dev}`. `self.amplifier` / `self.pressure` are the first entry.
  - [L68](../patcherbot/interface/patch.py#L68): one `PipetteDAQView` and one `AutoPatcher` per `calibrated_units` key.
  - Devices are paired by exact key in [`_device_for_pipette`](../patcherbot/interface/patch.py#L124): a single device serves all pipettes, and a missing key with several devices raises `KeyError`.
  - [`_daq_key_for_pipette`](../patcherbot/interface/patch.py#L110): exact match, otherwise share the DAQ's first channel set, with a warning.
- **`current_autopatcher`** is the first pipette's AutoPatcher.
- **`protocol_selections[key][protocol]`:** seeded from the protocol config, then edited by the checkbox grid via [`set_protocol_selection`](../patcherbot/interface/patch.py#L132).
- **[`_PROTOCOL_RUNNERS`](../patcherbot/interface/patch.py#L101):** the single ordered list of selection names → prepare/finish suffix. Both opto names map to `optogenetic_protocol`.
- **[`_run_protocol_matrix`](../patcherbot/interface/patch.py#L193):** for each protocol in order, with the pipettes selected for it:
  1. call [`_prepare_protocol`](../patcherbot/interface/patch.py#L158) on every pipette in a `ThreadPoolExecutor` ([L208](../patcherbot/interface/patch.py#L208));
  2. [`_merge_protocol_steps`](../patcherbot/interface/patch.py#L173): step methods must match across pipettes, otherwise `ValueError`. Kwargs should match (shared config); a mismatch warns and uses the first pipette's. Opto uses the first pipette's randomized sequence for everyone;
  3. one `self.daq.<method>(pipettes=daq_keys, **kwargs)` per step (`daq_keys` deduped, so shared-channel pipettes become one key);
  4. [`_finish_protocol`](../patcherbot/interface/patch.py#L169) on every pipette in parallel, in `finally`.
- **`run_protocols` command ([L414](../patcherbot/interface/patch.py#L414)):** saves one cell-metadata entry, with hold flags from "any pipette selected" ([L254](../patcherbot/interface/patch.py#L254)), then executes the matrix.

### 3.8 PipetteInterface: [interface/pipettes.py](../patcherbot/interface/pipettes.py)
- [`pipette_units`](../patcherbot/interface/pipettes.py#L38): a dict of all selected units.
- [`calibrated_units`](../patcherbot/interface/pipettes.py#L64): one `CalibratedUnit` per key, all sharing the stage, microscope, camera and calibration config. The first is the existing `calibrated_unit`.
- [`get_pipette_positions`](../patcherbot/interface/pipettes.py#L113).
- Everything else (calibration routines, Observer, movement recording) still uses the primary unit.

### 3.9 GUI
- **[patch_gui.py](../patch_gui.py):**
  - passes the selected count into `build_devices` ([L44](../patch_gui.py#L44));
  - FakeCalCamera gets all manipulators;
  - the pipette camera is the primary one only;
  - one `GraphInterface` per pipette, built from the AutoPatcher's own amplifier and pressure plus its DAQ view ([L109](../patch_gui.py#L109)) and labelled `pipette_label(key)`;
  - one set of 6 protocol windows per pipette ([L133](../patch_gui.py#L133)).
- **Ephys window, [gui/graph.py](../patcherbot/gui/graph.py):**
  - [`PipetteEphysPanel`](../patcherbot/gui/graph.py#L615) is one column: 4 plots, its own deques and its own `FileLogger`. Its [`refresh`](../patcherbot/gui/graph.py#L698) fetches and plots one pipette. Size policy is Ignored, so columns share the window width ([L668](../patcherbot/gui/graph.py#L668)).
  - [`EPhysGraph`](../patcherbot/gui/graph.py#L748) lays the panels out left to right (pipette 0 leftmost) above one control bar.
  - The "Control:" picker ([L811](../patcherbot/gui/graph.py#L811), shown with more than one pipette) picks which pipette the bar acts on: cell mode, pressure, zap, Ra/Rm/Cm and the noise check. [`_select_pipette`](../patcherbot/gui/graph.py#L957) carries the shared laser cache over via `adopt_laser_state`.
  - One timer: [`update_plot`](../patcherbot/gui/graph.py#L985) refreshes all panels, then [`_update_bar`](../patcherbot/gui/graph.py#L995).
  - **Recorder:** pipette 0 writes `graph_recording.csv` (format unchanged); the others write `graph_recording_pipette_<n>.csv` ([L792](../patcherbot/gui/graph.py#L792)).
  - **Simulated pressure:** "(simulated)" in the bar and "sim. pressure" in the panel title.
- **Protocol windows:** [`ProtocolGraph`](../patcherbot/gui/graph.py#L42) takes `pipette_name`. The title becomes "… - Pipette n", and the EPhysLogger filename gets the suffix `_pipette_n` ([L90](../patcherbot/gui/graph.py#L90)). Opto windows pop entries from their pipette's own queue through the view.
- **[gui/patch.py](../patcherbot/gui/patch.py#L105):** the Protocols tab now has a "Run Protocols" grid (rows: protocols; columns: pipettes) above the parameter editor. The six protocol flags were removed from `ProtocolConfig.categories`, so they're no longer in the editor.
- **[interface/graph.py](../patcherbot/interface/graph.py):**
  - [`get_last_data(pipette_key=None)`](../patcherbot/interface/graph.py#L66) returns the view's pipette packet;
  - [`pressure_is_simulated`](../patcherbot/interface/graph.py#L46);
  - [`adopt_laser_state`](../patcherbot/interface/graph.py#L132).
- **[EPhysLogger.py](../patcherbot/utils/EPhysLogger.py#L103):** the opto basename uses `ephys_filename`, which now carries the pipette suffix.

### 3.10 Rig selector: [rig_selector.py](../rig_setup/rig_selector.py)
- **Rig editor (`RigBuilderDialog`):**
  - shows one group of per-pipette device rows per pipette (`pipette_count` from the config, [`_rebuild_pipette_sections`](../rig_setup/rig_selector.py#L617));
  - caches and restores rows;
  - saves `pipette_count`, with lists for per-pipette slots and a single shared DAQ.
- **Start-up dialog (`RigSelectorDialog`):** a combo of `1..pipette_count` ([L860](../rig_setup/rig_selector.py#L860)) → `selected_pipette_count` ([L925](../rig_setup/rig_selector.py#L925)).

### 3.11 Pressure, fake camera and detector
- **Simulated flag:** `PressureController.simulated = False` and `FakePressureController.simulated = True` ([BasePressureController.py#L50](../patcherbot/devices/pressurecontroller/BasePressureController.py#L50)). Fake controllers can be listed per pipette on real rigs.
- **Fake camera:** [`FakeCalCamera.set_pipette_manipulators`](../patcherbot/devices/camera/FakeCalCamera.py#L46) accepts a dict, list or single manipulator and draws every pipette. [`get_fake_pipette_tip_xy`](../patcherbot/devices/camera/FakeCalCamera.py#L96) takes a key, index or manipulator.
- **Fake detector:** [`FakePipetteDetector`](../patcherbot/devices/manipulator/helpers/PipetteCalHelper.py#L22) is used automatically when the camera can report fake tips ([L99](../patcherbot/devices/manipulator/helpers/PipetteCalHelper.py#L99)).

### 3.12 Tests and `.gitignore`
- **`.gitignore`:** ignores `/tests/*` except `test_fake_pipette_geometry.py` and `test_parallel_protocol_selection.py`.
- **Untracked tests** (need `git add -f`): `test_fake_daq_multichannel.py`, `test_ephys_side_by_side.py`, `test_multi_pipette_rig_config.py`, `test_nidaq_multichannel.py`.
  - NiDAQ tests run against a stub `nidaqmx` that records tasks, channels, writes and reads; there is no hardware coverage.
- **Pre-existing failure:** `test_fake_pipette_geometry.py::test_fake_pipette_approach_direction_comes_from_starting_geometry` fails (`FakePipette.geometry_direction` missing). This is unrelated to the multi-pipette work.

---

## 4. Flagged future problems (most important first)

1. **A failed protocol setup can leave the live test pulse dead.**
   - **Where:** NiDAQ `getLeakSubtraction` ([pause L1936](../patcherbot/devices/amplifier/DAQ.py#L1936) → [setup L1951](../patcherbot/devices/amplifier/DAQ.py#L1951) → [try L1963](../patcherbot/devices/amplifier/DAQ.py#L1963)), `getVoltageClampSweep` ([L2069](../patcherbot/devices/amplifier/DAQ.py#L2069)/[L2082](../patcherbot/devices/amplifier/DAQ.py#L2082)/[L2092](../patcherbot/devices/amplifier/DAQ.py#L2092)), `getDataFromCurrentProtocol` ([L2212](../patcherbot/devices/amplifier/DAQ.py#L2212)/[L2216](../patcherbot/devices/amplifier/DAQ.py#L2216)/[L2219](../patcherbot/devices/amplifier/DAQ.py#L2219)).
   - **Problem:** task setup happens after `pause_acquisition()` but before the `try/finally` that calls `resume_acquisition()`. If NI-DAQmx rejects the setup (bad or placeholder channel, resource reserved, rate too high), the live thread stays parked with its tasks closed, and every ephys panel freezes until restart.
   - **Fix:** move the setup inside the `try`.
2. **One failed `prepare_*` leaves the other amplifiers in protocol state.**
   - **Where:** [`_run_protocol_matrix`](../patcherbot/interface/patch.py#L208).
   - **Problem:** if any pipette's prepare raises, `list(executor.map(...))` raises before the `try`, so `finish_*` runs for **no** pipette, including those that already prepared. For example, one amplifier is left in current clamp with holding switched on.
   - **Fix:** collect per-pipette futures, and run finish for every pipette whose prepare succeeded.
3. **Parallel MultiClamp calls are untested.**
   - **Where:** prepare/finish run in threads on two `MultiClampChannel`s.
   - **Problem:** each channel has its own DLL message handle, but the class tracks `selected_device` with check-then-select (`needs_select`, `multiclamp.py` ~L72), and the DLL's thread safety is unknown.
   - **Fix:** verify on the rig, or serialize amplifier calls with a lock.
4. **The automated patching flow ignores the per-pipette checkboxes.**
   - **Where:** `AutoPatcher.run_protocols` ([controller/patch.py#L897](../patcherbot/controller/patch.py#L897)), called from the hunt → gigaseal → break-in flow ([L236](../patcherbot/controller/patch.py#L236), [L308](../patcherbot/controller/patch.py#L308)).
   - **Problem:** it reads `protocol_config.voltage_protocol` etc., which are now hidden from the parameter editor and only seeded from YAML. Only the "Run Protocols" button uses the checkbox grid via the matrix. Unchecking a protocol in the grid has no effect on automated runs.
5. **Only pipette 0 can be patched.**
   - **Problem:** `current_autopatcher` is fixed to the first pipette, and nothing switches it. So hunt/gigaseal/break-in, the standalone "Run Optogenetic Protocol" command ([L436](../patcherbot/interface/patch.py#L436)), the Observer/movement recording and cell metadata all act on pipette 0 only.
   - **Effect:** other pipettes take part only in the protocol matrix and the ephys display.
6. **Stopping the live thread while paused hangs.**
   - **Where:** [`stop_acquisition`](../patcherbot/devices/amplifier/DAQ.py#L600) joins a thread blocked on `_pause_evt.wait()`. This was pre-existing; the tests work around it.
   - **Fix:** in `stop()`, set `running=False` and then `_pause_evt.set()`.
7. **The rig editor can break a `channels`-based NiDAQ config.**
   - **Where:** the NiDAQ editor fields are the six legacy keys ([rig_config.py#L318](../rig_setup/rig_config.py#L318)), and [`get_params`](../rig_setup/rig_selector.py#L145) writes empty strings for blank required fields.
   - **Problem:** opening DAQ settings on `Norwich_multiple_rig.json` and pressing OK adds `readDev: ""` etc. alongside `channels`, and NiDAQ then raises "not both". Re-selecting the DAQ class ([`_apply_option_defaults`](../rig_setup/rig_selector.py#L384)) replaces params and drops `channels`.
8. **The Moscow multi config records pipette 1 on pipette 0's channels.**
   - **Problem:** `daq_keys` dedupes both pipettes to `pipette_0`, so both views read pipette 0's state. Both pipettes' protocol windows and log files show pipette 0's data, while pipette 1's amplifier is still switched by its prepare/finish.
   - **Fix:** give Moscow a `channels` list once its wiring is known.
9. **The Norwich multi config can't load with 2 pipettes yet.** Its `COM??` port and `..._TODO` channels make construction fail (intentional). With 1 pipette selected it builds, using entry 0 of each list.
   - **Wiring note:** in DIFF mode `aiN` uses `ai(N+8)` as its negative input, so `ai17`/`ai19` also occupy `ai25`/`ai27`.
10. **You can only run the first N pipettes.** Truncation keeps the first N ([L634](../rig_setup/rig_config.py#L634)), so you can't run pipette 1 alone, or 0 and 2.
11. **Gains are assumed identical.** The `C_CLAMP_*` / `V_CLAMP_*` gains are class constants, the same for every pipette. Different headstages or gain settings per MultiClamp channel would be scaled wrongly.
12. **Check the board's sample rate.** Total rate = rate × AI channels: 2 pipettes = 200 kS/s live, 400 kS/s in current clamp. Multiplexed boards (M/X-series) also skew channels within one sample by a few µs. On the live pulse, CPU time per cycle grows with pipettes in cell mode (one scipy fit each).
13. **Behaviour changes downstream code should tolerate:**
    - the membrane test falls back to an unfit first trace, where previously no data was stored;
    - a bad-capacitance pipette in NiDAQ current clamp returns `([], [], 0)`, so the current protocol window and logger see an empty sweep list for it.
14. **Opto randomization is shared.** Each pipette builds its own randomized sequence, but only the first pipette's is run. The others' stim logs show the sequence that was actually used (the shared `stim_used`), but the unused sequences are silently discarded.
15. **Label and key inconsistencies.**
    - The rig selector and editor say "Pipette 1/2" while the GUI, keys and files use 0-based "Pipette 0/1".
    - `PipetteInterface` ([L38](../patcherbot/interface/pipettes.py#L38)) and `FakeCalCamera` hard-code `"pipette_0"` / `f"pipette_{index}"` instead of `pipette_key_for`.
16. **Finite AI tasks open while the live tasks still exist.** NiDAQ holding and opto create their finite AI task while the live tasks are stopped but not closed. This is unchanged from baseline single-pipette behaviour, but now spans all channels; watch for "resource reserved" errors on some boards.
17. **The matrix pause isn't abortable.** The 0.25 s gap between DAQ steps ([L216](../patcherbot/interface/patch.py#L216)) uses `time.sleep`, which doesn't check for abort.
18. **Ephys recordings: one file per pipette.** Files are named by panel position, not key. Analysis code that globs `graph_recording*.csv` will now pick up the extra files.

---

## 5. Seen in the diff but not multi-pipette (skip for this audit)
- Docstring and formatting churn in `rig_config.py` and `rig_selector.py`.
- New optional editor fields (manipulator speed/accel/polling, cell-sorter params, Waynesboro and IBB pressure options).
- `load_config(load_overlays=...)`.
- Default fake-rig template fields.
- `FakDAQ_graph_recording.csv` (data file for FakeDAQ).
- `setup_test_rig.json` reformatting.
