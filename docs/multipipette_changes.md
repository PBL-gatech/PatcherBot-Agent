# Multi-pipette changes on `MultiNIDaq`: audit guide

**Scope.** Everything on `MultiNIDaq` that differs from where it branched (`0cf69cd`, "more testing", 2026-10-30, from `big_update`), limited to multi-pipette work. Code added and later removed within the branch isn't described.

- **State audited:** commit `2214d8f` ("current protocol fix"), clean working tree. All earlier "uncommitted" work is now committed.
- **Branch commits:** `d6adb86` select pipettes at start · `84e403f` multi protocol · `1a511da` single-DAQ fix · `ec75795` synchronous FakeDAQ · `d72c942` cleanup + selector · `da5d5ce` multi ephys + fake pressure · `6d18f2b` NiDAQ multi-channel rewrite · `e8b0406`/`c95289f`/`86a4f3c`/`ad18f8c` Norwich COM ports, detector, config · `cd8b61c` pipette detector · `30f241c` `MultiClampAmplifier`, combined protocol windows, task setup inside `try` · `693ed16` numpy channel fix · `b45d0c6` shared MultiClamp DLL handle · `2214d8f` read/resp swap fix, parallel amplifier prepare, signal-path test.
- **Reproduce any change:** `git diff 0cf69cd -- <file>`.
- **Line numbers** are for `2214d8f`. If they have drifted, search for the function name given next to each link.
- **Refactor after `2214d8f`** (§11): `pipette_keys.py` became `multipipette_helper.py`, the multi-pipette bookkeeping moved out of `DAQ.py`, and the protocol matrix moved into a new controller (`MultiPipettePatcher`). §1, §5.1, §5.4, §5.8 and §8.1 are updated; other sections still name the `2214d8f` locations.

Sections 2 and 3 answer "is acquisition really parallel, and can it do a paired (synaptic) recording?". Sections 4–6 are the file-by-file audit. Section 7 lists the open problems. Section 8 checks the code against the layer rules (gui ↔ interface ↔ controller ↔ devices). Section 10 reviews the DAQ information flow (`DAQ.py`, `pipette_keys.py`): what each multi-pipette piece is for, where it should live, and what changes to run different protocols on different pipettes at once. Section 11 records the refactor that followed it.

---

## 1. The design in one pass

```
rig JSON (pipette_count, per-pipette device lists, ONE DAQ, amplifier = list OR one shared mapping)
  → RigSelectorDialog: user picks how many pipettes (always the first N)
  → RigConfigManager.build_devices
       per-pipette slots   → {"pipette_0": dev, "pipette_1": dev, ...}
       shared amplifier    → one MultiClampAmplifier (one headstage per pipette)
       ONE DAQ, constructed with pipette_keys=[...] and one channel set per pipette
  → PipetteInterface: one CalibratedUnit per pipette key
  → AutoPatchInterface → MultiPipettePatcher.from_devices (controller): one AutoPatcher
       per pipette key, each given that pipette's headstage (device_for_pipette →
       MultiClampAmplifier.for_pipette), pressure and daq.for_pipette(key) (a PipetteDAQView)
  → "Run Protocols" button: interface executes MultiPipettePatcher.run_protocol_matrix
       (with a copy of the checkbox selections), for each protocol IN A FIXED ORDER:
       prepare_<protocol>() for every pipette ticked for it (one thread per headstage)
       ONE daq.<method>(pipettes=[keys], **kwargs) call per DAQ step, all ticked pipettes in
          the same hardware tasks
       finish_<protocol>() for every ticked pipette (one thread per headstage)
  → DAQ stores results per pipette (per-pipette state keyed pipette_<n>)
  → GUI: side-by-side live panels (one per pipette); one combined window per protocol
       type (both pipettes, two colours); optogenetic windows per pipette
```

**Two keys, two layers:**
- **Pipette key `pipette_<n>`** (0-based) is the public identity everywhere: devices, DAQ arguments and results, per-pipette state, GUI and file names. It is defined in [multipipette_helper.py](../patcherbot/utils/multipipette_helper.py), together with the rest of the multi-pipette bookkeeping (§11).
- **Channel name** (`ai0` on fake, `Dev1/ai17` on NI) is used only by the DAQ hardware layer. `DAQ.channel_map` (`{pipette_key: PipetteChannels(read, resp, cmd)}`) translates. `read` = amplifier command monitor; `resp` = amplifier response (primary output); `cmd` = AO driving the amplifier's external command.

**`pipettes=` / `pipette_key=` convention.** Every DAQ protocol and `getDataFromSquareWave` take `pipettes=[keys]` and return `{key: result}`; single-pipette DAQ methods (`setCellMode`, `resistance`, `get_last_acquisition`, …) take `pipette_key=`. `None` means the primary pipette (`pipette_0`). Per-pipette state is `daq.pipette_state(key)`. The live thread and the matrix pass keys explicitly; `AutoPatcher.run_<protocol>()` goes through its `PipetteDAQView`, which fills in its own key, so it records that pipette alone.

---

## 2. Is acquisition actually parallel?

**Short answer.** Within one protocol call, yes: every ticked pipette is recorded in the same samples on one clock, and the code is truly parallel at that level. Across protocols, no. The matrix runs protocol types one after another, so different protocols on different pipettes never overlap in time. Amplifier setup now runs in one thread per headstage, but it is not part of the data path.

| Level | What the code does | Parallel? | Where to check |
|---|---|---|---|
| **Samples within one protocol call** | All requested pipettes' `read`+`resp` channels go into **one** AI task (one sample clock). All their `cmd` channels go into **one** AO task at the same rate, armed on `/<dev>/ai/StartTrigger`. Each read returns every pipette's rows from the same samples. | **Yes**, sample-synchronous. Remaining skew is the multiplexed ADC's inter-channel delay, a few µs (§3.4). | [`_new_ai_task`](../patcherbot/devices/amplifier/DAQ.py#L1406), [`_new_ao_task`](../patcherbot/devices/amplifier/DAQ.py#L1413), trigger [L1540](../patcherbot/devices/amplifier/DAQ.py#L1540) / [L1599](../patcherbot/devices/amplifier/DAQ.py#L1599) / [L1663](../patcherbot/devices/amplifier/DAQ.py#L1663), [`_read_train`](../patcherbot/devices/amplifier/DAQ.py#L1451), [`_split_rows`](../patcherbot/devices/amplifier/DAQ.py#L1435) |
| **Stimulus per pipette** | Leak/sweep: the **same** voltage wave goes to every pipette ([`_sendSquareWaveVoltage`](../patcherbot/devices/amplifier/DAQ.py#L1776) broadcasts). Current protocol: per-pipette amplitude per train ([`_sendSquareWaveCurrent`](../patcherbot/devices/amplifier/DAQ.py#L1754) with `{key: amp}`). Holding/opto: **no AO task at all**. | Partly. The plumbing for per-pipette waves exists ([`_ao_block`](../patcherbot/devices/amplifier/DAQ.py#L1419) accepts `{key: wave}`, with `None` meaning 0 V), but only the current protocol uses it. | as linked |
| **Different protocols on different pipettes** | [`_run_protocol_matrix`](../patcherbot/interface/patch.py#L234) loops over `_PROTOCOL_RUNNERS` in order (voltage → current → sweep → holding → opto). Each iteration acquires only the pipettes ticked for that protocol. Every DAQ protocol holds `_protocol_lock` ([`multi_pipette_protocol`](../patcherbot/devices/amplifier/DAQ.py#L130)). [`_merge_protocol_steps`](../patcherbot/interface/patch.py#L181) raises if two pipettes need different DAQ methods. | **No, serialized.** A pipette ticked only for "holding" is not recorded while another runs "current", and vice versa. | as linked |
| **Amplifier setup (prepare/finish)** | Since `2214d8f`, [`_run_amplifier_workers`](../patcherbot/interface/patch.py#L201) groups by headstage object, so the two headstages of the one MultiClamp prepare in **two threads**. Every DLL select+call is serialized by `MultiClampChannel.dll_lock` ([`needs_select`](../patcherbot/devices/amplifier/multiclamp.py#L73)). Only the `sleep`s between calls overlap. | Interleaved, not simultaneous. Not part of the data path. | as linked |
| **Live test pulse and display** | One [`DAQAcquisitionThread.run`](../patcherbot/devices/amplifier/DAQ.py#L197) cycle reads all pipettes in one read and stores one `{key: packet}` dict. Each GUI panel then fetches its own packet separately. The thread is parked during every protocol. | Data: yes. Display: no (§2.2). | [`PipetteEphysPanel.refresh`](../patcherbot/gui/graph.py#L767) |

**What `2214d8f` actually changed.** The NI acquisition was already one AI/AO task pair per protocol call; that came from `6d18f2b`. `2214d8f` fixed the swapped `read`/`resp` inputs in both Norwich configs. Before that fix the current protocol couldn't run, and the live edge detection (below) used the wrong channel. It also let the two headstages *prepare* in parallel threads. It did not change how data is acquired.

### 2.1 Evidence trail for the sample-level claim (audit these)
1. Live stream: [`_setupAcquisition`](../patcherbot/devices/amplifier/DAQ.py#L1490) creates the AI task over `self.pipette_keys` (all pipettes) and the AO task over all `cmd` channels, at one rate (`self._wave_rate`). AO is primed with `_ao_block(self.wave)` (the same test pulse on every channel), then started armed, then AI starts.
2. Protocol tasks: [`_setupAcquisitionCurrent`](../patcherbot/devices/amplifier/DAQ.py#L1564) and [`_setupAcquisitionVoltage`](../patcherbot/devices/amplifier/DAQ.py#L1607) do the same for `keys` only. Holding ([L2298](../patcherbot/devices/amplifier/DAQ.py#L2298)) and opto ([`_setup_optogenetic_ai_task`](../patcherbot/devices/amplifier/DAQ.py#L1824)) build one finite AI task over `keys`.
3. Reads: protocol tasks read sequentially from the start of acquisition (default `CURR_READ_POS`), one train at a time ([`_read_train`](../patcherbot/devices/amplifier/DAQ.py#L1451)). The live task reads the most recent `_wave_samples` ([L1519–L1521](../patcherbot/devices/amplifier/DAQ.py#L1519)). Either way it is one `task.read` for all channels.
4. Row order: `_split_rows` / `_rows_to_channel_dict` ([L434](../patcherbot/devices/amplifier/DAQ.py#L434)) rebuild per-pipette rows from the order the task was built in (`_ai_task_keys`). **Check:** every split passes the same `keys` the task was built with. Holding uses `_split_rows(raw, keys)` and opto uses `_read_optogenetic_ai(..., keys)`.
5. AO timing: AO uses its own onboard sample clock at the same rate as AI, started by the AI start trigger. Both clocks come from the board's one timebase, so they don't drift relative to each other. **Check:** every protocol uses one `rate` for both tasks (L1589/L1595, L1640/L1650).

### 2.2 Why the live graphs look out of sync during protocols
These are display effects. None of them means the samples themselves are out of sync.
1. **Frozen traces.** Every protocol parks the live thread: `pause_acquisition` or, for the membrane test, `_park_live_thread`. The panels keep re-plotting the **last** packet. The pressure plot keeps scrolling, and the resistance deque keeps appending that same stale value every 42 ms ([`refresh`](../patcherbot/gui/graph.py#L767)).
2. **Panels can show different cycles.** Each panel calls `get_last_data()` on its own ([L778](../patcherbot/gui/graph.py#L778)). The live loop runs with `interval=0` ([L1356](../patcherbot/devices/amplifier/DAQ.py#L1356)), so a new cycle can land between panel 0's fetch and panel 1's fetch. Fix: fetch `daq.get_last_acquisition_all()` ([L522](../patcherbot/devices/amplifier/DAQ.py#L522)) once per tick and hand each panel its packet.
3. **Each trace is cut around its own edge.** [`_squareWaveProcessor`](../patcherbot/devices/amplifier/DAQ.py#L980) finds the step from `argmax(gradient(read))` ([L990–L992](../patcherbot/devices/amplifier/DAQ.py#L990)) and starts each pipette's time axis at that pipette's own edge. With identical simultaneous commands the cuts coincide. They won't if one `read` channel is noisy, flat, wired wrong (as it was before `2214d8f`), or that headstage is in I-clamp.
4. **The recorder writes stale data.** While recording is enabled, each panel writes a row every tick ([L804](../patcherbot/gui/graph.py#L804)) with `datetime.now()`, even when the packet hasn't changed. During protocols, `graph_recording*.csv` therefore holds repeated copies of the last live packet with fresh timestamps. The two pipettes' files can also be one cycle apart (point 2).

---

## 3. Paired recording (presynaptic current steps, postsynaptic holding)

**Goal:** inject current into cell A (I-clamp) while recording cell B in voltage clamp at a holding potential, on the same clock, to detect a synaptic connection and measure its strength.

### 3.1 What blocks it today
1. **The matrix can't combine protocols.** Ticking A for "current" and B for "holding" gives two separate acquisitions, the current protocol and then the 60 s holding protocol. B isn't in the AI task while A is stimulated. Ticking both for "current" puts **both** in I-clamp ([`prepare_current_protocol`](../patcherbot/controller/patch.py#L1061)) and gives B its own capacitance-scaled steps.
2. **`_merge_protocol_steps` forbids different methods** per pipette ([L191–L195](../patcherbot/interface/patch.py#L191)).
3. **Each protocol applies one unit conversion to every pipette.** The current protocol scales every `resp` row as membrane voltage (`raw[1] / C_CLAMP_VOLT_PER_VOLT`, [L2254](../patcherbot/devices/amplifier/DAQ.py#L2254)). Holding scales every row as current (`V_CLAMP_VOLT_PER_AMP`, [L2316–L2320](../patcherbot/devices/amplifier/DAQ.py#L2316)). A mixed I-clamp/V-clamp pair needs per-pipette scaling chosen by each pipette's mode.
4. **Command lines outside the task hold stale values** (§7, P1). B's AO line would sit at whatever the live test pulse last output: 0 V, or +0.5 V (= +10 mV in V-clamp).
5. **Per-pipette files with no shared acquisition ID.** Each pipette's traces go to `<Protocol>_pipette_<n>` ([`_PipetteTraces`](../patcherbot/gui/graph.py#L57)). Pairing A and B later relies on the sample number and train index. Sample index *i* is simultaneous in both, but nothing records that explicitly.

### 3.2 What already exists and can be reused
- `_new_ai_task(keys)` / `_new_ao_task(keys)` put both pipettes in one task pair.
- `_ao_block({pre: wave, post: None})` gives a per-pipette AO row, with `None` meaning explicit 0 V.
- `_read_train` / `_split_rows` read both pipettes in one call.
- The queued, **non-regenerating** train pattern of [`getVoltageClampSweep`](../patcherbot/devices/amplifier/DAQ.py#L2034) (`DONT_ALLOW_REGENERATION`, [L1654](../patcherbot/devices/amplifier/DAQ.py#L1654)) gives deterministic train ordering. Prefer it to the current protocol's regeneration mode (§7, P2).
- Per-headstage amplifier control: `MultiClampAmplifier.for_pipette(key)` → `current_clamp()`, `voltage_clamp()`, `set_holding()`, `switch_holding()`, `auto_bridge_balance()`.
- `@multi_pipette_protocol`, per-pipette state via `for_pipette`, and `_run_amplifier_workers` for prepare/finish.

### 3.3 Shape of the new protocol (sketch, not a spec)
1. `NiDAQ.getDataFromPairedProtocol(keys, *, roles={pre_key: "stim", post_key: "hold"}, ...)`, decorated `@multi_pipette_protocol`.
2. `pause_acquisition()` → `_close_tasks()` → one continuous AI task with both pipettes and one AO task with both `cmd` channels, same rate, AO armed on the AI start trigger, `DONT_ALLOW_REGENERATION`. All of it goes inside the `try`, so the `finally` always runs `_close_tasks()` + `resume_acquisition()`.
3. AO rows: the pre row is the I-clamp stimulus (A/V via `C_CLAMP_AMP_PER_VOLT`). The post row is **explicit zeros**; B's holding potential comes from the MultiClamp holding, not the DAQ.
4. Readout per role: pre `resp` → Vm (`C_CLAMP_VOLT_PER_VOLT`); post `resp` → Im (`V_CLAMP_VOLT_PER_AMP`). Store both under each pipette's state, plus one shared record (train index, stimulus amplitude, sample rate) so the pair can be re-joined.
5. Controller: `prepare_paired_protocol(role)`. The pre side runs I-clamp, bridge balance and holding current; the post side runs V-clamp, `set_holding(V)` and `switch_holding(True)`. `finish_` returns both to V-clamp.
6. Orchestration belongs in the **controller** layer, not the interface (§8, L1–L2). Use a controller-layer runner that owns the autopatchers and the DAQ, runs role-specific prepares, makes the one shared DAQ call, and runs finishes. The interface only stores the per-pipette roles/selections and `execute`s that runner. The GUI gets the results through `GraphInterface` methods, not `graph_interface.daq` (§8, L4). The GUI needs one window with pre Vm and post Im on a shared time axis.
7. Test: extend `ModelCellDriver` in [test_nidaq_signal_path.py](../tests/test_nidaq_signal_path.py) with a coupling term from A's command to B's response. Assert that the latency and amplitude survive the pipeline, and that B's AO row is exactly 0.

### 3.4 Can the sampling rate be maintained?
- **Aggregate rate the board must sustain:** per-channel rate × 2 inputs × pipettes.
  - Live, membrane test, leak, sweep, holding and opto: 50 kS/s × 4 = **200 kS/s**.
  - Current protocol: 100 kS/s × 4 = **400 kS/s**. A paired protocol at 100 kS/s needs the same.
- **The board model isn't recorded anywhere in the repo.** `Dev1` exposes `ai17`–`ai23` in differential mode, so it is a 32-channel multiplexed multifunction board (M/X-series class).
  - 6259/6363-class boards: ~1 MS/s multichannel aggregate. Fine.
  - 6229-class boards: 250 kS/s aggregate. The 2-pipette current protocol would be **rejected by DAQmx at setup**. That gives an error, not silent decimation. The setup is now inside `try`, so the live stream recovers and the matrix reports the failure.
  - **Check the model in NI MAX.**
- **No silent rate change:** 50 kHz and 100 kHz divide the board timebase exactly, so DAQmx won't coerce them.
- **Inter-channel skew:** one ADC is multiplexed across 4 channels, so within one "sample" the channels are converted a few µs apart, bounded by the sample period (10–20 µs). That is negligible against ~1 ms synaptic latency, and correctable if needed: the order is `pre.read, pre.resp, post.read, post.resp` as built by `_add_ai_channels`. To minimise it, set `ai_task.timing.ai_conv_rate` to the board maximum.
- **Host throughput** (≤ 3.2 MB/s) and buffers (`input_buf_size` ≥ 10 s of samples) are not limiting. The live stream uses `OVERWRITE_UNREAD_SAMPLES`. Protocol tasks don't, so a stall over ~10 s would raise -200279 rather than corrupt data.
- **Amplifier bandwidth:** Commander's output Bessel filter (default 10 kHz) is below Nyquist at both rates. **Check:** set both channels identically.

### 3.5 Is MultiClamp in the way of the data?
- **Not in the data path.** Data flows analog: amplifier outputs → NI AI, and NI AO → amplifier external command. The DLL only sets amplifier state. Its calls are serialized (`dll_lock`) and happen in prepare/finish, before and after acquisition, never during a DAQ read.
- **It does decide what the numbers mean,** and the code never reads that back:
  - DAQ scaling constants are per class, not per pipette or mode ([NiDAQ L1293–L1296](../patcherbot/devices/amplifier/DAQ.py#L1293)): 20 mV/V and 400 pA/V command sensitivities, 2 mV/pA and 10 mV/mV output gains. Both Commander channels must match them: primary/secondary signal selection, output gain, external command sensitivity, and headstage feedback resistor.
  - The code sets only primary/secondary gain = 1 at start-up ([`_init_selected`](../patcherbot/devices/amplifier/multiclamp.py#L219)). Telegraphs are not used.
- **What is on the primary output depends on mode** (Im in V-clamp, Vm in I-clamp). This is why a mixed pair needs per-pipette, per-mode scaling (§3.1.3).
- **Commander auto routines** (auto fast/slow compensation, bridge balance, pipette offset) inject their own test signals. Since `2214d8f` the two headstages prepare in parallel threads, so channel 1's routine may still be running in Commander when channel 2 is selected. Whether these DLL calls block until the routine finishes is **unverified**. **Check on the rig,** or keep the two headstages sequential (group by `shared_amplifier` again).
- **Commander's own Seal Test / tuning generator** must be off on both channels, or it adds its pulse to the data. `start_patch` turns it off; the protocol prepares don't.

### 3.6 Is the NI DAQ in the way?
- **One AI timing engine and one AO timing engine per device.** Only one hardware-timed AI task (and one AO task) can run at a time. Two different protocols therefore can't run as two simultaneous tasks. The only way to run "different protocols at once" is **one** task holding every channel, with a per-channel AO waveform, which is what §3.3 does.
- **Stale AO level** on channels not in the current AO task (§7, P1). This is the main data-integrity problem for a postsynaptic cell.
- **Regeneration-mode train updates** in the current protocol (§7, P2). Cross-pipette sync holds either way, because both rows go in one `write`.
- **Differential pairs:** `aiN` uses `ai(N+8)` as its negative input, so `ai17/19/21/23` also occupy `ai25/27/29/31`. **Check** that those are wired to the amplifier output grounds and not reused.

### 3.7 Rig checks before trusting connectivity data
1. Board model and aggregate rate (§3.4).
2. Commander settings identical on both channels and matching the NiDAQ constants (§3.5).
3. **Crosstalk control:** run the paired protocol with B in the bath, no cell (or on a model cell). Any B response time-locked to A's steps at **zero latency** is electrical or capacitive crosstalk (pipette–pipette, headstage, or DAQ ghosting). A chemical synapse shows ~1 ms latency.
4. In each saved train, check the `read` (command-monitor) row against the intended step. This catches both the stale-AO and regeneration problems.
5. Measure B's command line while A's protocol runs (scope or `read` row). It should be 0 V.

---

## 4. Suggested audit order

| # | File | What to look at |
|---|------|-----------------|
| 1 | [multipipette_helper.py](../patcherbot/utils/multipipette_helper.py) (was `pipette_keys.py`) | keys, per-pipette state, `MultiPipetteDevice`, channel model, DAQ view (§11) |
| 2 | [rig_config.py](../rig_setup/rig_config.py) | slot split, per-pipette build, DAQ given keys, shared amplifier validation, numeric-list conversion, pressure calibration |
| 3 | rig configs | `fake_rig.json`, `moscow_multiple_rig.json`, `Norwich_multiple_rig.json`, `Norwich_rig.json` |
| 4 | [multiclamp.py](../patcherbot/devices/amplifier/multiclamp.py) | `dll_lock`, shared DLL handle, `MultiClampAmplifier` |
| 5 | [DAQ.py](../patcherbot/devices/amplifier/DAQ.py) base | channel model, per-pipette state, decorator, live thread, pause, membrane test, view |
| 6 | [DAQ.py](../patcherbot/devices/amplifier/DAQ.py) `NiDAQ` | constructor, task helpers, each protocol (§2.1 checklist) |
| 7 | [DAQ.py](../patcherbot/devices/amplifier/DAQ.py) `FakeDAQ` | replay clock, protocols |
| 8 | [controller/patch.py](../patcherbot/controller/patch.py) | prepare / DAQ steps / finish split |
| 9 | [interface/patch.py](../patcherbot/interface/patch.py) | per-pipette AutoPatchers, amplifier workers, protocol matrix |
| 10 | [interface/pipettes.py](../patcherbot/interface/pipettes.py) | per-pipette calibrated units |
| 11 | [patch_gui.py](../patch_gui.py), [gui/graph.py](../patcherbot/gui/graph.py), [gui/patch.py](../patcherbot/gui/patch.py), [interface/graph.py](../patcherbot/interface/graph.py) | GUI wiring, live panels, combined protocol windows, checkbox grid |
| 12 | [rig_selector.py](../rig_setup/rig_selector.py) | per-pipette editor, shared amplifier row, pipette count |
| 13 | Pressure, calibration, fake camera, logger | smaller supporting changes |
| 14 | tests, `.gitignore` | coverage and tracking |

---

## 5. Changes by area

### 5.1 Multi-pipette helper: [multipipette_helper.py](../patcherbot/utils/multipipette_helper.py) (new; was `pipette_keys.py`)
- **Keys:** `pipette_key_for(i)` → `"pipette_i"`; `pipette_index(key)` → `i` or `None`; `pipette_label(key)` → `"Pipette i"` (0-based); [`validate_pipette_keys`](../patcherbot/utils/multipipette_helper.py#L57) (default `[pipette_0]`, no duplicates).
- **[`device_for_pipette(devices, key)`](../patcherbot/utils/multipipette_helper.py#L73):** exact key, else the single device; then `device.for_pipette(key)` if the device has one (headstage for `MultiClampAmplifier`, view for the DAQ).
- **Per-pipette state:** [`PER_PIPETTE_FIELDS`](../patcherbot/utils/multipipette_helper.py#L96) (name → default; a callable default such as `deque` is called per pipette) and [`PipetteState`](../patcherbot/utils/multipipette_helper.py#L128), an attribute container that rejects undeclared names.
- **[`MultiPipetteDevice`](../patcherbot/utils/multipipette_helper.py#L159):** base class for one device serving several pipettes: `pipette_keys`, `primary_pipette_key`, `require_pipette`, [`resolve_pipettes`](../patcherbot/utils/multipipette_helper.py#L192) (None = primary; dedupes; validates), [`pipette_state(key)`](../patcherbot/utils/multipipette_helper.py#L202), abstract `for_pipette(key)`. `DAQ` and `MultiClampAmplifier` inherit it.
- **DAQ channel model:** [`PipetteChannels`](../patcherbot/utils/multipipette_helper.py#L219), [`build_channel_map`](../patcherbot/utils/multipipette_helper.py#L242), [`enumerated_channels`](../patcherbot/utils/multipipette_helper.py#L270), moved verbatim from `DAQ.py`.
- **DAQ method tags and view:** [`multi_pipette_protocol`](../patcherbot/utils/multipipette_helper.py#L278), [`takes_pipettes`](../patcherbot/utils/multipipette_helper.py#L294), [`per_pipette`](../patcherbot/utils/multipipette_helper.py#L303) and [`PipetteDAQView`](../patcherbot/utils/multipipette_helper.py#L322) (see §5.4).
- **Check:** every producer of a pipette key goes through `pipette_key_for`. The remaining exceptions are in §7 (P15).

### 5.2 Rig configuration and builder: [rig_config.py](../rig_setup/rig_config.py)
- **Slot lists:** [`SHARED_DEVICE_SLOTS`](../rig_setup/rig_config.py#L47) and [`PIPETTE_DEVICE_SLOTS`](../rig_setup/rig_config.py#L59) (`pipette_controller`, `pipette_camera`, `pressure`, `amplifier`). [`DEVICE_SLOTS`](../rig_setup/rig_config.py#L66) puts the pipette slots first, so the DAQ is built after them. The DAQ is shared: exactly one per rig.
- **[`build_devices`](../rig_setup/rig_config.py#L593):**
  - takes `active_pipette_count`, validated against `pipette_count`;
  - a per-pipette **list** is cut to the first N ([L634](../rig_setup/rig_config.py#L634)) and stored as `{pipette_key_for(i): device}` ([L652](../rig_setup/rig_config.py#L652));
  - a per-pipette **mapping** (only the amplifier may be one when `pipette_count > 1`) is built once as a single device ([L657](../rig_setup/rig_config.py#L657));
  - the DAQ gets `pipette_keys=self._selected_pipette_keys(...)` ([L662](../rig_setup/rig_config.py#L662)), passed only if the constructor names that parameter and the config doesn't set it;
  - [`_selected_pipette_keys`](../rig_setup/rig_config.py#L723) returns the pipette controller dict's keys, or `[pipette_key_for(0)]`.
- **Numeric lists become numpy float arrays:** [`_resolve_value`](../rig_setup/rig_config.py#L874) converts every flat numeric list. So `"channels": [1, 2]` arrives as `array([1., 2.])`. `MultiClampAmplifier` normalises this ([L796–L803](../patcherbot/devices/amplifier/multiclamp.py#L796)). Any other constructor that takes an int list must do the same (§7, P17).
- **[`_validate_config`](../rig_setup/rig_config.py#L900):** with `pipette_count > 1`, per-pipette slots must be lists of exactly `pipette_count`. The exception is that `amplifier` may be one mapping ([L931–L933](../rig_setup/rig_config.py#L931)), meaning one device serving every pipette. A shared slot may never be a list.
- **Indexed refs:** `$ref:slot[i]` resolves positionally into a per-pipette dict, relying on dict insertion order.
- **Pressure calibration, [`_apply_pressure_calibration`](../rig_setup/rig_config.py#L1109):** `devices.pressure` may be a dict or a list. Every non-fake entry gets the top-level `native_zero` / `native_per_mbar` / reader offset and scale. An optional `pressure_calibrations[i]` overrides entry *i* ([L1126](../rig_setup/rig_config.py#L1126)). Fake entries are skipped. The inline-override check counts `pressure_calibrations` ([L1042](../rig_setup/rig_config.py#L1042)). [CalibrationConfig.py L80](../patcherbot/configs/CalibrationConfig.py#L80) adds `pressure_calibrations`, deliberately not shown in the editor.
- **Configs:**
  - **`fake_rig.json`:** `pipette_count: 3`, fake devices per pipette, one `FakeDAQ`.
  - **`moscow_multiple_rig.json`:** `pipette_count: 2`; amplifier `[MultiClampChannel 1, MultiClampChannel 2]` (a list: two channel objects sharing the DLL handle); pressure `[Moscow, Fake]`. The NiDAQ uses the **legacy** six params, so pipette 1 shares pipette 0's channels (§7, P10).
  - **`Norwich_multiple_rig.json`:** `pipette_count: 2`.
    - Manipulators on COM8 / COM7, stage on COM12.
    - Pressure `[SerialPressureController, Fake]`.
    - Amplifier: one `MultiClampAmplifier` with `channels: [1, 2]` (pipette_0 → headstage 1, pipette_1 → headstage 2).
    - DAQ `channels`: pipette_0 `read Dev1/ai19, resp Dev1/ai17, cmd Dev1/ao0`; pipette_1 `read Dev1/ai23, resp Dev1/ai21, cmd Dev1/ao3`. The read/resp swap was fixed in `2214d8f`.
  - **`Norwich_rig.json`:** single pipette, legacy params; the same read/resp fix (`readChannel ai19`, `respChannel ai17`).
- **Check:**
  - entry *i* of every per-pipette list and of the DAQ `channels` list belongs to the same physical pipette;
  - the shared-amplifier `channels` order matches the pipette order;
  - `read` is wired to the command monitor and `resp` to the primary output on the rig.

### 5.3 MultiClamp: [multiclamp.py](../patcherbot/devices/amplifier/multiclamp.py)
- **[`needs_select`](../patcherbot/devices/amplifier/multiclamp.py#L73):** holds the class-wide `dll_lock` ([L185](../patcherbot/devices/amplifier/multiclamp.py#L185), an RLock) around "select this channel if it isn't selected, then call". The select and the command can't be split by another thread.
- **One shared DLL handle** ([L186–L217](../patcherbot/devices/amplifier/multiclamp.py#L186)). The first `MultiClampChannel` creates the message handler and runs `FindFirst/FindNext` on it. Later channels reuse both the handler and `all_devices`. The DLL remembers found Commander windows *per handler*, so a fresh handler per channel failed with "MultiClamp Commander is not open". [`close`](../patcherbot/devices/amplifier/multiclamp.py#L764) reference-counts and destroys the handler with the last channel, then clears `all_devices`.
- **[`_init_selected`](../patcherbot/devices/amplifier/multiclamp.py#L219):** selects the channel and sets primary/secondary gain to 1, under the lock.
- **[`MultiClampAmplifier`](../patcherbot/devices/amplifier/multiclamp.py#L782):** one 700B serving several pipettes.
  - `channels[n]` is pipette_n's headstage, normalised to ints.
  - Each headstage gets `.shared_amplifier = self`.
  - A failing headstage raises `RuntimeError("MultiClamp channel N: ...")` and closes those already built.
  - [`for_pipette(key)`](../patcherbot/devices/amplifier/multiclamp.py#L815) returns that pipette's `MultiClampChannel`.
- **Check:** every amplifier method a protocol uses is `@needs_select`. **Not decorated:** `set_bridge_balance` (L662), and `start_patch`/`stop_patch`, which call decorated methods. `set_bridge_balance` runs on whichever channel was last selected (§7, P9).

### 5.4 DAQ core (base class): [DAQ.py](../patcherbot/devices/amplifier/DAQ.py)
*Updated after the §11 refactor; links in this subsection point at the current file.*
- **`class DAQ(MultiPipetteDevice, TaskController)`** ([L179](../patcherbot/devices/amplifier/DAQ.py#L179)): keeps only what talks to the DAQ — `channel_map`, `ai_channels`/`ao_channels`, [`channels_for(key)`](../patcherbot/devices/amplifier/DAQ.py#L232), the reads, the tasks, the protocols. The channel model, state fields, decorator and view live in [multipipette_helper.py](../patcherbot/utils/multipipette_helper.py) (§5.1).
- **Per-pipette state** is one `PipetteState` per pipette, reached explicitly with `self.pipette_state(key)`. There is no "active pipette" any more: the thread-local, `_PerPipetteAttribute` descriptors and the module-level `setattr` loop are gone.
  - Single-pipette methods take `pipette_key=` (None = primary) and are tagged `@per_pipette`: `setCellMode`, `getCellMode`, `resistance`, `capacitance`, `accessResistance`, `get_last_acquisition`, `compute_noise_metrics`, `pop_optogenetic_entry`.
  - Analysis helpers that need per-pipette data take the key as an argument: `_squareWaveProcessor` (cell mode), `_getParamsfromCurrent` (writes `holding_current`), `_store_membrane_test`, `_current_pulses`.
  - **Check:** every attribute the GUI, controllers and phases read through a view is in `PER_PIPETTE_FIELDS` or is a tagged method; an unknown name on `PipetteState` raises `AttributeError`.
- **[`for_pipette(key)`](../patcherbot/devices/amplifier/DAQ.py#L240)** returns a `PipetteDAQView`. A key the DAQ has no channels for falls back to the primary pipette's channels with a warning (moved from the interface's `_daq_key_for_pipette`; keeps the legacy single-channel-set NiDAQ working, P10).
- **`@multi_pipette_protocol`** (helper): the body is `method(self, keys, ...)`; the wrapper resolves `pipettes=` with `resolve_pipettes` and holds `_protocol_lock` ([L214](../patcherbot/devices/amplifier/DAQ.py#L214)) for the whole call. One protocol call at a time; a call may record several pipettes simultaneously (§2, §10.5).
- **Reads:**
  - [`_read_pipettes`](../patcherbot/devices/amplifier/DAQ.py#L276) does **one** `_readAnalogInput` for every listed pipette.
  - [`getDataFromSquareWave`](../patcherbot/devices/amplifier/DAQ.py#L487) (`@takes_pipettes`) sends the wave and reads under `_deviceLock`, then processes each pipette with `pipette_key=key`.
- **Live thread, [`DAQAcquisitionThread.run`](../patcherbot/devices/amplifier/DAQ.py#L197):**
  - one cycle covers all pipettes and stores `{key: packet}`;
  - per-pipette live values go into each pipette's state;
  - it restarts the DAQ only when **every** pipette failed its fit 3 times in a row ([L272–L279](../patcherbot/devices/amplifier/DAQ.py#L272));
  - [`get_last_acquisition(key)`](../patcherbot/devices/amplifier/DAQ.py#L508) returns one pipette's packet, [`get_last_acquisition_all()`](../patcherbot/devices/amplifier/DAQ.py#L522) all of them from the same cycle.
- **Pause handshake** (marked `REVIEW(pause)`):
  - the thread clears `_idle_evt` only while touching hardware ([L217](../patcherbot/devices/amplifier/DAQ.py#L217));
  - [`_park_live_thread`](../patcherbot/devices/amplifier/DAQ.py#L609) clears `_pause_evt` and waits for idle;
  - [`pause_acquisition`](../patcherbot/devices/amplifier/DAQ.py#L623) parks, then **stops** (doesn't close) AI and then AO;
  - [`resume_acquisition`](../patcherbot/devices/amplifier/DAQ.py#L636) rebuilds the live tasks (`_restartAcquisition`) and sets `_pause_evt`.
- **Membrane test, shared by NiDAQ and FakeDAQ, [`getDataFromVoltageProtocol`](../patcherbot/devices/amplifier/DAQ.py#L732):**
  - parks the live thread **without stopping the tasks**, so every pipette's AO keeps the test pulse, ticked or not;
  - cycles `getDataFromSquareWave` over the pipettes still collecting, until each has 5 fits in 0–250 pF or `max_attempts`;
  - stores the first acceptable trace, else the first trace (`fit_ok` False);
  - returns `{key: mean Cm or None}`.
  - The step comes from [`_membrane_step_mV`](../patcherbot/devices/amplifier/DAQ.py#L796); [`_store_membrane_test`](../patcherbot/devices/amplifier/DAQ.py#L800) also updates `latest*` and `totalResistance`.
- **[`PipetteDAQView`](../patcherbot/utils/multipipette_helper.py#L322)** (now in the helper): one pipette's handle on the shared DAQ, a reference rather than a copy.
  - Per-pipette fields read and write that pipette's `PipetteState`.
  - `@multi_pipette_protocol` / `@takes_pipettes` methods get `pipettes=[key]`, `@per_pipette` methods get `pipette_key=key`, when the caller leaves them out or passes None ([`__getattr__`](../patcherbot/utils/multipipette_helper.py#L339)). Wrapped calls expose `__self__ = daq` so `TaskInterface.execute` finds the owner (L8, unchanged).
  - Untagged methods are the shared DAQ's bound methods; other attributes are set on the shared DAQ.
- **Removed vs baseline:** `latest_protocol_data`, `isRunningProtocol`, `equalizer`, and plain `self.cellMode` etc. (now `pipette_state(key).cellMode`).

### 5.5 NiDAQ: [DAQ.py](../patcherbot/devices/amplifier/DAQ.py#L1285)
**Principle:** every pipette in a call shares **one** AI task (one sample clock) and **one** AO task at the same rate, armed on the AI start trigger (§2.1). Baseline NiDAQ was single-pipette.
- **[Constructor](../patcherbot/devices/amplifier/DAQ.py#L1298):**
  - `channels=[{"read","resp","cmd"}, ...]` with full physical names, one per pipette, **or** the legacy six params (extra pipettes share one set, with a warning). Both forms, or neither, raise `ValueError`.
  - [`_check_single_task_devices`](../patcherbot/devices/amplifier/DAQ.py#L1371) warns if the AI channels span unsynchronised devices.
  - It then builds the live tasks and starts the live thread at 40 Hz / 50 kS/s / 25 ms, `interval=0`.
- **Task helpers:**
  - [`_start_trigger_terminal`](../patcherbot/devices/amplifier/DAQ.py#L1380): first AI channel's device, chassis rule for cDAQ.
  - [`_add_ai_channels`](../patcherbot/devices/amplifier/DAQ.py#L1386) / [`_add_ao_channels`](../patcherbot/devices/amplifier/DAQ.py#L1399): `channel_map` order, DIFF ±10 V.
  - [`_new_ai_task`](../patcherbot/devices/amplifier/DAQ.py#L1406) / [`_new_ao_task`](../patcherbot/devices/amplifier/DAQ.py#L1413) record `_ai_task_keys` / `_ai_task_channels` / `_ao_task_keys`.
  - [`_ao_block`](../patcherbot/devices/amplifier/DAQ.py#L1419): one wave (broadcast) or `{key: wave}` (None/missing → zeros).
  - [`_split_rows`](../patcherbot/devices/amplifier/DAQ.py#L1435) gives `{key: [read, resp]}`; [`_read_train`](../patcherbot/devices/amplifier/DAQ.py#L1451) polls until one train is available, then reads it; [`_close_tasks`](../patcherbot/devices/amplifier/DAQ.py#L1457) stops and closes, ignoring errors.
- **Live stream:**
  - [`_setupAcquisition`](../patcherbot/devices/amplifier/DAQ.py#L1490): continuous AI with `OVERWRITE_UNREAD_SAMPLES` and `MOST_RECENT_SAMPLE` minus one wave (each live read is the latest 25 ms, in no fixed phase); regenerating AO with the same test pulse on every channel.
  - [`_readAnalogInput`](../patcherbot/devices/amplifier/DAQ.py#L1675) maps rows by the running task's channel order and raises if a requested channel isn't in it.
  - [`_sendSquareWave`](../patcherbot/devices/amplifier/DAQ.py#L1693): unchanged wave → no-op; same length → in-place regenerating write; otherwise stop, re-time and restart in trigger order.
- **Protocols** (all `@multi_pipette_protocol`; results stored per pipette; per-pipette return shapes unchanged):
  - **[`getLeakSubtraction`](../patcherbot/devices/amplifier/DAQ.py#L1851) / [`getVoltageClampSweep`](../patcherbot/devices/amplifier/DAQ.py#L2034):**
    - `pause_acquisition` → `_close_tasks` → (inside `try`) continuous AI + **non-regenerating** AO for `keys`.
    - The first real train is queued behind a 0 V prime, then each next train is queued before reading the current one.
    - The **same** wave goes to every pipette; each `_read_train` is split per pipette.
  - **[`getDataFromCurrentProtocol`](../patcherbot/devices/amplifier/DAQ.py#L2167):**
    - per-pipette steps from [`_current_pulses`](../patcherbot/devices/amplifier/DAQ.py#L2140), scaled by that pipette's `voltageMembraneCapacitance` (−2…+10 pA/pF), or custom steps;
    - a pipette without usable capacitance is skipped (error log; result `([], array([]), 0)`); if **no** pipette is usable it raises before touching the hardware;
    - continuous AI + **regenerating** AO ([L1597](../patcherbot/devices/amplifier/DAQ.py#L1597)) for the usable pipettes;
    - in train *i* each AO row gets its own step, and pipettes that ran out of steps get 0 V;
    - 1 prime + max(step count) trains; responses scaled as Vm for every row ([L2254](../patcherbot/devices/amplifier/DAQ.py#L2254)).
  - **[`getDataFromHoldingProtocol`](../patcherbot/devices/amplifier/DAQ.py#L2274):** `pause_acquisition` (live tasks stopped, not closed), one finite **AI-only** task for `keys`, one read. Scaled as V-clamp current.
  - **[`getDataFromOptogeneticProtocol`](../patcherbot/devices/amplifier/DAQ.py#L2329):** one finite AI-only task per "on" step covering `keys`. The laser fires once per step for every pipette; its timing is software (`perf_counter` + spin, ms-level jitter relative to the samples). Each pipette gets its own traces and opto-queue entry.
  - **Pipettes not in `keys`** are not in the protocol tasks. Their command line is **not** forced to 0 V; it holds its last value (§7, P1). The live stream is paused for every pipette during any protocol.
- **Check:**
  - every AO write's row order equals `_ao_task_keys`;
  - every split uses the key order the task was built with;
  - every protocol's task setup sits inside the `try` whose `finally` calls `resume_acquisition` (now true for all five).

### 5.6 FakeDAQ: [DAQ.py](../patcherbot/devices/amplifier/DAQ.py#L2573)
- [`__init__`](../patcherbot/devices/amplifier/DAQ.py#L2585) takes `pipette_keys` and optional `channels` (default `enumerated_channels`) and maps each AI channel to `(pipette index, row)`.
- [`_read_recorded`](../patcherbot/devices/amplifier/DAQ.py#L2681): each read advances **one** shared replay tick for every channel, like one hardware task. Pipette *i* replays trace (tick + *i*).
- Protocols: leak, sweep, current and holding are each one `_read_pipettes` over all keys per step; the fake current protocol supports different step counts. `_setupAcquisitionCurrent/Voltage` don't take `keys`.
- `FakDAQ_graph_recording.csv` was recorded before a seal, so fake Ra/Rm/Cm are `None` (known).
- `_membrane_step_mV` is overridden at [L3006](../patcherbot/devices/amplifier/DAQ.py#L3006). `ArduinoDAQ` takes `pipette_keys` but stays single-pipette.

### 5.7 Controller: [controller/patch.py](../patcherbot/controller/patch.py)
- **The split:** each protocol is `prepare_<name>()` (amplifier setup; returns `[(daq_method, kwargs)]`) plus `finish_<name>()`.
  - Voltage [L992](../patcherbot/controller/patch.py#L992): V-clamp, auto fast/slow comp, holding → membrane test.
  - Sweep [L1029](../patcherbot/controller/patch.py#L1029): V-clamp, holding → leak then sweep.
  - Current [L1061](../patcherbot/controller/patch.py#L1061): V-clamp, auto comp, then I-clamp, optional neutralisation/bridge balance, holding current.
  - Holding [L1142](../patcherbot/controller/patch.py#L1142): V-clamp at `vclamp_hold`.
  - Opto [L1173](../patcherbot/controller/patch.py#L1173).
  - `run_<name>()` = [`_run_split_protocol`](../patcherbot/controller/patch.py#L982) (prepare → [`_run_daq_steps`](../patcherbot/controller/patch.py#L972) → finish in `finally`); this path covers this pipette alone.
- **Cell mode** is set by each prepare/finish (voltage, sweep, holding: True; current: False, then True in finish).
- **[`amplifier_group`](../patcherbot/controller/patch.py#L891):** property naming the object whose channel this pipette uses (its headstage), so `MultiPipettePatcher` can group prepares without reading `autopatcher.amplifier` (L5). The old `run_protocol_matrix(protocol_runner)` callback wrapper is gone (L2).
- [`run_protocols`](../patcherbot/controller/patch.py#L897) (the automated flow) still reads the `protocol_config` flags (§7, P5).

### 5.7a Multi-pipette controller: [controller/multi_patch.py](../patcherbot/controller/multi_patch.py) (new, §11)
- **[`MultiPipettePatcher`](../patcherbot/controller/multi_patch.py#L19)** (`TaskController`) owns one `AutoPatcher` per pipette and the shared DAQ.
- **[`from_devices(...)`](../patcherbot/controller/multi_patch.py#L38)** pairs each pipette with its devices: `device_for_pipette` for amplifier and pressure (a `MultiClampAmplifier` hands out the pipette's headstage), `daq.for_pipette(key)` for the DAQ view (L3).
- **[`PROTOCOL_NAMES`](../patcherbot/controller/multi_patch.py#L23):** the ordered selection names → prepare/finish suffix. Both opto names map to `optogenetic_protocol`.
- **[`_run_amplifier_workers`](../patcherbot/controller/multi_patch.py#L124):** groups pipettes by `autopatcher.amplifier_group` (one group per headstage) and runs the groups in a `ThreadPoolExecutor`. Returns `({key: result}, {key: exception})`; one failure doesn't stop the others.
- **[`run_protocol_matrix(selections)`](../patcherbot/controller/multi_patch.py#L156),** for each protocol in order:
  1. prepare every ticked pipette (`_run_amplifier_workers` + [`_prepare_protocol`](../patcherbot/controller/multi_patch.py#L89));
  2. if any prepare failed, raise, but still run step 4 for **every** ticked pipette;
  3. [`_merge_protocol_steps`](../patcherbot/controller/multi_patch.py#L104): methods must match across pipettes; differing kwargs warn and use the first pipette's; opto uses the first pipette's sequence;
  4. one `self.daq.<method>(pipettes=daq_keys, **kwargs)` per step, with an abortable `self.sleep(0.25)` between steps (P19 fixed); `daq_keys` come from each AutoPatcher's view (`autopatcher.daq.pipette_key`);
  5. in `finally`, [`_finish_protocol`](../patcherbot/controller/multi_patch.py#L100) for every ticked pipette, logging cleanup errors.
  - **Abort** is honoured between protocols and between DAQ steps. It is deliberately not forwarded to the AutoPatchers, so every finish (amplifier cleanup) still runs. (Before, the matrix ran as pipette 0's AutoPatcher task, so an abort could also interrupt pipette 0's prepare **and its finish**.)

### 5.8 Interface: [interface/patch.py](../patcherbot/interface/patch.py)
*Updated after the §11 refactor.*
- **Per-pipette devices:** the constructor normalises `amplifier` / `pressure` to `{pipette_key: device}` and hands them, with the DAQ and the calibrated units, to [`MultiPipettePatcher.from_devices`](../patcherbot/interface/patch.py#L67) (L3). `self.autopatchers` is the controller's dict; `self.amplifier` / `self.pressure` / `self.daq` stay as the primary devices for pre-branch code.
- `current_autopatcher` is fixed to the first pipette.
- `protocol_selections[key][protocol]` is seeded from the config (`MultiPipettePatcher.PROTOCOL_NAMES`, [L80](../patcherbot/interface/patch.py#L80)) and edited by the checkbox grid ([`set_protocol_selection`](../patcherbot/interface/patch.py#L94)).
- **[`run_protocols`](../patcherbot/interface/patch.py#L287) command:** one cell-metadata entry, with hold values from "any pipette ticked" (`_protocol_holding_parameters`), then `execute(multi_patcher.run_protocol_matrix, argument=<copy of protocol_selections>)` (L1, L2).
- Removed: `protocol_daQs`, `amplifiers`, `pressures`, `_device_for_pipette`, `_daq_key_for_pipette`, `_PROTOCOL_RUNNERS`, `_opto_params`, `_prepare_protocol`, `_finish_protocol`, `_merge_protocol_steps`, `_run_amplifier_workers`, `_run_protocol_matrix`.

### 5.9 PipetteInterface: [interface/pipettes.py](../patcherbot/interface/pipettes.py)
- `pipette_units` is a dict ([L38](../patcherbot/interface/pipettes.py#L38)). `calibrated_units` holds one `CalibratedUnit` per key, sharing stage, microscope, camera and calibration. [`get_pipette_positions`](../patcherbot/interface/pipettes.py#L113).
- Calibration routines, the Observer and movement recording still use the primary unit.

### 5.10 GUI
- **[patch_gui.py](../patch_gui.py):**
  - passes the selected count into `build_devices` ([L44](../patch_gui.py#L44));
  - one `GraphInterface` per pipette, from that AutoPatcher's headstage, pressure and DAQ view ([L109](../patch_gui.py#L109));
  - one `EPhysGraph` ([L121](../patch_gui.py#L121));
  - **one combined window** each for current, voltage (membrane test), leak subtraction and holding, showing every pipette ([L127](../patch_gui.py#L127));
  - optogenetic stim/wavelength windows **per pipette** ([L137](../patch_gui.py#L137)).
- **Protocol windows ([gui/graph.py](../patcherbot/gui/graph.py)):**
  - [`ProtocolGraph`](../patcherbot/gui/graph.py#L77) takes one `GraphInterface` or a `{label: GraphInterface}` dict. [`_PipetteTraces`](../patcherbot/gui/graph.py#L57) holds each pipette's data source, items and its own `EPhysLogger` (files stay per pipette: `<Protocol>_pipette_<n>`).
  - Pipette 0 is drawn in blue shades, pipette 1 in orange ([`PIPETTE_GRADIENTS`](../patcherbot/gui/graph.py#L44)), with a legend.
  - A new sample number clears the plot.
  - The voltage window marks unfit membrane tests ([`_show_fit_state`](../patcherbot/gui/graph.py#L319)).
  - Windows poll every 1.5 s.
- **Live window:**
  - [`PipetteEphysPanel`](../patcherbot/gui/graph.py#L684) is one column per pipette, with its own deques and recorder; [`refresh`](../patcherbot/gui/graph.py#L767) fetches that pipette's packet.
  - [`EPhysGraph`](../patcherbot/gui/graph.py#L817) lays panels out left to right.
  - The recorder writes `graph_recording.csv` for pipette 0 and `graph_recording_pipette_<n>.csv` for the others ([L861](../patcherbot/gui/graph.py#L861)).
  - The "Control:" picker ([L878](../patcherbot/gui/graph.py#L878)) chooses which pipette the bottom bar acts on. [`_select_pipette`](../patcherbot/gui/graph.py#L1026) carries the laser state over.
  - One 42 ms timer runs [`update_plot`](../patcherbot/gui/graph.py#L1054).
  - See §2.2 for the sync caveats.
- **[gui/patch.py](../patcherbot/gui/patch.py#L109):** a "Run Protocols" checkbox grid (rows: protocols; columns: pipettes). The six protocol flags were removed from the parameter editor.
- **[interface/graph.py](../patcherbot/interface/graph.py):** `get_last_data(pipette_key=None)` ([L66](../patcherbot/interface/graph.py#L66)), `pressure_is_simulated`, `adopt_laser_state`.

### 5.11 Rig selector: [rig_selector.py](../rig_setup/rig_selector.py)
- **Rig editor:**
  - one group of per-pipette rows per pipette ([`_rebuild_pipette_sections`](../rig_setup/rig_selector.py#L638)), labelled with `pipette_label` (0-based);
  - a config whose amplifier is one mapping loads as **shared** ([L571](../rig_setup/rig_selector.py#L571)): pipette 0's row is "Amplifier (shared)" and the other rows are hidden ([L629–L634](../rig_setup/rig_selector.py#L629));
  - saving writes one mapping ([L486](../rig_setup/rig_selector.py#L486)).
- **Start-up dialog:** the pipette count, `1..pipette_count` → `selected_pipette_count` ([L948](../rig_setup/rig_selector.py#L948)).

### 5.12 Pressure, fake camera, detector, logger
- `PressureController.simulated = False`, `FakePressureController.simulated = True` ([BasePressureController.py L50](../patcherbot/devices/pressurecontroller/BasePressureController.py#L50)).
- [`FakeCalCamera.set_pipette_manipulators`](../patcherbot/devices/camera/FakeCalCamera.py#L46) draws every pipette. [`FakePipetteDetector`](../patcherbot/devices/manipulator/helpers/PipetteCalHelper.py#L24) is used automatically when the camera can report fake tips.
- `EPhysLogger` names opto files from `ephys_filename`, which carries the pipette suffix.

### 5.13 Tests and `.gitignore`
- **Tracked** (`.gitignore` un-ignores them): `test_fake_pipette_geometry.py`, `test_parallel_protocol_selection.py`, `test_norwich_pipette_detector.py`, `test_shared_amplifier.py`, `test_nidaq_signal_path.py`.
- **Untracked** (need `git add -f`): `test_cell_queue_coordinator.py`, `test_pipette_calibration_per_pipette.py`, `test_pipette_detector_multiple.py`. The test files named in the previous version of this doc no longer exist.
- **How to run** (the `patcher_bot` conda env; `tests/` is not a package): `PYTHONPATH=. python tests/<file>.py`.
- **Status at `2214d8f` on the dev machine:**

  | Test file | Result |
  |---|---|
  | `test_shared_amplifier.py` | 18 OK |
  | `test_nidaq_signal_path.py` | 3 OK |
  | `test_norwich_pipette_detector.py` | 2 errors: `BoRigCombo.pt` missing on that machine |
  | `test_fake_pipette_geometry.py`, `test_parallel_protocol_selection.py` | pytest-style, and pytest isn't installed in that env, so **not run** |

- **What `test_nidaq_signal_path.py` covers:** a DAQmx double with a model cell on both headstages. It checks the live read, the membrane test on the shared stream, and the current-protocol task layout and waveforms.
- **Coverage gaps:**
  - The double treats a regenerating AO write as replacing the output instantly, and the current-protocol test mocks `_read_train`. Train/step labelling under real regeneration is untested (P2).
  - No test checks the AO level of channels outside a task (P1).
  - Nothing runs on hardware.

---

## 6. Resolved since the previous version of this doc
1. **Failed protocol setup killed the live stream.** Setup now sits inside the `try` in leak ([L1959](../patcherbot/devices/amplifier/DAQ.py#L1959)), sweep ([L2090](../patcherbot/devices/amplifier/DAQ.py#L2090)) and current ([L2230](../patcherbot/devices/amplifier/DAQ.py#L2230)). Holding and opto already had it.
2. **One failed `prepare_*` skipped every `finish_*`.** Finish now runs for every ticked pipette ([L261–L267](../patcherbot/interface/patch.py#L261)).
3. **Parallel MultiClamp calls were unguarded.** Fixed by `dll_lock` in `needs_select` plus one shared DLL handle. P8 has the remaining open question.
4. **Norwich multi config couldn't load.** COM ports and channels are filled in, the amplifier is a `MultiClampAmplifier`, and read/resp are fixed.
5. **MultiClampAmplifier failed on real hardware:** first "truth value of an array is ambiguous" (numpy channels), then "MultiClamp Commander is not open" (one DLL handle per channel). Both fixed.
6. **Rig selector said "Pipette 1/2".** It now uses 0-based `pipette_label`.

---

## 7. Open problems (most important first)

**For paired / simultaneous recording**

- **P1. Command lines not in the running AO task hold a stale voltage.**
  - **Where:** `pause_acquisition` stops (doesn't close) the live AO ([L631–L632](../patcherbot/devices/amplifier/DAQ.py#L631)). Leak, sweep and current then close it and build AO tasks for `keys` only. Holding and opto build no AO task.
  - **Problem:** an NI AO channel keeps its last generated value after its task stops or closes. The live wave is a 0 / 0.5 V square wave at 50 % duty ([`createSquareWave`](../patcherbot/devices/amplifier/DAQ.py#L848)), so each line sits at 0 V or +0.5 V at random. That is **+10 mV in V-clamp, or +200 pA in I-clamp**.
  - **Affects:** every pipette during the holding and opto protocols, and un-ticked pipettes during leak, sweep and current. This would bias a postsynaptic holding recording.
  - **Fix:** after stopping the live tasks, write 0 V to every `cmd` channel (an on-demand AO task). Better still, include every `cmd` channel in protocol AO tasks with explicit zeros.
- **P2. The current protocol's train-to-step labelling relies on regeneration-mode writes.**
  - **Where:** [`_setupAcquisitionCurrent`](../patcherbot/devices/amplifier/DAQ.py#L1564) uses `ALLOW_REGENERATION` with a one-train buffer. [`_sendSquareWaveCurrent`](../patcherbot/devices/amplifier/DAQ.py#L1754) rewrites that buffer while the task runs, one train ahead of the read.
  - **Problem:** with regeneration, how a mid-run write lines up with train boundaries depends on DAQmx write-position and FIFO behaviour. If it doesn't land on a boundary, a train can start with the old amplitude and end with the new one, or be labelled with the wrong step. Both pipettes' rows go in one write, so they always change together, and cross-pipette sync is unaffected. Untested (§5.13).
  - **Check:** compare each saved trace's `read` row with its labelled step.
  - **Fix:** use the non-regenerating queued pattern that leak and sweep use.
- **P3. Unit conversion is per protocol, not per pipette.** Gains are class constants ([L1293–L1296](../patcherbot/devices/amplifier/DAQ.py#L1293)), the same for every headstage, and every row of a protocol is scaled as one mode. Mixed I-clamp/V-clamp pairs, different feedback resistors, or different Commander gains per channel come out scaled wrongly. Nothing is read back from Commander.
- **P4. Board aggregate rate unverified.** The 2-pipette current protocol needs 400 kS/s aggregate (§3.4). Record the board model in the rig config or this doc.
- **P5. The automated patching flow ignores the checkbox grid.** [`AutoPatcher.run_protocols`](../patcherbot/controller/patch.py#L897), called from the hunt → gigaseal → break-in flow, reads `protocol_config.*_protocol` flags. Those are hidden from the editor and only seeded from YAML. Only the "Run Protocols" button uses the grid. That path also runs a single pipette through its view.
- **P6. Only pipette 0 can be patched.** `current_autopatcher` is fixed ([L85](../patcherbot/interface/patch.py#L85)). Hunt/gigaseal/break-in, the standalone opto command ([L482](../patcherbot/interface/patch.py#L482)), Observer/movement recording and cell metadata all act on pipette 0. Other pipettes take part only in the matrix and the live display.
- **P7. Live display and recorder desync** (§2.2): panels fetch separately; the recorder writes stale packets with fresh timestamps during protocols; traces are cut at each pipette's own detected edge.
- **P8. Parallel prepare on one MultiClamp.** Since `2214d8f`, headstages 1 and 2 prepare in two threads. Each DLL call is atomic, but Commander auto routines (fast/slow comp, bridge balance) may still be running when the other channel is selected. Verify on the rig, or go back to grouping by `shared_amplifier` (the pre-`2214d8f` behaviour, which ran one headstage after the other).
- **P9. `set_bridge_balance` lacks `@needs_select`** ([multiclamp.py L662](../patcherbot/devices/amplifier/multiclamp.py#L662)). It acts on whichever headstage was selected last. With parallel prepare that can be the other pipette's.

**Other**

- **P10. The Moscow multi config records pipette 1 on pipette 0's channels.** Its legacy NiDAQ params give one channel set, so `daq_keys` dedupes both pipettes to `pipette_0` and both views read pipette 0's state. Pipette 1's amplifier is still switched by its prepare/finish. **Fix:** a `channels` list.
- **P11. The rig editor can break a `channels`-based NiDAQ config.** The NiDAQ editor fields are the six legacy keys ([rig_config.py L318–L340](../rig_setup/rig_config.py#L318)), and [`get_params`](../rig_setup/rig_selector.py#L146) writes empty strings. Opening DAQ settings on `Norwich_multiple_rig.json` and pressing OK adds `readDev: ""` etc. next to `channels`, and NiDAQ then raises "not both". Re-selecting the class ([`_apply_option_defaults`](../rig_setup/rig_selector.py#L388)) drops `channels`.
- **P12. Stopping the live thread while it is paused hangs.** [`stop_acquisition`](../patcherbot/devices/amplifier/DAQ.py#L600) joins a thread blocked on `_pause_evt.wait()`. **Fix:** in `stop()`, set `running=False`, then `_pause_evt.set()`.
- **P13. Only the first N pipettes can run** ([L634](../rig_setup/rig_config.py#L634)). You can't run pipette 1 alone.
- **P14. Behaviour changes downstream code must tolerate:** the membrane test falls back to an unfit first trace (`fit_ok` False), and a skipped current-protocol pipette returns `([], array([]), 0)`.
- **P15. Hard-coded keys:** `PipetteInterface` ([L38](../patcherbot/interface/pipettes.py#L38)) and `FakeCalCamera` ([L53](../patcherbot/devices/camera/FakeCalCamera.py#L53), [L57](../patcherbot/devices/camera/FakeCalCamera.py#L57)) build `"pipette_0"` / `f"pipette_{index}"` instead of using `pipette_key_for`.
- **P16. Opto randomisation is shared.** Each pipette builds its own sequence, but only the first pipette's is run. The others are discarded; their logs show the sequence actually used.
- **P17. Numeric lists in rig JSON become float arrays** ([L874–L879](../rig_setup/rig_config.py#L874)). Any constructor doing `if not param:` or using the values as ints breaks.
- **P18. Finite AI tasks open while the live tasks are stopped but not closed** (holding, opto). This is baseline behaviour, now spanning all channels. Watch for "resource reserved" on some boards.
- ~~**P19. The 0.25 s gap between matrix DAQ steps is not abortable**~~ — fixed in §11: `MultiPipettePatcher` uses `self.sleep`.
- **P20. One recorder file per pipette, named by panel position.** Analysis that globs `graph_recording*.csv` picks up the extra files.

### 7.1 Scaling to four pipettes

**Verdict.** Synchronization is the part that scales best. Every pipette in a protocol call is one more pair of rows in the same AI task and one more row in the same AO task, so 4 pipettes are as sample-locked as 2 (§2.1). Nothing in the DAQ, key or state code assumes 2. What blocks 4 pipettes is **identifying two MultiClamp amplifiers**, the **DAQ's channel and rate budget**, and some 2-pipette assumptions in the config tooling and GUI.

**Hardware and MultiClamp**
- **Two 700Bs, two Commander instances.** A 700B has two headstage channels, so 4 pipettes means two amplifiers. Each 700B is run by its own Commander instance. The AxMultiClampMsg API enumerates every open instance as (serial, channel) pairs through `FindFirst/FindNext` ([`find_amplifiers`](../patcherbot/devices/amplifier/multiclamp.py#L335)) and selects one at a time. The shared handle and `dll_lock` already handle that. **Verify on the rig** that both Commander instances enumerate (expect four "Found a MultiClamp 700B" log lines).
- **S1. Selecting by serial number is broken.** With two amplifiers, `channel=1` alone matches two devices, so [`select_amplifier`](../patcherbot/devices/amplifier/multiclamp.py#L375) raises "2 devices identified" and the serial must be given. But [`_identify_amplifier`](../patcherbot/devices/amplifier/multiclamp.py#L105) stores `serial.value`, which is **bytes** (`b'00123456'`), while a serial written in rig JSON arrives as a `str`, or as an `int`/float if unquoted. The equality check at [L381–L383](../patcherbot/devices/amplifier/multiclamp.py#L381) never matches, so the result is "No device identified". **Fix:** normalise both sides to a decoded, stripped string.
- **S2. `MultiClampAmplifier` covers one amplifier.** It passes one `identification` to all of its channels ([L796](../patcherbot/devices/amplifier/multiclamp.py#L796)). `for_pipette` indexes by pipette number ([L815](../patcherbot/devices/amplifier/multiclamp.py#L815)), so a 2-channel instance can't serve `pipette_2`/`pipette_3`.
  - **Workable today** (once S1 is fixed): the per-pipette **list** form with 4 `MultiClampChannel` entries, each `{"serial": ..., "channel": 1|2}`. `_validate_config` accepts this.
  - **Cleaner:** let `MultiClampAmplifier` take `channels=[{"serial": A, "channel": 1}, ...]`.
- **S3. One lock for both amplifiers.** `dll_lock` is class-wide, so the two amplifiers' prepares (4 headstages, 4 threads) all queue on it. Setup time grows roughly linearly with the number of headstages: several seconds each for auto compensation and bridge balance. Data is not affected. P8 (Commander auto routines overlapping) applies across 4 channels.

**NI DAQ**
- **S4. Channels.** 4 pipettes need 8 differential AI and 4 AO channels.
  - **AI:** on a 32-channel board in DIFF mode the usable inputs are `ai0–7` and `ai16–23`. Norwich uses `ai17/19/21/23`, leaving `ai16/18/20/22` and `ai0–7`. Each also occupies `aiN+8`.
  - **AO:** the commands use `ao0` and `ao3`. 4 pipettes need four AO channels, which is all of them on a 4-AO board. Nothing is left for other analog stimuli.
- **S5. Aggregate rate doubles.**
  - Live, leak, sweep, holding and opto: 8 × 50 kS/s = **400 kS/s**.
  - Current protocol: 8 × 100 kS/s = **800 kS/s**. That is near the ~1 MS/s multichannel limit of 6259/6363-class boards and well over 6229-class (P4).
  - Inter-channel skew within a sample grows to ~7 conversions (still µs; §3.4).
  - If the rate doesn't fit, the per-protocol rate needs to become configurable.
- **S6. More than one DAQ device.** If channels run out, a second board means a multi-device AI task. The code only **warns** about this ([`_check_single_task_devices`](../patcherbot/devices/amplifier/DAQ.py#L1371)); it configures no shared timebase or trigger (RTSI cable / PXI backplane / cDAQ chassis). The rig config also allows exactly one DAQ (shared slot). Without hardware sync, the guarantees in §2.1 stop holding across boards.

**Software**
- **S7. Wasted work in the timing-critical loop.** Leak and sweep run `_squareWaveProcessor` per pipette per train and **discard** the result ([L1993–L1995](../patcherbot/devices/amplifier/DAQ.py#L1993), [L2119–L2121](../patcherbot/devices/amplifier/DAQ.py#L2119)). This happens between reads, while the next non-regenerating train must already be queued. 4 fits per train eat into the slack before an AO underrun. Remove the call.
- **S8. Memory in the holding protocol.** One finite read of 60 s × 50 kS/s × 8 channels is ~192 MB as float64. The row split (`_rows_to_channel_dict` copy, `vstack`) and the scaled result add 2–3 more copies.
- **S9. Live loop CPU.** The live thread fits every pipette in cell mode serially on each cycle ([L239](../patcherbot/devices/amplifier/DAQ.py#L239)). With 4 the live display refreshes more slowly. Reads are "most recent" ([L1519](../patcherbot/devices/amplifier/DAQ.py#L1519)), so there's no overflow risk, but resistance readouts lag. The GUI also redraws 4 panels × 4 plots and writes 4 recorder files every 42 ms.
- **S10. Config tooling assumes one amplifier.**
  - The rig editor's amplifier options are `FakeAmplifier` and `MultiClampChannel` with a `channel` field only: no `serial` field and no `MultiClampAmplifier` ([rig_config.py L344–L358](../rig_setup/rig_config.py#L344)).
  - "Shared amplifier" mode means **one** device for every pipette ([rig_selector.py L486](../rig_setup/rig_selector.py#L486), [L571](../rig_setup/rig_selector.py#L571)), so "two amplifiers, two headstages each" can't be expressed or edited there.
- **S11. GUI colours repeat.** [`PIPETTE_GRADIENTS`](../patcherbot/gui/graph.py#L42) has 2 entries, indexed `position % 2`, so pipettes 2 and 3 are drawn in the same colours as 0 and 1 in the combined protocol windows. Add 2 more gradients.
- **S12. Problems that grow with N:**
  - P6: only pipette 0 can be patched.
  - P13: only the first N pipettes can run, e.g. no running pipettes 1 and 3 alone.
  - P3: per-pipette gains, now possibly across two amplifiers with different headstages.
  - P1: stale AO lines, now 4 lines.
  - `_merge_protocol_steps`: the first pipette's kwargs win.
- **Already N-generic:**
  - pipette keys and per-pipette state;
  - `build_channel_map`, `_ao_block`, `_split_rows` and every protocol's task construction;
  - the membrane test (collects until each has 5 fits);
  - the current protocol's per-pipette step lists (train count = longest list);
  - the rig validator (list of `pipette_count`);
  - side-by-side live panels;
  - the start-up pipette-count picker;
  - `fake_rig.json`, which already runs 3 pipettes on `FakeDAQ`.
- **Paired protocol design (§3.3):** define roles as a `{key: role}` mapping, not a pre/post pair. With 4 cells that allows one presynaptic cell with three postsynaptic, or rotating the presynaptic role to map connectivity within one task setup.

---

## 8. Layering audit (gui ↔ interface ↔ controller ↔ devices)

**The rule** (from the [holypipette developer guide](https://holypipette.readthedocs.io/en/latest/developer.html)):
- Each layer talks only to its neighbours. The GUI calls interfaces (`@command` / `@blocking_command`, run through `TaskInterface.execute`). Interfaces call controllers. Controllers call devices.
- The backend (controller, devices) contains no Qt.
- `utils/` and `configs/` are shared helpers. The composition root ([patch_gui.py](../patch_gui.py), [rig_config.py](../rig_setup/rig_config.py)) builds every layer and is allowed to touch all of them.

**How to re-check:** `grep -rn -E "^\s*(from|import) patcherbot\.(gui|interface|controller|devices)" patcherbot/<layer>` finds cross-layer imports. Most violations, though, are attribute chains (`interface.daq.x`, `autopatcher.calibrated_unit.camera.y`), so also grep each layer for the lower layers' attribute names, as below.

**What is clean:**
- No Qt (`PyQt5`, `pyqtSignal`) anywhere in `controller/` or `devices/`.
- No import of `interface` or `gui` from `controller/` or `devices/`.

### 8.1 Introduced or extended by the multi-pipette branch
| # | Where | Violation | Suggested fix |
|---|---|---|---|
| **L1** ✅ | [`_run_protocol_matrix`](../patcherbot/interface/patch.py#L234), DAQ call at [L260](../patcherbot/interface/patch.py#L260) | **Interface → devices.** The interface runs the DAQ acquisition itself (`getattr(self.daq, method_name)(pipettes=...)`). It also merges steps and schedules prepare/finish, which is controller work. | Move the matrix into a controller-layer class (e.g. a `TaskController` owning the autopatchers + DAQ). The interface keeps `protocol_selections` and `execute`s it. **Resolved (§11):** the matrix lives in `MultiPipettePatcher.run_protocol_matrix` (controller); the interface only `execute`s it with its selections. |
| **L2** ✅ | [`AutoPatcher.run_protocol_matrix`](../patcherbot/controller/patch.py#L890), called from [`run_protocols`](../patcherbot/interface/patch.py#L474) | **Controller → interface (upward).** The interface passes its own bound `_run_protocol_matrix` as the argument, and the controller calls it (`protocol_runner(self)`). So the controller runs interface code, which then calls the device. | Same fix as L1. The controller method should contain the logic, not a callback. **Resolved (§11):** `AutoPatcher.run_protocol_matrix(protocol_runner)` was deleted; no controller calls interface code. |
| **L3** ✅ | [`AutoPatchInterface.__init__`](../patcherbot/interface/patch.py#L55) L55–L85, [`_daq_key_for_pipette`](../patcherbot/interface/patch.py#L110), [`_device_for_pipette`](../patcherbot/interface/patch.py#L124) | **Interface → devices.** It builds `PipetteDAQView` (a device class, imported at [L15](../patcherbot/interface/patch.py#L15)), reads `daq.channel_map` / `daq.primary_pipette_key`, and calls `amplifier.for_pipette(key)`. | Do the per-pipette pairing in the composition root (`build_devices` already has the keys) or in a controller factory. The interface then receives ready-made per-pipette controllers. **Resolved (§11):** `MultiPipettePatcher.from_devices` does the pairing (`device_for_pipette`, `daq.for_pipette`); the interface no longer imports `PipetteDAQView` or reads `channel_map` / `primary_pipette_key`. |
| **L4** | Protocol windows in [gui/graph.py](../patcherbot/gui/graph.py): [L202](../patcherbot/gui/graph.py#L202), [L231](../patcherbot/gui/graph.py#L231), [L248](../patcherbot/gui/graph.py#L248), [L348](../patcherbot/gui/graph.py#L348), [L396](../patcherbot/gui/graph.py#L396), [L438](../patcherbot/gui/graph.py#L438) | **GUI → devices** through `graph_interface.daq`. It reads protocol results, and **writes** device state: `daq.current_protocol_data = None` at L231. This existed at the branch base (18 such refs in the single-pipette windows), but the combined multi-pipette windows rebuilt and kept it. | Add `GraphInterface` methods: `get_protocol_result(name)`, `clear_protocol_result(name)`, `pop_optogenetic_entry(key)`. The GUI calls only those. |
| **L5** ✅ | [`AutoPatchInterface._run_amplifier_workers`](../patcherbot/interface/patch.py#L201) | **Interface reaches through the controller to its device.** It groups pipettes by `self.autopatchers[key].amplifier` object identity. | Have the controller expose what the interface needs (e.g. `autopatcher.amplifier_group`), or move it with L1. **Resolved (§11):** `AutoPatcher.amplifier_group` exposes the grouping key; the controller groups by it. |
| **L6** | [`GraphInterface.pressure_is_simulated`](../patcherbot/interface/graph.py#L46) (new) and `get_last_data(pipette_key)` ([L66](../patcherbot/interface/graph.py#L66), extended) | **Interface → devices.** It reads the device attribute `pressure.simulated` and calls `daq.get_last_acquisition`. The whole of `GraphInterface` wraps devices directly (see L9). | Same as L9. |
| **L7** | [patch_gui.py L109–L118](../patch_gui.py#L109) | **Composition root reaches into interface internals.** It builds each `GraphInterface` from `patch_controller.autopatchers[key].amplifier/.pressure` and `patch_controller.protocol_daQs[key]`. The root may wire every layer, but this depends on `AutoPatchInterface`'s private structure. | Build the per-pipette device bundles once in the root and pass them to both interfaces. |
| **L8** | [`PipetteDAQView`](../patcherbot/devices/amplifier/DAQ.py#L1237), [L1266–L1269](../patcherbot/devices/amplifier/DAQ.py#L1266) | **Device shaped around interface internals.** It sets `call_for_pipette.__self__ = daq` only so that `TaskInterface.execute` will accept it. Not a call, but the devices layer is coupled to how the interface layer is implemented. | Disappears if the GUI-facing paths go through controllers (L3, L9). |

### 8.2 Pre-existing (present at `0cf69cd`), listed for completeness
| # | Where | Violation |
|---|---|---|
| **L9** | [interface/graph.py](../patcherbot/interface/graph.py) L39–L198 | Interface → devices. `GraphInterface` has no controller. It calls `pressure.*`, `daq.*`, `amplifier.set_zap_duration/zap` and `laser.*` directly (several through `self.execute(device.method)`). |
| **L10** | [interface/patch.py](../patcherbot/interface/patch.py): [L333](../patcherbot/interface/patch.py#L333), [L371–L400](../patcherbot/interface/patch.py#L371), [L442–L455](../patcherbot/interface/patch.py#L442), [L492–L532](../patcherbot/interface/patch.py#L492), [L555–L673](../patcherbot/interface/patch.py#L555), [L800](../patcherbot/interface/patch.py#L800), [L846–L900](../patcherbot/interface/patch.py#L846) | Interface → devices, often through a controller or another interface: `current_autopatcher.lamp.get_filter()`, `current_autopatcher.calibrated_unit.camera.get_16bit_image()`, `pipette_controller.calibrated_stage.*`, `calibrated_cellsorter.set_led_status`, `microscope.position()`. It also writes device configs (`calibrated_unit.config.safe_position = ...`). |
| **L11** | [interface/pipettes.py](../patcherbot/interface/pipettes.py) (~68 refs, e.g. [L128](../patcherbot/interface/pipettes.py#L128), [L376–L478](../patcherbot/interface/pipettes.py#L376)) | Interface → devices. It `execute`s methods of `CalibratedUnit`, `CalibratedStage`, `Microscope` and `CalibratedCellSorter` directly; these live in `devices/` but inherit `TaskController`. |
| **L12** | [interface/camera.py](../patcherbot/interface/camera.py) L45–L168 | Interface → devices: `camera.get_exposure/auto_exposure/normalize/...`. |
| **L13** | [gui/patch.py](../patcherbot/gui/patch.py) [L403](../patcherbot/gui/patch.py#L403), [L1056](../patcherbot/gui/patch.py#L1056), [L1165–L1236](../patcherbot/gui/patch.py#L1165); [gui/manipulator.py](../patcherbot/gui/manipulator.py) [L79](../patcherbot/gui/manipulator.py#L79), [L227](../patcherbot/gui/manipulator.py#L227), [L283](../patcherbot/gui/manipulator.py#L283) | GUI → devices (and GUI → controller) through interface attributes. Examples: `patch_interface.current_autopatcher.calibrated_stage.reference_position()`, `pipette_interface.calibrated_unit.unit.position()`, `pipette_interface.microscope.position()`, `self.camera.get_16bit_image()`. |
| **L14** | [gui/livefeed.py](../patcherbot/gui/livefeed.py) L13/L97, [gui/patch.py](../patcherbot/gui/patch.py#L36) constructor, [gui/manipulator.py](../patcherbot/gui/manipulator.py#L16) | GUI → devices. The GUI is handed `Camera` objects and reads frames from them (`last_frame_pair`). It imports `devices.camera.camera.Camera` and `devices.manipulator.calibratedunit.CalibrationError`. |
| **L15** | [gui/camera.py](../patcherbot/gui/camera.py#L28), [gui/manipulator.py](../patcherbot/gui/manipulator.py#L13), [gui/patch.py](../patcherbot/gui/patch.py#L13) | GUI imports `controller.TaskController`, only as a `pyqtSignal(TaskController)` type. Low severity. Could use `object` or re-export it from `interface`. |
| **L16** | `devices/` classes subclassing `TaskController`: amplifier, DAQ, lamp, laser, manipulator, pressure, cell sorter (`from patcherbot.controller... import TaskController`) | Devices → controller import, for the shared logging/abort base class. Not a call, but it makes every device "executable", which is what lets interfaces skip the controller layer (L9–L12). Moving `TaskController`'s base into `utils/` would make the dependency direction explicit. |

**Priority for the paired protocol work:** L1, L2, L3 and L5 are resolved (§11); L4 is next. They sit exactly on the code path a paired/role-based protocol will extend (§3.3). L3, L5 and L7 follow naturally from the same refactor. L9–L16 are older and can be scheduled separately.

---

## 9. In the diff but not multi-pipette (skip for this audit)
- Docstring and formatting churn in `rig_config.py` and `rig_selector.py`.
- New optional editor fields (manipulator speed/accel/polling, cell-sorter params, Waynesboro and IBB pressure options).
- `load_config(load_overlays=...)`; default fake-rig template fields.
- `FakDAQ_graph_recording.csv` (FakeDAQ data file); `setup_test_rig.json` reformatting.
- Pipette detector / focuser model routing (`cd8b61c`, tested by `test_norwich_pipette_detector.py`).

---

## 10. DAQ information flow: review and restructuring proposal

**Scope.** Review questions about [DAQ.py](../patcherbot/devices/amplifier/DAQ.py) and `pipette_keys.py`. When written, no code had been changed; this was a proposal. **§11 records what was then implemented, which differs from §10.2 in places** (one helper file that also holds the DAQ channel model; no DAQ package split yet). Line numbers are for `2214d8f`. "Baseline" means the branch point `0cf69cd` (`git show 0cf69cd:patcherbot/devices/amplifier/DAQ.py`).

**Summary.**
- The multi-pipette DAQ works, but it was built so that single-pipette code would run unchanged. Most of the "messy" pieces (`_PER_PIPETTE_FIELDS`, the `setattr` loop, `_new_pipette_state`, `PipetteDAQView`) exist only for that purpose. They can all be replaced by explicit per-pipette state passed around by key.
- `DAQ.py` (3008 lines) mixes five things: the channel model, per-pipette state, the live thread, hardware-agnostic protocol planning and analysis, and three hardware back ends. FakeDAQ re-implements every protocol. Splitting it into a package with a hardware-agnostic base and a small per-device "engine" fixes the dumping-ground problem and the duplication at once.
- Protocols are **serialized across calls and simultaneous within a call**. Two cells in current clamp at the same time already works. Current clamp on one cell with holding on another does not, and removing the lock would not make it work. The only way that fits the hardware is **one AI task and one AO task with a different waveform and scaling per row**, which the code is half-way to already (§10.5).
- Current and voltage acquisition differ in waveform layout, unit scaling, sample rate and regeneration mode. Only the scaling is tied to the clamp mode. Regeneration mode is a leftover inconsistency from the baseline, not a property of current clamp (§10.4).

### 10.1 What each multi-pipette piece does, and how it was done before

| Piece | What it does | Before (baseline) | Verdict |
|---|---|---|---|
| [`PipetteChannels`](../patcherbot/devices/amplifier/DAQ.py#L30) | Frozen `(read, resp, cmd)` record for one pipette. `from_spec` checks a config entry has exactly those three keys. | Did not exist. NiDAQ took six loose constructor params (`readDev`, `readChannel`, …). | Small and useful: config typos fail at start-up and code says `spec.resp`, not `spec["resp"]`. Its real value is the validation. Worth keeping, and a natural home for per-pipette gains later (§10.6, item 3). |
| [`build_channel_map`](../patcherbot/devices/amplifier/DAQ.py#L53) | Builds the ordered `{pipette_key: PipetteChannels}`. Checks for duplicate keys, too few channel sets, and a physical channel used twice. | Did not exist. | Keep, but move (§10.2). |
| [`_PER_PIPETTE_FIELDS`](../patcherbot/devices/amplifier/DAQ.py#L91), [`_new_pipette_state`](../patcherbot/devices/amplifier/DAQ.py#L123) | The list of DAQ attributes that need one value per pipette, and a function that builds one fresh dict of them. A default that is callable (`collections.deque`) is called, so each pipette gets its own queue. | Each was a plain attribute set in `DAQ.__init__` (baseline L164–L194: `self.cellMode = False`, `self.current_protocol_data = None`, …). One DAQ meant one pipette, so one value was enough. | **New, and needed in some form**: one DAQ now holds N pipettes' results. The dict-plus-callable-default form is what makes it look messy. A dataclass does the same job cleanly (§10.2). |
| [`_PerPipetteAttribute`](../patcherbot/devices/amplifier/DAQ.py#L146) + the [`for _field_name in _PER_PIPETTE_FIELDS` loop](../patcherbot/devices/amplifier/DAQ.py#L1232) | The loop installs one descriptor per field name on the `DAQ` class, after the class body. Each descriptor sends `self.cellMode` (etc.) to the *active* pipette's dict. The active pipette is a **thread-local** set by [`for_pipette`](../patcherbot/devices/amplifier/DAQ.py#L388) and defaults to `pipette_0`. | Not needed. | **Remove.** It exists so that code like [`_squareWaveProcessor`](../patcherbot/devices/amplifier/DAQ.py#L980) (reads `self.getCellMode()`, L1007) and [`_getParamsfromCurrent`](../patcherbot/devices/amplifier/DAQ.py#L1176) (writes `self.holding_current`, L1193) didn't have to be edited. The cost is hidden state. Which pipette `daq.resistance()` refers to depends on the calling thread. Any thread that forgets `for_pipette` silently gets pipette 0. Any new attribute left off the list is silently shared between pipettes. |
| [`PipetteDAQView`](../patcherbot/devices/amplifier/DAQ.py#L1237) | A proxy that looks like a single-pipette DAQ. Per-pipette fields go to that pipette's state. Every method call runs inside `for_pipette(key)`. Anything else is set on the shared DAQ. | Not needed. Each AutoPatcher held the real DAQ. | See §10.3. |
| [`multi_pipette_protocol`](../patcherbot/devices/amplifier/DAQ.py#L130) | Turns `pipettes=` into a validated key list (None = active pipette), then holds `_protocol_lock` for the whole call. | Did not exist. | Keep the idea. It belongs in the base class (§10.2). |

### 10.2 Where the code should live

**Does the channel map belong in `pipette_keys`?** **No.** Keep the *definitions* in the DAQ code and move a *generic* part into the helper:
- **`channel_map` should stay owned by the DAQ instance.** It is the DAQ's wiring: which AI/AO terminals belong to which pipette. FakeDAQ and ArduinoDAQ have their own defaults. `MultiClampAmplifier` follows the same pattern for headstages (`for_pipette` → headstage channel). Each device knows how *its* hardware maps to pipette keys. Only the keys themselves are global.
- **`read` / `resp` / `cmd` are DAQ concepts.** If `utils/` defined them, every layer importing the helper would depend on DAQ wiring, and the helper would stop being general.
- **What is generic and should move to the helper:** the part of `build_channel_map` that isn't about channels (default to `[pipette_0]`, reject duplicate keys, require at least N entries, zip into an ordered mapping). The same logic is repeated in `rig_config.build_devices` (per-pipette dicts), `MultiClampAmplifier.for_pipette`, `AutoPatchInterface._device_for_pipette` / `_daq_key_for_pipette`, and `DAQ._resolve_pipettes`.

**Proposed helper:** rename `pipette_keys.py` to `patcherbot/utils/multi_pipette.py`. It stays free of device imports.
- Existing: `pipette_key_for`, `pipette_index`, `pipette_label`.
- New: `resolve_pipettes(requested, known, default)`, the dedupe-and-validate logic now in [`DAQ._resolve_pipettes`](../patcherbot/devices/amplifier/DAQ.py#L423).
- New: a small `PipetteMap` class: an ordered `{key: value}` mapping that validates its keys. Methods `require(key)` (with the "known pipettes: …" error), `subset(keys)`, and a constructor from a list in rig order. The channel map, rig builder, amplifier and interface all use it.
- A class called `MultiPipetteHelper` that only holds static methods would be a module with extra steps. The module is the namespace (`from patcherbot.utils import multi_pipette`). `PipetteMap` is the one real class.

**Proposed DAQ package** (replacing the single `DAQ.py`):

| Module | Contents | Notes |
|---|---|---|
| `daq/channels.py` | `PipetteChannels`, `build_channel_map` (on top of `PipetteMap`), `enumerated_channels` | Device-specific half of the channel model. |
| `daq/state.py` | `@dataclass PipetteState`: `cell_mode`, live metrics, protocol results, `optogenetic_queue: deque = field(default_factory=deque)` | Replaces `_PER_PIPETTE_FIELDS`, `_new_pipette_state`, the descriptor and the `setattr` loop. `default_factory` is the standard way to give each instance its own deque. The DAQ holds `self.pipettes: PipetteMap[PipetteState]`. Code asks for `self.pipettes[key].cell_mode` explicitly. |
| `daq/analysis.py` | `square_wave_processor`, `params_from_current`, `filter_data`, `optimizer`, `calc_param`, `compute_noise_metrics`, waveform builders | Pure functions. Cell mode and gains go **in** as arguments, and holding current comes **out** as a return value. This removes the main reason for the thread-local "active pipette". It also makes them testable without hardware. |
| `daq/live.py` | `DAQAcquisitionThread` | Unchanged apart from writing to `PipetteState`. |
| `daq/base.py` | `DAQ`: owns the channel map and `pipettes` state, `pipettes=` resolution and the protocol decorator, `_protocol_lock`, pause/resume, the membrane test, **protocol planning** (step lists, sweep targets, per-pipette waveforms and unit conversion), and storing results | Everything shared by NI, Fake and a future Arduino. |
| `daq/nidaq.py`, `daq/fake.py`, `daq/arduino.py` | Only the hardware **engine** (see below) | |
| `daq/pipette_handle.py` | The per-pipette handle (§10.3) | |

**Base class vs back end: where to draw the line so Arduino or other DAQs fit.**
- Today each NiDAQ protocol mixes planning (build the step list from capacitance, compute targets, scale to volts) with hardware (create tasks, queue trains, read). FakeDAQ therefore re-implements leak, sweep, current and holding ([L2806–L3006](../patcherbot/devices/amplifier/DAQ.py#L2806)), including its own copy of `_current_pulses`.
- Suggested split: the base class turns a protocol into **a list of trains, each a `{pipette_key: AO waveform in volts}` block, plus a sample rate**. It hands that to one abstract back-end method, roughly `_run_trains(keys, rate, train_samples, trains) -> iterator of {key: (read, resp)}`, plus `_read_finite(keys, rate, n)` for holding/opto. Then it splits and scales the returned rows per pipette.
- NiDAQ implements `_run_trains` once (one continuous AI task + one non-regenerating AO task, queue one train ahead, read one train at a time: the existing leak/sweep loop). FakeDAQ implements it by replay. ArduinoDAQ can implement it for one pipette, or raise `NotImplementedError` for more than one.
- Every protocol, including a future paired protocol, is then written once in the base class.

### 10.3 `PipetteDAQView`: what it is for, and what should replace it

- **Why it looks unused:** it is *constructed* once ([interface/patch.py L71](../patcherbot/interface/patch.py#L71)). That object is then the `daq` of every `AutoPatcher` ([controller/patch.py L53](../patcherbot/controller/patch.py#L53)) and of every per-pipette `GraphInterface` ([patch_gui.py L112](../patch_gui.py#L112)). All single-pipette code goes through it: `setCellMode`, `resistance()`, `capacitance()`, `accessResistance()`, `holding_current`, `get_last_acquisition`, `compute_noise_metrics`, `AutoPatcher.run_<protocol>()`. The protocol windows also read and clear results through it (`graph_interface.daq.current_protocol_data`, …, [gui/graph.py L202–L574](../patcherbot/gui/graph.py#L202)).
- **Its function:** a single-pipette DAQ handle for code written before multi-pipette, so that code didn't change.
- **Why it is fragile:**
  - `__getattr__` wraps *every* method (so you can't tell from the class what a pipette can do).
  - `__setattr__` on a non-per-pipette name silently changes the shared DAQ for every pipette.
  - It fakes `__self__` ([L1269](../patcherbot/devices/amplifier/DAQ.py#L1269)) so that `TaskInterface.execute` accepts it (§8, L8).
- **Can it be rolled into something else?** Yes. Replace it with an explicit `PipetteDAQ(daq, key)` class that has only the members used today (listed above, about 10). Results are read with `daq.pipettes[key]`, and protocols with `daq.<protocol>(pipettes=[key])[key]`. No `__getattr__`, no thread-local.
  - Until interfaces stop `execute`-ing device methods directly (§8, L3/L9), the handle can subclass `TaskController` so the `__self__` trick isn't needed.
  - Once protocol results reach the GUI through `GraphInterface` methods (§8, L4), most of the handle's attribute surface disappears.

### 10.4 Current vs voltage acquisition, mechanically

**The DAQ side is the same physical path in both modes:** an AO voltage goes into the amplifier's external command BNC, and two AI voltages come back (command monitor → `read`, primary output → `resp`). The DAQ never knows the clamp mode. The MultiClamp decides what those volts *mean*, and the protocol's prepare step sets it. So "both just send square waves with different scaling" is essentially right. Here are all the differences:

| | Current protocol (I-clamp) | Voltage sweep / leak (V-clamp) | Tied to the clamp mode? |
|---|---|---|---|
| **AO scaling** (command sensitivity) | pA → V via `C_CLAMP_AMP_PER_VOLT` = 400 pA/V ([L2212](../patcherbot/devices/amplifier/DAQ.py#L2212)) | V → V via `V_CLAMP_VOLT_PER_VOLT` = 20 mV/V ([L2101](../patcherbot/devices/amplifier/DAQ.py#L2101)) | **Yes.** Commander has a separate external-command sensitivity per mode. |
| **`resp` scaling** (primary output) | Vm: `raw / C_CLAMP_VOLT_PER_VOLT` (10 V/V) ([L2254](../patcherbot/devices/amplifier/DAQ.py#L2254)) | Im: `raw × V_CLAMP_VOLT_PER_AMP` (2e-9 A/V) ([L2123](../patcherbot/devices/amplifier/DAQ.py#L2123)) | **Yes.** The primary output carries Vm in I-clamp and Im in V-clamp. |
| **`read` scaling** | Stored **raw** (unscaled volts) | `raw × V_CLAMP_VOLT_PER_VOLT` | Should be, but the current protocol doesn't convert it. Inconsistent. |
| **Waveform per train** ([`createSquareWaveCurrent`](../patcherbot/devices/amplifier/DAQ.py#L928) / [`createSquareWaveVoltage`](../patcherbot/devices/amplifier/DAQ.py#L884)) | 4 segments of `recordingTimeMs` (default 500 ms, so a **2 s** train): 0 → −20 pA calibrator for half a segment → step for half a segment → 0 | 3 segments of `recordingTime/3` (default 750 ms train): 0 → step for half a segment (125 ms) → 0 | No. This is protocol design. |
| **Sample rate** | 100 kS/s, hard-coded ([L2188](../patcherbot/devices/amplifier/DAQ.py#L2188)) | 50 kS/s (argument) | No. |
| **AO regeneration mode** | `ALLOW_REGENERATION`, one-train buffer ([L1597](../patcherbot/devices/amplifier/DAQ.py#L1597)) | `DONT_ALLOW_REGENERATION`, buffer of all trains ([L1654](../patcherbot/devices/amplifier/DAQ.py#L1654)) | **No.** Both were already this way at the baseline (baseline L1039 / L1110). |

Both protocols put out **0 V between steps**. The holding potential or holding current comes from the amplifier, not the DAQ. The voltage step is `(target − holding)`.

**Stale comments in the current protocol:** [L2190–L2192](../patcherbot/devices/amplifier/DAQ.py#L2190) say "0.25 s / segment", "4 Hz" and "1.0 s / train". With the default `recordingTimeMs=500` it is 0.5 s, 2 Hz and 2.0 s.

**Scaling constant to verify on the rig.** [NiDAQ L1296](../patcherbot/devices/amplifier/DAQ.py#L1296) is commented "2 mV per pA", and §3.5 repeats that. But the code *multiplies* raw volts by `2e-9`, which means 2 nA per V, i.e. **0.5 mV/pA**. That matches a 500 MΩ feedback resistor at gain 1. The name `V_CLAMP_VOLT_PER_AMP` also holds amps per volt. Either the comment is wrong (most likely) or currents are off by 4×. Check against Commander's primary-output scale.

**What "regeneration" means.** It is the DAQmx term for whether the device may **re-output samples already in its buffer** once it has played them.
- `ALLOW_REGENERATION`: the buffer loops forever with no further writes. This is right for the live test pulse ([L1537](../patcherbot/devices/amplifier/DAQ.py#L1537)): one 25 ms pulse repeating without Python feeding it. New writes replace buffer contents in place.
- `DONT_ALLOW_REGENERATION`: every sample is output once. Python must queue new data ahead of the hardware. If it falls behind, the task **stops with an error** (DAQmx -200290) instead of repeating old data.
- The current protocol probably inherited regeneration from the live-stream code it was copied from. It is the riskier choice for a stepped protocol. If a write is late (a Python/GC stall longer than one train), the device silently **replays the previous step**. The read loop labels trains by index, so from then on every trace carries the wrong step label, and nothing raises (§7, P2). Non-regeneration fails loudly instead.
- **For mixed protocols:** regeneration mode is set per *task*, not per channel, so every row of a shared AO task must use the same mode. Non-regeneration works for every stepped protocol, so standardise on it. Regeneration does not stop different pipettes running different protocols; it is just the one setting they must share.

**Why the two setup functions are formatted differently.** [`_setupAcquisitionCurrent`](../patcherbot/devices/amplifier/DAQ.py#L1564) and [`_setupAcquisitionVoltage`](../patcherbot/devices/amplifier/DAQ.py#L1607) were already this way at the baseline (the branch only added `keys=`).
- The current one takes `recordingTime` in **ms**, hard-codes the rate, builds its own prime wave positionally and sizes the AI buffer as 10 s.
- The voltage one takes everything keyword-only, including `exp_samples` and `buffer_trains`, and sizes both buffers from them.
- The current prime train is not zeros: it contains the −20 pA calibrator segment. It is discarded, so this is harmless.
- **Suggestion:** one `_setup_protocol_tasks(keys, rate, train_samples, buffer_trains)` that primes with zeros. This is the NiDAQ `_run_trains` engine from §10.2. Both protocols then use it, and the differences become plain parameters.

### 10.5 Parallelism, `_protocol_lock`, and running different protocols on different pipettes

**What the lock means.** `_protocol_lock` ([L347](../patcherbot/devices/amplifier/DAQ.py#L347)) allows **one protocol call at a time**. One call may cover several pipettes, and they are recorded in the same samples (§2.1). So:

| Wanted | Today |
|---|---|
| Current clamp on two cells at once | **Works.** Tick both pipettes for "current". One `getDataFromCurrentProtocol(pipettes=[0, 1])` call, one AI task and one AO task, each pipette with its own capacitance-scaled steps (row *i* of each AO write). |
| V-clamp sweep on two cells at once | **Works.** Same mechanism, but both get the same voltage wave. |
| Current on A **and** holding on B at once | **Does not work.** The matrix runs "current" and then "holding" as two calls ([interface/patch.py L234](../patcherbot/interface/patch.py#L234)), and [`_merge_protocol_steps`](../patcherbot/interface/patch.py#L181) rejects different DAQ methods within one protocol. |

The code comment "Protocols use the shared hardware one at a time" would be clearer as "One protocol call at a time; a call may record several pipettes simultaneously."

**Why removing the lock would not help:**
1. **Hardware.** An NI multifunction board has **one AI timing engine and one AO timing engine**. A second hardware-timed AI or AO task on the same board fails with "resource reserved". It does not run alongside the first. Separate per-pipette AO tasks triggered from the AI start trigger therefore can't exist on one board: all AO channels running at once must be in **one** AO task.
2. **Software.** Every protocol keeps its tasks in the same attributes (`self.ai_task`, `self.ao_task`, `_ai_task_keys`, …), so two concurrent protocols would overwrite each other's tasks.

**What the hardware does allow (your last idea, adjusted):** unify **both** AI and AO.
- **AI:** one task with every pipette's `read` + `resp`, as now. Scaling becomes **per row**, chosen by that pipette's mode (Vm for the I-clamp pipette, Im for the V-clamp pipette).
- **AO:** one task with every pipette's `cmd`, each row carrying **that pipette's own waveform**, converted with **that pipette's own** command sensitivity. [`_ao_block`](../patcherbot/devices/amplifier/DAQ.py#L1419) already accepts `{key: wave}`, and `None` means 0 V.
- **Sync is then free:** one sample clock, one start trigger, and both rows go out in the same write.
- **The one-task constraints:** every row shares **one sample rate**, **one train length** (an AO write is a rectangle: channels × samples) and **one regeneration mode**. For a paired recording that is natural. B's "holding" row is simply zeros (or a small test pulse) of A's train length, and B is recorded over exactly the same samples.
- A second, separately clocked set of tasks would need a second board sharing a timebase and start trigger (RTSI cable, PXI or cDAQ chassis). That isn't worth it for this.

**So the model should change from "protocol × pipette" to "one acquisition, with a role per pipette".** A role says, for each pipette: amplifier mode (I-clamp / V-clamp), AO waveform per train in physical units (A or V), and whether to record. The existing protocols become special cases:
- current = every pipette in I-clamp with step waveforms;
- sweep = every pipette in V-clamp;
- holding = V-clamp with a zero waveform;
- paired = A in I-clamp with a stimulus, B in V-clamp with zeros.

This is the `_run_trains` engine from §10.2 plus per-pipette scaling. It extends the §3.3 sketch from a fixed pre/post pair to a `{key: role}` map, as §7.1 already suggests.

### 10.6 Things that will come up with current on A and holding on B (synaptic connection)

Ordered roughly by when you will hit them.

1. **The run model and GUI.** The checkbox grid (protocol rows × pipette columns) can't express "A stimulates while B records". `_merge_protocol_steps` raises on it by design. A paired acquisition needs its own row, where each pipette column chooses a role (stimulate / record / off). Orchestration belongs in a controller-layer runner, not the interface (§8, L1/L2). The single-pipette path (`AutoPatcher.run_<protocol>()` through the view) can't express it at all. A paired run is inherently multi-pipette.
2. **Different amplifier prepare per pipette.**
   - A: I-clamp, bridge balance, optional holding current.
   - B: V-clamp, `set_holding(V)`, `switch_holding(True)`, compensation.
   - The prepares already run per pipette in threads. Finish must restore **both** even if one fails (it does today).
   - P8 matters more here: B's auto compensation running in Commander while A's channel is selected.
3. **Per-pipette, mode-aware scaling** (§7, P3).
   - Today scaling is per protocol and comes from class constants.
   - The DAQ must take the mode from each pipette's role. Ideally it should also check the mode against the amplifier (telegraph or a mode query) so a pipette accidentally left in the wrong mode is caught, not mis-scaled.
   - Per-headstage gains could live on `PipetteChannels` (renamed, e.g. `PipetteSignalPath`). That gives it a clearer reason to exist.
   - Resolve the 0.5 vs 2 mV/pA question (§10.4) first.
4. **B's command line must be held at 0 V explicitly** (§7, P1). If B isn't in the AO task, its `cmd` sits at the last live test-pulse value: 0 or +0.5 V, i.e. **+10 mV** on B's holding potential, chosen at random each run. Put B in the AO task with a zero row.
5. **One rate and one train length for both.**
   - Pick the rate the stimulus needs (100 kS/s if you keep the current protocol's rate). Then check the aggregate rate: 2 pipettes × 2 inputs × 100 kS/s = 400 kS/s (§3.4, P4).
   - The post-synaptic events (ms rise times) don't need more than 20–50 kS/s. If the board can't do 400 kS/s, run the whole acquisition at 50 kS/s.
6. **Non-regenerating AO is a requirement here,** not a nicety. A silently replayed train would pair a presynaptic step with the wrong postsynaptic sweep (§10.4).
7. **Gap-free postsynaptic record.** Protocol reads are sequential from the start of acquisition (not "most recent"), so as long as each train is read before the 10 s AI buffer fills, B's record is continuous across trains and can be concatenated. Keep it that way. The live stream's `MOST_RECENT_SAMPLE` read mode would drop data.
8. **The stimulus shape is different from the F-I protocol.**
   - Connectivity tests usually use brief suprathreshold pulses on A (about 1–3 ms at ~1–2 nA, single or in a short train, e.g. 5 at 20 Hz) to fire one spike per pulse. They also leave rest between sweeps (several seconds) so the synapse recovers.
   - `createSquareWaveCurrent`'s fixed baseline/calibrator/step layout and the capacitance-scaled −2…+10 pA/pF steps don't fit. A generic per-pipette waveform builder (a list of `(start, duration, amplitude)` segments) does, and the rest interval is just zero-filled train time.
9. **AO range.** At 400 pA/V, ±10 V on the AO is **±4 nA** at most. [`_current_pulses`](../patcherbot/devices/amplifier/DAQ.py#L2156) clamps steps to ±20 nA, which would ask for ±50 V, and DAQmx would reject the write. The clamp should come from `10 V × C_CLAMP_AMP_PER_VOLT`. Brief 1–2 nA spike pulses fit. If you need more, switch Commander to 2 nA/V.
10. **Monitoring B during long runs.**
    - The live test pulse is parked for **every** pipette during a protocol, so B's access resistance isn't monitored. Over minutes of paired recording Ra drift changes EPSC amplitude.
    - Since B has its own AO row, add a short −5 mV step at the end of each of B's trains and fit it with the existing square-wave analysis. That gives per-sweep Ra and Rm.
    - For pipettes in neither role (3–4 pipette rigs), consider still putting them in the AI task, and in the AO task with zeros, so they are recorded and not stale.
11. **Artifacts and crosstalk.** A fast current step on A couples capacitively into B at **zero latency**, and brief large spike pulses make it worse than F-I steps. Run the §3.7 bath control, and treat anything time-locked to A's step at under ~0.5 ms latency as artifact.
12. **What to save.**
    - One shared acquisition ID and the role map.
    - The sample rate.
    - The **actual AO rows written** for each train (not just a step amplitude), so analysis can find stimulus onsets exactly.
    - Both pipettes' rows with the same train index.
    - Today each pipette is saved to its own `<Protocol>_pipette_<n>` file with nothing linking them (§3.1, item 5).
13. **Analysis timing.** A's spike (from Vm) and B's response come from the same sample clock. The only offset is the multiplexed ADC's inter-channel delay, a few µs (§3.4). That is negligible against ~1 ms synaptic latency, and the channel order is known if you want to correct it.
14. **Abort and cleanup.** A paired run can take minutes. Check for abort between trains (`_read_train` uses `self.sleep`). Always `finally` → close tasks, write 0 V to every `cmd`, resume the live stream, and run both finishes.

### 10.7 Suggested order of work
1. **No behaviour change:** split `DAQ.py` into the package (§10.2). Rename `pipette_keys` → `multi_pipette` and add `PipetteMap` / `resolve_pipettes`. Fix the stale comments and the scaling-constant comment once checked on the rig.
2. Replace the descriptor, `setattr` loop and thread-local with `PipetteState` and explicit keys. Make the analysis functions pure. Replace `PipetteDAQView` with the explicit `PipetteDAQ` handle.
3. Build the NiDAQ `_run_trains` engine (non-regenerating, zero-primed) and the FakeDAQ replay version. Port leak, sweep and current onto it. This removes the regeneration risk (P2) and the FakeDAQ protocol copies.
4. Add per-pipette modes and gains to the scaling (P3) and include every `cmd` in the AO task with zeros (P1).
5. Add a role-based acquisition (paired protocol), a controller-layer runner (§8, L1/L2) and the paired GUI row and window.

---

## 11. Refactor: multi-pipette helper, explicit keys, multi-pipette controller

**Goal.** `DAQ.py` only talks to DAQs, keyed only by arguments passed to specific functions; all multi-pipette bookkeeping lives in one helper; layer violations L1, L2, L3 and L5 (§8.1) are removed. No acquisition behaviour changed (task setup, regeneration modes, scaling and protocol order are as at `2214d8f`). Pre-branch violations (L9–L16) and L4, L6–L8 were deliberately left alone.

**Decisions (differences from the §10 proposal):**
- **One helper file**, [`patcherbot/utils/multipipette_helper.py`](../patcherbot/utils/multipipette_helper.py) (renamed from `pipette_keys.py`, no compatibility shim). It holds the DAQ channel model too, because the user preferred one place for everything multi-pipette over keeping `utils/` free of DAQ concepts. It imports nothing from `patcherbot.devices`.
- **A base class,** `MultiPipetteDevice`, gives every multi-pipette device the same key handling and `for_pipette(key)`. `DAQ` (keys + per-pipette state) and `MultiClampAmplifier` (keys only) inherit it; other devices can follow.
- **Per-pipette state** is `PipetteState` built from `PER_PIPETTE_FIELDS`, kept as a dict-of-defaults (not a dataclass) so non-DAQ devices can declare their own field sets.
- **Explicit keys instead of a thread-local active pipette.** `_PerPipetteAttribute`, the module-level `setattr` loop and `DAQ.for_pipette` as a context manager are gone. Methods are tagged (`@multi_pipette_protocol`, `@takes_pipettes`, `@per_pipette`) so `PipetteDAQView` knows which argument to fill in.
- **`PipetteDAQView` and `multi_pipette_protocol` were kept** (moved to the helper). The view's `__self__` trick (L8) stays.
- **Not done in this pass:** splitting `DAQ.py` into a package, moving the analysis/waveform code out, the `_run_trains` engine (§10.2, §10.7 items 1 and 3–5).

**What moved where:**

| Was | Now |
|---|---|
| `utils/pipette_keys.py` | `utils/multipipette_helper.py` (keys unchanged) |
| `DAQ.py`: `PipetteChannels`, `build_channel_map`, `enumerated_channels` | helper (verbatim) |
| `DAQ.py`: `_PER_PIPETTE_FIELDS`, `_new_pipette_state` | helper: `PER_PIPETTE_FIELDS`, `PipetteState` |
| `DAQ.py`: `_PerPipetteAttribute` + `setattr` loop, thread-local `_active`, `active_pipette_key`, `for_pipette` context manager | deleted; `daq.pipette_state(key)` and `pipette_key=` arguments |
| `DAQ.py`: `_resolve_pipettes`, `pipette_keys`, `primary_pipette_key`, `_pipette_state` | `MultiPipetteDevice.resolve_pipettes` / `pipette_keys` / `primary_pipette_key` / `pipette_state` |
| `DAQ.py`: `multi_pipette_protocol`, `PipetteDAQView` | helper |
| interface: `_device_for_pipette` | helper: `device_for_pipette` |
| interface: `_daq_key_for_pipette` | `DAQ.for_pipette(key)` (returns the view; unknown key → primary with a warning) |
| interface: building AutoPatchers + views | `MultiPipettePatcher.from_devices` |
| interface: `_PROTOCOL_RUNNERS`, `_opto_params`, `_prepare_protocol`, `_finish_protocol`, `_merge_protocol_steps`, `_run_amplifier_workers`, `_run_protocol_matrix` | `MultiPipettePatcher` (`PROTOCOL_NAMES`, `run_protocol_matrix(selections)`) |
| controller: `AutoPatcher.run_protocol_matrix(protocol_runner)` | deleted |
| interface reads `autopatcher.amplifier` for grouping | `AutoPatcher.amplifier_group` |
| `patch_gui.py`: `patch_controller.protocol_daQs[key]` | `autopatcher.daq` |

**Behaviour notes:**
- `pipettes=None` / `pipette_key=None` on the DAQ itself now always means the primary pipette (before: the thread's active pipette, which was the primary unless inside `for_pipette`). Through a view, None means the view's pipette, as before.
- Abort during "Run Protocols" now lands on `MultiPipettePatcher`: checked between protocols and between DAQ steps (abortable sleep, P19), never inside an amplifier finish (§5.7a).
- Setting an attribute on `PipetteState` that isn't a declared field raises `AttributeError` (before, a typo silently created a shared DAQ attribute).

**Tests** (`PYTHONPATH=. python tests/<file>.py`, `patcher_bot` env):

| Test | Result |
|---|---|
| `test_shared_amplifier.py` | 19 OK (adds: abort between DAQ steps still finishes every pipette) |
| `test_nidaq_signal_path.py` | 3 OK |
| `test_multipipette_helper.py` (new, tracked) | 16 OK: state isolation and typo rejection, `resolve_pipettes`, view key injection, `device_for_pipette`, channel map |
| `test_parallel_protocol_selection.py` (run its function directly; no pytest) | OK |
| Headless `fake_rig.json` stack, 2 pipettes (offscreen Qt, wired like `patch_gui.py`) | AutoPatchers get views; cell-mode toggle on Pipette 1 affects only Pipette 1; "Run Protocols" (voltage + current) runs as a `MultiPipettePatcher` task with one DAQ call per protocol covering both pipettes |

`test_cell_queue_coordinator.py` (missing module), `test_pipette_calibration_per_pipette.py` (pre-existing circular import when `patcherbot.devices` is imported first) and `test_norwich_pipette_detector.py` (2 errors, missing `BoRigCombo.pt`) fail identically on clean `2214d8f`. Not yet run on the Norwich rig.

