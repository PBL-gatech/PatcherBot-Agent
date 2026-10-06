# Multiple pipette movement status and routing notes

This note describes the current state of the repo as it relates to multi-pipette movement, calibration, and routing. This is intentionally a code-level map for future debugging, not a claim that the system is fully complete or production-safe.

## 1. High-level architecture

The multi-pipette setup is built around a few key layers:

- A rig config defines the physical devices and how many pipettes are active.
- `RigConfigManager.build_devices()` instantiates the shared stage, camera, pressure, DAQ, amplifier, and one or more pipette controllers.
- `patch_gui.py` creates one `PipetteInterface` and one `AutoPatchInterface` per pipette.
- Each `PipetteInterface` owns a `CalibratedStage` and a `CalibratedUnit` for that specific pipette.
- The GUI switches the active object (`self.active_pipette` / `self.active_patch_interface`) when the user changes the selected pipette.
- A shared `CellQueueCoordinator` assigns cells to specific pipette workspaces based on geometry and/or pipette axis direction.

Most of this logic is split across:

- [patch_gui.py](../patch_gui.py)
- [patcherbot/interface/pipettes.py](../patcherbot/interface/pipettes.py)
- [patcherbot/interface/patch.py](../patcherbot/interface/patch.py)
- [patcherbot/devices/manipulator/calibratedunit.py](../patcherbot/devices/manipulator/calibratedunit.py)
- [patcherbot/devices/manipulator/CalibrationConfig.py](../patcherbot/devices/manipulator/CalibrationConfig.py)
- [patcherbot/interface/cell_queue.py](../patcherbot/interface/cell_queue.py)
- [rig_setup/rig_config.py](../rig_setup/rig_config.py)

---

## 2. How the rig is expanded from one pipette to many

The actual multi-pipette expansion begins in the rig config layer.

### 2.1 `rig_setup/rig_config.py`

`RigConfigManager.build_devices()` reads `config["pipette_count"]` and validates the requested count. It then slices each list-valued slot to the active number of pipettes:

- `pipette_controller`
- `pressure`
- `daq`
- `amplifier`
- `pipette_camera`

The code does roughly this:

- read `max_pipette_count = int(config.get("pipette_count", 1))`
- if `active_pipette_count` is missing, use the config default
- if the requested count is outside the allowed range, raise `RigConfigError`
- for each list-valued slot, instantiate the first `active_pipette_count` items
- assign a canonical key like `pipette_0`, `pipette_1`, etc.

This means hardware is built as a mapped device set rather than as one flat array. In a multi-pipette rig, each device instance is indexed per pipette.

The fake rig is a good example of the intended structure: see [rig_setup/rig_configs/fake_rig.json](../rig_setup/rig_configs/fake_rig.json). It declares:

- `"pipette_count": 3`
- three separate `pipette_controller` entries
- three separate `pipette_camera` entries
- three separate `pressure` entries
- three separate `daq` entries
- three separate `amplifier` entries

The generated `pipette_unit` object is then a dict keyed by the per-pipette device keys.

### 2.2 `patch_gui.py`

`patch_gui.py` is the runtime glue layer. After rig creation, it does this:

- gets `unit = rig_devices["pipette_unit"]`
- if `unit` is a dict, loops over `unit.keys()` and creates a dedicated `PipetteInterface` for each pipette
- stores a corresponding `AutoPatchInterface`
- stores a corresponding `GraphInterface`

Pseudo-flow:

```python
for i, id in enumerate(unit.keys()):
    pipette_controllers[id] = PipetteInterface(..., pipette_id=id)
    patch_controllers[id] = AutoPatchInterface(..., pipette_interface=pipette_controllers[id])
```

This means the pipette identity is attached early and then gets propagated into the patch task, calibration config, queue assignment, and GUI selection logic.

---

## 3. How GUI commands are routed to the correct pipette

This is one of the most important pieces of the current multi-pipette support.

### 3.1 Selecting the active pipette in the GUI

The main patch window keeps a `switch_manipulator_box` and connects it to `switch_active_pipette`:

- in [patcherbot/gui/patch.py](../patcherbot/gui/patch.py), `self.switch_manipulator_box.currentTextChanged.connect(self.switch_active_pipette)`
- `switch_active_pipette(id)` looks up `self.pipette_interfaces[id]`
- then it sets:
  - `self.active_pipette = self.pipette_interfaces[id]`
  - `self.active_patch_interface = self.patch_interfaces[id]`
- then it calls `tab.set_pipette(id)` for the active config tabs
- then it clears and rebuilds the registered keyboard/mouse actions

This is the central routing point: once the user selects a pipette, the live command dispatcher points at that pipette’s object.

### 3.2 The active object drives the button/key action

The actual key/mouse actions in [patcherbot/gui/manipulator.py](../patcherbot/gui/manipulator.py) are registered against `self.active_pipette` and `self.active_patch_interface`.

Examples:

- stage move: `self.active_pipette.move_stage_vertical`
- pipette move: `self.active_pipette.move_pipette_x`
- calibration: `self.active_pipette.calibrate_manipulator`
- patch actions: `self.active_patch_interface.break_in`

The `register_commands()` method is re-run after switching pipettes, so the same button/keyboard bindings point at the newly selected device.

### 3.3 Pipeline-level pipette ownership

`AutoPatchInterface` remembers its pipette identity:

- `self.pipette_controller = pipette_interface`
- `self.pipette_id = getattr(pipette_interface, "pipette_id", "pipette_0")`
- `self.pipette_index = getattr(pipette_interface, "pipette_index", 0)`

Then when cells are queued, the interface asks the queue for the next cell belonging to the current pipette:

```python
cell = self._cell_queue.next_cell(self.pipette_id)
```

This is how the queue and patch logic remain pipette-specific rather than globally shared.

---

## 4. How calibration matrices are generated and stored

There are two distinct but related calibration layers:

1. Stage calibration: maps stage encoder space to camera image space.
2. Pipette calibration: maps pipette manipulator encoder space to camera image space.

### 4.1 Stage calibration

The stage wrapper is a `CalibratedStage`, created in `PipetteInterface.__init__()`:

```python
self.calibrated_stage = CalibratedStage(stage, None, microscope, camera, config=self.calibration_config)
```

This stage is calibrated by `StageCalHelper` in [patcherbot/devices/manipulator/StageCalHelper.py](../patcherbot/devices/manipulator/StageCalHelper.py).

The flow is:

- move the stage a known diagonal distance
- record image frames while the stage moves
- compute optical flow between frames using OpenCV
- collect pairs of `(stage_um_delta, image_px_delta)`
- convert that to an affine transform using `cv2.estimateAffinePartial2D` or the equivalent stage calibration helper logic
- apply the transform with `CalibratedStage.calibrate()`

The key resulting matrices are:

- `self.M` : stage microns -> image pixels
- `self.M_inv` : inverse mapping
- `self.r0` : offset in pixel space
- `self.r0_inv` : inverse offset

The final stage calibration in `CalibratedStage.calibrate()` is:

```python
mat = self.stageCalHelper.calibrate(dist=self.config.stage_diag_move)
mat = np.append(mat, np.array([[0,0,1]]), axis=0)
mat_inv = pinv(mat)
self.r0 = mat[0:2, 2]
self.r0_inv = mat_inv[0:2, 2]
self.M = mat[0:2, 0:2]
self.M_inv = mat_inv[0:2, 0:2]
```

These are the stage matrices used later in `reference_position()` and `reference_relative_move()`.

### 4.2 Pipette calibration

The pipette wrapper is a `CalibratedUnit`, created in the same `PipetteInterface.__init__()` path:

```python
self.calibrated_unit = CalibratedUnit(unit, self.calibrated_stage, microscope, camera, config=self.calibration_config)
```

`CalibratedUnit` owns a `PipetteCalHelper`, which is responsible for generating a per-pipette transform.

Flow:

- `PipetteCalHelper.collect_cal_points()` walks the pipette around a small region
- for each point, it detects the pipette tip in camera pixels using the pipette detector (`detect_pipette`)
- it pairs that pixel coordinate with the current manipulator encoder position (`self.pipette.position()`)
- after enough points, `calibrate()` runs `cv2.estimateAffine2D(encoder_points, image_points)`
- the `2x3` affine result is embedded into a `3x4` homogeneous transform

The exact conversion appears in [patcherbot/devices/manipulator/PipetteCalHelper.py](../patcherbot/devices/manipulator/PipetteCalHelper.py):

```python
M2x3, inliers = cv2.estimateAffine2D(encoder_points, image_points)
mat3x4 = np.zeros((3, 4), dtype=np.float64)
mat3x4[0, 0:2] = M2x3[0, 0:2]
mat3x4[0, 3] = M2x3[0, 2]
mat3x4[1, 0:2] = M2x3[1, 0:2]
mat3x4[1, 3] = M2x3[1, 2]
mat3x4[2, 2] = 1.0
```

Then `CalibratedUnit.finish_calibration()` takes that and converts it into the runtime transform:

```python
mat = self.pipetteCalHelper.calibrate()
mat = np.vstack((mat, np.array([0,0,0,1])))
mat_inv = pinv(mat)
self.r0 = -mat[0:3, 3]
self.r0_inv = -mat_inv[0:3, 3]
self.M = mat[0:3, 0:3]
self.M_inv = mat_inv[0:3, 0:3]
self.calibrated = True
```

These matrices are what support the later calls such as:

- `pixels_to_um()`
- `um_to_pixels()`
- `reference_move()`
- `direct_pipette()`
- `center_pipette()`

### 4.3 Where the calibration is stored on disk

The calibration file is written by `PipetteInterface.write_calibration()` in [patcherbot/interface/pipettes.py](../patcherbot/interface/pipettes.py). It creates a folder like:

```python
experiments/Data/calibration_data/YYYY_MM_DD-HH_MM/{pipette_id}/calibration.pickle
```

and writes a dict with:

- `manip`: `self.calibrated_unit.save_configuration()`
- `stage`: `self.calibrated_stage.save_configuration()`
- `home`: concatenated home positions
- `safe`: concatenated safe positions
- `bath`: cleaning bath position

The save format for each device is small and explicit:

```python
config = {'up_direction': self.up_direction,
          'M': self.M,
          'r0': self.r0}
```

So the on-disk calibration is a runtime matrix snapshot, not a full config schema.

---

## 5. Where pipette-specific geometry is stored

There are two important kinds of per-pipette geometry in the repo:

1. runtime transform values used by motion code
2. user-configured workspace geometry used by queue assignment and motion planning

### 5.1 Runtime geometry in `CalibrationConfig`

`CalibrationConfig` in [patcherbot/devices/manipulator/CalibrationConfig.py](../patcherbot/devices/manipulator/CalibrationConfig.py) defines a dict-based interface for pipette-specific transforms:

- `pipette_rotation_matrix = param.Dict(default={}, doc="3x3 rotation matrix for each pipette keyed by pipette ID")`
- `pipette_affine_matrix = param.Dict(default={}, doc="3x4 affine transform for each pipette keyed by pipette ID")`

The lookup logic normalizes pipette IDs with `_normalise_pipette_id()`. That helper accepts values like:

- `pipette_0`
- `0`
- `pipette_0`

and resolves them to canonical pipette IDs as needed.

The resolution functions are:

- `resolve_pipette_rotation_matrix(pipette_id=None)`
- `resolve_pipette_axis_direction(pipette_id=None)`
- `resolve_pipette_affine_matrix(pipette_id=None)`

These are used by `CalibratedUnit.rotate()`:

```python
pipette_id = pipette_index
default_rotation = self.config.resolve_pipette_rotation_matrix(pipette_id)
if default_rotation is not None:
    coords = np.dot(default_rotation, coords)
```

So a pipette-specific rotation matrix changes the direction/orientation assumptions for that manipulator before a move is issued.

### 5.2 Workspace geometry in patch config

The user-facing workspace geometry is defined in [patcherbot/interface/patchConfig.py](../patcherbot/interface/patchConfig.py):

```python
pipette_geometry = List(default=[], doc='Per-pipette workspace geometry, e.g. [{id, x_um, y_um, angle_deg}]')
```

This is fed into `CellQueueCoordinator` in [patcherbot/interface/cell_queue.py](../patcherbot/interface/cell_queue.py):

- `self.geometry` is parsed as a list or dict
- each item is keyed by `id` or `pipette_id`
- `x_um`, `y_um`, and `angle_deg` are used to compute a geometric direction vector
- if no geometry is present, it falls back to the pipette rotation matrix for direction

The actual assignment logic is:

```python
angle = math.atan2(y_um, x_um)
...
rotation = config.resolve_pipette_rotation_matrix(pipette_id)
direction = rotation @ np.array([1.0, 0.0, 0.0])
angles[pipette_id] = math.atan2(direction[1], direction[0])
```

This is how cells are assigned to pipette workspaces rather than to a single global queue.

### 5.3 Hardcoded user values in the rig config

The user-configured positions are also hardcoded in the rig configuration files.

Examples:

- [rig_setup/rig_configs/fake_rig.json](../rig_setup/rig_configs/fake_rig.json) sets each pipette's starting position in the `devices.pipette_controller` array
- [patcherbot/interface/patchConfig.py](../patcherbot/interface/patchConfig.py) defines `pipette_geometry`
- the default rig setup in [rig_setup/rig_config.py](../rig_setup/rig_config.py) also includes fake manipulator parameters such as min/max ranges and initials

There is a strong split between:

- algorithmic calibration parameters (generated from images, stage movement, encoder data)
- static user metadata (rig geometry, pipette positions, nominal direction/angle, selected pipette count)

---

## 6. Current major issues that still need cleanup

The repo is clearly much farther along than a single-pipette implementation, but there are still several important rough edges.

### 6.1 Naming inconsistency between pipette IDs

The runtime now canonicalizes pipette ownership to `pipette_0`, `pipette_1`, ... and downstream code should use that scheme consistently.

There is explicit normalization logic in:

- `CalibrationConfig._normalise_pipette_id()`
- `CellQueueCoordinator._canonical_pipette_id()`

This is a workaround, not a single clean model. It means the system is tolerant of mismatched naming, but there is still no single canonical pipette ID convention across all code paths.

### 6.2 Runtime calibration and user configuration are not unified

There are two different sources of pipette direction/geometry truth:

- the generated transforms inside `CalibratedUnit.M`, `M_inv`, `r0`, `r0_inv`
- the user-configured dictionaries in `CalibrationConfig.pipette_rotation_matrix` and `pipette_affine_matrix`
- the workspace geometry list in `patch_config.pipette_geometry`

This is workable, but it is still fragmented. The system is not yet using one clean, explicit, per-pipette calibration record with a clear precedence order.

### 6.3 Shared camera assumptions are still fragile for multi-pipette operation

The pipette calibration helper frequently uses `self.camera.raw_frame_queue[0]` and a single detector path. That is fine for a single camera or a shared microscope image, but it is not a strong abstraction for a multi-camera rig where each pipette may have its own image source or different timing characteristics.

The multi-pipette runtime is partly wired for multiple device objects, but the calibration helpers still treat the camera as a shared global resource more often than a per-pipette resource.

### 6.4 The GUI routing is global-state-driven

The GUI dispatch mechanism relies on a single active object (`self.active_pipette`, `self.active_patch_interface`) and repopulates actions when a pipette is switched. That is functional, but it is inherently vulnerable to stale state, race conditions, and “one active object is wrong” bugs when commands are triggered while the UI is in the middle of switching.

This is acceptable as a prototype, but not a robust multi-pipette runtime model.

### 6.5 Collision guarding is directional, not physical

`CellQueueCoordinator._validate_workspace_geometry()` checks angle separation between pipettes. It rejects workspaces that are too similar. That is a useful first guard, but it does not validate actual reachability, stage clearance, or true physical overlap. It is a geometric heuristic, not a full robot-safety model.

### 6.6 There are signs of partially finished single-instance fallback logic

In the device builder, the camera injection path contains logic that is still clearly mixed between single-instance and multi-instance cases. In [rig_setup/rig_config.py](../rig_setup/rig_config.py), the `else` block for the non-dict case contains an undefined variable reference (`pipette_controller`), which indicates that the multi-pipette path and the single-pipette path have not been fully normalized.

This is not necessarily fatal in the current fake-rig path, but it is a real code smell and a likely source of edge-case failures.

---

## 7. Bottom line

The current codebase has moved a significant amount of functionality toward true multi-pipette support:

- device construction supports multiple pipettes
- each pipette gets its own `PipetteInterface`
- GUI selection switches the active pipette
- queue assignment is pipette-aware
- stage and pipette calibration each generate their own runtime transform matrices
- pipette-specific direction/geometry is stored in config dicts and workspace geometry metadata

What is still missing is a cleaner canonical model for pipette identity, a single explicit calibration record per pipette, and stricter handling of the assumptions that still leak through from the original single-pipette design.

That is the current state of the implementation: functional enough to route commands by pipette and estimate per-pipette geometry, but not yet fully simplified or hardened for production-scale multi-pipette operation.
