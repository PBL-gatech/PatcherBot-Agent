"""Collect one package per data source and retain a bounded observation history."""
from collections import deque
from copy import deepcopy
import time

import numpy as np


class Observer:
    def __init__(self, controller, attempt_id=None, *, maxlen=100):
        self.controller = controller
        self.unit = controller.calibrated_unit
        self.stage = controller.calibrated_stage
        self.attempt_id, self.attempt_token = attempt_id, object()
        self.observation_deque = deque(maxlen=maxlen)
        self.reference = None
        self.inference_image, self.inference_frame_number = None, None

    def set_reference(self, cell):
        """Bind existing selection/calibration data; do not acquire or infer."""
        if cell is None:
            self.reference = None
            return
        config = self.controller.config
        self.reference = deepcopy(dict(
            cell_position=cell[0], cell_capture_position=cell[2],
            home_position=self.controller.home_position,
            home_stage_position=self.controller.home_stage_position,
            cell_type=config.cell_type, cell_type_toggle=config.cell_type_toggle,
            cell_distance=config.cell_distance,
            slice_start_distance=config.slice_start_distance,
        ))

    def get_reference(self):
        return deepcopy(self.reference)

    def observe(self, *, pipette_inference=False, cell_inference=False, cell_masks=False,
                num_measurements=None, interval=None, raw_resistance=False, cached_ephys=False,
                pipette_anchor=None):
        timestamp = self.get_time()
        frame = self.get_image()
        image = frame.pop("image")
        self.inference_image, self.inference_frame_number = image, frame["frame_number"]
        if cell_inference or cell_masks:
            cell = self.get_cell(image, include_masks=cell_masks, detect=image is not None)
            stage = cell["stage"]
        else:
            cell = None
            stage = self.get_stage()
        pipette = self.get_pipette(image=image, detect=pipette_inference and image is not None,
                                   anchor=pipette_anchor)
        ephys = self.get_ephys(num_measurements=num_measurements, interval=interval,
                               raw_resistance=raw_resistance, cached=cached_ephys)
        pressure = self.get_pressure()
        observation = dict(time=timestamp, reference=self.get_reference(), image=frame, cell=cell, pipette=pipette, stage=stage, ephys=ephys, pressure=pressure)
        self.observation_deque.append(deepcopy(observation))
        return observation

    def get_inference_image(self, frame_number):
        """Return only the current inference frame, without acquiring another."""
        if frame_number != self.inference_frame_number:
            raise ValueError("Requested observation is not the current inference frame")
        if self.inference_image is None:
            raise ValueError("No image available for registration")
        return self.inference_image

    def get_time(self):
        """Return Unix time in seconds, rounded to millisecond precision."""
        return round(time.time(), 3)

    def get_image(self):
        """Return frame number, Unix seconds, and one shared image snapshot."""
        frame = self.stage.camera.last_raw_frame_data()
        number, timestamp, image = (None, None, None) if frame is None else frame
        return dict(frame_number=number,
                    width=None if image is None else image.shape[1],
                    height=None if image is None else image.shape[0],
                    time=None if timestamp is None else round(timestamp.timestamp(), 3),
                    image=None if image is None else image.copy())

    def get_ephys(self, *, num_measurements=None, interval=None, raw_resistance=False, cached=False):
        """Collect resistance, access resistance, and capacitance together."""
        if cached:
            daq = self.controller.daq
            return dict(resistance=daq.resistance(), access_resistance=daq.accessResistance(),
                        capacitance=daq.capacitance())
        resistance = (self.controller.daq.resistance() if raw_resistance else
            self.controller.resistanceRamp(num_measurements=1 if num_measurements is None else num_measurements,
                                           interval=.001 if interval is None else interval))
        count = 5 if num_measurements is None else num_measurements
        delay = .005 if interval is None else interval
        access = self.controller.accessRamp(num_measurements=count, interval=delay)
        capacitance = self.controller.capacitanceRamp(num_measurements=count, interval=delay)
        return dict(resistance=resistance, access_resistance=access, capacitance=capacitance)

    def get_pressure(self):
        """Collect measured/commanded pressure in mbar and atmospheric state."""
        measured = self.controller.pressure.get_last_acquisition_sample()[0]
        commanded = self.controller.pressure.get_pressure()
        atmospheric = self.controller.pressure.get_ATM()
        return dict(pressure=measured, commanded_pressure_mbar=commanded, pressure_atm_state=atmospheric)

    def get_stage(self, pixel_space=True):
        """Return measured XYZ in um and optional matrix-projected XY in pixels."""
        xy = self.stage.position()[:2]
        z = self.unit.microscope.position() / self.unit.config.microscope_units_per_um
        pixels = self.stage.reference_position()[:2] if pixel_space else None
        return dict(raw_position=np.append(xy, z), pixel_position=pixels)

    def get_pipette(self, detect=True, pixel_space=True, *, image=None, anchor=None):
        """Return measured um, projected pixels, and the complete detector result."""
        raw_position = self.unit.position()
        pixel_position = None
        if pixel_space:
            pixel_position = (self.unit.reference_position() if anchor is None
                              else self.unit.reference_position(anchor=anchor))[:2]
        detection = None
        if detect and image is None:
            image = self.get_image()["image"]
        if detect and image is not None:
            detection = self.unit.pipetteCalHelper.pipetteDetector.detect_pipette_details(image.copy())
        return dict(raw_position=raw_position, pixel_position=pixel_position, pipette_detection=detection)

    def get_cell(self, image=None, *, detect=True, stage_data=None, include_masks=True, seed_points=None,
                 include_interior_points=True, select_detection=None):
        """Collect detections, optional masks/centroids/interior targets, and their stage coordinates."""
        stage = self.get_stage() if stage_data is None else stage_data
        if image is None and detect:
            image = self.get_image()["image"]
        detections = None
        if detect and image is not None:
            detections = self.stage.cellDetectHelper._ensure_detector().detect_cells(
                image.copy(), include_boxes=True)
            if select_detection is not None and detections is not None and len(detections):
                selected = select_detection(dict(stage=stage, cell=dict(
                    cell_seed_points=[row[:2] for row in detections], cell_targets=detections)))
                detections = [] if selected is None else [selected]
        points = [] if detections is None else [row[:2] for row in detections]
        boxes = [] if detections is None else [row[3] for row in detections]
        if seed_points is not None:
            points = list(seed_points)
            boxes = [None] * len(points)  # Explicit seeds have no corresponding detector box.
        masks, centroids, targets = [], [], []
        if include_masks and points and image is not None:
            helper = self.stage._ensure_cell_track_helper()
            helper._prime_segmentor(image)  # Encode once; all prompts use this image.
            for point, box in zip(points, boxes):
                try:
                    if box is None:
                        mask = helper.segmentor.single_prediction(
                            np.asarray([point]), np.asarray([1]), multimask_output=False)
                    else:
                        predicted, _ = helper.segmentor.predict_mask_box(
                            np.asarray(box), multimask_output=False)
                        mask = predicted[0]
                except (ValueError, RuntimeError) as error:
                    self.controller.warning(f"Cell mask unavailable: {error}")
                    mask = None
                masks.append(mask)
                centroids.append(None if mask is None else helper._compute_centroid(mask))
                target = (helper.segmentor.get_mask_target(mask) if include_interior_points
                          else dict(mask=mask, target_points_xy=[]))
                target.update(box_xyxy=box, seed_point_xy=point)
                targets.append(target)
        return dict(cell_detections=detections, cell_masks=masks, cell_centroids=centroids,
                    cell_targets=targets, cell_seed_points=points, cell_boxes=boxes, stage=stage)
