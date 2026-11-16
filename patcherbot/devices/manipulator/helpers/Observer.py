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

    def observe(self, *, pipette_inference=False, cell_inference=False, cell_masks=False,
                num_measurements=None, interval=None, raw_resistance=False):
        timestamp = self.get_time()
        frame = self.get_image()
        image = frame.pop("image")
        if cell_inference or cell_masks:
            cell = self.get_cell(image, include_masks=cell_masks, detect=image is not None)
            stage = cell["stage"]
        else:
            cell = None
            stage = self.get_stage()
        pipette = self.get_pipette(image=image, detect=pipette_inference and image is not None)
        ephys = self.get_ephys(num_measurements=num_measurements, interval=interval,
                               raw_resistance=raw_resistance)
        pressure = self.get_pressure()
        observation = dict(time=timestamp, image=frame, cell=cell, pipette=pipette, stage=stage, ephys=ephys, pressure=pressure)
        self.observation_deque.append(deepcopy(observation))
        return observation

    def get_time(self):
        """Return Unix time in seconds, rounded to millisecond precision."""
        return round(time.time(), 3)

    def get_image(self):
        """Return frame number, Unix seconds, and one shared image snapshot."""
        frame = self.stage.camera.last_raw_frame_data()
        number, timestamp, image = (None, None, None) if frame is None else frame
        return dict(frame_number=number,
                    time=None if timestamp is None else round(timestamp.timestamp(), 3),
                    image=None if image is None else image.copy())

    def get_ephys(self, *, num_measurements=None, interval=None, raw_resistance=False):
        """Collect resistance, access resistance, and capacitance together."""
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

    def get_pipette(self, detect=True, pixel_space=True, *, image=None):
        """Return measured um, projected pixels, and the complete detector result."""
        raw_position = self.unit.position()
        pixel_position = self.unit.reference_position()[:2] if pixel_space else None
        detection = None
        if detect and image is None:
            image = self.get_image()["image"]
        if detect and image is not None:
            detection = self.unit.pipetteCalHelper.pipetteDetector.detect_pipette_details(image.copy())
        return dict(raw_position=raw_position, pixel_position=pixel_position, pipette_detection=detection)

    def get_cell(self, image=None, *, detect=True, stage_data=None, include_masks=True, seed_points=None):
        """Collect detections, optional masks/centroids/interior targets, and their stage coordinates."""
        stage = self.get_stage() if stage_data is None else stage_data
        if image is None and detect:
            image = self.get_image()["image"]
        detections = None
        if detect and image is not None:
            detections = self.stage.cellDetectHelper._ensure_detector().detect_cells(image.copy())
        points = [] if detections is None else [row[:2] for row in detections]
        if seed_points is not None:
            points = list(seed_points)
        masks, centroids, targets = [], [], []
        if include_masks and points and image is not None:
            helper = self.stage._ensure_cell_track_helper()
            masks, centroids = helper.segment_cells(image, points)
            targets = [helper.segmentor.get_mask_target(mask) for mask in masks]
        return dict(cell_detections=detections, cell_masks=masks, cell_centroids=centroids,
                    cell_targets=targets, cell_seed_points=points, stage=stage)
