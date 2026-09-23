# Reference only; not imported by the application.
# Source: patcherbot/controller/patch.py
# Source SHA256: 8d838678fe37ef92977418fcaa8ea4b4166911719b831bdcad3f05a11f2d44fa
# Original ranges: observe 810-902, _observe_legacy 904-943
# The reference class preserves the original method indentation and docstrings.

class ObserveReference:
    def observe(self, include_pressure_state: bool = False, *, fields=None, sampling=None, raw_measurements=False):
        """Read selected signals; omitted fields retain the legacy observation.

        Explicit fields return a dictionary and read only their dependencies.
        Image-derived signals share one queued frame per call. Sampling maps
        measurement field names to num_measurements and/or interval settings.
        raw_measurements preserves electrical scalars without float32 conversion.
        """
        if fields is None:
            if sampling is not None:
                raise ValueError("sampling requires explicit observation fields")
            return self._observe_legacy(include_pressure_state)

        from collections.abc import Mapping
        from numbers import Integral, Real

        supported = {
            "pipette_positions", "stage_positions", "camera_image", "resistance",
            "pressure", "commanded_pressure_mbar", "pressure_atm_state",
            "access_resistance", "capacitance", "manipulator_position",
        }
        if isinstance(fields, str):
            raise TypeError("fields must be an iterable of names, not a string")
        requested = tuple(dict.fromkeys(fields))
        unknown = set(requested) - supported
        if unknown:
            raise ValueError(f"Unknown observation fields: {unknown}")

        defaults = {
            "resistance": (1, 0.001),
            "access_resistance": (5, 0.200),
            "capacitance": (5, 0.200),
        }
        if sampling is None:
            sampling = {}
        if not isinstance(sampling, Mapping):
            raise TypeError("sampling must map measurement fields to settings")
        sample_options = {}
        for field, options in sampling.items():
            if field not in defaults or field not in requested:
                raise ValueError(f"Sampling requires a requested measurement field: {field}")
            if not isinstance(options, Mapping):
                raise TypeError(f"Sampling settings for {field} must be a mapping")
            unknown_options = set(options) - {"num_measurements", "interval"}
            if unknown_options:
                raise ValueError(f"Unknown sampling settings for {field}: {unknown_options}")
            count, interval = defaults[field]
            count = options.get("num_measurements", count)
            interval = options.get("interval", interval)
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ValueError("num_measurements must be a positive integer")
            if (
                isinstance(interval, bool) or not isinstance(interval, Real)
                or not np.isfinite(interval) or interval < 0
            ):
                raise ValueError("interval must be a finite non-negative number")
            sample_options[field] = (int(count), float(interval))

        values = {}
        if "camera_image" in requested or "pipette_positions" in requested:
            _, _, _, frame = self.calibrated_stage.camera._last_frame_queue[0]
            if "camera_image" in requested:
                values["camera_image"] = frame
            if "pipette_positions" in requested:
                position = self.calibrated_unit.pipetteCalHelper.pipetteDetector.detect_pipette(frame)
                focus = self.calibrated_unit.pipetteFocusHelper.pipetteFocuser.get_pipette_focus_value(frame)
                values["pipette_positions"] = np.append(position, focus)
        if "manipulator_position" in requested:
            values["manipulator_position"] = self.calibrated_unit.position()
        if "stage_positions" in requested:
            stage = self.calibrated_stage.position()[:2]
            z = self.calibrated_unit.microscope.position() / self.calibrated_unit.config.microscope_units_per_um
            values["stage_positions"] = np.append(stage, z)
        # Preserve the requested order of electrical measurements.
        for field in requested:
            if field in defaults:
                count, interval = sample_options.get(field, defaults[field])
                read = {
                    "resistance": self.resistanceRamp,
                    "access_resistance": self.accessRamp,
                    "capacitance": self.capacitanceRamp,
                }[field]
                measurement = read(num_measurements=count, interval=interval)
                values[field] = measurement if raw_measurements else np.asarray(
                    [measurement], dtype=np.float32,
                )
        if "pressure" in requested:
            values["pressure"] = np.asarray([self.pressure.get_last_acquisition()], dtype=np.float32)
        if "commanded_pressure_mbar" in requested:
            values["commanded_pressure_mbar"] = np.asarray([self.pressure.get_pressure()], dtype=np.float32)
        if "pressure_atm_state" in requested:
            values["pressure_atm_state"] = np.asarray([float(bool(self.pressure.get_ATM()))], dtype=np.float32)
        return {field: values[field] for field in requested}

    def _observe_legacy(self, include_pressure_state: bool = False):
        import time
        t0 = time.perf_counter()

        _, _, _, img = self.calibrated_stage.camera._last_frame_queue[0]
        t1 = time.perf_counter()

        cvpi = self.calibrated_unit.pipetteCalHelper.pipetteDetector.detect_pipette(img)
        # Pass the current frame to focus estimator (it now requires an image argument)
        cvpiz = self.calibrated_unit.pipetteFocusHelper.pipetteFocuser.get_pipette_focus_value(img)
        cvpi = np.append(cvpi, cvpiz)
        t2 = time.perf_counter()

        pi = self.calibrated_unit.position()
        st = self.calibrated_stage.position()[:2]
        stz = self.calibrated_unit.microscope.position() / self.calibrated_unit.config.microscope_units_per_um
        st = np.append(st, stz)
        t3 = time.perf_counter()

        res = self.resistanceRamp(num_measurements=1, interval=0.001)
        t4 = time.perf_counter()

        # self.info(f"[observe timing] frame={ (t1-t0)*1e3:.1f} ms | detect={ (t2-t1)*1e3:.1f} ms | coords={ (t3-t2)*1e3:.1f} ms | resistanceRamp={ (t4-t3):.3f} s | total={ (t4-t0):.3f} s")
        if not include_pressure_state:
            return [cvpi, st, img, res]

        actual_pressure = self.pressure.get_last_acquisition()
        commanded_pressure = self.pressure.get_pressure()
        pressure_atm_state = float(bool(self.pressure.get_ATM()))

        observation = {
            "pipette_positions": cvpi,
            "stage_positions": st,
            "camera_image": img,
            "resistance": np.asarray([res], dtype=np.float32),
            "pressure": np.asarray([actual_pressure], dtype=np.float32),
            "commanded_pressure_mbar": np.asarray([commanded_pressure], dtype=np.float32),
            "pressure_atm_state": np.asarray([pressure_atm_state], dtype=np.float32),
        }
        return observation
