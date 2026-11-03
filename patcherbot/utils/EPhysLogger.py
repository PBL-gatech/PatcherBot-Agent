import logging
import math
from datetime import datetime
import threading
import os
from PyQt5 import QtGui
import imageio
import numpy as np
import cv2

class EPhysLogger(threading.Thread):
    def __init__(self, recording_state_manager, folder_path="experiments/Data/patch_clamp_data/", ephys_filename="ephys"):
        super().__init__()
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()
        testMode = self.recording_state_manager.testMode
        if testMode:
            folder_path = folder_path.replace("Data/", "Data/TEST_")
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/" + f"{ephys_filename}" + "/"
        self.ephys_filename = ephys_filename
        self.filename = self.folder_path + f"{ephys_filename}"
        self.file = None

        # file used to store cell metadata such as coordinates
        self.cell_metadata_file = os.path.join(self.folder_path, "cell_metadata.csv")

        self.folder_created = False
        self.write_event = threading.Event()

        # Dictionary to track unique index and color combinations
        self.index_color_dict = {}
        # Lock for thread-safe access to the dictionary
        self.index_color_lock = threading.Lock()
        self.image_type  = "webp"

    def create_folder(self):
        if not self.folder_created:
            try:
                os.makedirs(os.path.dirname(self.folder_path), exist_ok=True)
                self.folder_created = True
                logging.info(f"Created folder at: {self.folder_path}")
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)
        else:
            pass # Folder already created, no need to create it again
            # logging.debug("Folder already created. Skipping creation.")

    def _write_to_file(self, index, timeData, readData, respData, color, filename_override=None):
        # Check if "CurrentProtocol" is in filename
        if filename_override is None and "CurrentProtocol" in self.filename:
            with self.index_color_lock:
                if index not in self.index_color_dict:
                    # Index is unique, create a new entry with an empty list for colors
                    self.index_color_dict[index] = []

                # Proceed only if the color is unique for the given index
                if color not in self.index_color_dict[index]:
                    # Append the color to the list for this index
                    self.index_color_dict[index].append(color)
                else:
                    # Color is not unique for this index, skip writing
                    logging.debug("Skipping write: Index %s and color %s are not unique", index, color)
                    return

        # If "CurrentProtocol" is not in filename, proceed as the original method
        lines = [f"{timeData[i]} {readData[i]} {respData[i]}\n" for i in range(len(timeData))]
        # Open the file in append mode and write the formatted strings
        if filename_override:
            target_path = os.path.join(self.folder_path, f"{filename_override}.csv")
        else:
            target_path = f"{self.filename}_{index}_{color}.csv"

        logging.debug("Writing to file %s", target_path)
        with open(target_path, 'a+') as file:
            file.writelines(lines)
        self.write_event.set()  # Signal that writing is done

    def write_ephys_data(self, index, timeData, readData, respData, color, *, filename_override=None):
        self.create_folder()  # Ensure folder is created if it hasn't been
        self.write_event.clear()
        threading.Thread(
            target=self._write_to_file,
            args=(index, timeData, readData, respData, color, filename_override)
        ).start()

    def save_ephys_plot(self, index, plot, *, filename_override=None):
        self.create_folder()  # Ensure folder is created if it hasn't been

        if filename_override:
            image_path = os.path.join(self.folder_path, f"{filename_override}.png")
        else:
            image_path = f"{self.filename}_{index}.png"
        exporter = QtGui.QImage(plot.width(), plot.height(), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(exporter)
        plot.render(painter)
        painter.end()

        if exporter.save(image_path):
            logging.info("Saved plot to %s", image_path)
        else:
            logging.error("Failed to save plot to %s", image_path)

    def _optogenetic_basename(self, index, protocol_type):
        protocol_type = self._normalize_protocol_type(protocol_type)
        return f"{self.ephys_filename}_{index}_{protocol_type}"

    def write_optogenetic_data(self, index, timeData, readData, respData, protocol_type):
        basename = self._optogenetic_basename(index, protocol_type)
        self.write_ephys_data(index, timeData, readData, respData, "k", filename_override=basename)

    def write_optogenetic_stim_data(self, index, stim_data, protocol_type):
        if not stim_data:
            return
        self.create_folder()
        basename = self._optogenetic_basename(index, protocol_type)
        target_path = os.path.join(self.folder_path, f"{basename}_stim.csv")
        header = "start_s,end_s,state,wavelength,power_percent,replicate\n"
        write_header = not os.path.exists(target_path)
        with open(target_path, "a+", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            for step in stim_data:
                start_s = self._format_stim_time_value(step.get("start_s"))
                end_s = self._format_stim_time_value(step.get("end_s"))
                state = step.get("state", "unknown")
                wavelength = self._format_optogenetic_value(step.get("wavelength"))
                power = self._format_metadata_value(step.get("power_percent"))
                replicate = self._format_metadata_value(step.get("replicate"))
                f.write(f"{start_s},{end_s},{state},{wavelength},{power},{replicate}\n")

    def save_optogenetic_plot(self, index, plot, protocol_type):
        self.create_folder()
        basename = self._optogenetic_basename(index, protocol_type)
        image_path = os.path.join(self.folder_path, f"{basename}.webp")
        exporter = QtGui.QImage(plot.width(), plot.height(), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(exporter)
        plot.render(painter)
        painter.end()

        if exporter.save(image_path):
            logging.info("Saved plot to %s", image_path)
            return

        fallback_path = os.path.join(self.folder_path, f"{basename}.png")
        if exporter.save(fallback_path):
            logging.info("Saved plot to %s", fallback_path)
        else:
            logging.error("Failed to save plot to %s", fallback_path)

    def _normalize_image(self, image):
            """Return an 8-bit version of ``image`` suitable for saving."""
            if image is None:
                return None
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                image = image.astype(np.uint8)
            return image

    def _format_metadata_value(self, value):
        if value is None:
            return "NaN"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NaN"
        if math.isnan(value):
            return "NaN"
        return f"{value:.6g}"

    def _format_stim_time_value(self, value):
        if value is None:
            return "NaN"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NaN"
        if math.isnan(value):
            return "NaN"
        return f"{value:.17g}"

    def _normalize_protocol_type(self, protocol_type):
        if protocol_type is None:
            return "unknown"
        name = getattr(protocol_type, "name", None)
        if isinstance(name, str):
            return name.lower()
        text = str(protocol_type).strip().lower()
        return text if text else "unknown"

    def _format_optogenetic_value(self, value):
        if value is None:
            return "NaN"
        if isinstance(value, str):
            return value.strip() or "NaN"
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name.lower()
        return self._format_metadata_value(value)

    def _append_metadata_row(self, target_path, header, row):
        write_header = not os.path.exists(target_path)
        with open(target_path, "a+", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            f.write(row)

    def save_cell_metadata(self, index, stage_coords, image=None, *, image_fluo=None, voltage_hold=None, current_hold=None):
        """Save cell image and stage coordinates for a given protocol index."""
        self.create_folder()

        if image is None:
            logging.warning("No cell image provided; skipping cell metadata save")
            return

        img_filename = f"cell_{index}.webp"
        image = self._normalize_image(image)
        imageio.imwrite(os.path.join(self.folder_path, img_filename), image)

        img_fluo_filename = "NaN"
        if image_fluo is not None:
            img_fluo_filename = f"cell_{index}_fluo.webp"
            image_fluo = self._normalize_image(image_fluo)
            imageio.imwrite(os.path.join(self.folder_path, img_fluo_filename), image_fluo)

        timestamp = int(datetime.now().timestamp() * 1000)
        voltage_hold_str = self._format_metadata_value(voltage_hold)
        current_hold_str = self._format_metadata_value(current_hold)
        header = "index;stage_x;stage_y;stage_z;image;image_fluo;timestamp;voltage_hold_mV;current_hold_pA\n"
        row = (
            f"{index};{stage_coords[0]};{stage_coords[1]};{stage_coords[2]};"
            f"{img_filename};{img_fluo_filename};{timestamp};{voltage_hold_str};{current_hold_str}\n"
        )
        self._append_metadata_row(self.cell_metadata_file, header, row)

    def hold_image(self, index, image):
        if image is None:
            logging.error("No image to hold")
            return
        else:
            self.image_path = f"cell_{index}.webp"
            self.image = image

    def close(self):
        if self.file is not None:
            logging.info("CLOSING FILE: %s", self.filename)
            self.write_event.wait()  # Wait for the last task to complete
            self.file.close()
            self.file = None
        logging.info("Closing csv recorder writing thread")
