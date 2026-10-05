'''
A generic camera class

TODO:
* A stack() method which takes a series of photos along Z axis
'''
from __future__ import print_function
import collections
import os
import datetime
import time
import threading
import imageio
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage.filters import gaussian_filter
# from scipy.ndimage import fourier_gaussian
import warnings
import traceback
from scipy.optimize import brentq
try:
    import cv2
except:
    warnings.warn('OpenCV not available')

__all__ = ['Camera', 'FakeCamera', 'RecordedVideoCamera']




class AcquisitionThread(threading.Thread):
    def __init__(self, camera, queues, raw_queues):
        self.camera = camera
        self.queues = queues
        self.raw_queues = raw_queues
        self.running = True
        
        self.last_frame_time = None
        self.fps = 0

        threading.Thread.__init__(self, name='image_acquire_thread')

    def get_frame_rate(self):
        # * A way to calculate FPS
        current_time = time.time()
        if self.last_frame_time is not None:
            self.fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time

        return self.fps

    def run(self):
        self.running = True

        start_time = time.time()

        last_report = start_time
        acquired_frames = 0
        last_frame = 0
        while self.running:
            snap_time = time.time()
            retrieval_started_at = time.monotonic()
            try:
                raw, processed = self.camera.snap()
                time.sleep(0.02)  # Simulate processing time
                # logging.debug(f"frame captured")
            except Exception as ex:
                print('something went wrong acquiring an image, waiting for 100ms: ')
                traceback.print_exception(type(ex), ex, ex.__traceback__)
                time.sleep(.1)
                continue
            frame_time = datetime.datetime.now()
            elapsed = snap_time - start_time
            processed_image = processed.copy() if hasattr(processed, "copy") else processed
            raw_image = raw.copy() if hasattr(raw, "copy") else raw
            processed_entry = (last_frame, frame_time, elapsed, processed_image)
            raw_entry = (last_frame, frame_time, elapsed, raw_image)
            self.camera._update_frame_pair(
                processed_entry, raw_entry, retrieval_started_at=retrieval_started_at)
            # Put image into queues for disk storage and display
            for queue in self.queues:
                queue.append(processed_entry)
            for queue in self.raw_queues:
                queue.append(raw_entry)

            # logging.debug(f"FPS in Acquisition Thread: {self.get_frame_rate():.2f}")

            last_frame += 1
            acquired_frames += 1
            if snap_time - last_report > 1:
                # frame_rate = acquired_frames / (snap_time - last_report)
                # logging.warning('Acquiring {:.1f} fps'.format(frame_rate))
                last_report = snap_time
                acquired_frames = 0
        
        # Put the end marker into the queues
        for queue in self.queues:
            queue.append((None, None, None, None))


class Camera(object):
    """
    Base class for all camera devices. At the end of the initialization, derived classes need to
    call self.start_acquisition() to start the thread that continously acquires images from the
    camera. Frames and their retrieval timing are made available to consumers.
    """
    def __init__(self):
        super(Camera, self).__init__()
        self._file_queue = None
        self._last_frame_queue = collections.deque(maxlen=1)
        self.raw_frame_queue = collections.deque(maxlen=1)
        self._acquisition_thread = None
        self._file_thread = None
        self._debug_write_delay = 0
        self.width = 1000
        self.height = 1000
        self.flipped = False # Horizontal flip
        self.auto_normalize = False

        self.stop_show_time = 0
        self.point_to_show = None
        self._transient_circles = ((), 0.0)
        self.cell_list = []
        self._frame_pair_lock = threading.Lock()
        self._last_frame_pair = None
        
        self.last_frame_time = None
        self.fps = 0

        # Default off; RigConfigManager applies calibration.use_ai_features after instantiation.
        self.use_ai_features = False

        self.Cellseg = None
        self._cellseg_error = None
        # testing flag
        

    def show_circle(self, point, color=(255, 255, 255), radius=10, duration=1.5, show_center=False):
        self.show_circles([point], color, radius, duration, show_center)

    def show_circles(self, points, color=(255, 255, 255), radius=10, duration=1.5, show_center=False):
        circles = tuple((tuple(map(int, p)), int(radius), color, bool(show_center)) for p in points)
        stop_show_time = time.time() + duration if circles else 0.0
        self._transient_circles = (circles, stop_show_time)
        self.point_to_show = list(circles[0]) if len(circles) == 1 else None
        self.stop_show_time = stop_show_time

    def start_acquisition(self):
        self._acquisition_thread = AcquisitionThread(camera=self,
                                                     queues=[self._last_frame_queue],
                                                     raw_queues=[self.raw_frame_queue])
        self._acquisition_thread.start()

    def stop_acquisition(self):
        if self._acquisition_thread is not None:
            self._acquisition_thread.running = False


    def flip(self):
        self.flipped = not self.flipped


    def segment(self, img, cell, label):
        segmentor = self._ensure_cellseg()
        mask = segmentor.segment(image = img, input_point = cell, input_label = label)
        return mask

    def _ensure_cellseg(self):
        if not self.use_ai_features:
            raise NotImplementedError(
                "SAM2 segmentation unavailable. AI features need to be enabled in "
                "calibration config before use. Set calibration.use_ai_features to true."
            )
        if self._cellseg_error is not None:
            raise NotImplementedError(f"SAM2 is not available: {self._cellseg_error}") from self._cellseg_error
        if self.Cellseg is None:
            try:
                from patcherbot.deepLearning.cellSegmentor import CellSegmentor2

                self.Cellseg = CellSegmentor2()
            except Exception as exc:
                self._cellseg_error = exc
                raise NotImplementedError(f"SAM2 is not available: {exc}") from exc
        return self.Cellseg
    
    def mask_test(self, mask,cell,img):
        if mask is not None:
            # save the image to a directory once for each unique cell coordinates
            mask_dir = r'C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\devices\camera\FakeMicroscopeImgs\mask_images'
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            filename = f'mask_cell_{cell[0]}_{cell[1]}.png'
            filepath = os.path.join(mask_dir, filename)
            if not os.path.exists(filepath):
                cv2.imwrite(filepath, mask * 255)  # multiply by 255 to convert boolean mask to visible image
            filename = f'raw_{cell[0]}_{cell[1]}.png'
            filepath = os.path.join(mask_dir, filename)
            if not os.path.exists(filepath):
                cv2.imwrite(filepath, img)

    # def inpaint(self, img, mask):

    def preprocess(self, input_img):
        img = input_img.copy()

        circles, stop_show_time = self._transient_circles
        if circles and time.time() < stop_show_time:
            for point, radius, color, show_center in circles:
                img = cv2.circle(img, point, radius, color, -1)
                if show_center:
                    img = cv2.circle(img, point, 2, color, 3)
        elif circles:
            self._transient_circles = ((), 0.0)
            self.point_to_show = None
            self.stop_show_time = 0.0

        # Process each cell's segmentation.
        for cell_coords, _, cell_img in self.cell_list:
            # cell_coords is assumed to be a 2D coordinate [x, y] in full-image space.
            x, y = cell_coords[0], cell_coords[1]
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue  # Skip if the cell is offscreen.
            # final_mask = None
            # if final_mask is None:
            #     if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            #         rgbimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            #     else:
            #         rgbimg = img

            #     # Convert rgbimg to float32 and normalize if needed.
            #     if rgbimg.dtype != np.float32:
            #         rgbimg = rgbimg.astype(np.float32) / 255.0

            #     cell_point_full = np.array(cell_coords, dtype=np.float32).reshape(1, 2)
            #     label_full = np.array([1], dtype=np.int32)
            #     mask = self.segment(rgbimg, cell_point_full, label_full)
            # else:
            #     mask = final_mask

            # if mask is not None:
            #     if mask.dtype != np.uint8:
            #         mask = mask.astype(np.uint8)
            #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #     total_area = sum(cv2.contourArea(c) for c in contours)
            #     image_area = self.width * self.height

            #     # Skip drawing if the mask area is too large.
            #     if total_area > 0.03 * image_area:
            #         continue

                # cv2.drawContours(img, contours, -1, (0, 255, 0), thickness=1)
            img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), 1)

        if self.flipped:
            img = img[:, ::-1]

        return img

    def new_frame(self):
        '''
        Returns True if a new frame is available
        '''
        return True

    def snap(self):
        '''
        Returns a processed and a raw image
        '''
        raw = self.raw_snap()
        return raw, self.preprocess(raw)

    def _update_frame_pair(self, processed_entry, raw_entry, *, retrieval_started_at=None) -> None:
        with self._frame_pair_lock:
            # snap() may return a buffered image; sensor capture time is unknown.
            timing = dict(acquired_at=None, retrieval_started_at=retrieval_started_at,
                          available_at=time.monotonic(), timestamp_basis="camera_retrieval")
            self._last_frame_pair = (processed_entry, raw_entry, timing)

    def raw_snap(self):
        return None

    def get_16bit_image(self):
        '''
        Returns the current image as a 16-bit color image
        '''
        return None

    def last_frame(self) -> None | tuple[int, np.ndarray]:
        '''
        Get the last snapped frame and its number

        Returns
        -------
        (frame_number, frame)
        '''
        try:
            # * the deque has a maxsize of 1, so we can do this
            # maybe we should use a list instead of a deque?
            # ? should we change this to grab the last element instead, as it is more intuitive?
            last_entry = self._last_frame_queue[0]
            # last_entry = self._last_frame_queue[-1]
            return last_entry[0], last_entry[-1]
        except IndexError:  # no frame (yet)
            return None
    
    def last_frame_data(self) -> None | tuple[int, datetime.datetime, np.ndarray]:
        '''
        Get the last snapped frame and its number

        Returns
        -------
        (frame_number, date, frame)
        '''
        try:
            last_entry = self._last_frame_queue[0]
            return last_entry[0], last_entry[1], last_entry[-1]
        except IndexError:  # no frame (yet)
            return None

    def last_raw_frame_data(self, *, include_timing=False):
        '''
        Get the last raw frame and its number

        Returns
        -------
        (frame_number, date, raw_frame), plus matching timing when requested.
        Timing describes retrieval/publication, not sensor exposure.
        '''
        with self._frame_pair_lock:
            if self._last_frame_pair is None:
                return None
            _, raw_entry, timing = self._last_frame_pair
        result = (raw_entry[0], raw_entry[1], raw_entry[-1])
        return result + (dict(timing),) if include_timing else result

    def last_frame_pair(self) -> None | tuple[int, datetime.datetime, np.ndarray, np.ndarray]:
        '''
        Return the latest processed/raw frame pair
        '''
        with self._frame_pair_lock:
            if self._last_frame_pair is None:
                return None
            processed_entry, raw_entry, _ = self._last_frame_pair
        return processed_entry[0], processed_entry[1], processed_entry[-1], raw_entry[-1]

    def close(self):
        """Shut down the camera device, free resources, etc."""
        pass

    def set_exposure(self, value):
        print('Setting exposure time not supported for this camera')

    def get_exposure(self):
        print('Getting exposure time not supported for this camera')
        return -1

    def change_exposure(self, change):
        if self.get_exposure() > 0:
            self.set_exposure(self.get_exposure() + change)

    def normalize(self):
        print('Normalizing not supported for this camera')

    def unnormalize(self):
        print('Unnormalizing not supported for this camera')
        
    def autonormalize(self, state):
        self.auto_normalize = state
        print('Autonormalizing not supported for this camera')
        return self.auto_normalize

    def auto_exposure(self):
        '''
        Auto exposure assumes frames are 8 bits.
        '''
        mean_luminance = 127

        def f(value):
            self.set_exposure(value)
            time.sleep(.2+.001*value) # wait for new frame with updated value
            while not self.new_frame():
                time.sleep(0.05)
            m = self.snap().mean()
            return m-mean_luminance
        exposure = brentq(f, 0.1,100., rtol=0.1)
        self.set_exposure(exposure)

    def get_frame_rate(self) -> None:
        # return super().get_frame_rate()
        # * A way to calculate FPS
        current_time = time.time()
        if self.last_frame_time:
            self.fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time

        logging.debug(f"FPS in Camera: {self.fps:.2f}")

    def reset(self):
        pass

    def get_frame_no(self):
        raise NotImplementedError('get_frame_no not implemented for this camera')

class FakeCamera(Camera):
    def __init__(self, manipulator=None, image_z=0, paramecium=False):
        super(FakeCamera, self).__init__()
        self.width = 1024
        self.height = 768
        self.exposure_time = 30
        self.manipulator = manipulator
        self.image_z = image_z
        self.scale_factor = .5  # micrometers in pixels
        self.depth_of_field = 2.
        self.frame = np.array(np.clip(gaussian_filter(np.random.randn(self.width * 2, self.height * 2) * 0.5, 10) * 50 + 128, 0, 255), dtype=np.uint8)
        
        self.start_acquisition()

    def set_exposure(self, value):
        if 0 < value <= 200:
            self.exposure_time = value

    def get_exposure(self):
        return self.exposure_time

    def get_microscope_image(self, x, y, z):
        frame = np.roll(self.frame, int(y), axis=0)
        frame = np.roll(frame, int(x), axis=1)
        frame = frame[self.height // 2:self.height // 2 + self.height,
                      self.width // 2:self.width // 2 + self.width]
        return np.array(frame, copy=True)

    def raw_snap(self):
        '''
        Returns the current image.
        This is a blocking call (wait until next frame is available)
        '''
        if self.manipulator is not None:
            # Use the part of the image under the microscope

            stage_x, stage_y, stage_z = self.manipulator.position_group([4, 5, 6])
            stage_z -= self.image_z
            stage_x *= self.scale_factor
            stage_y *= self.scale_factor
            stage_z *= self.scale_factor
            frame = self.get_microscope_image(stage_x, stage_y, stage_z)
            if self.paramecium is not None:
                self.paramecium.update_position()
                p_x, p_y, p_z = self.paramecium.position
                p_angle = self.paramecium.angle + np.pi / 2
                p_x *= self.scale_factor
                p_y *= self.scale_factor
                p_z *= self.scale_factor
                p_width = 30 * self.scale_factor
                p_height = 100 * self.scale_factor
                xx, yy = np.meshgrid(np.arange(-self.width // 2, self.width // 2), np.arange(-self.height // 2, self.height // 2))
                frame[((xx - (p_x - stage_x)) * np.cos(p_angle) + (yy - (p_y - stage_y)) * np.sin(p_angle)) ** 2 / (p_width / 2) ** 2 +
                      ((xx - (p_x - stage_x)) * np.sin(p_angle) - (yy - (p_y - stage_y)) * np.cos(p_angle)) ** 2 / (p_height / 2) ** 2 < 1] = 50
                frame[((xx - (p_x - stage_x)) * np.cos(p_angle) + (yy - (p_y - stage_y)) * np.sin(p_angle)) ** 2 / (p_width / 2) ** 2 +
                      ((xx - (p_x - stage_x)) * np.sin(p_angle) - (yy - (p_y - stage_y)) * np.cos(p_angle)) ** 2 / (p_height / 2) ** 2 < 0.8] = 100

            for direction, axes in [(-np.pi / 2, [1, 2, 3])]:
                manipulators = np.zeros((self.height, self.width), dtype=np.int16)
                x, y, z = self.manipulator.position_group(axes)
                x = np.cos(self.manipulator.angle) * (x + 50 / self.scale_factor)
                z = np.sin(self.manipulator.angle) * (x + 50 / self.scale_factor) + z
                x *= self.scale_factor
                y *= self.scale_factor
                z *= self.scale_factor
                # cut off a tip
                # Position relative to stage
                x -= stage_x
                y -= stage_y
                z -= stage_z
                X, Y = np.meshgrid(np.arange(self.width) - self.width // 2 + x,
                                   np.arange(self.height) - self.height // 2 + y)
                angle = np.arctan2(X, Y)
                dist = np.sqrt(X ** 2 + Y ** 2)
                border = (0.075 + 0.0025 * abs(z) / self.depth_of_field)
                manipulators[(np.abs(angle - direction) < border) & (dist > 50)] = 5
                edge_width = 0.02 if z > 0 else 0.04  # Make a distinction between below and above
                manipulators[(np.abs(angle - direction) < border) & (np.abs(angle - direction) > border-edge_width) & (dist > 50)] = 75
                frame[manipulators>0] = manipulators[manipulators>0]
        else:
            img = Image.fromarray(self.frame)
            frame = np.array(img.resize((self.width, self.height)))
        exposure_factor = self.exposure_time / 30.
        frame = frame + np.random.randn(self.height, self.width) * 5

        return np.array(np.clip(frame * exposure_factor, 0, 255),
                        dtype=np.uint8)


def text_phantom(text, size):
    # Availability is platform dependent
    font = 'Arial'
    
    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size[0] // len(text),
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', size, (0, 0, 0))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size[0] - text_width) // 2,
              (size[1] - text_height) // 2)
    white = "#ffffff"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    frame = np.asarray(canvas)
    return frame


class DebugCamera(Camera):
    '''A fake camera that shows the frame number'''
    def __init__(self, frames_per_s=20, write_delay=0):
        super(DebugCamera, self).__init__()
        self.width = 1024
        self.height = 768
        self.frameno = 0
        self.last_frame_time = None
        self.delay = 1 / frames_per_s
        self._debug_write_delay = write_delay
        self.start_acquisition()

    def get_frame_rate(self):
        return 1 / self.delay

    def raw_snap(self):
        '''
        Returns the current image.
        This is a blocking call (wait until next frame is available)
        '''
        frame = text_phantom('{:05d}'.format(self.frameno), (self.width, self.height))
        self.frameno += 1
        if self.last_frame_time is not None:
            if time.time() - self.last_frame_time < self.delay:
                time.sleep(self.delay - (time.time() - self.last_frame_time))
        self.last_frame_time = time.time()
        return frame


class RecordedVideoCamera(Camera):
    def __init__(self, file_name, pixel_per_um, slowdown=1):
        super(RecordedVideoCamera, self).__init__()
        self.file_name = file_name
        self.video = cv2.VideoCapture(file_name)
        self.video.open(self.file_name)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pixel_per_um = pixel_per_um
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.time_between_frames = 1 / self.frame_rate * slowdown
        self._last_frame_time = None
        self.start_acquisition()

    def get_frame_rate(self):
        return self.frame_rate

    def raw_snap(self):
        if self._last_frame_time is not None:
            if time.time() - self._last_frame_time < self.time_between_frames:
                # We are too fast, sleep a bit before returning the frame
                sleep_time = self.time_between_frames - (time.time() - self._last_frame_time)
                time.sleep(sleep_time)
        success, frame = self.video.read()
        self._last_frame_time = time.time()

        if not success and self._acquisition_thread.running:
            raise ValueError(f'Cannot read from file {self.file_name}.')

        return frame

    def close(self):
        self.video.release()
        super(RecordedVideoCamera, self).close()
