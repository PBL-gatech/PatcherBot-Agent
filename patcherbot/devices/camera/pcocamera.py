'''
Camera for a PCO Panda Camera
'''
import numpy as np

from . import *
import warnings
import pco


try:
    import cv2
except ImportError:
    warnings.warn('OpenCV is not installed.')

# ? See __init__.py for the following line
# __all__ = ['PcoCamera']


class PcoCamera(Camera):
    '''A camera class for the PCO Panda microscope camera.
       more info on the camera can be found here: https://www.pco.de/fileadmin/fileadmin/user_upload/pco-manuals/pco.panda_manual.pdf
    '''

    PCO_RECORDER_LATEST_IMAGE = 0xFFFFFFFF

    def __init__(self, width: int = 1280, height: int = 1280):
        """
        Initialize a PCO Panda camera and start continuous acquisition.

        Args:
            width (int, optional): Width of images to acquire. Defaults to 1280.
            height (int, optional): Height of images to acquire. Defaults to 1280.
        """
        super().__init__()

        self.width = width # update superclass img width / height vars
        self.height = height
        self.auto_normalize = False

        #setup the pco camera for continuous streaming
        self.cam = pco.Camera()

        print(f"CAMERA {self.cam}")

        # self.ca .sdk.set_timestamp_mode('binary & ascii')
        config = {'exposure time': 5e-3,
                    'roi': (385, 385, 1664, 1664),
                    'timestamp': 'off',
                    'trigger': 'auto sequence',
                    'acquire': 'auto',
                    'metadata': 'on',
                    'binning': (1, 1)}

        self.cam.configuration = config

        self.cam.record(number_of_images = 10, mode='ring buffer') #use "ring buffer" mode for continuous streaming from camera
        self.cam.wait_for_first_image()

        self.frameno = None

        self.currExposure = 0

        self.upperBound = 255
        self.lowerBound = 0

        self.last_frame_time = None
        self.fps = 0
        self.lastFrame = None



        self.normalize() #normalize image on startup

        self.start_acquisition() #start thread that updates camera gui

    def set_exposure(self, value: float) -> None:
        """
        Set the exposure time of the camera.

        Args:
            value (float): Desired exposure time in milliseconds.
        """
        self.cam.set_exposure_time(value / 1000)

    def get_exposure(self):
        '''
        return the exposure time of the camera in ms

        Returns:
            float: Exposure time in milliseconds.
        '''
        exposure = self.cam.get_exposure_time() # this is in seconds
        self.currExposure = exposure
        return exposure * 1000 #convert to ms

    def close(self):
        """
        Stop the camera and release resources.
        """
        if self.cam:
            self.cam.stop()
            self.cam.close()

    def reset(self) -> None:
        """
        Reset the camera to default configuration and restart streaming.
        """
        self.cam = pco.Camera()
        
        config = {'exposure time': 10e-3,
                    'roi': (0, 0, 2048, 2048),
                    'timestamp': 'off',
                    'pixel rate': 500_000_000,
                    'trigger': 'auto sequence',
                    'acquire': 'auto',
                    'metadata': 'on',
                    'binning': (1, 1)}
        self.cam.configuration = config

        self.cam.record(number_of_images=10, mode='ring buffer')
        self.cam.wait_for_first_image()

    def normalize(self, img = None) -> None:
        """
        Set the normalization bounds (upper and lower) based on an image.

        If no image is provided, the latest image from the camera is used.

        Args:
            img (np.ndarray, optional): Image used to compute normalization bounds. Defaults to None.
        """
        if not self.auto_normalize:
            print("NORMALIZING")   

        # print(f"BEFORE IMAGE: {img}")
        if img is None:
            img = self.get_16bit_image()
            # print(f"IMAGE after get_16bit_image: {img}")
            # print(type(img))
        if img is None:
            return
        # print(f"AFTER IMAGE: {img}")
        #is there a better way to do this?
        #maybe 2 stdevs instead?
        self.lowerBound = img.min()
        self.upperBound = img.max()

    def unnormalize(self, img = None) -> None:
        """
        Reset normalization bounds to default (0-255).

        Args:
            img (np.ndarray, optional): Image to reset (not used). Defaults to None.
        """
        if not self.auto_normalize:
            print("UNNORMALIZING")
        self.lowerBound = 0
        self.upperBound = 255

    def autonormalize(self,flag = None):
        """
        Enable or disable automatic normalization.

        Args:
            flag (bool, optional): True to enable auto-normalization, False to disable.

        Returns:
            bool: Current state of auto-normalization.
        """
        self.auto_normalize = flag
        return self.auto_normalize

    def get_frame_no(self) -> int:
        """
        Enable or disable automatic normalization.

        Args:
            flag (bool, optional): True to enable auto-normalization, False to disable.

        Returns:
            bool: Current state of auto-normalization.
        """
        return self.frameno
        
    def get_16bit_image(self) -> np.ndarray:
        '''get a 16 bit color image from the camera (no normalization)
           this compares to raw_snap which returns a 8 bit image with normalization
        
        Returns:
            np.ndarray: Latest image as a 16-bit array.
        '''
        # if self.frameno == self.cam.rec.get_status()['dwProcImgCount'] and self.lastFrame is not None:
        #     return self.lastFrame
        # else:
        # print('-----get 16 bit image----- PcoCamera.py')
        self.frameno = self.cam.rec.get_status()['dwProcImgCount']
        # self.get_frame_rate()
        
        try:
            # print(f"IMAGE NUMBER: {PcoCamera.PCO_RECORDER_LATEST_IMAGE}")
            # img, meta = self.cam.image(PcoCamera.PCO_RECORDER_LATEST_IMAGE)
            # this is the line that is causing an error if pco <= 2.1.2
            img, meta = self.cam.image(image_number=PcoCamera.PCO_RECORDER_LATEST_IMAGE)
            self.lastFrame = img
            # logging.debug(f"Got image from camera: {datetime.now()}")
            # print(meta)
        except Exception as e:
            print(f"ERROR in get_16bit_image: {e}")
            return self.lastFrame # there was an error grabbing the most recent frame

        return img

    def raw_snap(self):
        '''
        Returns the current image (8 bit color, with normalization).
        This is a blocking call (wait until next frame is available)
        
        Returns:
            np.ndarray: Normalized 8-bit image.
        '''
        img = self.get_16bit_image()

        if self.auto_normalize:
            # print("AutoNormalizing")
            self.normalize(img)

        if img is None:
            return None

        # apply upper / lower bounds (normalization)
        # span = np.maximum(self.upperBound - self.lowerBound, 1)  # Avoid division by zero

        # img = np.clip((img.astype(np.float32) - self.lowerBound) / span * 255, 0, 255).astype(np.uint8)
        img = self.apply_normalization(img)
        # # resize if needed
        # if self.width != None and self.height != None:
        #     img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_LINEAR)

        return img


    def apply_normalization(self, img: np.ndarray) -> np.ndarray:
        '''
        Apply normalization to a given image
        
        Args:
            img (np.ndarray): Image to normalize.

        Returns:
            np.ndarray: Normalized 8-bit image with values clipped between 0 and 255.
                None if image is None
        '''
        if img is None:
            return None

        span = np.maximum(self.upperBound - self.lowerBound, 1)  # Avoid division by zero

        img = np.clip((img.astype(np.float32) - self.lowerBound) / span * 255, 0, 255).astype(np.uint8)

        return img
