from numpy import random

# from patcherbot.deepLearning.virtualCell import VirtualCell
from numpy import random

# from patcherbot.deepLearning.virtualCell import VirtualCell
from patcherbot.devices.cellsorter.CellSorter import CellSorterManip
from patcherbot.devices.manipulator import Manipulator, FakeManipulator
from .camera import Camera
import numpy as np
import cv2
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

FAKE_STAGE_REFERENCE_POSITION = np.array([-235000.0, 55000.0, 285000.0])

class FakeCalCamera(Camera):
    """
    Simulated calibration camera that generates synthetic microscope images
    using stage, pipette, and optional cell sorter manipulators.
    """
    def __init__(self, stageManip=None, pipetteManip=None, image_z=0, targetFramerate=40, cellSorterManip=None, background_image=None):
        """
        Initialize the fake calibration camera.

        args:
            stageManip (Manipulator, optional): Stage manipulator providing position data.
            pipetteManip (Manipulator, optional): Pipette manipulator.
            image_z (float): Z-plane corresponding to focus.
            targetFramerate (float): Desired frame rate for simulation.
            cellSorterManip (CellSorterManip, optional): Cell sorter manipulator.
        """
        super(FakeCalCamera, self).__init__()
        self.width : int = 1024
        self.height : int = 1024
        self.exposure_time : int = 30
        self.pixels_per_micron : float = 1.25  # pixels / micrometers
        self.stageManip : Manipulator = stageManip
        if isinstance(pipetteManip, dict):
            self.pipetteManip = {}
            self.pipette = {}
            for i, (id, controller) in enumerate(pipetteManip.items()):
                self.pipetteManip[id] = controller
                self.pipette[id] = FakePipette(controller, self.pixels_per_micron)
        else:
            self.pipetteManip : Manipulator = pipetteManip
            self.pipette = FakePipette(self.pipetteManip, self.pixels_per_micron)
        self.image_z : float = image_z
        self.frameno : int = 0
        self.targetFramerate = targetFramerate
        self.cellSorterManip = cellSorterManip
        self.cellSorterHandler = FakeCellSorterHandler(self.stageManip, self.cellSorterManip, [400, 200, 0], self.pixels_per_micron)
        self.background_image = background_image

        curFile = str(Path(__file__).parent.absolute())

        #setup frame image (numpy because of easy rolling)
        if self.background_image is None:
            self.frame = cv2.imread(curFile + "/FakeMicroscopeImgs/background.tif", cv2.IMREAD_GRAYSCALE)
        else:
            self.frame = cv2.imread(f"patcherbot\devices\camera\FakeMicroscopeImgs\{background_image}", cv2.IMREAD_GRAYSCALE)
        # self.frame = cv2.imread(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\devices\camera\FakeMicroscopeImgs\213125_1733864762.600275.webp",cv2.IMREAD_GRAYSCALE)
        # self.frame = cv2.imread(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\devices\camera\FakeMicroscopeImgs\8ae5dd1d-c8de-4e00-8ba8-bc724275ee2f.webp",cv2.IMREAD_GRAYSCALE)
        # self.frame = cv2.imread(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\deepLearning\cellModel\example pictures\before.tiff", cv2.IMREAD_GRAYSCALE)
        # self.frame = cv2.imread(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\devices\camera\FakeMicroscopeImgs\cellsegtest.png", cv2.IMREAD_GRAYSCALE)
        # self.frame = cv2.imread(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\deepLearning\cellModel\sam2\notebooks\images\cars.jpg", cv2.IMREAD_GRAYSCALE)

        self.frame = cv2.resize(self.frame, dsize=(self.width * 2, self.height * 2), interpolation=cv2.INTER_NEAREST)


        self.last_img = None
        self.last_stage_pos = None

        #creating large noise arrays slows down fps, create 100 arrays at startup instead
        self.noiseArrs = []
        for _ in range(100):
            self.noiseArrs.append((np.random.random((self.width, self.height)) * 30).astype(np.uint16))

        #start image recording thread
        self.start_acquisition()

    def normalize(self):
        """
        Placeholder normalization method.

        notes:
            Not implemented for this simulated camera.
        """
        print('normalize not implemented for FakeCalCamera')

    def set_exposure(self, value):
        """
        Set exposure time.

        args:
            value (float): Exposure value.

        notes:
            Only values between 0 and 200 are accepted.
        """
        if 0 < value <= 200:
            self.exposure_time = value

    def get_exposure(self):
        """
        Get current exposure time.

        returns:
            float: Exposure time.
        """
        return self.exposure_time

    def get_microscope_image(self, x, y):
        """
        Generate a cropped background image based on stage position.

        args:
            x (float): X position in pixels.
            y (float): Y position in pixels.

        returns:
            PIL.Image.Image: Cropped microscope image.
        """
        if self.last_img is None or self.last_stage_pos[0] != x or self.last_stage_pos[1] != y:
            #we need to recalculate what the stage sees
            frame = np.roll(self.frame, int(y), axis=0)
            frame = np.roll(frame, int(x), axis=1)
            frame = frame[self.height//2:self.height//2+self.height,
                        self.width//2:self.width//2+self.width]
            
            #update cached frame
            self.last_stage_pos = [x, y]
            self.last_img = frame
        else:
            frame = self.last_img
            
        self.frameno += 1
        return Image.fromarray(frame)

    def get_16bit_image(self):
        """
        Get current frame as a 16-bit image.

        returns:
            np.ndarray: 16-bit image scaled from 8-bit frame.
        """
        #Note: use float 32 rather than int16 for opencv sobel filter compatability (focus score)
        return (self.raw_snap().astype(np.float32) / 255) * 65535 

    def get_frame_no(self):
        """
        Get current frame number.

        returns:
            int: Frame index.
        """
        return self.frameno

    def raw_snap(self):
        '''
        Returns the current image.
        This is a blocking call (wait until next frame is available)

        returns:
            np.ndarray: Simulated image frame.

        notes:
            Includes stage translation, focus blur, pipette overlay,
            noise injection, and frame rate limiting.
        '''
        start = time.time()
        # Use the part of the image under the microscope
        stage_x, stage_y, stage_z = self.stageManip.position_group([1, 2, 3])

        startPos = FAKE_STAGE_REFERENCE_POSITION
        # startPos = [0,0,0]
        stage_x = stage_x - startPos[0]
        stage_y = stage_y - startPos[1]
        stage_z = stage_z - startPos[2]

        #get background at current stage position
        img_x = -stage_x * self.pixels_per_micron
        img_y = -stage_y * self.pixels_per_micron
        frame = self.get_microscope_image(img_x, img_y)

        #blur cover slip proportionally to how far stage_z is from 0 (being focused in the img plane)
        focusFactor = abs(stage_z - self.image_z) / 10
        if focusFactor == 0:
            focusFactor = 0.1

        frame = cv2.GaussianBlur(np.array(frame), (63,63), focusFactor)
        frame = Image.fromarray(frame)

        #add pipette to image
        if isinstance(self.pipette, dict):
            for i, (id, pipette) in enumerate(self.pipette.items()):
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                frame = pipette.add_pipette_to_img(frame, [stage_x, stage_y, stage_z])
        else:
            frame = self.pipette.add_pipette_to_img(frame, [stage_x, stage_y, stage_z])


        #add cellsorter to image
        # frame = self.cellSorterHandler.add_cellsorter_to_img(frame, [stage_x, stage_y, stage_z])

        #add noise, exposure
        exposure_factor = self.exposure_time/30.

        frame = frame + self.noiseArrs[self.frameno % len(self.noiseArrs)] #use pregenerated noise to increase fps
        frame[np.where(frame >= 255)] = 255
        frame[np.where(frame < 0)] = 0
        frame = frame.astype(np.uint8)

        dt = time.time() - start
        if dt < (1/self.targetFramerate):
            time.sleep((1/self.targetFramerate) - dt)
        
        return frame
    
class FakeCellSorterHandler():
    """
    Simulates rendering of a cell sorter pipette overlay onto an image.
    """
    def __init__(self, stage : Manipulator, cellSorterManip : CellSorterManip, cellSorterOffset : np.ndarray, pixels_per_micron : float):
        """
        Initialize the cell sorter handler.

        args:
            stage (Manipulator): Stage manipulator.
            cellSorterManip (CellSorterManip): Cell sorter manipulator.
            cellSorterOffset (np.ndarray): Offset of sorter relative to stage.
            pixels_per_micron (float): Conversion factor.
        """
        self.stage = stage
        self.cellSorterManip = cellSorterManip
        self.cellSorterOffset = cellSorterOffset
        self.pixels_per_micron = pixels_per_micron

    def add_cellsorter_to_img(self, img : np.ndarray, stage_pos : np.ndarray):
        """
        Overlay a simulated cell sorter onto an image.

        args:
            img (np.ndarray): Input image.
            stage_pos (np.ndarray): Stage position.

        returns:
            np.ndarray: Image with cell sorter overlay.
        """
        stage_x, stage_y, stage_z = stage_pos

        # get cellsorter pos
        cellSorter_z = self.cellSorterManip.position()

        cellSorter_z *= 1000 #convert to um

        #covert to pixels
        cellSorterXY = [self.cellSorterOffset[0] * self.pixels_per_micron, self.cellSorterOffset[1] * self.pixels_per_micron]

        #else cell sorter pipette in frame: add it to image - 2 channel (grayscale, alpha)
        cellSorterImg = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)

        #pipette looks like a black ring while in focus
        cv2.circle(cellSorterImg, (int(cellSorterXY[0]), int(cellSorterXY[1])), thickness=40, radius=100, color=(0, 1))

        #blur proportionally to how far cellsorter_z is from 0 (being focused in the img plane)
        focusFactor = abs(cellSorter_z - stage_z - self.cellSorterOffset[2]) / 10
        if focusFactor == 0:
            focusFactor = 0.1
        
        cellSorterImg = cv2.GaussianBlur(cellSorterImg, (63,63), focusFactor)

        #add to image (with alpha blending)
        img = img.astype(np.float32)
        img = img * (1 - cellSorterImg[:, :, 1]) + cellSorterImg[:, :, 0] * cellSorterImg[:, :, 1]
        
        return img.astype(np.uint8)

            
class FakePipetteManipulator(FakeManipulator):
    """
    Simulated pipette manipulator with coordinate transformations between
    raw actuator space and real-world space.
    """
    def __init__(self, min=None, max=None, armAngle=np.pi/6):
        """
        Initialize the fake pipette manipulator.

        args:
            min (np.ndarray, optional): Minimum bounds.
            max (np.ndarray, optional): Maximum bounds.
            armAngle (float): Angle of the pipette arm.
        """
        super(FakePipetteManipulator, self).__init__(min, max)
        self.armAngle = armAngle

        self.raw_to_real_mat = np.array([
            [ np.cos(self.armAngle), 0,  np.sin(self.armAngle)],
            [ 0,                     1,  0],
            [-np.sin(self.armAngle), 0,  np.cos(self.armAngle)]
        ], dtype=np.float32)
        self.real_to_raw_mat = np.linalg.inv(self.raw_to_real_mat)


    def raw_to_real(self, raw_pos : np.ndarray):
        """
        Convert raw actuator coordinates to real-world coordinates.

        args:
            raw_pos (np.ndarray): Raw position.

        returns:
            np.ndarray: Real-world position.
        """
        raw_pos = raw_pos.copy()
        # raw_pos[0] = raw_pos[0] * np.cos(self.armAngle)
        real_pos = np.matmul(self.raw_to_real_mat, np.array(raw_pos).T)
        return real_pos

    def real_to_raw(self, real_pos : np.ndarray):
        """
        Convert real-world coordinates to raw actuator coordinates.

        args:
            real_pos (np.ndarray): Real-world position.

        returns:
            np.ndarray: Raw actuator position.
        """
        real_pos = real_pos.copy()
        raw_pos = np.matmul(self.real_to_raw_mat, np.array(real_pos).T)
        return raw_pos

    def position(self, axis=None):
        """
        Get current position in real-world coordinates.

        args:
            axis (int, optional): Specific axis (1-based).

        returns:
            np.ndarray | float: Position vector or axis value.
        """
        raw_pos = self.raw_position() 
        real_pos = self.raw_to_real(raw_pos)
        
        if axis == None:
            return real_pos
        else:
            return real_pos[axis-1]

    def raw_position(self, axis=None):
        """
        Get raw actuator position.

        args:
            axis (int, optional): Axis index.

        returns:
            np.ndarray | float: Raw position.
        """
        return super().position(axis).copy()


    def absolute_move(self, x, axis):
        """
        Move manipulator along a specified axis.

        args:
            x (float): Target position.
            axis (int): Axis index (1-based).
        """
        print(f"Moving axis {axis} to {x}\t{self.position()}\t{self.raw_position()}")
        new_setpoint_raw = np.empty((3,)) * np.nan
        if axis == 1:
            #we're dealing with the 'virtual' d-axis
            new_setpoint_raw = self.raw_position()
            curr_pos_real = self.position()
            dx = x - curr_pos_real[0]
            dVect = self.real_to_raw(np.array([dx, 0, 0]))
            new_setpoint_raw += dVect
        elif axis == 3:
            #we're dealing with z - needs to be handeled based on offset
            new_setpoint_raw = self.raw_position()
            curr_pos_real = self.position()
            dz = x - curr_pos_real[2]
            dVect = self.real_to_raw(np.array([0, 0, dz]))
            new_setpoint_raw += dVect
        else:
            #we're dealing with a physical axis, it can be commanded directly
            new_setpoint_raw[axis-1] = x

        for pos, axis in zip(new_setpoint_raw, range(3)):
            if not np.isnan(pos):
                super().absolute_move(pos, axis+1)

    def absolute_move_group(self, x, axes):
        """
        Move multiple axes simultaneously.

        args:
            x (list[float]): Target positions.
            axes (list[int]): Axes indices.
        """
        new_setpoint_raw = self.raw_position()
        curr_pos_real = self.position()
        x = np.array(x)
        axes = np.array(axes)
        if 1 in axes:
            #we're dealing with the 'virtual' d-axis
            indx = np.where(axes == 1)[0][0]
            dx = x[indx] - curr_pos_real[0]
            dVect = self.real_to_raw(np.array([dx, 0, 0]))
            new_setpoint_raw += dVect
        if 2 in axes:
            indy = np.where(axes == 2)[0][0]
            dy = x[indy] - curr_pos_real[1]
            new_setpoint_raw[1] += dy
        if 3 in axes:
            #we're dealing with z - needs to be handeled based on offset
            indz = np.where(axes == 3)[0][0]
            dz = x[indz] - curr_pos_real[2]
            dVect = self.real_to_raw(np.array([0, 0, dz]))
            new_setpoint_raw += dVect
        
        for x,i in zip(new_setpoint_raw, range(3)):
            print(f"Moving axis {i+1} to {x}\t{self.position()}\t{self.raw_position()}")
            super().absolute_move(x, i+1)

class FakePipette():
    """
    Simulates rendering of a pipette overlay onto microscope images.
    """
    def __init__(self, manipulator:Manipulator, microscope_pixels_per_micron, stage_to_pipette=np.eye(4,4), pipetteAngle=np.pi/6):
        """
        Initialize the fake pipette.

        args:
            manipulator (Manipulator): Pipette manipulator.
            microscope_pixels_per_micron (float): Conversion factor.
            stage_to_pipette (np.ndarray): Transformation matrix.
            pipetteAngle (float): Angle of pipette.
        """
        # stage_to_pipette = np.array([[0.7,  -0.3,   0,      0], 
        #                              [0.3,  1,      0,      0], 
        #                              [0,    0,      (1), 600], 
        #                              [0,    0,      0,      1]])
        # make a identify matrix
        stage_to_pipette = np.eye(4,4)

        # rotation matrix to make the x-axis to parallel to the pipette, rather than the stage
        # note: this is the rotation matrix about the y-axis, but only rotating the x axis (not z)
        # creates a non-orthogonal coordinate system, but that's the same as the real pipette  
        # self.rot_mat  = np.array([[np.cos(pipetteAngle), 0, 0, 0], 
        #                     [0, 1, 0, 0], 
        #                     [np.sin(pipetteAngle), 0, 1, 0], 
        #                     [0, 0, 0, 1]])

        self.rot_mat = np.array([
                        [1, 0, 0, 0],
                        [0,1, 0, 0],
                        [0,                   0, 1, 0],  
                        [0,                   0, 0, 1]
                    ])
        stage_to_pipette = np.matmul(np.linalg.inv(self.rot_mat), stage_to_pipette)

        
        self.manipulator = manipulator
        self.pixels_per_micron = microscope_pixels_per_micron
        self.stage_to_pipette = stage_to_pipette #homoegeneous transform matrix from stage to pipette
        self.pipette_to_stage = np.linalg.inv(self.stage_to_pipette)

        #setup pipette image (PIL b/c of easy pasting)
        curFile = str(Path(__file__).parent.absolute())
        self.pipetteImg = Image.open(curFile + "/FakeMicroscopeImgs/pipette.png").convert("L")
        self.pipetteImg = self.pipetteImg.resize((self.pipetteImg.size[0] * 4, self.pipetteImg.size[1] * 2), Image.Resampling.BILINEAR)
        filter = ImageEnhance.Brightness(self.pipetteImg)
        self.pipetteImg = filter.enhance(1.2)

        #create an alpha mask for the pipette (to make pipette see through)
        filter = ImageEnhance.Brightness(self.pipetteImg)
        self.alphaMask = filter.enhance(1.2)
        frame_center_um = np.array([512.0, 512.0]) / self.pixels_per_micron
        self.geometry_direction = np.asarray(manipulator.position(), dtype=float)[:2] - frame_center_um
        
    def add_pipette_to_img(self, frame:Image, stagePos:list):
        """
        Initialize the fake pipette.

        args:
            manipulator (Manipulator): Pipette manipulator.
            microscope_pixels_per_micron (float): Conversion factor.
            stage_to_pipette (np.ndarray): Transformation matrix.
            pipetteAngle (float): Angle of pipette.
        """
        # print(self.manipulator.position(), self.manipulator.raw_position())
        #get stage micron coords
        stage_x, stage_y, stage_z = stagePos

        #get stage pixel coords
        stage_img_x = stage_x * self.pixels_per_micron
        stage_img_y = stage_y * self.pixels_per_micron

        #get pipette micron coords
        pipette_x, pipette_y, pipette_z = self.manipulator.position()
        pipette_pos_h = np.array([pipette_x, pipette_y, pipette_z, 1])

        # startPos = [-235000, 55000, 285000]
        # pipette_x -= startPos[0]
        # pipette_y -= startPos[1]
        # pipette_z -= startPos[2]

        #get pipette position in stage coordinates
        pipette_pos_stage_coords_h = np.matmul(self.pipette_to_stage, pipette_pos_h.T)
        pipette_pos_stage_coords = pipette_pos_stage_coords_h[0:3] / pipette_pos_stage_coords_h[3]

        #get pipette position in image coordinates
        pipette_pos_img_coords = pipette_pos_stage_coords * self.pixels_per_micron

        #get x,y - convert to int, make relative to frame
        tip = np.array([self.pipetteImg.width - 1.0, 0.0])
        base = np.array([0.0, (self.pipetteImg.height - 1.0) / 2.0])
        source_angle = np.arctan2(tip[1] - base[1], tip[0] - base[0])
        geometry_angle = np.arctan2(self.geometry_direction[1], self.geometry_direction[0])
        rotation_angle = np.degrees(source_angle - geometry_angle)
        transform = cv2.getRotationMatrix2D(tuple(tip), rotation_angle, 1.0)
        corners = np.array([
            [0, 0],
            [self.pipetteImg.width - 1, 0],
            [0, self.pipetteImg.height - 1],
            [self.pipetteImg.width - 1, self.pipetteImg.height - 1],
        ], dtype=np.float32)
        rotated_corners = cv2.transform(corners[None, :, :], transform)[0]
        minimum = np.floor(rotated_corners.min(axis=0))
        maximum = np.ceil(rotated_corners.max(axis=0))
        transform[:, 2] -= minimum
        output_size = (int(maximum[0] - minimum[0] + 1), int(maximum[1] - minimum[1] + 1))
        tip_offset = transform @ np.array([tip[0], tip[1], 1.0])
        pipette_image = cv2.warpAffine(
            np.asarray(self.pipetteImg), transform, output_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        alpha_mask = cv2.warpAffine(
            np.asarray(self.alphaMask), transform, output_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        pipette_img_x = int(round(pipette_pos_img_coords[0] - stage_img_x - tip_offset[0]))
        pipette_img_y = int(round(pipette_pos_img_coords[1] - stage_img_y - tip_offset[1]))

        #blur pipette proportionally to distance between stage_z and pipette_z
        focusFactor = abs(stage_z - pipette_pos_stage_coords[2]) / 10
        if focusFactor == 0:
            focusFactor = 0.1 #resolve divide by 0 error

        #blur img
        pipetteImg = cv2.GaussianBlur(pipette_image, (63,63), focusFactor)
        pipetteImg = Image.fromarray(pipetteImg)

        #blur alpha channel
        alphaMask = cv2.GaussianBlur(alpha_mask, (63,63), focusFactor / 2)
        alphaMask = alphaMask / 1.3
        alphaMask = Image.fromarray(alphaMask.astype(np.uint8))

        #add pipette to frame
        frame.paste(pipetteImg, (pipette_img_x, pipette_img_y), alphaMask)
        frame = np.array(frame) #convert back to numpy for opencv support
        return frame