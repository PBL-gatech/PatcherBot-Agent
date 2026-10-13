from patcherbot.devices.cellsorter.CellSorter import CellSorterManip
from patcherbot.devices.manipulator import Manipulator, FakeManipulator
from .camera import Camera
import numpy as np
import cv2
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

class FakeCalCamera(Camera):
    def __init__(self, stageManip=None, pipetteManip=None, image_z=0, targetFramerate=40, cellSorterManip=None):
        super(FakeCalCamera, self).__init__()
        self.width : int = 1024
        self.height : int = 1024
        self.exposure_time : int = 30
        self.stageManip : Manipulator = stageManip
        self.pipetteManip : Manipulator = pipetteManip
        self.image_z : float = image_z
        self.pixels_per_micron : float = 1.25  # pixels / micrometers
        self.frameno : int = 0
        self.pipette = FakePipette(self.pipetteManip, self.pixels_per_micron)
        self.targetFramerate = targetFramerate
        self.cellSorterManip = cellSorterManip
        self.cellSorterHandler = FakeCellSorterHandler(self.stageManip, self.cellSorterManip, [400, 200, 0], self.pixels_per_micron)

        curFile = str(Path(__file__).parent.absolute())

        #setup frame image (numpy because of easy rolling)
        self.frame = cv2.imread(curFile + "/FakeMicroscopeImgs/background.tif", cv2.IMREAD_GRAYSCALE)
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
        print('normalize not implemented for FakeCalCamera')

    def set_exposure(self, value):
        if 0 < value <= 200:
            self.exposure_time = value

    def get_exposure(self):
        return self.exposure_time

    def get_microscope_image(self, x, y):
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
        #Note: use float 32 rather than int16 for opencv sobel filter compatability (focus score)
        return (self.raw_snap().astype(np.float32) / 255) * 65535 

    def get_frame_no(self):
        return self.frameno

    def raw_snap(self):
        '''
        Returns the current image.
        This is a blocking call (wait until next frame is available)
        '''
        start = time.time()
        # Use the part of the image under the microscope
        stage_x, stage_y, stage_z = self.stageManip.position_group([1, 2, 3])

        startPos = [-235000, 55000, 285000]
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
    def __init__(self, stage : Manipulator, cellSorterManip : CellSorterManip, cellSorterOffset : np.ndarray, pixels_per_micron : float):
        self.stage = stage
        self.cellSorterManip = cellSorterManip
        self.cellSorterOffset = cellSorterOffset
        self.pixels_per_micron = pixels_per_micron

    def add_cellsorter_to_img(self, img : np.ndarray, stage_pos : np.ndarray):
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
    def __init__(self, min=None, max=None, armAngle=np.pi/6):
        super(FakePipetteManipulator, self).__init__(min, max)
        self.armAngle = armAngle

        self.raw_to_real_mat = np.array([
            [ np.cos(self.armAngle), 0,  np.sin(self.armAngle)],
            [ 0,                     1,  0],
            [-np.sin(self.armAngle), 0,  np.cos(self.armAngle)]
        ], dtype=np.float32)

        self.real_to_raw_mat = np.linalg.inv(self.raw_to_real_mat)


    def raw_to_real(self, raw_pos : np.ndarray):
        raw_pos = raw_pos.copy()
        # raw_pos[0] = raw_pos[0] * np.cos(self.armAngle)
        real_pos = np.matmul(self.raw_to_real_mat, np.array(raw_pos).T)
        return real_pos

    def real_to_raw(self, real_pos : np.ndarray):
        real_pos = real_pos.copy()
        raw_pos = np.matmul(self.real_to_raw_mat, np.array(real_pos).T)
        return raw_pos

    def position(self, axis=None):
        raw_pos = self.raw_position() 
        real_pos = self.raw_to_real(raw_pos)
        
        if axis == None:
            return real_pos
        else:
            return real_pos[axis-1]

    def raw_position(self, axis=None):
        return super().position(axis).copy()


    def absolute_move(self, x, axis):
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

    def __init__(self, manipulator:Manipulator, microscope_pixels_per_micron, stage_to_pipette=np.eye(4,4), pipetteAngle=np.pi/6):

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
        
    def add_pipette_to_img(self, frame:Image, stagePos:list):

        # print(self.manipulator.position(), self.manipulator.raw_position())
        #get stage micron coords
        stage_x, stage_y, stage_z = stagePos

        #get stage pixel coords
        stage_img_x = stage_x * self.pixels_per_micron
        stage_img_y = stage_y * self.pixels_per_micron

        #get pipette micron coords
        pipette_x, pipette_y, pipette_z = np.asarray([0, 0, 0]) #obviously wrong but the fake manipulator is broken sooo #self.manipulator.position()
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
        pipette_img_x = int(pipette_pos_img_coords[0] - stage_img_x) - self.pipetteImg.size[0] #pipette_pos should correspond to tip of pipette (upper right) 
        pipette_img_y = int(pipette_pos_img_coords[1] - stage_img_y)

        #blur pipette proportionally to distance between stage_z and pipette_z
        focusFactor = abs(stage_z - pipette_pos_stage_coords[2]) / 10
        if focusFactor == 0:
            focusFactor = 0.1 #resolve divide by 0 error

        #blur img
        pipetteImg = cv2.GaussianBlur(np.array(self.pipetteImg), (63,63), focusFactor)
        pipetteImg = Image.fromarray(pipetteImg)

        #blur alpha channel
        alphaMask = cv2.GaussianBlur(np.array(self.alphaMask), (63,63), focusFactor / 2)
        alphaMask = alphaMask / 1.3
        alphaMask = Image.fromarray(alphaMask.astype(np.uint8))

        #add pipette to frame
        frame.paste(pipetteImg, (pipette_img_x, pipette_img_y), alphaMask)
        frame = np.array(frame) #convert back to numpy for opencv support
        return frame