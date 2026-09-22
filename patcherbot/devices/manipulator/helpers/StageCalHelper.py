import time
import cv2
import numpy as np
from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.manipulator import Manipulator
from patcherbot.devices.camera import Camera
from threading import Thread
import math


class FocusHelper():
    FOCUSING_MAX_SPEED = 25
    NORMAL_MAX_SPEED = 10000

    def __init__(self, microscope: Microscope, camera: Camera):
        self.microscope = microscope
        self.camera = camera

    def _distance_scale(self):
        if getattr(self.microscope, "config", None) is not None:
            return float(self.microscope.config.microscope_units_per_um)
        return float(getattr(self.microscope, "units_per_um", 5.0))

    def autofocusContinuous(self, distance, timeout=1):


        focusThread = FocusUpdater(self.microscope, self.camera)
        focusThread.start()
        scale = self._distance_scale()
        commandedPos = self.microscope.position() / scale + distance
        self.microscope.relative_move(distance)
        start_time = time.time()
        while abs(self.microscope.position() / scale - commandedPos) > 0.3:
            if time.time() - start_time > timeout:
                print("Timeout reached waiting for microscope movement")
                break
            time.sleep(0.05)

        focusThread.stop()
        focusThread.join()

        if len(focusThread.posFocusList) == 0:
            raise RuntimeError("No focus data collected!")

        bestIndex = np.argmax(focusThread.posFocusList[:, 1])
        bestPos = focusThread.posFocusList[bestIndex, 0]
        bestScore = focusThread.posFocusList[bestIndex, 1]

        return bestPos, bestScore

    def autofocus(self, dist=100):
        self.microscope.set_max_speed(self.FOCUSING_MAX_SPEED)
        scale = self._distance_scale()
        initPos = self.microscope.position() / scale
        # print("Moving downward to collect forward scores..")
        bestForwardPos, bestForwardScore = self.autofocusContinuous(-dist)
        # print(f"Focus values:{bestForwardPos/scale} um, {bestForwardScore} units")

        self.microscope.set_max_speed(self.NORMAL_MAX_SPEED)
        # self.microscope.absolute_move(initPos)

        # print ("moving up to reset")
        self.microscope.relative_move(dist)
        self.microscope.wait_until_still()
        self.microscope.set_max_speed(self.FOCUSING_MAX_SPEED)
        # print("Moving upward to collect backwards scores..")
        bestBackwardPos, bestBackwardScore = self.autofocusContinuous(dist)
        # print(f"Focus values:{bestBackwardPos/scale} um, {bestBackwardScore} units")

        self.microscope.set_max_speed(self.NORMAL_MAX_SPEED)

        finalPos = bestForwardPos if bestForwardScore >= bestBackwardScore else bestBackwardPos
        finalPos = finalPos / scale
        # print(f"FinalPosition: {finalPos}")
        self.microscope.absolute_move(finalPos)
        self.microscope.wait_until_still()


class FocusUpdater(Thread):
    def __init__(self, microscope: Microscope, camera: Camera):
        super().__init__()
        self.isRunning = True
        self.camera = camera
        self.microscope = microscope
        self.posFocusList = []
        self.lastFrame = -1

    def run(self):
        while self.isRunning:
            if len(self.camera.raw_frame_queue) == 0:
                time.sleep(0.01)
                continue

            frame_no, frametime, _, img = self.camera.raw_frame_queue[-1]
            if frame_no == self.lastFrame:
                time.sleep(0.005)
                continue

            self.lastFrame = frame_no
            score = self._getFocusScore(img)
            self.posFocusList.append([self.microscope.position(), score])

        self.posFocusList = np.array(self.posFocusList)

    def _getFocusScore(self, image):
        focusSize = 512
        x = image.shape[1] / 2 - focusSize / 2
        y = image.shape[0] / 2 - focusSize / 2
        crop_img = image[int(y):int(y + focusSize), int(x):int(x + focusSize)]

        xEdges = cv2.norm(cv2.Sobel(src=crop_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=7))
        yEdges = cv2.norm(cv2.Sobel(src=crop_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=7))

        score = xEdges ** 2 + yEdges ** 2
        return score

    def stop(self):
        self.isRunning = False

class StageCalHelper():
    '''A helper class to aid with Stage Calibration
    '''
    
    CAL_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 10000

    def __init__(self, stage: Manipulator, camera: Camera, frameLag: int):
        self.stage : Manipulator = stage
        self.camera : Camera = camera
        self.lastFrameNo : int = None
        self.frameLag = frameLag

    def calibrateContinuous(self, distance, video=False):
        '''Tell the stage to go a certain distance at a low max speed.
           Take a bunch of pictures and run optical flow. Use optical flow information
           to create a linear transform from stage microns to image pixels.
           if set, video creates an mp4 of the optical flow running in the project directory.
        '''
        #move the microscope a certain distance forward and up
        currPos = self.stage.position()
        commandedPos = np.array([currPos[0] + distance, currPos[1] - distance])
        axes = np.array([0, 1], dtype=int)
        self.stage.absolute_move_group(commandedPos, axes)

        #wait for the microscope to reach the pos, recording frames
        framesAndPoses = []
        currPos = self.stage.position()
        startPos = currPos
        _, _, _, firstFrame = self.camera.raw_frame_queue[0]
        p0 = self.calcOpticalFlowP0(firstFrame)
        while abs(currPos[0] - commandedPos[0]) > 0.3 or abs(currPos[1] - commandedPos[1]) > 0.3:
            while self.lastFrameNo == self.camera.get_frame_no():
                time.sleep(0.05) #wait for a new frame to be read from the camera
            self.lastFrameNo = self.camera.get_frame_no()
            currPos = self.stage.position()

            #get latest img
            _, _, _, frame = self.camera.raw_frame_queue[0]

            framesAndPoses.append([frame.copy(), currPos[0] - startPos[0], currPos[1] - startPos[1]])

        #run optical flow on the recorded frames
        print('running optical flow...')
        imgPosStagePosList = []
        x_pix_total = 0
        y_pix_total = 0

        if video:
            out = cv2.VideoWriter('opticalFlow.mp4', -1, 10.0, (1024,1024))

        #calculate the average image
        avgImg = np.zeros_like(framesAndPoses[0][0], dtype=np.float64)
        for frame, _, _ in framesAndPoses:
            avgImg += frame
        avgImg = avgImg / len(framesAndPoses)
        avgImg = avgImg.astype(np.uint8)

        #subtract average from all frames
        for frame, _, _ in framesAndPoses:
            frame -= avgImg

        for i in range(len(framesAndPoses) - 1):
            currFrame, x_microns, y_microns = framesAndPoses[i + 1]
            lastFrame, last_x_microns, last_y_microns = framesAndPoses[i]

            p0 = self.calcOpticalFlowP0(lastFrame)

            x_pix, y_pix = self.calcOpticalFlow(lastFrame, currFrame, p0)
            x_pix_total += x_pix
            y_pix_total += y_pix

            if math.isnan(x_pix) or math.isnan(y_pix): #if no corners can be found with optical flow, nan could be returned.  Don't add this to the list
                continue

            if video:
                vidFrame = cv2.cvtColor(currFrame.copy(), cv2.COLOR_GRAY2BGR)
                cv2.line(vidFrame, (512,512), (512 + int(x_pix_total), 512 + int(y_pix_total)), (255,0,0), 3)
                cv2.line(vidFrame, (600,100), (600 + int(x_pix_total), 100 + int(y_pix_total)), (255,0,0), 3)
                cv2.line(vidFrame, (900,400), (900 + int(x_pix_total), 400 + int(y_pix_total)), (255,0,0), 3)
                out.write(vidFrame)


            imgPosStagePosList.append([x_pix_total, y_pix_total, x_microns, y_microns])
        imgPosStagePosList = np.array(imgPosStagePosList)
        

        if video:
            out.release()
        
        #for some reason, estimateAffinePartial2D only works with int64
        #we can multiply by 100, to preserve 2 decimal places without affecting rotation / scaling portion of affline transform
        imgPosStagePosList = (imgPosStagePosList).astype(np.int64)
        print(imgPosStagePosList)

        #compute affine transformation matrix
        mat, inVsOut = cv2.estimateAffinePartial2D(imgPosStagePosList[:,2:4], imgPosStagePosList[:,0:2])

        #fix intercept - set image center --> stage center
        mat[0,2] = 0
        mat[1,2] = 0

        print('completed optical flow. matrix:')
        print(mat)

        #return transformation matrix
        return mat

    def calcOpticalFlowP0(self, firstFrame):
        #params for corner detector
        feature_params = dict(maxCorners = 100,
                                qualityLevel = 0.1,
                                minDistance = 10,
                                blockSize = 10)
        p0 = cv2.goodFeaturesToTrack(firstFrame, 70, 0.05, 25)
        # p0 = cv2.goodFeaturesToTrack(firstFrame, mask = None, **feature_params)

        return p0
    
    def calcMotionTranslation(self, lastFrame, currFrame):
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

        #compute transformation for image translation
        (cc, warp_matrix) = cv2.findTransformECC (lastFrame, currFrame,warp_matrix, warp_mode, criteria)

        #get x, y translation (pixels)
        print(warp_matrix)
        x_pix = warp_matrix[0,2]
        y_pix = warp_matrix[1,2]

        print(x_pix, y_pix)

        return x_pix, y_pix

    def calcOpticalFlow(self, lastFrame, currFrame, p0):
        #params for optical flow
        lk_params = dict(winSize  = (20, 20),
                    maxLevel = 20)

        # calculate optical flow from first frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(lastFrame, currFrame, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
         #find median movement vector
        dMovement = good_new - good_old
        medianVect = np.median(dMovement, axis=0)
        
        return medianVect[0], medianVect[1]


    def calibrate(self, dist=500):
        '''Calibrates the microscope stage using optical flow and stage encoders to create a um -> pixels transformation matrix
        '''

        self.stage.set_max_speed(self.CAL_MAX_SPEED)
        # self.stage.set_max_accel(10)

        initPos = self.stage.position()
        print('starting optical flow')
        mat = self.calibrateContinuous(dist)
        # commandedPos = np.array([initPos[0] + 200, initPos[1] - 200])
        # axes = np.array([0, 1], dtype=int)
        # self.stage.absolute_move_group(commandedPos, axes)
        self.stage.wait_until_still()
        currPos = self.stage.position()
        
        self.stage.set_max_speed(self.NORMAL_MAX_SPEED)
        self.stage.absolute_move(initPos)
        self.stage.wait_until_still()

        return mat
