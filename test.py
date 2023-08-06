# Other imports
import os
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor

from constants import *
from utils import getVideoOrientation

# Custom class imports
from classes.parameters import Parameters
from classes.video import Video
from classes.rtiAlgorithm import RTI
from classes.videoSynchronisation import VideoSynchronisation


# Obtain calibration data
rti = RTI()

kMoving = np.load('calibrations/calibration-23_08_06_16_05_1691330723/mv_cam_intrinsic.dat')

# Retrieve both videos
videoStatic = Video(STATIC_VIDEO_FILE_PATH)
videoMoving = Video(MOVING_VIDEO_FILE_PATH)
# Calculate the world homography using the points selected in the previous step and the frame size choosen before
worldHomography = rti.getWorldHomographyFromPts(points, int(params.getOutputImageSize()))

if worldHomography is None:
    print("An error occurred: World homography has not been set correctly. ")
    exit(-1)
else:
    print("World homography has been calculated correctly. ")
    
# Get the first frame from the static camera, which will be used during feature matching
_, firstStaticFrame = videoStatic.getCurrentFrame()


# Second video shifted. Set frame difference with it's own FPS
videoStatic.setVideoFrame()
videoMoving.setVideoFrame(abs(2))
print(f"Moving video is shifted. The frame difference is: {abs(2)}")

# Variable used to store the time calculated after each read on the video, in order to provide synchronisation
timeStaticVideo = 0.
timeMovingVideo = 0.
    
print("Starting calculation of the light directions in the videos..."
staticFrames = []
movingFrames = []

while videoStatic.isOpen() and videoMoving.isOpen():
    
    # Get frame from each video
    retStatic, staticFrame = videoStatic.getCurrentFrame()
    retMoving, movingFrame = videoMoving.getCurrentFrame()
    
    # For each iteration, sum the time for each video based on the tick (1 / FPS_video)
    timeStaticVideo += 1. / videoStatic.getFPS()
    timeMovingVideo += 1. / videoMoving.getFPS()
    
    # Now depends on which video has lower FPS
    if videoStatic.getFPS() < videoMoving.getFPS():
        # Video static is behind more than 1 frame, so skip it to recover the loss
        if timeStaticVideo > timeMovingVideo + (1. / videoMoving.getFPS()):
            retStatic, staticFrame = videoStatic.getCurrentFrame()
            
    else:    
        # Video moving is behind more than 1 frame, so skip it to recover the loss
        if timeMovingVideo > timeStaticVideo + (1. / videoMoving.getFPS()):
            retMoving, movingFrame = videoMoving.getCurrentFrame()
    
    # Check if static or moving frame is empty
    checkStaticFrame = (staticFrame is None or np.shape(staticFrame) == () or np.sum(staticFrame) == 0)
    checkMovingFrame = (movingFrame is None or np.shape(movingFrame) == () or np.sum(movingFrame) == 0)
    
    # Close loop in case of one video is over    
    if retStatic != True or retMoving != True or checkStaticFrame or checkMovingFrame:
        break
    
    staticFrames.append(staticFrame)
    movingFrames.append(movingFrame)
    
print("Extracted synchronised frames")
print("Computing feature matching for all frames...")

executor = ThreadPoolExecutor(max_workers = 4)

# Get the light and frame pair for each frame, and filter them to get only the valid ones
lightFramePair = list(executor.map(processFrameMatching, staticFrames, movingFrames))
validPairs = [pair for pair in lightFramePair if all(value is not None for value in pair)]

print(f"Valid pairs: {len(validPairs)}")
        
# Close the executor after all tasks are done
executor.shutdown()
            
def processFrameMatching(self, staticFrame, movingFrame):
        # Convert frames to grayscale
        staticFrame = cv.cvtColor(staticFrame, cv.COLOR_BGR2GRAY)
        movingFrame = cv.cvtColor(movingFrame, cv.COLOR_BGR2GRAY)
        
        print("Frames changed colors")
        
        # UniVE video
        _, _, homographyStaticToStatic = rti.getHomographyWithFeatureMatching(staticFrame, firstStaticFrame, "Static to Static", False, cutFrame1 = ((500, 1700), (1400, 2600)), cutFrame2 = ((500, 1700), (1400, 2600)))
        _, ptsMovingCam, homographyStaticToMoving = rti.getHomographyWithFeatureMatching(staticFrame, movingFrame, "Static to Moving", False, cutFrame1 = ((500, 1700), (1400, 2600)), cutFrame2 = ((450, 1150), (200, 900)))    
        
        print("Homographies computed")
        
        if homographyStaticToStatic is not None and homographyStaticToMoving is not None:
            
            # Homography mapping points from world reference system to moving camera ref. system
            hWorld2Moving = homographyStaticToMoving @ np.linalg.inv(homographyStaticToStatic) @ np.linalg.inv(worldHomography)
            
            # Homography mapping points from moving camera ref. system to world reference system 
            hMoving2World = worldHomography @ homographyStaticToStatic @  np.linalg.inv(homographyStaticToMoving)
            
            # Option 1: Use a meshgrid to shift points from one ref. system to world ref. system
            # # Create a grid in the moving camera ref. system
            # lx, ly = np.meshgrid(np.linspace(450., 1150., 11), np.linspace(200., 900., 11))   
            # # And plot the points             
            # points2d = np.vstack((lx.flatten(), ly.flatten())).T
            
            # Option 2: Use the features detected in the cam. ref. system and shift points to world ref. system
            points2d = ptsMovingCam
        
            # Add 1 to the source points
            points3d = np.hstack([np.squeeze(points2d), np.ones([points2d.shape[0], 1], dtype=points2d.dtype)])
            
            # Source points inside world reference system
            points3d = hMoving2World @ points3d.T 
            
            points3d /= points3d[2, :]
            
            points3d = points3d.T
            
            # Set last postion to 0
            points3d[:, 2] = 0
            
            # Now get world frame using static camera and homographies to move into the world reference system
            worldFrame = cv.warpPerspective(staticFrame, worldHomography @ homographyStaticToStatic, (params.getOutputImageSize(), params.getOutputImageSize()))
            
            # ... and do the same for moving camera, in order to get a similarity between frames
            warpedMoving = cv.warpPerspective(movingFrame,  hWorld2Moving, (params.getOutputImageSize(), params.getOutputImageSize()), flags = cv.WARP_INVERSE_MAP)
            
            # Now, let's try to cross-correlate the two warped images.
            # If the correlation is high, then the images are similar, so we can compute the light vector
            # Otherwise, skip the frame
            imgCorr = cv.matchTemplate(worldFrame, warpedMoving, cv.TM_CCORR_NORMED)
                            
            # Set as lower threshold 0.6 to have high confidentiality
            if imgCorr[0][0] >= 0.96:
                # Calculate the light vector using PnP
                lightVector = rti.getLigthWithSolvePnP(points3d, np.squeeze(points2d), kMoving)
            else:
                # If not enough confidence, remove this frame
                worldFrame = lightVector = None
        else: 
            # If no homographies, remove this frame
            worldFrame = lightVector = None
        
        return (worldFrame, lightVector)
                
            