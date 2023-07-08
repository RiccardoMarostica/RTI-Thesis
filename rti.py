# Import default modules
import cv2 as cv
import numpy as np
import sys
import os
import datetime

from PyQt6.QtWidgets import QApplication

# Import classes
from classes.videoSynchronisation import VideoSynchronisation
from classes.cameraCalibration import CameraCalibration
from classes.video import Video
from classes.rtiAlgorithm import RTI
from classes.gui import MainWindow

from constants import *

def main():
    
    print("Starting camera calibration...")
    
    # Get the two calibrations for both static and moving camera
    calibrationStatic = CameraCalibration(Video(STATIC_VIDEO_CALIBRATION_FILE_PATH), (9, 6))
    calibrationMoving = CameraCalibration(Video(MOVING_VIDEO_CALIBRATION_FILE_PATH), (9, 6))
    
    # Calibrate both cameras
    retCalibStatic = calibrationStatic.calibrateCamera()
    retCalibMoving = calibrationMoving.calibrateCamera()
    
    # Calibrate cameras and check result
    if (retCalibStatic == False or retCalibMoving == False):
        print("Error when calibrating one of the two cameras. ")
        exit(-1)
    else:
        print("Camera calibration completed without errors")
    
    # Create the two videos
    videoStatic = Video(STATIC_VIDEO_FILE_PATH)
    videoMoving = Video(MOVING_VIDEO_FILE_PATH)
    
    # Initialise RTI class
    rti = RTI()
    
    # Retrieve K for both cameras
    kStatic = calibrationStatic.getIntrinsicMatrix()
    # kMoving = calibrationMoving.getIntrinsicMatrix()
    kMoving = rti.getDefaultK(videoMoving)

    # Store the first frame of the Static Camera
    _, firstStaticFrame = videoStatic.getCurrentFrame()
    
    # And try to get the 4 points in the static video
    worldHomography = rti.getWorldHomography(videoStatic)

    if len(worldHomography) == 0:
        print("Error when computing homography to get world reference system. ")
        exit(-1)
    else:
        print("Homography calculated without errors")
    
    print("Starting video synchronisation...")
    
    # # Create class to synch the videos
    # videoSynchronisation = VideoSynchronisation(STATIC_VIDEO_FILE_PATH, MOVING_VIDEO_FILE_PATH)
    # # ... and them synch them
    # videoSynchronisation.synchroniseVideo()
    
    # print("Video synchronisation completed without errors")
    
    # # After synchronisation, get the offset between the two videos
    # # First get the default FPS
    # defaultFps = max(videoStatic.getFPS(), videoMoving.getFPS())
    # # ... and then compute the shift between the videos
    # frameDifference = videoSynchronisation.getFrameDifference(defaultFps)
    
    # frameDifference = -2    # Frame difference between unive (filename) video camera
    frameDifference = -10   # Frame difference between keys (filename) video camera
    # frameDifference = -6    # Frame difference between paperclip (filename) video camera
    # frameDifference = -9    # Frame difference between book (filename) video camera
    
    print("Frame difference: ", frameDifference)
    
    if (frameDifference > 0):
        print("Static Video shifted")
        # If the offset is positive, then the first video starts sooner.
        # So move its position in order to start as the second video
        videoStatic.setVideoFrame(abs(frameDifference))
        videoMoving.setVideoFrame()
    else:
        print("Moving Video shifted")
        # ... or vice versa
        videoStatic.setVideoFrame()
        videoMoving.setVideoFrame(abs(frameDifference))
        
    print("Starting calculation of the light directions in the videos...")
    
    # Variable used to store the time calculated after each read on the video, in order to provide synchronisation
    timeStaticVideo = 0.
    timeMovingVideo = 0.
    
    # Variable used to move both videos of a specific time (in ms), based on the current iteration
    iteration = 0
        
    while videoStatic.isOpen() and videoMoving.isOpen():
        
        # Move the video every [DEFAULT_MSEC_GAP_VIDEO] ms to obtain less frames
        videoStatic.setVideoPosition(int(iteration * DEFAULT_MSEC_GAP_VIDEO))
        videoMoving.setVideoPosition(int(iteration * DEFAULT_MSEC_GAP_VIDEO))
        
        # Get frame from each video
        retStatic, staticFrame = videoStatic.getCurrentFrame()
        retMoving, movingFrame = videoMoving.getCurrentFrame()
        
        if retStatic != True or retMoving != True:
            break
        
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
        
        # Convert frames to grayscale
        staticFrame = cv.cvtColor(staticFrame, cv.COLOR_BGR2GRAY)
        movingFrame = cv.cvtColor(movingFrame, cv.COLOR_BGR2GRAY)
        
        # Get the homography between static camera (src) and moving camera (dst) i-th frame
        
        # For keys
        _, _, homographyStaticToStatic = rti.getHomographyWithFeatureMatching(staticFrame, firstStaticFrame, "Static to Static", False, cutFrame1 = ((1500, 2600), (500, 1600)), cutFrame2 = ((1500, 2600), (500, 1600)))
        _, dstStaticToMoving, homographyStaticToMoving = rti.getHomographyWithFeatureMatching(staticFrame, movingFrame, "Static to Moving", True, cutFrame1 = ((1500, 2600), (500, 1600)), cutFrame2 = ((600, 1100), (250, 800)))
        
        # For paperclip
        # _, _, homographyStaticToStatic = rti.getHomographyWithFeatureMatching(staticFrame, firstStaticFrame, "Static to Static", False, cutFrame1 = ((500, 1700), (1400, 2600)), cutFrame2 = ((500, 1700), (1400, 2600)))
        # _, dstStaticToMoving, homographyStaticToMoving = rti.getHomographyWithFeatureMatching(staticFrame, movingFrame, "Static to Moving", True, cutFrame1 = ((500, 1700), (1400, 2600)), cutFrame2 = ((450, 1150), (200, 900)))
        
        # For books
        # _, _, homographyStaticToStatic = rti.getHomographyWithFeatureMatching(staticFrame, firstStaticFrame, "Static to Static", False, cutFrame1 = ((350, 1850), (1150, 2900)), cutFrame2 = ((300, 1900), (1200, 2950)))
        # _, dstStaticToMoving, homographyStaticToMoving = rti.getHomographyWithFeatureMatching(staticFrame, movingFrame, "Static to Moving", True, cutFrame1 = ((300, 1900), (1200, 2950)), cutFrame2 = ((400, 1350), (125, 1080)))
        
        if homographyStaticToStatic is not None and homographyStaticToMoving is not None:
            # Add 1 to the source points
            dstStaticToMoving_hom = np.hstack([np.squeeze(dstStaticToMoving), np.ones([dstStaticToMoving.shape[0], 1], dtype=dstStaticToMoving.dtype)])
            
            # Source points inside world reference system
            dstWorldFrame = worldHomography @ homographyStaticToStatic @  np.linalg.inv(homographyStaticToMoving) @ dstStaticToMoving_hom.T 
            
            dstWorldFrame /= dstWorldFrame[2, :]
            
            dstWorldFrame = dstWorldFrame.T
            
            # Set last postion to 0
            dstWorldFrame[:, 2] = 0
            
            lightVectorPnP = rti.getLigthWithSolvePnP(dstWorldFrame, np.squeeze(dstStaticToMoving), kMoving)
            
            Hworld2moving = homographyStaticToMoving @ np.linalg.inv(homographyStaticToStatic) @ np.linalg.inv(worldHomography)
            
            R, T = rti.getExtrinsicsParameters(Hworld2moving, kMoving)
            
            lightVectorEstimated = rti.getLightVector(R, T)
                
            # Now get world frame using static camera and homographies to move into the world reference system
            worldFrame = cv.warpPerspective(staticFrame, worldHomography @ homographyStaticToStatic, (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE))
            
            # ... and do the same for moving camera, in order to get a similarity between frames
            warpedMoving = cv.warpPerspective(movingFrame,  homographyStaticToMoving @ np.linalg.inv(homographyStaticToStatic), (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE), flags = cv.WARP_INVERSE_MAP)
            
            # Show the light plot of the calculated light vector
            cirlePlotPnP = rti.showCircleLightDirection(lightVectorPnP)
            cirlePlotEstimated = rti.showCircleLightDirection(lightVectorEstimated)
        
            cv.imshow('Light plot PnP', cirlePlotPnP)
            cv.imshow('Light plot Estimated', cirlePlotEstimated)
            cv.imshow('World frame', worldFrame)
            cv.imshow('World frame moving', warpedMoving)
        else:
            lightVectorPnP = []
        
        if len(lightVectorPnP) != 0:
            rti.storeLightVector(worldFrame, lightVectorPnP)      
        
        iteration += 1
        
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
        

    cv.destroyAllWindows()
    
    lightDirections = rti.getLightDirections()

    print("Frames aquired: ", len(lightDirections))
    
    # print("Calculation of the light directions completed without errors")
    
    # print("Starting with RBF Interpolation...")
    
    # # rti.applRBFInterpolation(11, 11, DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE)
    
    # print("RBF Interpolation done")
    
    rti.applyRelighting()
    
    # release videos and destroy windows
    videoStatic.releaseVideo()
    videoMoving.releaseVideo()
    cv.destroyAllWindows()
    
def initaliseMainWindow():
    # Build the Application (only one instance can exsists)
    app = QApplication([])
    
    # Show the main window
    mainWindow = MainWindow()
    
    # And shows it
    sys.exit(app.exec())
    
    
    
if __name__ == "__main__":
    # initaliseMainWindow()
    main()