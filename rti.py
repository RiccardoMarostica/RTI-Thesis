# Import default modules
import cv2 as cv
import numpy as np
import sys
import time

from PyQt6.QtWidgets import QApplication

# Import classes
from classes.videoSynchronisation import VideoSynchronisation
from classes.cameraCalibration import CameraCalibration
from classes.video import Video
from classes.rtiAlgorithm import RTI
from classes.gui import MainWindow

from constants import *

def main():
    
    # print("Starting camera calibration...")
    
    # # Get the two calibrations for both static and moving camera
    # calibrationStatic = CameraCalibration(Video(STATIC_VIDEO_CALIBRATION_FILE_PATH), (9, 6))
    # calibrationMoving = CameraCalibration(Video(MOVING_VIDEO_CALIBRATION_FILE_PATH), (9, 6))
    
    # # Calibrate cameras and check result
    # if (calibrationStatic.calibrateCamera() == False or calibrationMoving.calibrateCamera() == False):
    #     print("Error when calibrating one of the two cameras. ")
    #     exit(-1)
    # else:
    #     print("Camera calibration completed without errors")
        
    # Create the two videos
    videoStatic = Video(STATIC_VIDEO_FILE_PATH)
    videoMoving = Video(MOVING_VIDEO_FILE_PATH)
    
    # Initialise RTI class
    rti = RTI()    
    
    # TODO: Only for testing purpose
    defaultK = rti.getDefaultK(videoMoving)
    
    # And try to get the 4 points in the static video
    worldHomography = rti.getWorldHomography(videoStatic)

    if len(worldHomography) == 0:
        print("Error when computing homography to get world reference system. ")
        exit(-1)
    else:
        print("Homography calculated without errors")
    
    # print("Starting video synchronisation...")
    
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
    
    frameDifference = 33
    
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
    
        # TODO: Apply undistortion
        
        # Convert frames to grayscale
        staticFrame = cv.cvtColor(staticFrame, cv.COLOR_BGR2GRAY)
        movingFrame = cv.cvtColor(movingFrame, cv.COLOR_BGR2GRAY)
        
        # Now get world frame using static camera and the homography
        worldFrame = cv.warpPerspective(staticFrame, worldHomography, (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE))
        
        # Now, it's possible to get homography between the world and the moving camera
        # Important: The order of parameters is important. In our case the mapping of the features to calculate the homography
        # are from the world frame to the moving frame.
        # Changing the order will change the computation of the ligth direction
        homographyWorldMoving = rti.getHomographyWithFeatureMatching(worldFrame, movingFrame)
        
        if (len(homographyWorldMoving) != 0):
            # If the homography is defined, it's possible to retrieve the extrinsic parameters R and T
            R, T = rti.getExtrinsicsParameters(homographyWorldMoving, defaultK)
        else:
            R = T = []
            
        if len(R) != 0 and len(T) != 0:
            # Get the light vector
            lightVector = rti.getLightVector(R, T)
            # ... and store it inside the light directions
            rti.storeLightVector(worldFrame, lightVector)
        else:
            lightVector = []
        
        iteration += 1
        
        print("Ligth vector: ", lightVector)
        
        cirlePlot = rti.showCircleLightDirection(lightVector)
        
        cv.imshow('Light plot', cirlePlot)
        cv.imshow('World frame', worldFrame)
        cv.imshow('Static camera', cv.resize(staticFrame,(480, 960)))
        cv.imshow('Moving camera', cv.resize(movingFrame,(480, 960)))
        
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
        
        # Press S to continue to the next view
        if cv.waitKey(0) & 0xFF == ord('s'):
            continue

    
    # lightDirections = rti.getLightDirections()

    # print("Frames aquired: ", len(lightDirections))
    
    # print("Calculation of the light directions completed without errors")
    
    # print("Starting with RBF Interpolation...")
    
    # # rti.applRBFInterpolation(11, 11, DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE)
    
    # print("RBF Interpolation done")
    
    # rti.applyRelighting()
    
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