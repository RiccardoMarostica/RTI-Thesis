# Import default modules
import os, h5py, cv2 as cv, numpy as np
from datetime import datetime

# Import classes
from classes.videoSynchronisation import VideoSynchronisation
from classes.cameraCalibration import CameraCalibration
from classes.video import Video
from classes.rtiAlgorithm import RTI
from classes.pca import PCAClass
from classes.neuralNetwork import NeuralNetwork

from constants import *
from utils import *

from concurrent.futures import ThreadPoolExecutor

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
    kMoving = calibrationMoving.getIntrinsicMatrix()

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
    
    frameDifference = -2    # Frame difference between unive (filename) video camera
    # frameDifference = -10   # Frame difference between keys (filename) video camera
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
    
    # Number of frames read
    nFrames = 0
    
    # Array with shape (400, 400, 2) which contains the sum of the U and V value of each pixel along the video
    sumUV = np.zeros((400, 400, 2))
    
    # Array with shape (nFrames, 400, 400, 3) where, for each pixel, stores the intensity of it, and the light value X and Y (costant along the frame)
    lightData = []
        
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
        
        checkStaticFrame = (staticFrame is None or np.shape(staticFrame) == () or np.sum(staticFrame) == 0)
        checkMovingFrame = (movingFrame is None or np.shape(movingFrame) == () or np.sum(movingFrame) == 0)
            
        if retStatic != True or retMoving != True or checkStaticFrame or checkMovingFrame:
            break
        
        # Convert frames to grayscale
        staticFrame = cv.cvtColor(staticFrame, cv.COLOR_BGR2GRAY)
        movingFrame = cv.cvtColor(movingFrame, cv.COLOR_BGR2GRAY)
        
        # UniVE video
        _, _, homographyStaticToStatic = rti.getHomographyWithFeatureMatching(staticFrame, firstStaticFrame, "Static to Static", False, cutFrame1 = ((500, 1700), (1400, 2600)), cutFrame2 = ((500, 1700), (1400, 2600)))
        _, ptsMovingCam, homographyStaticToMoving = rti.getHomographyWithFeatureMatching(staticFrame, movingFrame, "Static to Moving", False, cutFrame1 = ((500, 1700), (1400, 2600)), cutFrame2 = ((450, 1150), (200, 900)))    
        
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
            worldFrame = cv.warpPerspective(staticFrame, worldHomography @ homographyStaticToStatic, (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE))
            
            # ... and do the same for moving camera, in order to get a similarity between frames
            warpedMoving = cv.warpPerspective(movingFrame,  hWorld2Moving, (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE), flags = cv.WARP_INVERSE_MAP)
            
            # Now, let's try to cross-correlate the two warped images.
            # If the correlation is high, then the images are similar, so we can compute the light vector
            # Otherwise, skip the frame
            imgCorr = cv.matchTemplate(worldFrame, warpedMoving, cv.TM_CCORR_NORMED)
                            
            # Set as lower threshold 0.6 to have high confidentiality
            if imgCorr[0][0] >= 0.96:
                # Calculate the light vector using PnP
                lightVector = rti.getLigthWithSolvePnP(points3d, np.squeeze(points2d), kMoving)
            else:
                lightVector = None
        
            # Show the light plot of the calculated light vector
            cirlePlotPnP = rti.showCircleLightDirection(lightVector)
        
            # Plot images
            cv.imshow('Light plot PnP', cirlePlotPnP)
            cv.imshow('World frame', worldFrame)
            # cv.imshow('World frame moving', warpedMoving)
            
        else:
            # Otherwise, if one of the two homographies is not defined, then the light vector is None
            lightVector = None
        
        if lightVector is not None:
            # First, convert the frame from GRAY to BGR
            # Then from BGR to YUV, to extract the intensity and calculate U and V mean
            worldFrameBGR = cv.cvtColor(worldFrame, cv.COLOR_GRAY2BGR)
            worldFrameYUV = cv.cvtColor(worldFrameBGR, cv.COLOR_BGR2YUV)
            
            # Get Y, U, V
            Y, U, V = cv.split(worldFrameYUV)
            
            # Get the light position X and Y (Z can be removed now)            
            light = np.tile(lightVector[:2].flatten(), (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE, 1))
        
            # Now, store an array containing the intensity of the pixels and the respective light
            data = np.dstack((Y, light))
            
            # Now get the current UV with shape (400, 400, 2), and sum their values inside the sumUV matrix with shape (400, 400, 2)
            sumUV = sumUV + np.dstack((U, V))
            
            # Append the data
            lightData.append(data)
            
            # Increament number of frames acquired
            nFrames += 1
        
        print(f"Frame stored: {nFrames}, on the total frames of: {videoStatic.getTotalFrames()}")
            
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
        
    # Release videos and destroy windows
    videoStatic.releaseVideo()
    videoMoving.releaseVideo()
    cv.destroyAllWindows()  
    
    # Calculate UVMean
    meanUV = sumUV / nFrames
    
    # Convert from list 2 array
    lightData = np.stack(lightData)
    
    # Now, store this values inside a file    
    now_string = datetime.now().strftime("%y_%m_%d_%H_%M")
    dataName = 'unive'
    
    # First get the base dir
    BASE_DIR = "examples/%s_example"%dataName + "_%s/"%now_string
    
    try:
        # Creating the base dir
        os.mkdir(BASE_DIR)
    except:
        print("Error creating the new folder. ")
        # Not possible to create the dir, close the app
        exit(-1)    
    
    # Set the name of the file containing the inital dataset for training
    fileName = BASE_DIR + "%s.h5"%dataName
    
    # Then create the train dataset
    storeTrainDataset(fileName, dataName, lightData, meanUV)
    
    # Create PCA class
    pca = PCAClass(BASE_DIR, dataName, 8)
    # Read the dataset and get ready to perform PCA
    pca.readDataset()
    # Perform PCA on training dataset
    pca.applyPCA()
    
    pca = PCAClass(BASE_DIR, dataName, 8)

    print("Reading dataset...")
    pca.readDataset()
    print("Reading dataset: DONE")

    print("Applying PCA...")
    pca.applyPCA()
    print("Applying PCA: DONE")

    nn = NeuralNetwork(BASE_DIR, 8)

    print("Extracting dataset...")
    nn.extractDatasets()
    print("Extracting dataset: DONE")

    print("Shufflings dataset...")
    nn.shuffleDataset()
    print("Shuffling dataset: DONE")

    print("Executing NN training...")
    nn.executeTraining()
    print("Executing NN trainin: DONE")
    
    print("Showing results")    
    nn.showNNResults()
        
        
if __name__ == "__main__":
    main()