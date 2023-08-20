# Import default modules
import os, h5py, cv2 as cv, numpy as np, time
from datetime import datetime

# Import classes
from classes.cameraCalibration import CameraCalibration
from classes.video import Video
from classes.videoAnalysis import VideoAnalysis
from classes.threadPool import ThreadPool
from classes.pca import PCAClass
from classes.neuralNetwork import NeuralNetwork


from constants import *
from utils import *

def main():
    
    print("Starting camera calibration...")
    
    # Get the two calibrations for both static and moving camera
    calibrationStatic = CameraCalibration(Video(STATIC_VIDEO_CALIBRATION_FILE_PATH), (9, 6))
    calibrationMoving = CameraCalibration(Video(MOVING_VIDEO_CALIBRATION_FILE_PATH), (9, 6))
    
    # Calibrate both cameras
    retCalibStatic = calibrationStatic.calibrateCamera((1080, 1920))
    retCalibMoving = calibrationMoving.calibrateCamera((1920, 1080))
    
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
    videoAnalysis = VideoAnalysis()
    
    # Retrieve K for both cameras
    kStatic = calibrationStatic.getIntrinsicMatrix()
    kMoving = calibrationMoving.getIntrinsicMatrix()

    # Store the first frame of the Static Camera
    _, firstStaticFrame = videoStatic.getCurrentFrame()
    firstStaticFrame = cv.resize(firstStaticFrame, (1080, 1920))
    firstStaticFrame = cv.cvtColor(firstStaticFrame, cv.COLOR_BGR2GRAY)
    
    # And try to get the 4 points in the static video
    worldHomography = videoAnalysis.getWorldHomography(videoStatic)

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
        
    # print("Starting calculation of the light directions in the videos...")
    
    # Variable used to store the time calculated after each read on the video, in order to provide synchronisation
    timeStaticVideo = 0.
    timeMovingVideo = 0.
    
    # Number of frames read
    nFrames = 0
    
    # Array with shape (400, 400, 2) which contains the sum of the U and V value of each pixel along the video
    sumUV = np.zeros((400, 400, 2))
    
    # Array with shape (nFrames, 400, 400, 3) where, for each pixel, stores the intensity of it, and the light value X and Y (costant along the frame)
    lightData = []
    
    staticFrames = []
    movingFrames = []
    
    # Get the FPS of both frames
    videoStaticFPS = videoStatic.getFPS()
    videoMovingFPS = videoMoving.getFPS()
    
    while True:
        
        # Get frame from each video
        retStatic, staticFrame = videoStatic.getCurrentFrame()
        retMoving, movingFrame = videoMoving.getCurrentFrame()
        
        # For each iteration, sum the time for each video based on the tick (1 / FPS_video)
        timeStaticVideo += 1. / videoStaticFPS
        timeMovingVideo += 1. / videoMovingFPS
        
        # Now depends on which video has lower FPS
        if videoStaticFPS < videoMovingFPS:
            # Video static is behind more than 1 frame, so skip it to recover the loss
            if timeStaticVideo > timeMovingVideo + (1. / videoMovingFPS):
                retStatic, staticFrame = videoStatic.getCurrentFrame()
        else:    
            # Video moving is behind more than 1 frame, so skip it to recover the loss
            if timeMovingVideo > timeStaticVideo + (1. / videoStaticFPS):
                retMoving, movingFrame = videoMoving.getCurrentFrame()
        
        checkStaticFrame = (staticFrame is None or np.shape(staticFrame) == () or np.sum(staticFrame) == 0)
        checkMovingFrame = (movingFrame is None or np.shape(movingFrame) == () or np.sum(movingFrame) == 0)
            
        if retStatic != True or retMoving != True or checkStaticFrame or checkMovingFrame:
            break
                    
        staticFrame = cv.resize(staticFrame, (1080, 1920))
        
        # Convert frames to grayscale
        staticFrame = cv.cvtColor(staticFrame, cv.COLOR_BGR2GRAY)
        movingFrame = cv.cvtColor(movingFrame, cv.COLOR_BGR2GRAY)
        
        staticFrames.append(staticFrame)
        movingFrames.append(movingFrame)
    
    print(f"Images acquired. {len(staticFrames)}")
    
    # Release videos and destroy windows
    videoStatic.releaseVideo()
    videoMoving.releaseVideo()
    cv.destroyAllWindows()
    
    # Create thread pool, with 4 threads
    pool = ThreadPool(4)
    
    # Get keypoints and descriptor for first frame
    featuresFirstStaticFrame = videoAnalysis.extractFeaturesFromFrame(firstStaticFrame, 0)
    
    start = time.time()
    
    for i in range(len(staticFrames)):
        frame = staticFrames[i]
        # For each static frame, calculate its features
        pool.add_task(videoAnalysis.extractFeaturesFromFrame, frame, i)
        
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    featuresStaticFrames = pool.get_results()
        
    end = time.time()
    
    print(f"Time of execution to get features from static frames: {int(end - start)} seconds")
    
    start = time.time()
    
    for i in range(len(movingFrames)):
        frame = movingFrames[i]
        # For each static frame, calculate its features
        pool.add_task(videoAnalysis.extractFeaturesFromFrame, frame, i)
        
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    featuresMovingFrames = pool.get_results()
        
    end = time.time()
    
    print(f"Time of execution to get features from moving frames: {int(end - start)} seconds")
    
    # Now sort features based on index
    featuresStaticFrames = sorted(featuresStaticFrames, key = lambda x: x[0])
    featuresMovingFrames = sorted(featuresMovingFrames, key = lambda x: x[0])
    
    # Replicate first frame features to then match it to all the other frames in the static camera 
    featuresFirstFrame = [featuresFirstStaticFrame] * len(featuresStaticFrames)
    
    # Create a list of features that will be matched using feature matching technique
    featuresStaticStatic = list(zip(featuresStaticFrames, featuresFirstFrame))
    featuresStaticMoving = list(zip(featuresStaticFrames, featuresMovingFrames))
        
    start = time.time()
    
    print("Extracting matches between first static frame and other static frames...")
    
    for i in range(len(featuresStaticStatic)):
        feature = featuresStaticStatic[i]
        # For each static frame, calculate its features
        pool.add_task(videoAnalysis.matchFeatures, feature)
        
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    matchingStaticStatic = pool.get_results()
    
    print("Extracting matches between static frames and moving frames...")
    
    # Repeat the process
    for i in range(len(featuresStaticMoving)):
        feature = featuresStaticMoving[i]
        # For each static frame, calculate its features
        pool.add_task(videoAnalysis.matchFeatures, feature)
        
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    matchingStaticMoving = pool.get_results()
    
    end = time.time()
    
    print(f"Time of execution to extract good matches: {int(end - start)} seconds")
    
    # Now sort matching based on index
    matchingStaticStatic = sorted(matchingStaticStatic, key = lambda x: x[0])
    matchingStaticMoving = sorted(matchingStaticMoving, key = lambda x: x[0])
    
    start = time.time()
    
    for i in range(len(matchingStaticStatic)):
        # Get frames
        staticFrame = staticFrames[i]
        movingFrame = movingFrames[i]
        # Get homographies and dst pts
        _, _, _, homographyStatic = matchingStaticStatic[i]
        _, _, dstPts, homographyMoving = matchingStaticMoving[i]
        
        # Calculate light given parameters
        pool.add_task(videoAnalysis.getLight, staticFrame, movingFrame, homographyStatic, dstPts, homographyMoving, worldHomography, kMoving)
    
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    lightFramePair = pool.get_results()
    
    end = time.time()
    
    print(f"Time of execution to get light from all frames: {int(end - start)} seconds")
    
    validPairs = [pair for pair in lightFramePair if all(value is not None for value in pair)]
    
    for worldFrame, light in validPairs:
        
        # Show the light plot of the calculated light vector
        cirlePlotPnP = getLightDirectionPlot(light, DEFAULT_SQUARE_SIZE)
        
        # First, convert the frame from GRAY to BGR
        # Then from BGR to YUV, to extract the intensity and calculate U and V mean
        worldFrameBGR = cv.cvtColor(worldFrame, cv.COLOR_GRAY2BGR)
        worldFrameYUV = cv.cvtColor(worldFrameBGR, cv.COLOR_BGR2YUV)
        
        # Get Y, U, V
        Y, U, V = cv.split(worldFrameYUV)
        
        # Get the light position X and Y (Z can be removed now)            
        light = np.tile(light[:2].flatten(), (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE, 1))
    
        # Now, store an array containing the intensity of the pixels and the respective light
        data = np.dstack((Y, light))
        
        # Now get the current UV with shape (400, 400, 2), and sum their values inside the sumUV matrix with shape (400, 400, 2)
        sumUV = sumUV + np.dstack((U, V))
        
        # Append the data
        lightData.append(data)
        
        # Increament number of frames acquired
        nFrames += 1
        
        # Plot images
        cv.imshow('Light plot PnP', cirlePlotPnP)
        cv.imshow('World frame', worldFrame)
                    
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
    
    # Calculate UVMean
    meanUV = sumUV / nFrames
    
    # Convert from list 2 array
    lightData = np.stack(lightData)
    
    # Now, store this values inside a file    
    now_string = datetime.now().strftime("%y_%m_%d_%H_%M")
    
    # First get the base dir
    BASE_DIR = "relights/relight_%s/"%now_string
    
    try:
        # Creating the base dir
        os.mkdir(BASE_DIR)
    except:
        print("Error creating the new folder. ")
        # Not possible to create the dir, close the app
        exit(-1)
    
    # Set the name of the file containing the inital dataset for training
    fileName = BASE_DIR + "data.h5"
    
    # Then create the train dataset
    storeTrainDataset(fileName, lightData, meanUV)
    
def testNNWithThreads():
    
    # First get the base dir
    BASE_DIR = "relights/relight_23_08_20_14_49/"    
    # Set the name of the file containing the inital dataset for training
    fileName = BASE_DIR + "data.h5"
    
    pca = PCAClass(BASE_DIR, fileName, 8)

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
    # main()
    testNNWithThreads()
    