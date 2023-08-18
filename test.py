# Import default modules
import os, h5py, cv2 as cv, numpy as np, time
from datetime import datetime

# Import classes
from classes.cameraCalibration import CameraCalibration
from classes.video import Video
from classes.rtiAlgorithm import RTI

from constants import *
from utils import *

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    rti = RTI()
    
    # Retrieve K for both cameras
    kStatic = calibrationStatic.getIntrinsicMatrix()
    kMoving = calibrationMoving.getIntrinsicMatrix()

    # Store the first frame of the Static Camera
    _, firstStaticFrame = videoStatic.getCurrentFrame()
    firstStaticFrame = cv.resize(firstStaticFrame, (1080, 1920))
    
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
    
    # frameDifference = -2    # Frame difference between unive (filename) video camera
    # # frameDifference = -10   # Frame difference between keys (filename) video camera
    # # frameDifference = -6    # Frame difference between paperclip (filename) video camera
    # # frameDifference = -9    # Frame difference between book (filename) video camera
    
    # print("Frame difference: ", frameDifference)
    
    # if (frameDifference > 0):
    #     print("Static Video shifted")
    #     # If the offset is positive, then the first video starts sooner.
    #     # So move its position in order to start as the second video
    #     videoStatic.setVideoFrame(abs(frameDifference))
    #     videoMoving.setVideoFrame()
    # else:
    #     print("Moving Video shifted")
    #     # ... or vice versa
    #     videoStatic.setVideoFrame()
    #     videoMoving.setVideoFrame(abs(frameDifference))
        
    # print("Starting calculation of the light directions in the videos...")
    
    # # Variable used to store the time calculated after each read on the video, in order to provide synchronisation
    # timeStaticVideo = 0.
    # timeMovingVideo = 0.
    
    # # Number of frames read
    # nFrames = 0
    
    # # Array with shape (400, 400, 2) which contains the sum of the U and V value of each pixel along the video
    # sumUV = np.zeros((400, 400, 2))
    
    # # Array with shape (nFrames, 400, 400, 3) where, for each pixel, stores the intensity of it, and the light value X and Y (costant along the frame)
    # lightData = []
    
    # staticFrames = []
    # movingFrames = []
    
    # # Get the FPS of both frames
    # videoStaticFPS = videoStatic.getFPS()
    # videoMovingFPS = videoMoving.getFPS()
    
    # while True:
        
    #     # Get frame from each video
    #     retStatic, staticFrame = videoStatic.getCurrentFrame()
    #     retMoving, movingFrame = videoMoving.getCurrentFrame()
        
    #     # For each iteration, sum the time for each video based on the tick (1 / FPS_video)
    #     timeStaticVideo += 1. / videoStaticFPS
    #     timeMovingVideo += 1. / videoMovingFPS
        
    #     # Now depends on which video has lower FPS
    #     if videoStaticFPS < videoMovingFPS:
    #         # Video static is behind more than 1 frame, so skip it to recover the loss
    #         if timeStaticVideo > timeMovingVideo + (1. / videoMovingFPS):
    #             retStatic, staticFrame = videoStatic.getCurrentFrame()
    #     else:    
    #         # Video moving is behind more than 1 frame, so skip it to recover the loss
    #         if timeMovingVideo > timeStaticVideo + (1. / videoStaticFPS):
    #             retMoving, movingFrame = videoMoving.getCurrentFrame()
        
    #     checkStaticFrame = (staticFrame is None or np.shape(staticFrame) == () or np.sum(staticFrame) == 0)
    #     checkMovingFrame = (movingFrame is None or np.shape(movingFrame) == () or np.sum(movingFrame) == 0)
            
    #     if retStatic != True or retMoving != True or checkStaticFrame or checkMovingFrame:
    #         break
            
    #     staticFrame = cv.resize(staticFrame, (1080, 1920))
        
    #     staticFrames.append(staticFrame)
    #     movingFrames.append(movingFrame)
        
    #     if cv.waitKey(20) == ord('q'):
    #         break
    
    # print(f"Images acquired. {len(staticFrames)}")
    
    # # Release videos and destroy windows
    # videoStatic.releaseVideo()
    # videoMoving.releaseVideo()
    # cv.destroyAllWindows()
    
    # np.save('frames/static/frames', staticFrames)
    # np.save('frames/moving/frames', movingFrames)
    
    videoStatic.releaseVideo()
    
    staticFrames = np.load('frames/static/frames.npy')
    movingFrames = np.load('frames/moving/frames.npy')
    
    # Get keypoints and descriptor for first frame
    featuresFirstStaticFrame = extractFeaturesFromFrame(firstStaticFrame)

    # staticFramesIdxs = [i for i in range(len(staticFrames))]
    # movingFrameIdxs = [i for i in range(len(movingFrames))]

    start = time.time()
    
    featuresStaticFrames = []
    executor = ThreadPoolExecutor()
    for result in executor.map(extractFeaturesFromFrame, staticFrames):
        featuresStaticFrames.append(result)
        
    end = time.time()
    
    print(f"Time of execution to get features from static frames: {((end - start) * 10**3)}ms")
    
    start = time.time()
    
    featuresMovingFrames = []
    for result in executor.map(extractFeaturesFromFrame, movingFrames):
        featuresMovingFrames.append(result)
        
    end = time.time()
    
    print(f"Time of execution to get feature from moving frames: {((end - start) * 10**3)}ms")
    
    start = time.time()
    
    featuresFirstFrame = [featuresFirstStaticFrame] * len(featuresStaticFrames)
    matchingStaticStatic = []
    for result in executor.map(matchFeatures, featuresFirstFrame, featuresStaticFrames):
        matchingStaticStatic.append(result)
        
    end = time.time()
    
    print(f"Time of execution to extract good matches between first static frame and current static frame: {((end - start) * 10**3)}ms")
    
    start = time.time()
    
    matchingStaticMoving = []
    for result in executor.map(matchFeatures, featuresStaticFrames, featuresMovingFrames):
        matchingStaticMoving.append(result)
        
    end = time.time()
    
    print(f"Time of execution to extract good matches between current static and moving frame: {((end - start) * 10**3)}ms")
    print(f"Matches extracted between static and static: {len(matchingStaticStatic)}, and static and moving: {len(matchingStaticMoving)}")
        
    start = time.time()
    
    lightFramePair = []
    submittedTasks = []
    
    for idx in range(len(staticFrames)):
        # Get frames
        frame1 = staticFrames[idx]
        frame2 = movingFrames[idx]
        # Get static matching
        _, _, homographyStaticToStatic = matchingStaticStatic[idx]
        _, ptsMovingCam, homographyStaticToMoving = matchingStaticMoving[idx]
        
        future = executor.submit(getLight, frame1, frame2, homographyStaticToStatic, ptsMovingCam, homographyStaticToMoving, worldHomography, kMoving)
        submittedTasks.append(future)
    
    for task in submittedTasks:
        res = task.result()
        lightFramePair.append(res)
        
    
    end = time.time()
    
    print(f"Time of execution to get light from all frames: {((end - start) * 10**3)}ms")
    
    validPairs = [pair for pair in lightFramePair if all(value is not None for value in pair)]
    
    print(f"Valid pairs: {len(validPairs)}")
    
    for frame, light in validPairs:
        # Show the light plot of the calculated light vector
        cirlePlotPnP = rti.showCircleLightDirection(light)
        
        # Plot images
        cv.imshow('Light plot PnP', cirlePlotPnP)
        cv.imshow('World frame', frame)
                    
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
    
    
def extractFeaturesFromFrame(frame):
        sift = cv.SIFT_create(nfeatures=1000)
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        return keypoints, descriptors
    
def matchFeatures(features1, features2):
        try:
            # Get both features
            # features1, features2 = features
            
            # Extract keypoints and descriptors of both features
            keypoints1, descriptors1 = features1
            keypoints2, descriptors2 = features2
            
            flann = cv.FlannBasedMatcher_create()
        
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        except:
            return None, None, None
            
        src = []
        dst = []
        
        for m1, m2 in matches:
            if m1.distance < 0.7 * m2.distance:
                
                src_pt = keypoints1[m1.queryIdx].pt
                dst_pt = keypoints2[m1.trainIdx].pt
                
                src.append(src_pt)
                dst.append(dst_pt)
                    
        # Set a treshold (MIN_MATCH_COUNT) which denotes the minimum number of matches to get the Homography
        if len(src) >= MIN_MATCH_COUNT:
            
            # Get source and destination points found inside the good matches to build the homography between the two frames
            src = np.float32(src).reshape(-1, 1, 2)
            dst = np.float32(dst).reshape(-1, 1, 2)
            
            # Get the Homography. In this case the method used to findthe transformation is through RANSAC, a consensus-based approach. Since RANSAC is used, it's necessary to set a treshold in which a point pair is considered as an inlier.
            homography, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)
            
            return src, dst, homography
        else:
            return None, None, None
    
def getLight(staticFrame, movingFrame, homographyStaticToStatic, ptsMovingCam, homographyStaticToMoving, worldHomography, kMoving):  
        
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
                
                # Set a treshold (MIN_MATCH_COUNT) which denotes the minimum number of matches to get the Homography
                src = points3d
                dst = np.squeeze(points2d)
                
                if len(src) > MIN_MATCH_COUNT:
                
                    ret, rvec, tvec = cv.solvePnP(src, dst, kMoving, None, flags=cv.SOLVEPNP_IPPE)
                    
                    if not ret:
                        # if solvePnP fails, then return an empty array, corresponding to no light
                        lightVector = worldFrame = None
                    
                    # Get rotation
                    R, _ = cv.Rodrigues(rvec)
                    
                    # then compute light vector
                    lightVector = -R.T @ tvec
                    lightVector = lightVector / np.linalg.norm(lightVector)      
                    
                    # If any of the position is Nan, then skip it
                    if np.isnan(lightVector).any():
                        lightVector = worldFrame = None
                    
                else:
                    lightVector = worldFrame = None
            else:
                lightVector = worldFrame = None
            
        else:
            # Otherwise, if one of the two homographies is not defined, then the light vector is None
            lightVector = worldFrame = None
    
        return (worldFrame, lightVector)
        
if __name__ == "__main__":
    main()
    