# Import default modules
import os, h5py, cv2 as cv, numpy as np, time
from datetime import datetime

# Import classes
from classes.cameraCalibration import CameraCalibration
from classes.video import Video
from classes.rtiAlgorithm import RTI
from classes.threadPool import ThreadPool


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
    firstStaticFrame = cv.cvtColor(firstStaticFrame, cv.COLOR_BGR2GRAY)
    
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
        
    # print("Starting calculation of the light directions in the videos...")
    
    # Variable used to store the time calculated after each read on the video, in order to provide synchronisation
    timeStaticVideo = 0.
    timeMovingVideo = 0.
    
    # # Number of frames read
    # nFrames = 0
    
    # # Array with shape (400, 400, 2) which contains the sum of the U and V value of each pixel along the video
    # sumUV = np.zeros((400, 400, 2))
    
    # # Array with shape (nFrames, 400, 400, 3) where, for each pixel, stores the intensity of it, and the light value X and Y (costant along the frame)
    # lightData = []
    
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
    
    # np.save('frames/static/frames', staticFrames)
    # np.save('frames/moving/frames', movingFrames)
    # staticFrames = np.load('frames/static/frames.npy')
    # movingFrames = np.load('frames/moving/frames.npy')
    
    # Create thread pool, with 4 threads
    pool = ThreadPool(4)
    
    # Get keypoints and descriptor for first frame
    featuresFirstStaticFrame = extractFeaturesFromFrame(firstStaticFrame, 0)
    
    start = time.time()
    
    for i in range(len(staticFrames)):
        frame = staticFrames[i]
        # For each static frame, calculate its features
        pool.add_task(extractFeaturesFromFrame, frame, i)
        
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    featuresStaticFrames = pool.get_results()
        
    end = time.time()
    
    print(f"Time of execution to get features from static frames: {int(end - start)} seconds")
    print(f"Lenght of result array: {len(featuresStaticFrames)}")
    
    start = time.time()
    
    for i in range(len(movingFrames)):
        frame = movingFrames[i]
        # For each static frame, calculate its features
        pool.add_task(extractFeaturesFromFrame, frame, i)
        
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    featuresMovingFrames = pool.get_results()
        
    end = time.time()
    
    print(f"Time of execution to get features from moving frames: {int(end - start)} seconds")
    print(f"Lenght of result array: {len(featuresMovingFrames)}")
    
    # Now sort features based on index
    featuresStaticFrames = sorted(featuresStaticFrames, key = lambda x: x[0])
    featuresMovingFrames = sorted(featuresMovingFrames, key = lambda x: x[0])
    
    # Replicate first frame features to then match it to all the other frames in the static camera 
    featuresFirstFrame = [featuresFirstStaticFrame] * len(featuresStaticFrames)
    
    # Create a list of features that will be matched using feature matching technique
    featuresStaticStatic = list(zip(featuresFirstFrame, featuresStaticFrames))
    featuresStaticMoving = list(zip(featuresStaticFrames, featuresMovingFrames))
        
    start = time.time()
    
    for i in range(len(featuresStaticStatic)):
        feature = featuresStaticStatic[i]
        # For each static frame, calculate its features
        pool.add_task(matchFeatures, feature, ((500, 1700), (1400, 2600)), ((500, 1700), (1400, 2600)))
        
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    matchingStaticStatic = pool.get_results()
    
    # Repeat the process
    for i in range(len(featuresStaticMoving)):
        feature = featuresStaticMoving[i]
        # For each static frame, calculate its features
        pool.add_task(matchFeatures, feature, ((500, 1700), (1400, 2600)), ((450, 1150), (200, 900)))
        
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
        pool.add_task(getLight, staticFrame, movingFrame, homographyStatic, dstPts, homographyMoving, worldHomography, kMoving)
    
    # Wait completion of the queue
    pool.wait_completion()
    
    # Get the results
    lightFramePair = pool.get_results()
    
    end = time.time()
    
    print(f"Time of execution to get light from all frames: {int(end - start)} seconds")
    
    # validPairs = [pair for pair in lightFramePair if all(value is not None for value in pair)]
    
    print(f"Valid pairs: {len(lightFramePair)}")
    
    for worldFrame, movingFrame in lightFramePair:
        
        # Plot images
        cv.imshow('Moving warped frame', movingFrame)
        cv.imshow('World frame', worldFrame)
                    
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
    
    # for frame, light in lightFramePair:
    #     # Show the light plot of the calculated light vector
    #     cirlePlotPnP = rti.showCircleLightDirection(light)
        
    #     # Plot images
    #     cv.imshow('Light plot PnP', cirlePlotPnP)
    #     cv.imshow('World frame', frame)
                    
    #     # Press Q on the keyboard to exit.
    #     if (cv.waitKey(25) & 0xFF == ord('q')):
    #         break
    
    
def extractFeaturesFromFrame(frame, idx):
        sift = cv.SIFT_create(nfeatures=3000)
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        return idx, keypoints, descriptors
    
def matchFeatures(features, cutFrame1, cutFrame2):
        try:
            # Get both features
            features1, features2 = features
            
            # Extract keypoints and descriptors of both features
            idx1, keypoints1, descriptors1 = features1
            idx2, keypoints2, descriptors2 = features2
            
            flann = cv.FlannBasedMatcher_create()
        
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        except:
            print("Error in match feature")
            return idx2, None, None, None
            
        src = []
        dst = []
        
        for m1, m2 in matches:
            if m1.distance < 0.7 * m2.distance:
                
                src_pt = keypoints1[m1.queryIdx].pt
                dst_pt = keypoints2[m1.trainIdx].pt
                
                if cutFrame1 is not None and cutFrame2 is not None:
                    
                    # Get cut points for src
                    srcCutX = cutFrame1[0]
                    srcCutY = cutFrame1[1]
                    
                    # Get cut points for dst
                    dstCutX = cutFrame2[0]
                    dstCutY = cutFrame2[1]
                    
                    isInsideSrcCut = (srcCutX[0] <= src_pt[0] <= srcCutX[1]) and (srcCutY[0] <= src_pt[1] <= srcCutY[1])
                    isInsideDstCut = (dstCutX[0] <= dst_pt[0] <= dstCutX[1]) and (dstCutY[0] <= dst_pt[1] <= dstCutY[1])
                    
                    if isInsideSrcCut == True and isInsideDstCut == True:
                        src.append(src_pt)
                        dst.append(dst_pt)
                                
                else:            
                    src.append(src_pt)
                    dst.append(dst_pt)
                    
        # Set a treshold (MIN_MATCH_COUNT) which denotes the minimum number of matches to get the Homography
        if len(src) >= MIN_MATCH_COUNT:
            
            # Get source and destination points found inside the good matches to build the homography between the two frames
            src = np.float32(src).reshape(-1, 1, 2)
            dst = np.float32(dst).reshape(-1, 1, 2)
            
            # Get the Homography. In this case the method used to findthe transformation is through RANSAC, a consensus-based approach. Since RANSAC is used, it's necessary to set a treshold in which a point pair is considered as an inlier.
            homography, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)
            
            return idx2, src, dst, homography
        else:
            print("Not enough pts")
            return idx2, None, None, None
    
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
            # worldFrame = cv.warpPerspective(staticFrame, worldHomography @ homographyStaticToStatic, (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE))
            worldFrame = cv.warpPerspective(staticFrame, worldHomography, (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE))
            
            # ... and do the same for moving camera, in order to get a similarity between frames
            warpedMoving = cv.warpPerspective(movingFrame,  hWorld2Moving, (DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE), flags = cv.WARP_INVERSE_MAP)
        
        return worldFrame, warpedMoving
            
        #     # Now, let's try to cross-correlate the two warped images.
        #     # If the correlation is high, then the images are similar, so we can compute the light vector
        #     # Otherwise, skip the frame
        #     imgCorr = cv.matchTemplate(worldFrame, warpedMoving, cv.TM_CCORR_NORMED)
                            
        #     # Set as lower threshold 0.6 to have high confidentiality
        #     if imgCorr[0][0] >= 0.96:
        #         # Calculate the light vector using PnP
                
        #         # Set a treshold (MIN_MATCH_COUNT) which denotes the minimum number of matches to get the Homography
        #         src = points3d
        #         dst = np.squeeze(points2d)
                
        #         if len(src) > MIN_MATCH_COUNT:
                
        #             ret, rvec, tvec = cv.solvePnP(src, dst, kMoving, None, flags=cv.SOLVEPNP_IPPE)
                    
        #             if not ret:
        #                 # if solvePnP fails, then return an empty array, corresponding to no light
        #                 lightVector = worldFrame = None
                    
        #             # Get rotation
        #             R, _ = cv.Rodrigues(rvec)
                    
        #             # then compute light vector
        #             lightVector = -R.T @ tvec
        #             lightVector = lightVector / np.linalg.norm(lightVector)      
                    
        #             # If any of the position is Nan, then skip it
        #             if np.isnan(lightVector).any():
        #                 lightVector = worldFrame = None
                    
        #         else:
        #             lightVector = worldFrame = None
        #     else:
        #         lightVector = worldFrame = None
            
        # else:
        #     # Otherwise, if one of the two homographies is not defined, then the light vector is None
        #     lightVector = worldFrame = None
    
        # return (worldFrame, lightVector)
        
if __name__ == "__main__":
    main()
    