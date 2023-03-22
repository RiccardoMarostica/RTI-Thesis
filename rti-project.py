import cv2 as cv
import numpy as np
import functions.helpers as helpers
import moviepy.editor as mp
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"

from constants import *

# Define main function to calibrate both camera parameters
def main():
    
    # helpers.consoleLog("Starting camera calibrations", "MAIN")
    
    # # First, calibrate the static camera
    # calibrateCamera(True)

    # # Then calibrate the moving camera    
    # calibrateCamera(False)
    
    # helpers.consoleLog("Ended camera calibrations", "MAIN")
    
    # # After retrieving the parameters of the camera, let's compute the offset between the two videos    
    # # Get the videos
    videoStatic = cv.VideoCapture(STATIC_VIDEO_FILE_PATH)
    videoMoving = cv.VideoCapture(MOVING_VIDEO_FILE_PATH)
    
    # helpers.consoleLog("Video synchronisation starts", "MAIN")
    
    # synchroniseVideos(videoStatic, STATIC_VIDEO_FILE_PATH, videoMoving, MOVING_VIDEO_FILE_PATH)
    
    # helpers.consoleLog("Video synchronisation ends", "MAIN")
    
    # After setting the videos and synchronise them, the next step is to remove the distortion made by the camera lens
    # To do so, we need to use the undistortion values obtained in the previous step (camera calibration)
    # Apply the corresponding array to the proper video (for each frame) to remove any distortion
    
    # Get the distortion params from the file (for static camera)
    distVideoStatic = np.loadtxt(STATIC_VIDEO_PARAMS_PATH + "distortionCoeffs.dat")
    mtxVideoStatic = np.loadtxt(STATIC_VIDEO_PARAMS_PATH + "intrinsicMatrix.dat")
    
    # Get the distortion params from the file (for moving camera)
    distVideoMoving = np.loadtxt(MOVING_VIDEO_PARAMS_PATH + "distortionCoeffs.dat")
    mtxVideoMoving = np.loadtxt(MOVING_VIDEO_PARAMS_PATH + "intrinsicMatrix.dat")

    # Using Width and Height of the static camera, compute the new camera matrix for the static camera
    wStatic = int(videoStatic.get(cv.CAP_PROP_FRAME_WIDTH))
    hStatic = int(videoStatic.get(cv.CAP_PROP_FRAME_HEIGHT))
    staticNewCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtxVideoStatic, distVideoStatic, (wStatic, hStatic), 1, (wStatic, hStatic))

    # # And this is done for the moving camera too
    # wMoving = int(videoMoving.get(cv.CAP_PROP_FRAME_WIDTH))
    # hMoving = int(videoMoving.get(cv.CAP_PROP_FRAME_HEIGHT))
        
    # movingNewCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtxVideoMoving, distVideoMoving, (wMoving, hMoving), 1, (wMoving, hMoving))

    print(staticNewCameraMtx)
    print(roi)

    while videoStatic.isOpened():
        ret, frame = videoStatic.read()
        
        if not ret:
            break
        
        # Get undistortion
        undistFrame = cv.undistort(frame, mtxVideoStatic, distVideoStatic, None, staticNewCameraMtx)
        # ... and crop the image
        x, y, w, h = roi
        undistFrame = undistFrame[y:y+h, x:x+w]
    
        cv.imshow("Undistorted Video static", undistFrame)
        
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
    
    # # While one of the two videos is open, then read frame by frame
    # while videoStatic.isOpened() or videoMoving.isOpened():
        
    #     # Get the frame from each video
    #     ret1, frame1 = videoStatic.read()
    #     ret2, frame2 = videoMoving.read()

    #     # If one of the two does not return a frame, then exit the loop
    #     if not ret1 or not ret2:
    #         break
        
    #     cv.imshow("Frames Static Camera", frame1)
    #     cv.imshow("Frames Moving Camera", frame2)

    #     # Press Q on the keyboard to exit.
    #     if (cv.waitKey(25) & 0xFF == ord('q')):
    #         break

    # Release videos
    videoStatic.release()
    videoMoving.release()

    # And destroy windows
    cv.destroyAllWindows()
    
def calibrateCamera(isStatic: bool):
    if isStatic:  # If param is true, then we compute parameters for the static camera
        videoCapture = cv.VideoCapture(STATIC_VIDEO_CALIBRATION_FILE_PATH)
        paramsPath = STATIC_VIDEO_PARAMS_PATH
    else:  # Otherwise, we compute parameters for moving camera
        videoCapture = cv.VideoCapture(MOVING_VIDEO_CALIBRATION_FILE_PATH)
        paramsPath = MOVING_VIDEO_PARAMS_PATH

    # Video Capture has problem opening the Video stream, so close it
    if (videoCapture.isOpened() == False):
        helpers.consoleLog(
            "VideoCapture not opened. Not possible to calculate camera paramters",
            "calibrateCamera",
            True
        )
        exit(-1)
    else:
        helpers.consoleLog(
            "VideoCapture is opened. Start reading frames to calibrate the camera",
            "calibrateCamera"
        )

    # Variable used to jump over the next frame in the Video every 2 seconds and collect a new frame
    milliseconds = 0

    # Termination criteria used for method cornerSubPix, which is used to refine the corner locations
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, DEFAULT_FPS_RATE, 0.001)

    # Preprare object points, like: (0, 0, 0), (1, 0, 0), ..., (8, 5, 0)
    # For simplicity we can consider that z = 0
    objPoint = np.zeros((6 * 9, 3), np.float32)
    objPoint[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    #  Arrays to store object and image points from all frames
    objPoints = []  # Object Points : 3D points -> World space
    imgPoints = []  # Image Points : 2D points -> Image plane

    helpers.consoleLog(
        "Starting video analysis to compute corners of chessboard",
        "calibrateCamera"
    )

    # Loop over the VideoCapture until it becomes closed (at the end of the Video)
    while (videoCapture.isOpened()):
        # Read frame.
        # ret is used to check presence of the next frame (False otherwise)
        # frame is used to grab the current frame
        ret, frame = videoCapture.read()

        # Frame has been grabbed
        if ret == True:
            # Now, convert the frame into grayscale, which will be useful to perform the methods and retrieve
            # the image and object point
            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Given the gray frame, find the chessboard corners
            ret, corners = cv.findChessboardCorners(grayFrame, (9, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objPoints.append(objPoint)

                # Now, before adding image points, refine the corners
                refinedCorners = cv.cornerSubPix(grayFrame, corners, (11, 11), (-1, -1), criteria)

                # ... and add the refined corners to image points
                imgPoints.append(refinedCorners)

            # Now, jump over 2 seconds to get next frame
            milliseconds += 2000
            videoCapture.set(cv.CAP_PROP_POS_MSEC, milliseconds)

            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                break
        else:
            # Otherwise, no frame grabbed, so exit the loop
            break

    helpers.consoleLog(
        "Completed video analysis to compute corners of chessboard",
        "calibrateCamera"
    )

    # Now, check if object points (World Space) and image points (Camera plane) are different
    # If, so we need to throw an error since it's not possible to execute the camera calibration method
    if (len(objPoints) != len(imgPoints)):
        helpers.ConsoleLog(
            "Object and Image points are not equal. Impossible to retrieve intrisic parameters of the camera",
            "calibrateCamera",
            True
        )
        exit(-1)

    # Otherwise, we have all the information to perform camera calibration and retrieve the intrinsic and extrinsic parameters
    # First, read the image containing the calibration pattern, and convert into grayscale
    calibrationPattern = cv.imread(CALIBRATION_PATTERN_IMG_PATH, cv.IMREAD_GRAYSCALE)
    
    # Parameters:
    # matrix -> 3x3 floating point camera intrinsic matrix (remember that scale is equal to 0 by default)
    # dist -> vector of distortion coefficients
    # rvecs -> vector of rotation vectors
    # tvecs -> vector of translation vectors
    ret, matrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, calibrationPattern.shape[::-1], None, None)

    try:
        
        # At the end, store the results in different files
        # N.B: It's necessary to save only the intrinsic matrix and the distortion coeffs
        np.savetxt(paramsPath + "intrinsicMatrix.dat", matrix)
        np.savetxt(paramsPath + "distortionCoeffs.dat", dist)
        
        helpers.consoleLog("Camera parameters stored", "calibrateCamera")
        
    except Exception as error:
        print(str(error))
        # An error occurred
        helpers.consoleLog("An error occurred while storing the camera parameters", "calibrateCamera", True)
        exit(-1)
        
def synchroniseVideos(video1, video1Path, video2, video2Path):
    
    helpers.consoleLog("Started audio extraction", "SynchroniseVideos")
    
    # After getting the path of the videos, retrieve the audios
    audio1 = mp.AudioFileClip(video1Path)
    audio2 = mp.AudioFileClip(video2Path)

    # Now, get the inital 5 seconds in order to calculate the offset.
    # From these 5 seconds, retrieve the sound array and convert their values using as sampling rate 44100 Hz.
    # That value represnts the best value for audio sampling.
    audio1 = audio1.subclip(0, 5).to_soundarray(fps=DEFAULT_SAMPLING_AUDIO_RATE)
    audio2 = audio2.subclip(0, 5).to_soundarray(fps=DEFAULT_SAMPLING_AUDIO_RATE)
    
    helpers.consoleLog("Ended audio extraction", "SynchroniseVideos")

    # Compute the cross-correlation of the left audio channels
    corr = np.correlate(audio1[:, 0], audio2[:, 0], mode='full')
    offset = corr.argmax() - (len(corr) // 2)

    # Get offset in seconds
    offset_sec = offset / DEFAULT_SAMPLING_AUDIO_RATE

    helpers.consoleLog("Offset in seconds is: " + str(offset_sec), "SynchroniseVideos")

    # Then, get the fps of both video, and check if the both are equal to the default fps rate
    fpsVideo1 = int(round(video1.get(cv.CAP_PROP_FPS)))
    fpsVideo2 = int(round(video2.get(cv.CAP_PROP_FPS)))

    # If one of them is not as the default fps rate, then modify the video
    if (fpsVideo1 != DEFAULT_FPS_RATE or fpsVideo2 != DEFAULT_FPS_RATE):
        # TODO: Aggiungere parte per modifica FPS del video
        print("Different Video FPS")

    # Now, we can compute the frame count to shift the two videos
    frameCount = abs(int(round(offset_sec * DEFAULT_FPS_RATE)))

    helpers.consoleLog("Frame difference between two videos is: " + str(frameCount), "SynchroniseVideos")

    if offset > 0:
        # If the offset is positive, then the first video starts sooner.
        # So move its position in order to start as the second video
        helpers.consoleLog("The first video starts sooner, so shift it", "SynchroniseVideos")

        video1.set(cv.CAP_PROP_POS_FRAMES, frameCount)
        video2.set(cv.CAP_PROP_POS_FRAMES, 0)
    elif offset < 0:
        # Otherwise, if the offset is negative, then we have the opposite scenario.
        helpers.consoleLog("The second video starts sooner, so shift it", "SynchroniseVideos")
        
        video1.set(cv.CAP_PROP_POS_FRAMES, 0)
        video2.set(cv.CAP_PROP_POS_FRAMES, frameCount)
        
# Main method call
if __name__ == "__main__":
    main()
