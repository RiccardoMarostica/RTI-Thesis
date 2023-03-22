import cv2 as cv
import numpy as np
import functions.helpers as helpers

# Define main function to calibrate both camera parameters
def main():
    
    helpers.consoleLog("Starting calibrating the static camera", "Main")
    
    # In the main, call the method to calibrating the camera, starting from the static one.
    calibrateCamera(True)
    
    helpers.consoleLog("Complete calibrating the static camera", "Main")

    helpers.consoleLog("Starting calibrating the moving camera", "Main")
    
    # In the main, call the method to calibrating the camera, starting from the static one.
    calibrateCamera(False)
    
    helpers.consoleLog("Complete calibrating the moving camera", "Main")

def calibrateCamera(isStatic: bool):
    if isStatic:  # If param is true, then we compute parameters for the static camera
        videoCapture = cv.VideoCapture("./video/cam-static/calibration.mp4")
        paramsPath = "parameters/static/"
    else:  # Otherwise, we compute parameters for moving camera
        videoCapture = cv.VideoCapture("video/cam-moving/calibration.mp4")
        paramsPath = "parameters/moving/"

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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
                refinedCorners = cv.cornerSubPix(
                    grayFrame, corners, (11, 11), (-1, -1), criteria)

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

    # At the end, release the Video Capture
    videoCapture.release()

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
    calibrationPattern = cv.imread("video/calibration_pattern.png", cv.IMREAD_GRAYSCALE)
    
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
        
# Main method call
if __name__ == "__main__":
    main()
