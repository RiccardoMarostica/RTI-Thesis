import cv2 as cv
import numpy as np

from classes.video import Video


class CameraCalibration:

    # This boolean value permits to enable/disable debug features, like show images, etc...
    debug = False

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    milliseconds = 0

    # Construction, just take the video from the main
    def __init__(self, video : Video, corners : tuple):
        # Only field used to calibrate the camera
        self.video: Video = video

        # Get number of corners in the chessboard
        self.cornersX = corners[0]
        self.cornersY = corners[1]

        # Preprare object points, like: (0, 0, 0), (1, 0, 0), ..., (8, 5, 0)
        # For simplicity we can consider that z = 0
        self.objectPoint = np.zeros((self.cornersX * self.cornersY, 3), np.float32)
        self.objectPoint[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # These fields are the results of the camera calibration
        # We need to store them for future use
        self.intrinsicParameters = []
        self.distortionCoefficients = []

    def calibrateCamera(self) -> bool:

        #  Arrays to store object and image points from all frames
        objectPoints = [] # Object Points : 3D points
        imagePoints = [] # Image Points : 2D points 

        # While the video is open, get frame by frame
        while(self.video.isOpen()):
            # Get current frame
            ret, frame = self.video.getCurrentFrame()

            # Means there is a frame in the buffer
            if ret == True:
                # Convert image from BRG to GRAY
                grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # ... and find chessboard corners from the calibration video
                ret, corners = cv.findChessboardCorners(grayFrame, (self.cornersX, self.cornersY), None)

                if ret == True:
                    # Now found the corners, just store object points and image points
                    objectPoints.append(self.objectPoint)

                    # But before, refine them with specific criteria
                    refinedCorners = cv.cornerSubPix(
                        grayFrame, corners, (11, 11), (-1, -1), self.criteria)

                    # ... and store them
                    imagePoints.append(refinedCorners)

                # Now, jump 2 seconds to next frame in order to get different view of the chessboard
                self.milliseconds += 2000
                self.video.setVideoPosition(self.milliseconds)
                
                self.video.showFrame(frame, debug = self.debug)
                
            else:
                break

        # Release video and destroy windows
        self.video.releaseVideo()
        cv.destroyAllWindows()

        # Now, check if object points (World Space) and image points (Camera plane) are different
        # If, so we need to throw an error since it's not possible to execute the camera calibration method
        if(len(objectPoints) != len(imagePoints)):
            return False

        # After having points, compute camera calibration
        # Parameters:
        # matrix -> 3x3 floating point camera intrinsic matrix (remember that scale is equal to 0 by default)
        # dist -> vector of distortion coefficients
        # rvecs -> vector of rotation vectors
        # tvecs -> vector of translation vectors
        ret, matrix, dist, vecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, grayFrame.shape[::-1], None, None)

        # Lastly, store in the class both instrinsic matrix and distortion coefficients
        self.intrinsicParameters = matrix
        self.distortionCoefficients = dist

        return True

    def getIntrinsicMatrix(self) -> list:
        # Return the 3 x 3 Matrix K containing intrinsic parameters of the camera
        return self.intrinsicParameters

    def getDistortionCoefficients(self) -> list:
        # Return distortion coefficients (5 coefficients) vector
        return self.distortionCoefficients
