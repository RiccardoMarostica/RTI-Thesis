import cv2 as cv
import numpy as np

from utils import getVideoOrientation
from classes.video import Video


class CameraCalibration:
    """This class is used to perform Camera Calibration, in order to obtain an estimate of the Intrinsic Parameters Matrix K (3 x 3 Matrix) and the distortion coefficients (5 parameters distortion coefficients).\n
    In the constructor of the class, it's necessary to get a Video class instance, used to initialise the calibration video and the number of corners in the Calibration target.\n
    At the end of the Camera Calibration it's possible to retrieve both Intrinsic Parameters Matrix and the distortion coefficients.
    """

    # This boolean value permits to enable/disable debug features, like show images, etc...
    debug = False

    # Termination criteria used for method cornerSubPix, which is used to refine the corner locations.
    # The process stops either after the max iterations is reached or when the currentposition moves by less than the epsilon value
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Variable used to move forward to the next frame in the Video instance every [x] seconds and grab a new frame.
    milliseconds = 0

    def __init__(self, video : Video, corners : tuple, debug = False):
        """The constructor takes in input a Video class, containing a Video instance from OpenCv and a tuple indicating the number of corners in the Calibration target (A chessboard, etc...)

        Args:
            video (Video): Video Class instance, containing the Video for Camera Calibration
            corners (tuple): Tuple indicating the number of corners in the Calibration target.
            debug (bool): Enable/Disable debug mode. Default to False.
        """
        # Set debug mode
        self.debug = debug
        
        # Store Video instance in the class field
        self.video: Video = video

        # ... and also the number of corners in the Calibration target (Chessboard)
        # The tuple (cornersX, cornersY) denotes the pattern of the Calibration target
        self.cornersX = corners[0]
        self.cornersY = corners[1]

        # Preprare object points, like: (0, 0, 0), (1, 0, 0), ..., (cornersX, cornersY, 0)
        # This denotes the location of the points (corners)
        # For simplicity, consider that z = 0, meaning the chessboard was kept stationary at XY plane
        self.objectPoint = np.zeros((self.cornersX * self.cornersY, 3), np.float32)
        self.objectPoint[:, :2] = np.mgrid[0:self.cornersX, 0:self.cornersY].T.reshape(-1, 2)

        # These fields are the results of the camera calibration
        self.intrinsicParameters = []
        self.distortionCoefficients = []

    def storeCalibrationDataFromFiles(self, matrix, dist, debug = False):
        self.setIntrinsicMatrix(matrix)
        self.setDistortionCoefficients(dist)

    def calibrateCamera(self, size = None) -> bool:
        """The function performs camera calibration, using the Video instance passed in the constructor of the class.
        The function loops over the Calibration video (in which the Calibration target is kept stationary at XY plane), detect the corners in the Calibration target, and store them inside the Image Points list, which denotes 2D Image Points.

        Returns:
            bool: True if the Calibration is done without errors (K and dist coefficients are retrieved successfully). False, otherwise.
        """

        objectPoints = [] # Object Points <--> 3D points from real world space
        imagePoints = []  # Image Points  <--> 2D points 

        while(self.video.isOpen()):
            # Grab a frame
            ret, frame = self.video.getCurrentFrame()

            # Means there is a frame in the buffer
            if ret == True:
                
                if size is not None:
                    # Based on the size in input, resize the frame
                    frame = cv.resize(frame, size)
                    
                # Convert image from BRG to GRAY
                grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Find chessboard corners from the calibration video
                # The corners are placed in the order: left-to-right, top-to-bottom
                ret, corners = cv.findChessboardCorners(grayFrame, (self.cornersX, self.cornersY), None)

                # Means the pattern is obtained from the frame
                if ret == True:
                    
                    # Store object points (3D points), using for simplicity the points instantiated at the beginning
                    objectPoints.append(self.objectPoint)

                    # Refine corners location before store them
                    refinedCorners = cv.cornerSubPix(grayFrame, corners, (11, 11), (-1, -1), self.criteria)

                    # Store the refined corners (2D points)
                    imagePoints.append(refinedCorners)
                    
                    if (self.debug == True  and ret == True):
                        # If we are in debugging and there is a pattern, draw chessboard corners in the image
                        cv.drawChessboardCorners(frame, (self.cornersX, self.cornersY), refinedCorners, ret)

                # Jump 2 seconds ahead, to next frame, to get different view of the Calibration target
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

        # After retrieving the 3D <--> 2D correspondences, it's possible to calibrate the camera
        # Parameters:
        # matrix -> 3x3 floating point camera intrinsic parameters matrix (remember that scale is equal to 0 by default)
        # dist -> vector of distortion coefficients
        ret, matrix, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, grayFrame.shape[::-1], None, None)

        # Before returning, calculate the re-projection error, which is a good estimation of just how exact the found parameters are.
        # The closer the error is to zero, the more accurate the parameters found are.
        self.meanError = 0
        for i in range(len(objectPoints)):
            imagePoints2, _ = cv.projectPoints(objectPoints[i], rvecs[i], tvecs[i], matrix, dist)
            error = cv.norm(imagePoints[i], imagePoints2, cv.NORM_L2) / len(imagePoints2)
            self.meanError += error

        # Store in the class field both instrinsic matrix and distortion coefficients
        self.setIntrinsicMatrix(matrix)
        self.setDistortionCoefficients(dist)

        # Camera calibration done without errors
        return True
    
    def setIntrinsicMatrix(self, matrix):
        self.intrinsicParameters = matrix
    
    def setDistortionCoefficients(self, dist):
        self.distortionCoefficients = dist

    def getIntrinsicMatrix(self) -> list:
        """The function retuns the Camera Intrinsic Parameters Matrix (K - 3 x 3 Matrix).\n
        The Matrix includes the focal lenght (fx, fy) and the optical centers (cx, cy).
        
        Returns:
            list: A 3x3 Matrix denoting the Camera Intrinsic Parameters.
        """
        return self.intrinsicParameters

    def getDistortionCoefficients(self) -> list:
        """The function returns the 5 parameters distortion coefficients (k1, k2, k3, p1, p2).
        These parameters are used to do image undistortion.

        Returns:
            list: A 5 parameters list used to perform iamge undistortion.
        """
        return self.distortionCoefficients

    def getReProjectionError(self) -> float:
        """The function returns the re-projection error, which is a good estimation of just how exact the found parameters are.
        The closer the error is to zero, the more accurate the parameters found are.

        Returns:
            float: Re-projection error value
        """
        return self.meanError