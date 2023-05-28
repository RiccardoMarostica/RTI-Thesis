import cv2 as cv
import numpy as np
import video 

class CameraCalibration:
    # Construction, just take the video from the main
    def __init(self, video):
        # Only field used to calibrate the camera
        self.video = video
        
        # These fields are the results of the camera calibration
        # We need to store them for future use
        self.intrinsicParameters = []
        self.distortionCoefficients = []
