# Import default modules
import cv2 as cv
import numpy as np

# Import classes
from classes.cameraCalibration import CameraCalibration
from classes.video import Video


def main():
    
    # Get the two calibrations for both static and moving camera
    calibrationStatic = CameraCalibration(Video("./video/cam-static/calibration.mp4"), 9, 6)
    calibrationMoving = CameraCalibration(Video("./video/cam-moving/calibration.mp4"), 9, 6)
    
    # Calibrate cameras and check result
    if (calibrationStatic.calibrateCamera() == False or calibrationMoving.calibrateCamera() == False):
        print("Error when calibrating one of the two cameras. ")
        exit(-1)
    
    print("K for Static Camera: \n", calibrationStatic.getIntrinsicMatrix())
    print("K for Moving Camera: \n", calibrationMoving.getIntrinsicMatrix())
    
if __name__ =="__main__":
    main()