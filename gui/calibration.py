# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QFileDialog, QSpinBox
from PyQt6 import uic

# Other imports
import os, numpy as np
from datetime import datetime

from utils import getVideoOrientation, videoNeedsResize

# Custom class imports
from classes.parameters import Parameters
from classes.cameraCalibration import CameraCalibration
from classes.video import Video

class Calibration(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super(Calibration, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/cameraCalibration.ui")
        
        # Load the UI
        uic.loadUi(path, self)
        
        # Hide by default 
        self.hide()
        
        # Instantiate singleton class for parameters
        self.params = Parameters()

        # Instantiate the number of corners for the calibration
        self.cornersX = 0
        self.cornersY = 0
        
    def setSpinBoxes(self):

        # Set spin boxes in order to get their value
        self.xAxisSb = self.findChild(QSpinBox, 'xAxisInput')
        self.yAxisSb = self.findChild(QSpinBox, 'yAxisInput')
        
        # Connect to the function when value changed
        self.xAxisSb.valueChanged.connect(lambda: self.setAxisValue('X'))
        self.yAxisSb.valueChanged.connect(lambda: self.setAxisValue('Y'))
        
        # Check if it's possible to start calibration
        self.enableStartCalibBtn()
        
    def setUploadBtns(self, camId):
        if camId == "stCamBtn":
            # Set button to upload static camera calibration video (no light video)
            self.stCamBtn = self.findChild(QPushButton, camId)
            
            self.stCamBtn.clicked.connect(lambda: self.uploadVideos(camId=camId))
            
        if camId == 'mvCamBtn':
            # Set button to upload moving camera calibration video (light video)
            self.mvCamBtn = self.findChild(QPushButton, camId)
            
            self.mvCamBtn.clicked.connect(lambda: self.uploadVideos(camId=camId))
    
    def setAxisValue(self, axis):
        # Based on the axis, store the current value of the Spin Box
        if axis == "X":
            self.cornersX = self.xAxisSb.value()
        if axis == "Y":
            self.cornersY = self.yAxisSb.value()  
                
    def setCalibrationBtn(self, dstPage: QWidget):
        # Set button to start with camera calibration
        self.calibBtn = self.findChild(QPushButton, 'startBtn')
        self.calibBtn.clicked.connect(lambda: self.startCalibration(dstPage))
        
    def uploadVideos(self, camId):
        # Open file dialog, to get Video file path
        dialog = QFileDialog()
        folderPath = dialog.getOpenFileName(None, "Select Calibration video file", "", "Video files (*.mov *.mp4)")
        
        # Pass the folder path to parameters class to store it
        self.params.setCamCalibPath(camId, folderPath[0])
        
        # Check if it's possible to start calibration
        self.enableStartCalibBtn()
        
        
    def startCalibration(self, dstPage: QWidget):
        
        # Initalise the video, passing the paths to open the video with OpenCV
        stCamVideo = Video(self.params.getStCamCalibPath())
        mvCamVideo = Video(self.params.getMvCamCalibPath())
        
        # Get the orientation of both videos
        stCamOrientation = getVideoOrientation(stCamVideo)
        mvCamOrientation = getVideoOrientation(mvCamVideo)
        
        # Set defulat value for static and moving camera size.
        # None is used so when processing, no operations are necessary
        stCamSize = mvCamSize = None
        
        if videoNeedsResize(stCamVideo, self.params.getFrameDefaultSize(stCamOrientation)):
            # Check if static camera size needs to be resized. And if so, retrieve the default dimension
            stCamSize = self.params.getFrameDefaultSize(stCamOrientation)
            
        if videoNeedsResize(mvCamVideo, self.params.getFrameDefaultSize(mvCamOrientation)):
            # Check if moving camera size needs to be resized. And if so, retrieve the default dimension
            mvCamSize = self.params.getFrameDefaultSize(mvCamOrientation)
        
        print("Starting with calibration...")
        
        corners = (self.cornersX, self.cornersY)
                
        # Then, start with calibration
        stCamCalibration = CameraCalibration(stCamVideo, corners)
        mvCamCalibration = CameraCalibration(mvCamVideo, corners)
        
        if not stCamCalibration.calibrateCamera(stCamSize) or not mvCamCalibration.calibrateCamera(mvCamSize):
            # Something went wrong in one of the two cameras
            print("Something went wrong when calibrating the two cameras. ")
            exit(-1)
        
        print("Camera calibration completed. ")
        
        # Otherwise, store the two instances inside the parameter class
        self.params.setCamsCalibData(stCamCalibration, mvCamCalibration)
        
        # Get current time in format: YY_MM_DD_hh_mm
        now = datetime.now().strftime("%y_%m_%d_%H_%M_%s")
        path = 'calibrations/calibration-%s/'%now
        
        if not os.path.exists(path):
            # If the path not exsists, create new
            os.mkdir(path)
            
            # Then store the two interesting informations (K and dist) for both cameras
            np.savetxt(path + "st_cam_intrinsic.dat", stCamCalibration.getIntrinsicMatrix())
            np.savetxt(path + "st_cam_distCoeff.dat", stCamCalibration.getDistortionCoefficients())
            np.savetxt(path + "mv_cam_intrinsic.dat", mvCamCalibration.getIntrinsicMatrix())
            np.savetxt(path + "mv_cam_distCoeff.dat", mvCamCalibration.getDistortionCoefficients())

        # Then, hide the current widget and show the new one
        self.hide()
        dstPage.show()
    
    def enableStartCalibBtn(self):
        # Check if it's possible to enable the button to start camera calibration
        if self.params.getStCamCalibPath() is not None and self.params.getMvCamCalibPath() is not None and self.cornersX > 0 and self.cornersY > 0:
            # Enable the start calibration button, in case both paths are set and the corners are positive
            self.calibBtn.setEnabled(True)
            return