from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from constants import *

from gui.homepage import Homepage
from gui.calibration import Calibration
from gui.videoAnalysis import VideoAnalysis
from gui.pointSelection import PointSelection

class MainWindow (QMainWindow):
        
    def __init__(self) -> None:
        # Call the init of the father class (QWidget, which is the window)
        super(MainWindow, self).__init__()
        
        # Then, initalise the window
        self.initialiseWindow()
        
    def initialiseWindow(self):
        # Set window size using the dimension given by constants
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        self.setWindowTitle('Reflectance Transformation Imaging')
        
        # Initalise the pages
        self.homepage = self.initHomepage()
        self.calibration = self.initCalibration()
        self.videoAnalysis = self.initVideoAnalysis()
        self.pointSelection = self.initPointSelection()
        
        # Then, set all their informations
        self.setHomepage()
        self.setCalibration()
        self.setVideoAnalysis()
        self.setPointSelection()
        
        self.homepage.show()
            
        # Show window since by default windows are hidden
        self.show()
        
    def initHomepage(self):
        return Homepage(self)
    
    def initCalibration(self):
        return Calibration(self)
    
    def initVideoAnalysis(self):
        return VideoAnalysis(self)
    
    def initPointSelection(self):
        return PointSelection(self)
    
    def setHomepage(self):
        # Set start btn to open the calibration page
        self.homepage.setStartBtn(self.videoAnalysis)
        
    def setCalibration(self):
        # Set up both the upload buttons
        self.calibration.setSpinBoxes()
        self.calibration.setUploadBtns("stCamBtn")
        self.calibration.setUploadBtns("mvCamBtn")
        self.calibration.setCalibrationBtn(self.videoAnalysis)
        
    def setVideoAnalysis(self):
        self.videoAnalysis.setSpinBoxes()
        self.videoAnalysis.setUploadBtns("stCamBtn")
        self.videoAnalysis.setUploadBtns("mvCamBtn")
        self.videoAnalysis.setStartBtn(self.pointSelection)
        
    def setPointSelection(self):
        self.pointSelection.geometryChanged.connect(self.handleGeometry)
        self.pointSelection.setStartBtn()
        
    def handleGeometry(self, geometry):
        self.setGeometry(geometry)