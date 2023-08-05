# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QFileDialog, QSpinBox
from PyQt6 import uic

# Other imports
import os

# Custom class imports
from classes.parameters import Parameters

# Import point selection class, to see the methods
from gui.pointSelection import PointSelection

class VideoAnalysis(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super(VideoAnalysis, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/videoAnalysis.ui")
        
        # Load the UI
        uic.loadUi(path, self)
        
        # Hide by default 
        self.hide()
        
        # Instantiate singleton class for parameters
        self.params = Parameters()

        self.defaultSize = 0
            
    def setSpinBoxes(self):
        # Set spin boxes in order to get their value
        self.frameSizeSb = self.findChild(QSpinBox, 'frameSizeInput')
        
        # Get the default value of the spin box
        self.defaultSize = self.frameSizeSb.value()
        
        # Connect to the function when value changed
        self.frameSizeSb.valueChanged.connect(self.setFrameSizeValue)
        
    def setUploadBtns(self, camId):
        if camId == "stCamBtn":
            # Set button to upload static camera calibration video (no light video)
            self.stCamBtn = self.findChild(QPushButton, camId)
            
            self.stCamBtn.clicked.connect(lambda: self.uploadVideos(camId=camId))
            
        if camId == 'mvCamBtn':
            # Set button to upload moving camera calibration video (light video)
            self.mvCamBtn = self.findChild(QPushButton, camId)
            
            self.mvCamBtn.clicked.connect(lambda: self.uploadVideos(camId=camId))
            
    def setStartBtn(self, dstPage: QWidget):
        self.startBtn = self.findChild(QPushButton, 'startBtn')
        self.startBtn.clicked.connect(lambda: self.startPointSelection(dstPage))    
    
    def setFrameSizeValue(self):
        # Store the current value of the frame size
        self.defaultSize = self.frameSizeSb.value()
        
        self.enableStartBtn()  
        
    def uploadVideos(self, camId):
        # Open file dialog, to get Video file path
        dialog = QFileDialog()
        folderPath = dialog.getOpenFileName(None, "Select video file", "", "Video files (*.mov *.mp4)")
        
        # Pass the folder path to parameters class to store it
        self.params.setCamVideoPath(camId, folderPath[0])
        
        self.enableStartBtn()
         
    def startPointSelection(self, dstPage: PointSelection):
        # Before moving to next widget (point selection), store the last information
        self.params.setWorldDefaultSize(self.defaultSize)

        # Then, hide the current widget and show the new one
        self.hide()
    
        # Before showing the point selection page, set the image to choose points
        dstPage.setImage()
        dstPage.show()
       
    def enableStartBtn(self):
        # Check if it's possible to enable the button to start camera calibration
        if self.params.getStCamVideoPath() is not None and self.params.getMvCamVideoPath() is not None and self.defaultSize >= self.frameSizeSb.minimum():
            self.startBtn.setEnabled(True)
            return
            