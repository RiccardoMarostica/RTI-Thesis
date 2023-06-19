from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

from constants import *

from classes.cameraCalibration import CameraCalibration
from classes.video import Video
from classes.parameters import Parameters
from classes.rtiAlgorithm import RTI
from classes.videoSynchronisation import VideoSynchronisation

class MainWindow (QWidget):
        
    def __init__(self) -> None:
        # Call the init of the father class (QWidget, which is the window)
        super().__init__()
                
        self.parameters = Parameters()
        
        self.rti = RTI()
        
        # Then, initalise the window
        self.initialiseWindow()
        
    def initialiseWindow(self):
        # Set window size using the dimension given by constants
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        self.setWindowTitle('Reflectance Transformation Imaging')
        
        # Layout to switch widgets
        self.layoutWidgets = QStackedLayout()
        
        self.homepage = self.initialiseHomepage()
        # self.help = self.initaliseHelp()
        self.calibration = self.initialiseCalibration()
        self.videoAnalysis = self.initaliseVideoAnalysis()
        
        self.layoutWidgets.addWidget(self.homepage)
        self.layoutWidgets.addWidget(self.calibration)
        self.layoutWidgets.addWidget(self.videoAnalysis)
        # self.layoutWidgets.addWidget(self.help)
        
        self.setLayout(self.layoutWidgets)
        
        self.showHomepage()
        
        # Show window since by default windows are hidden
        self.show()
    
    def initialiseHomepage(self) -> QWidget:
        # Create layout widget
        homepage = QWidget()
        
        # Create vertical box layout instance
        layout = QVBoxLayout()
        
        # Set RTI Start button
        start = QPushButton('Begin with RTI')
        start.clicked.connect(self.showCalibration)

        # # Set Help button
        # help = QPushButton('Help & Information')
        # help.clicked.connect(self.showHelp)
        
        # Add everything to the layout
        layout.addWidget(start)
        # layout.addWidget(help)
        
        homepage.setLayout(layout)
        
        return homepage
        
    def initaliseHelp(self) -> QWidget:
        window = QWidget()
        
        label = QLabel(window)
        label.setText("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\nUt enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\nDuis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.\nExcepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum")
        
        fontText = QFont()
        fontText.setPointSize(16)
        
        label.setFont(fontText)
        label.setWordWrap(True)  # Enable word wrapping for long text

        # Adjust the size of the window to fit the QLabel content
        label.adjustSize()
                
        return window
    
    def initialiseCalibration(self):
        # Window widget
        window = QWidget()
        # Set window size as the parent
        window.setGeometry(self.geometry())
        
        # Create the master vertical box
        # Inside here add all the components of the window
        masterVBoxLayout = QVBoxLayout()
        
        # Static Camera horizontal box
        stcmHBoxLayout = QHBoxLayout()
        # Description label
        stcmLabel = QLabel()
        stcmLabel.setText('Static Camera calibration video')
        # Input button
        stcmUploadBtn = QPushButton('Load video')
        stcmUploadBtn.clicked.connect(lambda:self.getFile('calibration-static'))
        # Add widgets to layout
        stcmHBoxLayout.addWidget(stcmLabel)
        # stcmHBoxLayout.addStretch()
        stcmHBoxLayout.addWidget(stcmUploadBtn)
        
        # Static Camera horizontal box
        mvcmHBoxLayout = QHBoxLayout()
        # Description label
        mvcmLabel = QLabel()
        mvcmLabel.setText('Moving Camera calibration video')
        # Input button
        mvcmUploadBtn = QPushButton('Load video')
        mvcmUploadBtn.clicked.connect(lambda:self.getFile('calibration-moving'))
        # Add widgets to layout
        mvcmHBoxLayout.addWidget(mvcmLabel)
        # mvcmHBoxLayout.addStretch()
        mvcmHBoxLayout.addWidget(mvcmUploadBtn)
        
        # Now, add confirm button to start with the calibration
        confirmBtn = QPushButton('Start camera calibration')
        confirmBtn.clicked.connect(self.startCalibrationCamera)
        
        # Lastly, add layouts to master layout
        masterVBoxLayout.addLayout(stcmHBoxLayout)
        masterVBoxLayout.addStretch()
        masterVBoxLayout.addLayout(mvcmHBoxLayout)
        masterVBoxLayout.addStretch()
        masterVBoxLayout.addWidget(confirmBtn)
        
        # Use the layout on the widget
        window.setLayout(masterVBoxLayout)
        
        # ... and return it
        return window
    
    def initaliseVideoAnalysis(self):
        # Window widget
        window = QWidget()
        # Set window size as the parent
        window.setGeometry(self.geometry())
        
        # Create the master vertical box
        # Inside here add all the components of the window
        masterVBoxLayout = QVBoxLayout()
        
        # Static Camera horizontal box
        stcmHBoxLayout = QHBoxLayout()
        # Description label
        stcmLabel = QLabel()
        stcmLabel.setText('Static Camera Video')
        # Input button
        stcmUploadBtn = QPushButton('Load video')
        stcmUploadBtn.clicked.connect(lambda:self.getFile('video-static'))
        # Add widgets to layout
        stcmHBoxLayout.addWidget(stcmLabel)
        # stcmHBoxLayout.addStretch()
        stcmHBoxLayout.addWidget(stcmUploadBtn)
        
        # Static Camera horizontal box
        mvcmHBoxLayout = QHBoxLayout()
        # Description label
        mvcmLabel = QLabel()
        mvcmLabel.setText('Moving Camera Video')
        # Input button
        mvcmUploadBtn = QPushButton('Load video')
        mvcmUploadBtn.clicked.connect(lambda:self.getFile('video-moving'))
        # Add widgets to layout
        mvcmHBoxLayout.addWidget(mvcmLabel)
        # mvcmHBoxLayout.addStretch()
        mvcmHBoxLayout.addWidget(mvcmUploadBtn)
        
        # Now, add confirm button to start with the calibration
        confirmBtn = QPushButton('Start Video Analysis')
        confirmBtn.clicked.connect(self.startCalibrationCamera)
        
        # Lastly, add layouts to master layout
        masterVBoxLayout.addLayout(stcmHBoxLayout)
        masterVBoxLayout.addStretch()
        masterVBoxLayout.addLayout(mvcmHBoxLayout)
        masterVBoxLayout.addStretch()
        masterVBoxLayout.addWidget(confirmBtn)
        
        # Use the layout on the widget
        window.setLayout(masterVBoxLayout)
        
        # ... and return it
        return window  
    
    def showHomepage(self):
        # Show page of interest
        self.homepage.show()
        # ... and hide all the others
        self.videoAnalysis.hide()
        self.calibration.hide()
        # self.help.hide()
        pass
    
    def showCalibration(self):
        # Show page of interest
        self.calibration.show()
        # ... and hide all the others
        self.videoAnalysis.hide()
        self.homepage.hide()
        # self.help.hide()
        pass
    
    def showVideoAnalysis(self):
        # Show page of interest
        self.videoAnalysis.show()
        # ... and hide all the others
        self.calibration.hide()
        self.homepage.hide()
        pass
    
    def showHelp(self):
        self.help.show()
        self.homepage.hide()
        pass
    
    def startCalibrationCamera(self):
        
        print("Starting camera calibration...")
        
        # Get the two calibrations for both static and moving camera
        self.calibrationStatic = CameraCalibration(Video(self.parameters.getStaticCameraCalibrationPath()), (9, 6)).calibrateCamera()
        self.calibrationMoving = CameraCalibration(Video(self.parameters.getMovingCameraCalibrationPath()), (9, 6)).calibrateCamera()
    
        # Calibrate cameras and check result
        if (self.calibrationStatic == False or self.calibrationMoving == False):
            print("Error when calibrating one of the two cameras. ")
            exit(-1)
        else:
            print("Camera calibration completed without errors")
        
        # TODO: Show next page (Video Upload) 
        self.showVideoAnalysis()   
        pass
    
    def startVideoAnalysis(self):
        
        print("Starting video analysis...")
        
        videoStaticPath = self.parameters.getStaticCameraAnalysisPath()
        videoMovingPath = self.parameters.getMovingCameraAnalysisPath()
        
        # Create the two videos
        videoStatic = Video(videoStaticPath)
        videoMoving = Video(videoMovingPath)
            
        # TODO: Only for testing purpose
        defaultK = self.rti.getDefaultK(videoStatic)
        
        print("First: Perform Video Synchonisation...")
            
        # Create class to synch the videos
        videoSynchronisation = VideoSynchronisation(videoStaticPath, videoMovingPath)
        
        # ... and them synch them
        videoSynchronisation.synchroniseVideo()
        
        print("Video Synchonisation done without errors")
        
        # After synchronisation, get the offset between the two videos
        # First get the default FPS
        defaultFps = max(videoStatic.getFPS(), videoMoving.getFPS())
        
        # ... and then compute the shift between the videos
        frameDifference = videoSynchronisation.getFrameDifference(defaultFps)
    
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
        
        print("Video analysis completed without errors")
        
        pass
    
    def getFile(self, type: str):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        
        if type == 'calibration-static':
            self.parameters.setStaticCameraCalibrationPath(fname[0])
        elif type == 'calibration-moving':
            self.parameters.setMovingCameraCalibrationPath(fname[0])
        elif type == 'video-static':
            self.parameters.setStaticCameraAnalysisPath(fname[0])
        elif type == 'video-moving':
            self.parameters.setMovingCameraAnalysisPath(fname[0])
                        
        pass