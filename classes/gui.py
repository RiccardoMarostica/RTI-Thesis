from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

from constants import *

class MainWindow (QWidget):
        
    def __init__(self) -> None:
        # Call the init of the father class (QWidget, which is the window)
        super().__init__()
                
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
        
        self.layoutWidgets.addWidget(self.homepage)
        self.layoutWidgets.addWidget(self.calibration)
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
        
    
    def showHomepage(self):
        self.homepage.show()
        # self.help.hide()
        pass
    
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
    
    def showHelp(self):
        self.help.show()
        self.homepage.hide()
        pass
    
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
        # Add widgets to layout
        mvcmHBoxLayout.addWidget(mvcmLabel)
        # mvcmHBoxLayout.addStretch()
        mvcmHBoxLayout.addWidget(mvcmUploadBtn)
        
        # Lastly, add layouts to master layout
        masterVBoxLayout.addLayout(stcmHBoxLayout)
        masterVBoxLayout.addStretch()
        masterVBoxLayout.addLayout(mvcmHBoxLayout)
        masterVBoxLayout.addStretch()
        
        # Use the layout on the widget
        window.setLayout(masterVBoxLayout)
        
        # ... and return it
        return window
    
    def showCalibration(self):
        self.calibration.show()
        self.homepage.hide()
        pass
    
    def showWaiting(self):
        pass
    
    def showVideo(self):
        pass
    
    def showResult(self):
        pass