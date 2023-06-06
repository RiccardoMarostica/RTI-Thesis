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
        self.help = self.initaliseHelp()
        
        self.layoutWidgets.addWidget(self.homepage)
        self.layoutWidgets.addWidget(self.help)
        
        # # Button to switch between widgets
        # self.swithButton = QPushButton("Switch window")
        # self.swithButton.clicked.connect(self.switchWidgets)
        # self.swithButton.setFixedSize(100, 100)

        # self.layoutWidgets.addWidget(self.swithButton)
        # self.layoutWidgets.addWidget(self.wid1)
        # self.layoutWidgets.addWidget(self.wid2)
        
        self.setLayout(self.layoutWidgets)
        
        # self.frontWidget = 1
        
        self.showHomepage()
        
        # Show window since by default windows are hidden
        self.show()
    
    def initialiseHomepage(self) -> QWidget:
        # Create layout widget
        homepage = QWidget()
        # homepage.setGeometry(0, 0, WINDOW_WIDTH - 10, WINDOW_HEIGHT - 10)
        
        # Create vertical box layout instance
        layout = QVBoxLayout()
        
        # Set RTI Start button
        start = QPushButton('Begin with RTI')
        start.clicked.connect(self.showCalibration)

        # Set Help button
        help = QPushButton('Help & Information')
        help.clicked.connect(self.showHelp)
        
        # Add everything to the layout
        layout.addWidget(start)
        layout.addWidget(help)
        
        homepage.setLayout(layout)
        
        return homepage
        
    
    def showHomepage(self):
        self.homepage.show()
        self.help.hide()
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
    
    def showCalibration(self):
        print("Calibration clicked")
        pass
    
    def showWaiting(self):
        pass
    
    def showVideo(self):
        pass
    
    def showResult(self):
        pass