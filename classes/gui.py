from PyQt6.QtWidgets import *
from constants import *

class MainWindow (QWidget):
    
    frontWidget = None
    
    def __init__(self) -> None:
        # Call the init of the father class (QWidget, which is the window)
        super().__init__()
                
        # Then, initalise the window
        self.initialiseWindow()
        
    def initialiseWindow(self):
        # Set window size using the dimension given by constants
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        self.setWindowTitle('Reflectance Transformation Imaging')
        
        # Layout to switch widgets
        self.layoutWidgets = QStackedLayout()
        
        # Button to switch between widgets
        self.swithButton = QPushButton("Switch window")
        self.swithButton.clicked.connect(self.switchWidgets)
        self.swithButton.setFixedSize(100, 100)
        
        # Define widgets
        self.wid1 = QWidget()
        self.wid1.setStyleSheet("""background: blue;""")
        self.wid1.setFixedSize(200,200)
        self.wid1.move(100, 100)
        self.wid2 = QWidget()
        self.wid2.setStyleSheet("""background: green;""")
        self.wid2.setFixedSize(200, 200)
        self.wid2.move(100, 100)

        self.layoutWidgets.addWidget(self.swithButton)
        self.layoutWidgets.addWidget(self.wid1)
        self.layoutWidgets.addWidget(self.wid2)
        
        self.setLayout(self.layoutWidgets)
        
        self.frontWidget = 1
        
        # Show window since by default windows are hidden
        self.show()
        
    def switchWidgets(self):
        if self.frontWidget == 1:
            self.wid1.hide()
            self.wid2.show()
            self.frontWidget = 2
        else:
            self.wid1.show()
            self.wid2.hide()
            self.frontWidget = 1
    
    def showHomepage(self):
        self.homepage = QWidget()
        pass
    
    def showHelp(self):
        pass    
    
    def showCalibration(self):
        pass
    
    def showWaiting(self):
        pass
    
    def showVideo(self):
        pass
    
    def showResult(self):
        pass