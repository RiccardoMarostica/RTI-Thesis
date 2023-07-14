from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from constants import *

from gui.classes.homepage import Homepage
from gui.classes.calibration import Calibration

class MainWindow (QMainWindow):
        
    def __init__(self) -> None:
        # Call the init of the father class (QWidget, which is the window)
        super().__init__()
        
        # Then, initalise the window
        self.initialiseWindow()
        
    def initialiseWindow(self):
        # Set window size using the dimension given by constants
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        self.setWindowTitle('Reflectance Transformation Imaging')
        
        # Initalise the pages
        self.homepage = self.initHomepage()
        self.calibration = self.initCalibration()
        
        # Then, set all their informations
        self.setHomePageFeatures()
        
        self.homepage.show()
            
        # Show window since by default windows are hidden
        self.show()
        
    def initHomepage(self):
        return Homepage(self)
    
    def initCalibration(self):
        return Calibration(self)
    
    def setHomePageFeatures(self):
        # Set start btn to open the calibration page
        self.homepage.setStartBtn(self.calibration)