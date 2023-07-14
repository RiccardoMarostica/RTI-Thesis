from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

import os

class Calibration(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super(Calibration, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/cameraCalibration.ui")
        
        # Load the UI
        uic.loadUi(path, self)
        
        self.hide()
    
    def showWidget(self):
        self.show()
        
    def hideWidget(self):
        self.hide()