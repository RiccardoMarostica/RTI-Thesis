from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

import os

from gui.pointSelection import PointSelection

class Homepage(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super(Homepage, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/homepage.ui")
        
        uic.loadUi(path, self)
        
        self.hide()
                
    def setStartBtn(self, dstPage: QWidget):
        # Find the button responsible to start the calibration
        self.start : QPushButton = self.findChild(QPushButton, 'startBtn')
        
        # Now add the event to show the other widget, and hide this one
        self.start.clicked.connect(lambda: self.showCalibration(dstPage))

    def showCalibration(self, dstPage: QWidget):
        self.hide()
        dstPage.show()