# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QFileDialog, QSpinBox, QLabel
from PyQt6 import uic

# Other imports
import os

# Custom class imports
from classes.parameters import Parameters
from classes.video import Video

class PointSelection(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super(PointSelection, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/framePointSelection.ui")
        
        print(path)
        
        # Load the UI
        uic.loadUi(path, self)
        
        # Hide by default 
        self.hide()
        
        # Instantiate singleton class for parameters
        self.params = Parameters()
        
        print("Calling")
        
        self.setPointSelection()
        
    def setPointSelection(self):
        print("Setting")
        # self.label = self.findChild(QLabel, 'showFrame')
        # self.label.setText("Prova prova")