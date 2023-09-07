# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QScrollArea, QHBoxLayout
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6 import uic

# Other imports
import os

# Custom class imports
from classes.parameters import Parameters

class RelightingHistory(QWidget):
    
    geometryChanged = pyqtSignal(QRect)
    
    def __init__(self, parent: QWidget) -> None:
        super(RelightingHistory, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/relighting-history.ui")
                        
        # Load the UI
        uic.loadUi(path, self)
        
        # Hide by default 
        self.hide()
        
    def setScrollArea(self):
        # Create scroll area
        self.scrollArea : QScrollArea = self.findChild(QScrollArea, 'scrollArea')
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Add scroll content
        self.scrollContent = QWidget()
        self.scrollArea.setWidget(self.scrollContent)
        
        # Fetch the list of folders in a specific directory
        folder_path = os.path.dirname("../relights")
        folder_list = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        for folder_name in folder_list:
            self.addRelightingHistory(folder_name)

    def addRelightingHistory(self, folder_name):
        # Create a HorizontalLayout for each folder
        folder_layout = QHBoxLayout()

        # Add widgets or buttons for each folder
        folder_button = QPushButton(folder_name)
        folder_layout.addWidget(folder_button)

        # Add the HorizontalLayout to the scroll content
        self.scrollContent.layout().addWidget(folder_layout)