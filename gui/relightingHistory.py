# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QScrollArea, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6 import uic

# Other imports
import os
from datetime import datetime

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
        self.scrollArea.setWidgetResizable(True)

        # Add scroll content
        self.scrollContent = QWidget()
        self.scrollContent.setLayout(QVBoxLayout(self))
        self.scrollArea.setWidget(self.scrollContent)
        
        # Fetch the list of folders in a specific directory
        currentDir = os.path.dirname(os.path.abspath(__file__))
        parentDir = os.path.dirname(currentDir)
        relightDir = os.path.join(parentDir, 'relights')
                
        subFolders = [d for d in os.listdir(relightDir) if os.path.isdir(os.path.join(relightDir, d))]
        subFolders.sort()
        
        for i in range(len(subFolders)):
            folderName = subFolders[i]
            self.addRelightingHistory(i, folderName)

    def addRelightingHistory(self,idx, folderName):

        # Create widget for the relighting, where we add:
        # 1 - Title: Relighting - [REL_NUMBER_INC]
        # 2 - Date of calculation
        # 3 - Button to access the relighting page and apply relighting
        
        # Create the widget 
        relWidget = QWidget()
        relWidget.setFixedHeight(100)
        # And add with it a QHBoxLayout (meaning that we place objects in columns)
        hLayout = QHBoxLayout(relWidget)
        # Then we need to add 2 components:
        # 1 - A vertical Box containing textual information (Title and Date)
        # 2 - A button accessing to the relighting example
        vLayout = QVBoxLayout()
        title = QLabel(text= f"Relighting N° {idx}", parent=relWidget)
        # Get date
        date = folderName.split("-")[1]
        dateList = date.split("_")
        formattedDate = datetime(
            int(dateList[0]), 
            int(dateList[1]), 
            int(dateList[2]),
            int(dateList[3]),
            int(dateList[4])
        )
        # Add widgets or buttons for each folder
        openBtn = QPushButton("Open")  
        dateLbl = QLabel(text=f"Computed: {formattedDate.date()} at: {formattedDate.time()}")
        vLayout.addWidget(title)
        vLayout.addWidget(dateLbl)     
        vLayout.addWidget(openBtn) 

        # Add elements to horizontal Layout
        hLayout.addLayout(vLayout)

        # Add the HorizontalLayout to the scroll content
        self.scrollContent.layout().addWidget(relWidget)