# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QScrollArea, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6 import uic

# Other imports
import os
from datetime import datetime

# Custom class imports
from classes.parameters import Parameters
from gui.relighting import Relighting

class RelightingHistory(QWidget):
    
    geometryChanged = pyqtSignal(QRect)
    
    def __init__(self, parent: QWidget) -> None:
        super(RelightingHistory, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/relighting-history.ui")
                     
        self.params = Parameters()
           
        # Load the UI
        uic.loadUi(path, self)
        
        # Hide by default 
        self.hide()
        
    def setScrollArea(self, dstPage: Relighting):
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
        subFolders.sort(reverse = True)
        
        for i in range(len(subFolders)):
            folderName = subFolders[i]
            self.addRelightingHistory(i, folderName, dstPage)

    def addRelightingHistory(self,idx, folderName, dstPage: Relighting):

        # Create widget for the relighting, where we add:
        # 1 - Title: Relighting - [REL_NUMBER_INC]
        # 2 - Date of calculation
        # 3 - Button to access the relighting page and apply relighting
        
        # Create the widget 
        relWidget = QWidget()
        relWidget.setFixedHeight(120)
        # And add with it a QHBoxLayout (meaning that we place objects in columns)
        hLayout = QHBoxLayout(relWidget)
        # Then we need to add 2 components:
        # 1 - A vertical Box containing textual information (Title and Date)
        # 2 - A button accessing to the relighting example
        vLayout = QVBoxLayout()
        title = QLabel(text= f"Relighting N° {idx}")
        
        # Set title font
        titleFont = QFont()
        titleFont.setBold(True)
        titleFont.setPointSize(24)
        title.setFont(titleFont)
        
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
        dateLbl = QLabel(text=f"Computed: {formattedDate.date()} at: {formattedDate.time()}")
        
        vLayout.addWidget(title)
        vLayout.addWidget(dateLbl)
        
        openBtn = QPushButton("Open relighting")
        openBtn.setFixedHeight(50)
        openBtn.clicked.connect(lambda: self.setButtonPath(folderName, dstPage))
        
        openBtnFont = QFont()
        openBtnFont.setPointSize(18)
        openBtn.setFont(openBtnFont)
                
        vLayout.addWidget(openBtn) 

        # Add elements to horizontal Layout
        hLayout.addLayout(vLayout)

        # Add the HorizontalLayout to the scroll content
        self.scrollContent.layout().addWidget(relWidget)
        
    def setButtonPath(self, folderName, dstPage: Relighting):
        path = f"relights/{folderName}/"
        self.params.setRelightingBasePath(path)
    
        # Then, hide the current widget and show the new one
        self.hide()
        
        # Prepare dst page
        dstPage.setPlotImage()
        dstPage.setOutputImage()
        dstPage.show()