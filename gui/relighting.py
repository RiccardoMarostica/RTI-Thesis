# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QLabel
from PyQt6.QtGui import QPixmap, QImage, QPaintEvent, QPainter, QPen
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6 import uic

# Other imports
import os
import numpy as np
import torch
from torch import nn
import numpy as np
import math

# Custom class imports
from utils import *
from classes.parameters import Parameters
from classes.pixelEncoder import PixelEncoder

class Relighting(QWidget):
    
    geometryChanged = pyqtSignal(QRect)
    
    def __init__(self, parent: QWidget) -> None:
        super(Relighting, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/relighting.ui")
                
        # Load the UI
        uic.loadUi(path, self)
        
        # Hide by default 
        self.hide()
        
        # Instantiate singleton class for parameters
        self.params = Parameters()
        
        self.points = []
        
        self.resizedX = 0
        self.resizedY = 0
        
    def drawPlot(self, x = None, y = None):
        painter = QPainter(self.pixmapPlot)
        pen = QPen(Qt.GlobalColor.white, 2)
        painter.setPen(pen)
        
        margin = 10
        
        self.center_x, self.center_y = self.pixmapPlot.width() // 2, self.pixmapPlot.height() // 2
        self.radius = min(self.center_x, self.center_y) - margin # Leave some margin
        
        # Draw the circle
        painter.drawEllipse(self.center_x - self.radius, self.center_y - self.radius, self.radius * 2, self.radius * 2)
        
        # Draw axis
        painter.drawLine(self.center_x, margin, self.center_x, self.pixmapPlot.height() - margin)
        painter.drawLine(margin, self.center_y, self.pixmapPlot.width() - margin, self.center_y)
        
        if x is not None and y is not None:
            painter.drawLine(self.center_x, self.center_y, x, y)
            painter.drawEllipse(x, y, 2, 2)

        painter.end()
        self.plotImg.setPixmap(self.pixmapPlot)
    
    def setPlotImage(self):
        # From the params, retrieve the size of the images
        # self.defaultImgSize = self.params.getWorldDefaultSize()
        self.defaultImgSize = 400
        
        # Get the QLabel
        self.plotImg = self.findChild(QLabel, 'relightingPlotLbl')
        self.outputImg = self.findChild(QLabel, 'outputImgLbl')
        
        # Add set mouse tracking
        self.plotImg.setMouseTracking(True)
        
        # Create a value which is used by parent QWidget as resize value of its geometry
        geometryResize = (self.defaultImgSize * 2) + 100
        
        # Emit the signal to change the window size to the dimension of the frame
        self.geometryChanged.emit(QRect(0, 0, geometryResize, (self.defaultImgSize + 100)))
        
        # Init the pixmap with a black background
        self.pixmapPlot = QPixmap(self.defaultImgSize, self.defaultImgSize)
        self.pixmapPlot.fill(Qt.GlobalColor.black)
        self.drawPlot()
        
        # Resize the widget to get the image inside
        self.resize(geometryResize, (self.defaultImgSize + 100))
        
        self.plotImg.mouseMoveEvent = self.relightPlot
        
    def setOutputImage(self):
        # Get base directory to extract the data and show them
        dataDir = 'relights/unive_example_23_07_23_18_10/'
        
        # Retrieve the file paths to extract data
        projPixelsPCAFile = dataDir + 'pca_norm/proj_pixels_pca.npy'
        weights = dataDir + 'models/pixel-weights.pt'
        
        # Get projected pixels
        self.projPixels = np.load(projPixelsPCAFile)
        
        B = np.zeros((2, 10), dtype=np.float32)

        self.model = PixelEncoder(self.projPixels.shape[2] + 2, B)
        self.model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
        self.model.to(getDevice())
        
    def relightPlot(self, event):
        
        # Retrive points (in the resized space)
        x, y = event.pos().x(), event.pos().y()
        
        # Init the pixmap with a black background
        self.pixmapPlot = QPixmap(self.defaultImgSize, self.defaultImgSize)
        self.pixmapPlot.fill(Qt.GlobalColor.black)
        self.drawPlot(x, y)
        
        if math.dist([x,y], [self.center_x, self.center_y]) < self.radius:    
            light = np.array([[normaliseCoordinate(x, self.defaultImgSize), normaliseCoordinate(y, self.defaultImgSize)]])
            # Generate the output images (only one in this case)
            images = predictRelight(self.model, light, self.projPixels)
            
            # Extract the intensity of the image in each pixel, and convert them into integers
            outImg = images[0,:,:].astype(np.uint8)
            
            qImage = QImage(outImg.data, self.defaultImgSize, self.defaultImgSize, self.defaultImgSize, QImage.Format.Format_Grayscale8)
            
            self.outputImg.setPixmap(QPixmap(qImage))
        
        