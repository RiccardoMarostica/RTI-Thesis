# PyQt imports
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6 import uic

# Other imports
import os
import numpy as np
import torch
import numpy as np
import math
from datetime import datetime

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
        
        self.isMoving = True
        
    def drawPlot(self, x = None, y = None):
        painter = QPainter(self.pixmapPlot)
        pen = QPen(Qt.GlobalColor.white, 2)
        painter.setPen(pen)
        
        margin = 10
        
        self.center_x, self.center_y = self.pixmapPlot.width() // 2, self.pixmapPlot.height() // 2
        self.radius = min(self.center_x, self.center_y) - margin # Leave some margin
        
        # Draw the circle
        painter.drawEllipse(self.center_x - self.radius, self.center_y - self.radius, self.radius * 2, self.radius * 2)
        
        if x is not None and y is not None:
                
            # Retrive points (in the resized space)
            x = x * self.pixmapPlot.width() / self.plotImg.size().width()
            y = y * self.pixmapPlot.height() / self.plotImg.size().height()
            
            painter.drawLine(self.center_x, self.center_y, x, y)
            painter.drawEllipse(x, y, 2, 2)

        painter.end()
        self.plotImg.setPixmap(self.pixmapPlot)
    
    def setPlotImage(self):
        self.basePath = self.params.getRelightinBasePath()
        self.uvMean = getUVMean(self.basePath + "models/uvMean.h5").astype(np.uint8)
        
        # Get default size of the image
        self.defaultImgSize = self.uvMean.shape[0]
        
        # Get the QLabel
        self.plotImg = self.findChild(QLabel, 'relightingPlotLbl')
        self.outputImg = self.findChild(QLabel, 'outputImgLbl')
        self.downloadBtn = self.findChild(QPushButton, 'downloadBtn')
        
        self.downloadBtn.setEnabled(False)
        
        # Add set mouse tracking
        self.plotImg.setMouseTracking(True)
        
        # Create a value which is used by parent QWidget as resize value of its geometry
        geometryResize = (self.defaultImgSize * 2) + 200
        
        # Emit the signal to change the window size to the dimension of the frame
        self.geometryChanged.emit(QRect(0, 0, geometryResize, (self.defaultImgSize + 200)))
        
        # Init the pixmap with a black background
        self.pixmapPlot = QPixmap(self.plotImg.size())
        self.pixmapPlot.fill(Qt.GlobalColor.black)
        self.drawPlot()
        
        # Resize the widget to get the image inside
        self.resize(geometryResize, (self.defaultImgSize + 200))
        
        # Set events for the plotting image
        self.plotImg.mouseMoveEvent = self.relightPlot
        self.plotImg.mousePressEvent = self.getImageFrame
        
        # Set event to download the image
        self.downloadBtn.clicked.connect(self.downloadRelightedImg)
        
    def setOutputImage(self):
        
        # Retrieve the file paths to extract data
        projPixelsPCAFile = self.basePath + 'pca_norm/proj_pixels_pca.npy'
        weights = self.basePath + 'models/pixel-weights.pt'
        
        # Get projected pixels
        self.projPixels = np.load(projPixelsPCAFile)
        
        B = np.zeros((2, 10), dtype=np.float32)

        self.model = PixelEncoder(self.projPixels.shape[2] + 2, B)
        self.model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
        self.model.to(getDevice())
        
    def relightPlot(self, event):
        
        # Retrive points (in the resized space)
        x, y = event.pos().x(), event.pos().y()
        
        if self.isMoving:
            # Init the pixmap with a black background
            self.pixmapPlot = QPixmap(self.plotImg.size())
            self.pixmapPlot.fill(Qt.GlobalColor.black)
            
            self.drawPlot(x, y)
            
            if math.dist([x,y], [self.center_x, self.center_y]) < self.radius:
                light = np.array([[normaliseCoordinate(x, self.plotImg.size().width()), normaliseCoordinate(y, self.plotImg.size().height())]])
                # Generate the output images (only one in this case)
                images = predictRelight(self.model, light, self.projPixels)
                
                # Extract the intensity of the image in each pixel, and convert them into integers
                outImg = images[0,:,:].astype(np.uint8)
                outImg = np.expand_dims(outImg, axis=2)
                
                # Concatenate to get YUV Scale and convert to original BGR Scale
                outImg = np.dstack((outImg, self.uvMean))    
                outImg = cv.cvtColor(outImg, cv.COLOR_YUV2BGR)        
                
                # Create image with predicted relighting
                h, w, _ = outImg.shape
                qImage = QImage(outImg.data, w, h, 3 * w, QImage.Format.Format_BGR888)
                
                # Set pixmap for output image                        
                self.outputImg.setPixmap(QPixmap(qImage))
                
    def getImageFrame(self, event):
        if self.isMoving:
            self.isMoving = False
            
            self.downloadBtn.setEnabled(True)
        else: 
            self.isMoving = True
            
            self.downloadBtn.setEnabled(False)
    
    def downloadRelightedImg(self, event):
        # Then we recover the image from the pixmap
        pixmap : QPixmap = self.outputImg.pixmap()
        
        image = pixmap.toImage()
                
        # Now, store this values inside a file
        now_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
        folderPath = self.basePath + "outputs/"
        
        if not os.path.exists(folderPath):
            # If folder does not exsist, create it
            os.makedirs(folderPath)    
        
        # Set file name
        fileName = folderPath + "capture_%s.png"%(now_string)
        
        # Save the image to the chosen file path
        image.save(fileName)