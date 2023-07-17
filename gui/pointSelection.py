# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QLabel
from PyQt6.QtGui import QPixmap, QImage, QPaintEvent, QPainter, QPen
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6 import uic

# Other imports
import os

# Custom class imports
from classes.parameters import Parameters
from classes.video import Video

class PointSelection(QWidget):
    
    geometryChanged = pyqtSignal(QRect)
    
    def __init__(self, parent: QWidget) -> None:
        super(PointSelection, self).__init__(parent)

        # Compute the path
        basePath = os.path.dirname(__file__)       
        path = os.path.join(basePath, "templates/framePointSelection.ui")
                
        # Load the UI
        uic.loadUi(path, self)
        
        # Hide by default 
        self.hide()
        
        # Instantiate singleton class for parameters
        self.params = Parameters()
        
        self.points = []
        
        self.resizedX = 0
        self.resizedY = 0
        
    def setImage(self):
        # Get the video path from static camera
        self.video = Video(self.params.getStCamVideoPath())
        
        # Then, extract the first frame, and set it as Pixmap in the QLabel
        ret, frame = self.video.getCurrentFrame()
        
        if ret == True:
            # Get the QLabel
            self.img = self.findChild(QLabel, 'imgLbl')
            
            # Convert the frame from OpenCV to QImage
            
            # Based on the original dimension, resize the video
            windowType = 'Landscape' if self.video.getWidth() > self.video.getHeight() else 'Portrait'
            self.resizedW, self.resizedH = self.params.getDefaultFrameSize(windowType)
            
            # Then resize the frame to choosen dimension
            frame = self.video.resizeVideo(frame, (self.resizedW, self.resizedH))
            
            # ... and create the QImage
            qImage = QImage(frame.data, self.resizedW, self.resizedH, 3 * self.resizedW, QImage.Format.Format_RGB888).rgbSwapped()

            # Emit the signal to change the window size to the dimension of the frame
            self.geometryChanged.emit(QRect(0, 0, self.resizedW, self.resizedH))
            
            # Set the pixmap
            self.pixmap = QPixmap(qImage)
            # And add it to the label
            self.img.setPixmap(self.pixmap)
            
            # Resize the widget to get the image inside
            self.resize(self.pixmap.width(), self.pixmap.height())
                        
    def setStartBtn(self):
        self.startBtn = self.findChild(QPushButton, 'startBtn')
        self.startBtn.clicked.connect(lambda: self.startLightFrameExtraction())
    
    def mousePressEvent(self, event):
        # Retrive points (in the resized space)
        self.resizedX = event.pos().x()
        self.resizedY = event.pos().y()
        
        # Then, given the original (W, H) and the resized (W, H), compute the scale for X and Y axis
        scaleX = self.video.getWidth() / self.resizedW
        scaleY = self.video.getHeight() / self.resizedH
        
        # With the two scales, convert the pt into the original space and scale
        x = int(self.resizedX * scaleX)
        y = int(self.resizedY * scaleY)
        
        if len(self.points) < 4:
            # Store them only if there are less than 4 stored points
            self.points.append((x, y))
            print(f"Point added: ({x}, {y})")
            
            if len(self.points) == 4:
                # We reached the maximum elements, so enable the start btn
                self.startBtn.setEnabled(True)
                
            self.update()
            
        else:
            print("Maximum number reached")

    # TODO: Sistemare paint event
    # def paintEvent(self, event: QPaintEvent) -> None:
        
    #     # Create painter object responsible to paint things in the image
    #     painter = QPainter(self.img.pixmap())
        
    #     painter.drawPixmap(self.rect(), self.pixmap)
        
    #     # Create the pen
    #     pen = QPen()
    #     pen.setWidth(20)
    #     pen.setColor(Qt.GlobalColor.red)

    #     # Set the pen
    #     painter.setPen(Qt.GlobalColor.red)
        
    #     if self.resizedX > 0 and self.resizedY > 0:
            
    #         print("Calling paint event")
            
    #         print("Enters if")
    #         print(f"Points: ({self.resizedX}, {self.resizedY})")
    #         painter.drawEllipse(self.resizedX, self.resizedY, 50, 50)
                

            
    def startLightFrameExtraction(self):
        # TODO: Aggiungere algoritmo x feature matching e resto
        return