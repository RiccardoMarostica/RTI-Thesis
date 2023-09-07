# PyQt imports
from PyQt6.QtWidgets import QPushButton, QWidget, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QRect, pyqtSignal
from PyQt6 import uic

# Other imports
import os
import numpy as np
import cv2 as cv
import time
from datetime import datetime

from utils import getVideoOrientation, videoNeedsResize, getLightDirectionPlot, storeTrainDataset

# Custom class imports
from classes.parameters import Parameters
from classes.video import Video
from classes.videoAnalysis import VideoAnalysis
from classes.videoSynchronisation import VideoSynchronisation
from classes.threadPool import ThreadPool

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
                
        # Initialise RTI class, which performs the feature matching and the storage of informations
        self.videoAnalysis = VideoAnalysis()
        
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
            windowType = getVideoOrientation(self.video)
            self.resizedW, self.resizedH = self.params.getResizedSizePointCapture(windowType)
            
            # Then resize the frame to choosen dimension
            frame = self.video.resizeVideo(frame, (self.resizedW, self.resizedH))
            
            # ... and create the QImage
            qImage = QImage(frame.data, self.resizedW, self.resizedH, 3 * self.resizedW, QImage.Format.Format_RGB888).rgbSwapped()

            # Emit the signal to change the window size to the dimension of the frame
            self.geometryChanged.emit(QRect(0, 0, self.resizedW, self.resizedH + 200))
            
            # Set the pixmap
            self.pixmap = QPixmap(qImage)
            # And add it to the label
            self.img.setPixmap(self.pixmap)
            
            # Resize the widget to get the image inside
            self.resize(self.resizedW, self.resizedH)
            
            print("Geometry of point selection: ", self.geometry())
            
            self.img.mousePressEvent = self.getPoints
                        
    def setStartBtn(self):
        self.startBtn = self.findChild(QPushButton, 'startBtn')
        self.startBtn.clicked.connect(lambda: self.startLightFrameExtraction())
    
    def getPoints(self, event):
        pixmap = self.img.pixmap()
        image_size = pixmap.size()
        

        # Retrive points (in the resized space)
        self.resizedX = event.pos().x() * image_size.width() / self.img.size().width()
        self.resizedY = event.pos().y() * image_size.height() / self.img.size().height()
        
        camOrientation = getVideoOrientation(self.video)
        needsResize = videoNeedsResize(self.video, self.params.getFrameDefaultSize(camOrientation))

        defW = self.video.getWidth()
        defH = self.video.getHeight()
            
        if (needsResize):
            defSize = self.params.getFrameDefaultSize(camOrientation)
            defW = defSize[0]
            defH = defSize[1]
        
        # Then, given the original (W, H) and the resized (W, H), compute the scale for X and Y axis
        scaleX = defW / self.resizedW
        scaleY = defH / self.resizedH
        
        print(f"Scale X: {scaleX}")
        print(f"Scale Y: {scaleY}")
        
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

    def startLightFrameExtraction(self):
        
        # Disable the button
        self.startBtn.setEnabled(False)
        
        # Obtain calibration data
        stCamCalibration = self.params.getStCamCalibData()
        mvCamCalibration = self.params.getmvCamCalibData()
        
        self.kMoving = mvCamCalibration.getIntrinsicMatrix()
        
        # Retrieve both videos
        videoStatic = Video(self.params.getStCamVideoPath())
        videoMoving = Video(self.params.getMvCamVideoPath())

        # Calculate the world homography using the points selected in the previous step and the frame size choosen before
        self.worldHomography = self.videoAnalysis.getWorldHomographyFromPts(self.points, int(self.params.getOutputImageSize()))
        
        if self.worldHomography is None:
            print("An error occurred: World homography has not been set correctly. ")
            exit(-1)
        else:
            print("World homography has been calculated correctly. ")
            
        # Get the first frame from the static camera, which will be used during feature matching
        _, self.firstStaticFrame = videoStatic.getCurrentFrame()
        
        # Then, reset the position to the initial
        videoStatic.setVideoFrame()
            
        # Now, before starting, synchronise the two videos
        videoSynch = VideoSynchronisation(self.params.getStCamVideoPath(), self.params.getMvCamVideoPath())
        videoSynch.synchroniseVideo()
        
        # Then get the offset and check its value. If positive, then shift first video. Otherwise, do the opposite
        videoOffset = videoSynch.getOffset()
        
        if videoOffset > 0:
            # First video shifted. Set frame difference with it's own FPS
            frameDiff = videoSynch.getFrameDifference(videoStatic.getFPS())
            videoStatic.setVideoFrame(abs(frameDiff))
            videoMoving.setVideoFrame()
            print(f"Static video is shifted. The frame difference is: {abs(frameDiff)}")
        else:
            # Second video shifted. Set frame difference with it's own FPS
            frameDiff = videoSynch.getFrameDifference(videoMoving.getFPS())
            videoStatic.setVideoFrame()
            videoMoving.setVideoFrame(abs(frameDiff))
            print(f"Moving video is shifted. The frame difference is: {abs(frameDiff)}")
        
        # Variable used to store the time calculated after each read on the video, in order to provide synchronisation
        timeStaticVideo = 0.
        timeMovingVideo = 0.
            
        print("Starting calculation of the light directions in the videos...")

        staticFrames = []
        movingFrames = []
        
        # Get video orientation for both cases
        stCamOrientation = getVideoOrientation(videoStatic)
        mvCamOrientation = getVideoOrientation(videoMoving)
        
        # Set to None (so no resize is necessary)
        stCamSize = mvCamSize = None
        
        if videoNeedsResize(videoStatic, self.params.getFrameDefaultSize(stCamOrientation)):
            # Check if static camera size needs to be resized. And if so, retrieve the default dimension
            stCamSize = self.params.getFrameDefaultSize(stCamOrientation)
            
        if videoNeedsResize(videoMoving, self.params.getFrameDefaultSize(mvCamOrientation)):
            # Check if moving camera size needs to be resized. And if so, retrieve the default dimension
            mvCamSize = self.params.getFrameDefaultSize(mvCamOrientation)
        
        # First, acquire pair of static/moving camera frames
        while videoStatic.isOpen() and videoMoving.isOpen():
            
            # Get frame from each video
            retStatic, staticFrame = videoStatic.getCurrentFrame()
            retMoving, movingFrame = videoMoving.getCurrentFrame()
            
            # For each iteration, sum the time for each video based on the tick (1 / FPS_video)
            timeStaticVideo += 1. / videoStatic.getFPS()
            timeMovingVideo += 1. / videoMoving.getFPS()
            
            # Now depends on which video has lower FPS
            if videoStatic.getFPS() < videoMoving.getFPS():
                # Video static is behind more than 1 frame, so skip it to recover the loss
                if timeStaticVideo > timeMovingVideo + (1. / videoMoving.getFPS()):
                    retStatic, staticFrame = videoStatic.getCurrentFrame()
                    
            else:    
                # Video moving is behind more than 1 frame, so skip it to recover the loss
                if timeMovingVideo > timeStaticVideo + (1. / videoMoving.getFPS()):
                    retMoving, movingFrame = videoMoving.getCurrentFrame()
            
            # Check if static or moving frame is empty
            checkStaticFrame = (staticFrame is None or np.shape(staticFrame) == () or np.sum(staticFrame) == 0)
            checkMovingFrame = (movingFrame is None or np.shape(movingFrame) == () or np.sum(movingFrame) == 0)
            
            # Close loop in case of one video is over    
            if retStatic != True or retMoving != True or checkStaticFrame or checkMovingFrame:
                break
            
            if (stCamSize is not None):
                # Resize the static frame
                staticFrame = cv.resize(staticFrame, (stCamSize))
            
            if (mvCamSize is not None):
                # Resize the moving frame
                movingFrame = cv.resize(movingFrame, (mvCamSize))
                
            # Then apply the undistortion of the camera
            staticFrame = videoStatic.applyUndistortion(
                staticFrame, 
                stCamCalibration.getIntrinsicMatrix(),
                stCamCalibration.getDistortionCoefficients()
            )
            movingFrame = videoMoving.applyUndistortion(
                movingFrame, 
                mvCamCalibration.getIntrinsicMatrix(),
                mvCamCalibration.getDistortionCoefficients()
            )
            
            # Store current frame
            staticFrames.append(staticFrame)
            movingFrames.append(movingFrame)
            
        videoStatic.releaseVideo()
        videoMoving.releaseVideo()
        cv.destroyAllWindows()
         
        # Create thread pool, with 4 threads
        pool = ThreadPool(4)
            
        # Get keypoints and descriptor for first frame
        if (stCamSize is not None):
            # Resize the static frame
            self.firstStaticFrame = cv.resize(self.firstStaticFrame, (stCamSize))
        
        # Extract features from first frame
        featuresFirstStaticFrame = self.videoAnalysis.extractFeaturesFromFrame(self.firstStaticFrame, 0)
        
        for i in range(len(staticFrames)):
            # For each static frame, calculate its features
            pool.add_task(self.videoAnalysis.extractFeaturesFromFrame, staticFrames[i], i)
            
        # Wait completion of the queue, and get the results
        pool.wait_completion()
        featuresStaticFrames = pool.get_results()
        
        for i in range(len(movingFrames)):
            # For each static frame, calculate its features
            pool.add_task(self.videoAnalysis.extractFeaturesFromFrame, movingFrames[i], i)
            
        # Wait completion of the queue, and get the results
        pool.wait_completion()
        featuresMovingFrames = pool.get_results()
            
        # Now sort features based on index
        featuresStaticFrames = sorted(featuresStaticFrames, key = lambda x: x[0])
        featuresMovingFrames = sorted(featuresMovingFrames, key = lambda x: x[0])
        
        # Replicate first frame features to then match it to all the other frames in the static camera 
        featuresFirstFrame = [featuresFirstStaticFrame] * len(featuresStaticFrames)
        
        # Create a list of features that will be matched using feature matching technique
        featuresStaticStatic = list(zip(featuresStaticFrames, featuresFirstFrame))
        featuresStaticMoving = list(zip(featuresStaticFrames, featuresMovingFrames))
    
        for i in range(len(featuresStaticStatic)):
            feature = featuresStaticStatic[i]
            # For each static frame, calculate its features
            pool.add_task(self.videoAnalysis.matchFeatures, feature)
            
        # Wait completion of the queue, and get the results
        pool.wait_completion()
        matchingStaticStatic = pool.get_results()
        
        # Repeat the process
        for i in range(len(featuresStaticMoving)):
            feature = featuresStaticMoving[i]
            # For each static frame, calculate its features
            pool.add_task(self.videoAnalysis.matchFeatures, feature)
            
        # Wait completion of the queue, and get the results
        pool.wait_completion()
        matchingStaticMoving = pool.get_results()
        
        # Now sort matching based on index
        matchingStaticStatic = sorted(matchingStaticStatic, key = lambda x: x[0])
        matchingStaticMoving = sorted(matchingStaticMoving, key = lambda x: x[0])
        
        for i in range(len(matchingStaticStatic)):
            # Get frames
            staticFrame = staticFrames[i]
            movingFrame = movingFrames[i]
            # Get homographies and dst pts
            _, _, _, homographyStatic = matchingStaticStatic[i]
            _, _, dstPts, homographyMoving = matchingStaticMoving[i]
            
            # Calculate light given parameters
            pool.add_task(self.videoAnalysis.getLight, staticFrame, movingFrame, homographyStatic, dstPts, homographyMoving, self.worldHomography, self.kMoving)
        
        # Wait completion of the queue, and get the results
        pool.wait_completion()
        lightFramePair = pool.get_results()
                
        validPairs = [pair for pair in lightFramePair if all(value is not None for value in pair)]
            
        # Number of frames read
        nFrames = 0
        
        # Array with shape (400, 400, 2) which contains the sum of the U and V value of each pixel along the video
        sumUV = np.zeros((400, 400, 2))
        
        # Array with shape (nFrames, 400, 400, 3) where, for each pixel, stores the intensity of it, and the light value X and Y (costant along the frame)
        lightData = []
            
        for worldFrame, light in validPairs:
            
            # Show the light plot of the calculated light vector
            cirlePlotPnP = getLightDirectionPlot(light, self.params.getOutputImageSize())
            
            # First, convert the frame from GRAY to BGR
            # Then from BGR to YUV, to extract the intensity and calculate U and V mean
            # worldFrameBGR = cv.cvtColor(worldFrame, cv.COLOR_GRAY2BGR)
            worldFrameYUV = cv.cvtColor(worldFrame, cv.COLOR_BGR2YUV)
            
            # Get Y, U, V
            Y, U, V = cv.split(worldFrameYUV)
            
            # Get the light position X and Y (Z can be removed now)            
            light = np.tile(light[:2].flatten(), (self.params.getOutputImageSize(), self.params.getOutputImageSize(), 1))
        
            # Now, store an array containing the intensity of the pixels and the respective light
            data = np.dstack((Y, light))
            
            # Now get the current UV with shape (400, 400, 2), and sum their values inside the sumUV matrix with shape (400, 400, 2)
            sumUV = sumUV + np.dstack((U, V))
            
            # Append the data
            lightData.append(data)
            
            # Increament number of frames acquired
            nFrames += 1
        
        # Calculate UVMean
        meanUV = sumUV / nFrames
        
        # Convert from list 2 array
        lightData = np.stack(lightData)
        
        # Now, store this values inside a file    
        now_string = datetime.now().strftime("%y_%m_%d_%H_%M")
        
        # First get the base dir
        BASE_DIR = "relights/relight_%s/"%now_string
        
        try:
            # Creating the base dir
            os.mkdir(BASE_DIR)
        except:
            print("Error creating the new folder. ")
            # Not possible to create the dir, close the app
            exit(-1)
        
        # Set the name of the file containing the inital dataset for training
        fileName = BASE_DIR + "data.h5"
        
        print("Calculation of the light directions done")
        
        # Then create the train dataset
        storeTrainDataset(fileName, lightData, meanUV)