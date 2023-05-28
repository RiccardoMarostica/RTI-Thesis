import os
import cv2 as cv
import numpy as np

class Video:
    def __init__(self, videoPath):
        # Given a video path, just create a new video instance using OpenCV
        self.video = cv.VideoCapture(videoPath)
    
    def isOpen(self) -> bool:
        # Returns True if the video file is open. False otherwise
        return True if self.video.isOpened() else False
    
    def getFPS(self) -> int:
        # Return the FPS of the video is it is open. Otherwise, returns 0
        if self.isOpen():
            return int(round(self.video.get(cv.CAP_PROP_FPS)))
        return 0
    
    def getWidth(self) -> int:
        # Return the width of the video if it is open. Otherwise, return 0
        if self.isOpen():
            return int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        else:
            return 0
        
    def getHeight(self) -> int:
        # Return the height of the video if it is open. Otherwise, return 0
        if self.isOpen():
            return int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        else:
            return 0
        
    def getTotalFrames(self) -> int:
        # Return the total frames of the video if it is open. Otherwise, returns 0
        if self.isOpen():
            return int(round(self.video.get(cv.CAP_PROP_FRAME_COUNT)))
        return 0
    
    def getDuration(self) -> int:
        # Return video duration in seconds if video is open. Otherwise, return 0
        frames = self.getTotalFrames()
        fps = self.getFPS()
        
        if self.isOpen():
            return int(round(frames/fps))
        return 0