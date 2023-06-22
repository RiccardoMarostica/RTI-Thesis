import os
import cv2 as cv
import numpy as np


class Video:
    """
    This class contains all the methods to handle a Video file instance.\n
    A Video can be initialised using OpenCv, and with OpenCv methods it's possible to get/set settings from/to the Video instance.\n
    The Video class accepts only one construct, the one in which it takes a Video Path as input. From it, it is possible to create the Video instance using OpenCv.
    """
    
    def __init__(self, videoPath):
        # Given a video path, just create a new video instance using OpenCV
        self.video = cv.VideoCapture(videoPath)

    def isOpen(self) -> bool:
        """The function checks if the video has been initalised already.
        
        Returns:
            bool: True if video capturing has been initialized already. False otherwise
        """
        # Returns true if video capturing has been initialized already
        return True if self.video.isOpened() else False

    def getCurrentFrame(self) -> tuple[bool, list]:
        """The function grabs a frame from the video instance.\n
        If no frame has been grabbed (no more frames in video file), returns false and an empty object.
        
        Returns: 
            If the function grabs a frame, returns True and the current frame. Otherwise, returns False and an empty frame.
        """
        if self.isOpen():
            # If the video is initalised, grab a frame.
            # If no frame has been grabbed, return False and an empty image
            ret, frame = self.video.read()
            return ret, frame
        
        # Returns False and an empty frame in case the video is not initalised
        return False, []

    def getFPS(self) -> float:
        """The function returns the number of Frame Per Second(FPS) the video supports.
        
        Returns:
            float: Returns the FPS of the Video if it is initliased. Otherwise, returns zero.
        """
        return self.video.get(cv.CAP_PROP_FPS) if self.isOpen() else 0

    def getWidth(self) -> int:
        """The function returns the width of the Video.
        
        Returns:
            int: The width of the Video, if it is initliased. Zero, otherwise.
        """
        return int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)) if self.isOpen() else 0

    def getHeight(self) -> int:
        """The function returns the height of the Video.
        
        Returns:
            int: The height of the Video, if it is initliased. Zero, otherwise.
        """
        return int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)) if self.isOpen() else 0

    def getTotalFrames(self) -> int:
        """The function returns the total number of frames contained in the Video.\n
        Returns:
            int: The number of frames in the Video, if it is initalised. Zero, otherwise.
        """
        return int(round(self.video.get(cv.CAP_PROP_FRAME_COUNT))) if self.isOpen() else 0

    def getDuration(self) -> int:
        """The function returns the duration of the video, in seconds.\n
        To calculate the duration of the Video, it's necessary to obtain the frames in the Video, and the FPS of the Video.
        
        Returns:
            int: The duration of the Video, in seconds, if it is initalised. Zero, otherwise.
        """
        # Get total frames and FPS of the Video instance
        frames = self.getTotalFrames()
        fps = self.getFPS()
        
        return int(round(frames/fps)) if self.isOpen() else 0
    
    def applyUndistortion(self, frame, K, dist):
        """The function returns the undistorted image using cv2 method undistortion().\n
        
        Args:
            frame (Any): Current grabbed frame, in which undistortion will be applied.
            K (Any): A 3x3 matrix, denoting the intrinsic parameters of the Camera.
            dist (Any): An array of 5 elements, containing the undistortion coefficients (k1, k2, k3, p1, p2)

        Returns:
            Any: The undistorted image, applying the cv2 method undistortion() using K and dist
        """
        return cv.undistort(frame, K, dist)

    def setVideoPosition(self, ms: int) -> None:
        """The function sets the current postion of the video file using milliseconds.\n
        With this function it's possible to move forward in the video based on the number of milliseconds.

        Args:
            ms (int): Current position of the video file in milliseconds
        """
        if self.isOpen():
            self.video.set(cv.CAP_PROP_POS_MSEC, ms)
        
    def setVideoFrame(self, pos : int = 0) -> None:
        """The function sets the index inside the video instance, to decode/capute a frame.

        Args:
            pos (int, optional): Index of the frame to be decoded/captured next. Defaults to 0.
        """
        if self.isOpen():
            self.video.set(cv.CAP_PROP_POS_FRAMES, pos)

    def showFrame(self, frame, winname = "Frame", debug=False):
        """The function shows a frame using the OpenCv method imshow()

        Args:
            frame (Any): Current grabbed frame user wants to display
            winname (str, optional): Name of the window. Defaults to "Frame".
            debug (bool, optional): Debug option, used to show the image. If True, when the method is called, show the image, False. Defaults to False.
        """
        if debug == True:
            # Show the frame from OpenCV
            cv.imshow(winname, frame)

            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                return

    def releaseVideo(self) -> None:
        """The function closes the video file instance.
        """
        self.video.release()
