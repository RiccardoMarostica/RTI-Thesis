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

    def getCurrentFrame(self):
        if self.isOpen():
            ret, frame = self.video.read()
            return ret, frame
        return None, None

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
    
    def applyUndistortion(self, frame, K, dist):
        # Apply undistortion to a frame and return the resulted frame
        undistortedFrame = cv.undistort(frame, K, dist)
        return undistortedFrame

    def setVideoPosition(self, ms: int) -> None:
        # If the video is open, set the next position in the video, after ms value
        if self.isOpen():
            self.video.set(cv.CAP_PROP_POS_MSEC, ms)

    def showFrame(self, frame, debug=False):
        if debug == True:
            # Show the frame from OpenCV
            cv.imshow("Frame", frame)

            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                return

    def releaseVideo(self) -> None:
        # Release video file
        self.video.release()
