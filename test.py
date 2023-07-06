import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from constants import *

# Import classes
from classes.video import Video
from classes.rtiAlgorithm import RTI

rti = RTI()

videoStatic = Video(STATIC_VIDEO_FILE_PATH)
videoMoving = Video(MOVING_VIDEO_FILE_PATH)

print(f"Width: {videoMoving.getWidth()}")
print(f"Height: {videoMoving.getHeight()}")

while(videoMoving.isOpen()):
    ret, frame = videoMoving.getCurrentFrame()
    
    if ret == True:
        
        frame = frame[300: 800, 600: 1100]
        
        cv.imshow("Frame", frame)
        
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
# frameDifference = -9    # Frame difference between book (filename) video camera

# if (frameDifference > 0):
#     print("Static Video shifted")
#     # If the offset is positive, then the first video starts sooner.
#     # So move its position in order to start as the second video
#     videoStatic.setVideoFrame(abs(frameDifference))
#     videoMoving.setVideoFrame()
# else:
#     print("Moving Video shifted")
#     # ... or vice versa
#     videoStatic.setVideoFrame()
#     videoMoving.setVideoFrame(abs(frameDifference))
        
# timeStaticVideo = 0.
# timeMovingVideo = 0.
        
# while (videoStatic.isOpen() and videoMoving.isOpen()):
    
#     retStatic, staticFrame = videoStatic.getCurrentFrame()
#     retMoving, movingFrame = videoMoving.getCurrentFrame()
    
#     if retStatic != True or retMoving != True:
#         break
    
#     # For each iteration, sum the time for each video based on the tick (1 / FPS_video)
#     timeStaticVideo += 1. / videoStatic.getFPS()
#     timeMovingVideo += 1. / videoMoving.getFPS()
    
#     # Now depends on which video has lower FPS
#     if videoStatic.getFPS() < videoMoving.getFPS():
#         # Video static is behind more than 1 frame, so skip it to recover the loss
#         if timeStaticVideo > timeMovingVideo + (1. / videoMoving.getFPS()):
#             retStatic, staticFrame = videoStatic.getCurrentFrame()
#     else:    
#         # Video moving is behind more than 1 frame, so skip it to recover the loss
#         if timeMovingVideo > timeStaticVideo + (1. / videoMoving.getFPS()):
#             retMoving, movingFrame = videoMoving.getCurrentFrame()
    
#     staticFrame = staticFrame[1000:3200, 0: 2160]    # Crop for book
#     movingFrame = movingFrame[50: 1080, 300: 1400]  # crop for book
            
#     cv.imshow("static", staticFrame)
#     cv.imshow("moving", movingFrame)
    
#     # Press Q on the keyboard to exit.
#     if (cv.waitKey(25) & 0xFF == ord('q')):
#         break

cv.destroyAllWindows()
videoStatic.releaseVideo()
videoMoving.releaseVideo()