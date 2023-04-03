import cv2 as cv
import numpy as np

from constants import *

'''
Sizes information:
 - Inner square: 141 x 141 px
 - Outer square: 198 x 198 px
 - Bottom left circle diameter: 11 px
'''

# Load the two videos and extract their audio
video1 = cv.VideoCapture(STATIC_VIDEO_FILE_PATH)
        
# While one of the two videos is open, then read frame by frame
while video1.isOpened():
    
    # Get the frame from each video
    ret1, frame1 = video1.read()

    # If one of the two does not return a frame, then exit the loop
    if not ret1:
        break
    
    # Convert the frame in grayscale
    grayFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    
    # Now, let's apply the threshold to the frame
    _, threshold = cv.threshold(grayFrame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Then, let's find the contours of the black square marker
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key = cv.contourArea)
    
    contour = contours[-2]
    
    epsilon = 0.1 * cv.arcLength(contour, True)
    approxCorners = cv.approxPolyDP(contour, epsilon, True)
    
    distances = []
    for i in range(len(approxCorners)):
        j = (i + 1) % len(approxCorners)
        distance = cv.norm(approxCorners[i][0], approxCorners[j][0])
        distances.append(distance)
    
    minDistanceIndex = distances.index(min(distances))
    x, y = approxCorners[minDistanceIndex][0]

    mask = np.zeros_like(grayFrame)
    cv.fillPoly(mask, [approxCorners], (255, 255, 255))
    
    dstPoints = np.array([ [0, 0], [480, 0], [480, 480], [0, 480]])
    
    homography = cv.findHomography(approxCorners, dstPoints)
    
    warped = cv.warpPerspective(frame1, homography[0], (480, 480))
    
    cv.imshow("Frame", warped)
    
    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break

# Release videos
video1.release()

# And destroy windows
cv.destroyAllWindows()