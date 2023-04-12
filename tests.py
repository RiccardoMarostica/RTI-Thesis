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
    
    # contours = sorted(contours, key = cv.contourArea)
    
    # contour = contours[-2]
    
    squareContour = max(contours, key = cv.contourArea)
    
    epsilon = 0.1 * cv.arcLength(squareContour, True)
    approxCorners = cv.approxPolyDP(squareContour, epsilon, True)
    
    corners = approxCorners.reshape(-1, 2)
    
    distances = [cv.norm(corners[i], corners[i+1]) for i in range(len(corners)-1)]
    
    minDistance = int(np.min(distances))
    
    print(minDistance)
        
    x, y =corners[0]

    # Define ROI using coordinates of smallest circle and size of internal square
    roi_x, roi_y, roi_w, roi_h = x, y, minDistance, minDistance
    
    roi = frame1[roi_y : (roi_y + roi_h), roi_x : (roi_x + roi_w)]
    
    # cv.imshow("FRAME", roi)
    
    # meanDistanceInner = meanDistance / 1.4
    
    # distances = []
    # for i in range(len(approxCorners)):
        
    #     j = (i + 1) % len(approxCorners)
    #     print(approxCorners[i][0], approxCorners[j][0])
    #     frame1 = cv.line(frame1, (approxCorners[i][0][0], approxCorners[i][0][1]), (approxCorners[j][0][0], approxCorners[j][0][1]), (0, 255, 0), 3)
        
    #     distance = cv.norm(approxCorners[i][0], approxCorners[j][0])
    #     distances.append(distance)
        
        
    # cv.imshow("Frame", roi)
    
    # minDistanceIndex = distances.index(min(distances))
    # x, y = approxCorners[minDistanceIndex][0]
    
    # frame1 = cv.circle(frame1, (x,y), 1, (0, 255, 0), 3)
    
    # # Define ROI using coordinates of smallest circle and size of internal square
    # roi_x, roi_y, roi_w, roi_h = x, y, 670, 673
    
    # roi = frame1[roi_y : (roi_y + roi_h), (roi_x - roi_w) :roi_x]
    
    # Find the circles in the corners of the square
    # corners = approxCorners.reshape(-1, 2)
    # distances = [cv.norm(corners[i], corners[i+1]) for i in range(len(corners)-1)]
    # min_distance_index = distances.index(min(distances))

    # # Find the smallest circle in the left corner of the square
    # (x, y), radius = cv.minEnclosingCircle(approxCorners[min_distance_index])
    # circle = (int(x), int(y))

    # Crop the frame to only include the ROI
    # roi = frame1[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # cv.imshow('ROI', frame1)

    mask = np.zeros_like(grayFrame)
    cv.fillPoly(mask, [approxCorners], (255, 255, 255))
    
    minDistanceInner = int(minDistance / 1.4)
    
    dstPoints = np.array([ [0, 0], [0, minDistanceInner], [minDistanceInner, minDistanceInner], [minDistanceInner, 0]])
    
    homography = cv.findHomography(approxCorners, dstPoints, cv.RANSAC)
    
    warped = cv.warpPerspective(frame1, homography[0], (minDistanceInner, minDistanceInner))
    
    cv.imshow("Frame", warped)
    
    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break

# Release videos
video1.release()

# And destroy windows
cv.destroyAllWindows()