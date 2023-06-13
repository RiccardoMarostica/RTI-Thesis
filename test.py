import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from constants import *

rbfInterpolation = np.load("interpolationMatrix.npy")

rbfInterpolation = np.array(rbfInterpolation)

# Create a blank image
relightPlot = np.zeros((DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE, 3), dtype=np.uint8)

center_x = center_y = DEFAULT_SQUARE_SIZE // 2
radius = DEFAULT_SQUARE_SIZE // 2    

# Draw the circle border
cv.circle(relightPlot, (center_x, center_y), radius, (255, 255, 255), 1)
cv.line(relightPlot, (0, center_y), (DEFAULT_SQUARE_SIZE, center_y), (255, 255, 255), 1)
cv.line(relightPlot, (center_x, 0), (center_x, DEFAULT_SQUARE_SIZE), (255, 255, 255), 1)

def mouseCallback(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Choosen point: ({x}, {y})")
        dis = distanceCalculate((center_x, center_y), (x, y))
        print("Distance: ", dis)
        interpolationXY = rbfInterpolation[x * DEFAULT_SQUARE_SIZE + y]
        print(f"Interpolation at ({x}, {y}): \n", interpolationXY)
        nearest = find_nearest(interpolationXY, dis)
        print("Nearest: ", nearest)
           
def distanceCalculate(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def find_nearest(array, value):
    differences = np.abs(array - value)
    min_index = np.unravel_index(differences.argmin(), differences.shape)
    return array[min_index]

image = cv.imread("frame.jpg")

while (True):
    cv.imshow("Relight plot", relightPlot)
    cv.imshow("Relight image", image)
    
    cv.setMouseCallback("Relight plot", mouseCallback)
    
    
    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break
    
cv.destroyAllWindows()