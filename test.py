import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from constants import *

rbfInterpolation = np.load("interpolationMatrix.npy")

rbfInterpolation = np.array(rbfInterpolation)

for u in range(DEFAULT_SQUARE_SIZE):
    for v in range(DEFAULT_SQUARE_SIZE):     
        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break
        
cv.destroyAllWindows()