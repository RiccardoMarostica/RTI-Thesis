import numpy as np

from constants import *

rbfInterpolation = np.load("interpolationMatrix.npy")

rbfInterpolation = np.array(rbfInterpolation)

# print(rbfInterpolation.shape)

print(rbfInterpolation[1 * DEFAULT_SQUARE_SIZE + 100].shape)