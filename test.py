# import numpy as np
# # Creating the first array (400, 400)
# array_1 = np.random.rand(400, 400)

# # Creating the second array (2,)
# array_2 = np.array([10, 20])

# print("Array 2 shape: ", array_2.shape)

# # Reshaping the second array to match the shape of the first array (400, 400, 2)
# reshaped_array_2 = np.tile(array_2, (400, 400, 1))

# print("Array 1 shape: ", array_1.shape)
# print("Array 2 shape: ", reshaped_array_2.shape)

# # Now, stack both arrays together along a new axis to get the desired result
# result_array = np.dstack((array_1, reshaped_array_2))

# print(result_array.shape)  # Output: (400, 400, 3)

import os
from datetime import datetime
import h5py
import numpy as np

datafile = "examples/unive_example_23_07_22_19_05/unive.h5"

with h5py.File(datafile,"r") as f:
    # Array with 3 channels: intensity, light_x, light_y
    data_ = np.array( f["lightdata"] ) 
    # In this array we have the mean UV in each position 
    UVmean = np.array(f["UVMean"])
    #:, 0, 0, 1: => In all the 2756 frames takes the first array in the shape (400, 400) and take the last two elements which are constants (which are light_X, light_Y)
    all_lights = data_[:,0,0,1:] # light vector is constant in each image

print("UVmean: ", UVmean.shape)
print("all data: ", data_.shape)
print("all lights: ", all_lights.shape)
