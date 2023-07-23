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
  
# Now, store this values inside a file    
now_string = datetime.now().strftime("%y_%m_%d_%H_%M")
dataName = 'unive'

# First get the base dir, and out dir
BASE_DIR = "examples/%s_example"%dataName + "_%s/"%now_string

try:
    # Creating the base dir
    os.mkdir(BASE_DIR)
except:
    # Not possible to create the dir, close the app
    exit(-1)
# Open the file
fileName = BASE_DIR + "%s.h5"%dataName

print(fileName)

f = h5py.File(fileName, "w")
# ... and create datasets
f.create_dataset("lightdata", (1,))
f.create_dataset("UVMean", (1,))
# Then stop writing
f.close()