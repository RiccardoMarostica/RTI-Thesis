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

from classes.pca import PCAClass
from classes.neuralNetwork import NeuralNetwork

BASE_DIR = "examples/unive_example_23_07_23_18_10/"
datafile = BASE_DIR + "unive.h5"

pca = PCAClass(BASE_DIR, datafile, 8)

print("Reading dataset...")
pca.readDataset()
print("Reading dataset: DONE")


print("Applying PCA...")
pca.applyPCA()
print("Applying PCA: DONE")


nn = NeuralNetwork(BASE_DIR, 8)

print("Extracting dataset...")
nn.extractDatasets()
print("Extracting dataset: DONE")


print("Shufflings dataset...")
nn.shuffleDataset()
print("Shuffling dataset: DONE")

print("Executing NN training...")
nn.executeTraining()
print("Executing NN trainin: DONE")
