from classes.neuralNetwork import NeuralNetwork

from utils import *

BASE_DIR = "examples/unive_example_23_07_23_18_10/"
datafile = BASE_DIR + "unive.h5"

# pca = PCAClass(BASE_DIR, datafile, 8)

# print("Reading dataset...")
# pca.readDataset()
# print("Reading dataset: DONE")


# print("Applying PCA...")
# pca.applyPCA()
# print("Applying PCA: DONE")

nn = NeuralNetwork(BASE_DIR, 8)

# print("Extracting dataset...")
# nn.extractDatasets()
# print("Extracting dataset: DONE")


# print("Shufflings dataset...")
# nn.shuffleDataset()
# print("Shuffling dataset: DONE")

# print("Executing NN training...")
# nn.executeTraining()
# print("Executing NN trainin: DONE")

BASE_DIR = "examples/unive_example_23_07_23_18_10/"
nn.showNNResults()