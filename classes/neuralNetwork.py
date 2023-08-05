import os
import torch
from torch import nn
import numpy as np
import h5py
from tqdm.auto import tqdm

from utils import *
from constants import *
from classes.pixelEncoder import *

class NeuralNetwork:
    def __init__(self, baseDir, pcaNumber) -> None:
        
        # H value where H is a frequency domain space -> So we have a 20 dimensional space (2 x 10)
        self.light_freqs = 10

        # Value used to generate the random values in B. This value offers good results in terms of avg. PSNR and SSIM on the test set
        self.sigma_light = 0.3

        self.light_noise_sigma = 0.0

        # Epoch size
        self.epochs = 20

        # Number of PCA bases. Value in which PSNR and SSIM stabilise
        self.pcaNumber = pcaNumber
        
        # multiply 1024 for the 20-dimensional space
        self.batch_size = 1024*20
        
        # Get base dir, training data dir and output dir
        self.baseDir = baseDir
        self.trainDataDir = baseDir + "pca_norm"
        self.outDir = baseDir + "models/"
                
        # try:
        #     os.mkdir(self.outDir)
        # except:
        #     print("Output directory already present. ")
        #     exit(-1)
        
    def extractDatasets(self):        
        # Get projection pixels computed before and the training data of the model
        train_data_file = "%s/train_data_pca_%02d.h5"%(self.trainDataDir,self.pcaNumber)
                
        # Then open the training data file, which was computed in the previous step
        f = h5py.File(train_data_file,"r")

        # And get from the file X (compressed pixels and light direction) and Y (intensity values) dataset
        self.x_train = f["X"]
        self.y_train = f["Y"]
    
    def shuffleDataset(self):
        # Convert the train values into an array
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

        print("Shuffling...")
        perm_indices = np.random.permutation(self.x_train.shape[0])
        self.x_train = self.x_train[perm_indices,...]
        self.y_train = self.y_train[perm_indices,...]
        
    def train_one_epoch(self, x_train, y_train, model, loss_fn, optimizer, batch_size, device ):
        # Start the train of the model
        model.train()
        
        if self.light_noise_sigma > 0:
            # If there is noise, add it to x_train array (only the light direction)
            light_noise = np.random.randn( x_train.shape[0], 2 ) * self.light_noise_sigma
            x_train[:,-2:] += light_noise
        
        # Get the batches of the train
        n_batches = x_train.shape[0] // batch_size
        
        # ... and create the permutation over the number of batches
        perm_batches = np.random.permutation( n_batches-1 )
        
        loss_sum = 0.0

        # Now, loop over the permutations
        tq = tqdm(perm_batches, desc="Training progress", unit="batch", position = 0, leave = True )    
        for ii, batch in enumerate(tq):
            
            # A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
            # For each train set, get a subset of it (batch*batch_size:(batch+1)*batch_size) and create the tensor
            X = torch.tensor(x_train[ batch*batch_size:(batch+1)*batch_size, ... ])
            y = torch.tensor(y_train[ batch*batch_size:(batch+1)*batch_size, ... ])
            
            # Then convert the tensor based on the device used (CPU or CUDA)
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Sum the loss for each batch
            loss_sum += loss.item()
            
            tq.set_description("avg loss: %2.4f "%(loss_sum/(ii+1)))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Then calculate the average loss and return it
        loss_avg = loss_sum / n_batches
        return loss_avg
    

    def do_train(self, x_train, y_train, model, loss_fn, batch_size, learning_rate, epochs, device):
        
        train_losses = []

        # Initialise Adam optimiser, passing the parameters of the model and the input learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # For each epoch
        for t in range(epochs):

            print("Epoch %d"%t, end="")
            
            # Train the epoch and return the average loss of the training
            loss = self.train_one_epoch(x_train, y_train, model, loss_fn, optimizer, batch_size, device)
            
            print("Train loss: %f"%loss)
            
            train_losses.append(loss)

        print("Done!")
        return train_losses
    
    def executeTraining(self):        
        # This var is the learning rate for Adam optimiser (10^-3 for first epoch and 10^-4 for second epoch)
        lr = 1e-3
        # Get device
        device = getDevice()
                
        # create model
        B = np.random.randn(2, self.light_freqs) * self.sigma_light

        # Initalise the Neural Model
        model = PixelEncoder(self.x_train.shape[1], B).to(device)
        
        # Creates a criterion that measures the mean absolute error (MAE) between each element in the input X and target Y
        loss_fn = nn.L1Loss().to(device)
        
        # Set Neural Model to work with specific device (CPU or CUDA)
        model.to(device)
        
        # Start the train for the first epoch
        train1 = self.do_train(self.x_train, self.y_train, model, loss_fn, self.batch_size, lr, self.epochs, device)

        # Start the train for the second epoch
        train2 = self.do_train(self.x_train, self.y_train, model, loss_fn, self.batch_size, lr*1e-1, 10, device)
        
        # Build the optput dir to store the weights
        
        MODEL_PATH = self.outDir + "pixel-weights.pt"
        
        # Store them
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved in ", MODEL_PATH)  
        
    def showNNResults(self):
        # Get proj pixels file
        proj_pixels_pca_file = self.trainDataDir + "/proj_pixels_pca.npy"
        self.proj_pixels = np.load(proj_pixels_pca_file)
        
        # Get model weights file
        model_weights = self.outDir + "/pixel-weights.pt"
        
        B = np.zeros((2, 10), dtype=np.float32)

        self.model = PixelEncoder( self.proj_pixels.shape[2]+2, B )
        self.model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
        self.model.to(getDevice())
        
        # Now start to plot the light
        center_x = center_y = DEFAULT_SQUARE_SIZE // 2
        radius = DEFAULT_SQUARE_SIZE // 2    
        
        while(True):
            # Draw plot image
            relightPlot = np.zeros((DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE, 3), dtype=np.uint8)
            
            # Draw the circle border
            cv.circle(relightPlot, (center_x, center_y), radius, (255, 255, 255), 1)
            cv.line(relightPlot, (0, center_y), (DEFAULT_SQUARE_SIZE, center_y), (255, 255, 255), 1)
            cv.line(relightPlot, (center_x, 0), (center_x, DEFAULT_SQUARE_SIZE), (255, 255, 255), 1)
            
            cv.imshow("Relight plot", relightPlot)
            cv.setMouseCallback("Relight plot", self.calculateRelightingFrame)
            
            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                return

    def calculateRelightingFrame(self, event, x, y, flags, params):
        if event == cv.EVENT_MOUSEMOVE:
            light = np.array([[normaliseCoordinate(x, 400), normaliseCoordinate(y, 400)]])
            # Generate test set output for comparison
            images = predictRelight(self.model, light, self.proj_pixels)
            
            Y_img = images[0,:,:].astype(np.uint8)
            
            cv.imshow("Output: ", Y_img)