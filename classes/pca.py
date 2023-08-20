import h5py
import numpy as np
import os
from sklearn.decomposition import PCA
import tqdm

class HDF5Appender:

    def __init__( self, filename ):
        self.f = h5py.File( filename, "w")
        self.datasets = {}

    def add_dataset( self, dataset_name, shape, dtype=np.float32, chunk_size=32 ):
        shape_array = [ k for k in shape ]
        d = self.f.create_dataset( dataset_name, [0]+shape_array, maxshape=[None]+shape_array, dtype=dtype, chunks= tuple( [chunk_size]+shape_array) )
        self.datasets[dataset_name] = d

    def has_dataset( self, dataset_name ):
        return dataset_name in self.datasets

    def append( self, dataset_name, samples ):
        d = self.datasets[dataset_name]
        if samples.ndim == d.ndim - 1:
            samples = np.expand_dims( samples, axis=0 )

        assert samples.ndim == d.ndim, "Wrong sample shape"

        for ii in range(1,samples.ndim):
            assert d.shape[ii] == samples.shape[ii], "Wrong sample shape"

        d.resize( (d.shape[0] + samples.shape[0]), axis=0 )
        d[-samples.shape[0]:] = samples

    def close( self ):
        self.f.close()


class PCAClass:
    def __init__(self, baseDir, datafile, pcaNumber) -> None:
        # Store input data file
        self.datafile = datafile
        
        # Set the output directory
        self.outputDir = "%s/pca_norm/"%(baseDir)
        # Create it
        os.mkdir(self.outputDir)
                
        # Number of principal components to keep
        self.pcaNumber = pcaNumber
        
        # Pixel and light step number
        self.PX_S = 1 # pixel step
        self.LIGHT_S = 1 # light step
        
        # Normalize in 0..1 the compressed vectors
        self.NORM = True 

    def readDataset(self):
        # Now, read the coin_train file, created in the previous step (which contains the train data and the UVMean)
        with h5py.File(self.datafile,"r") as f:
            # Array with 3 channels: Intensity, LightX, LightY
            data_ = np.array( f["lightdata"] )
            # Array with UV mean for each pixel
            self.UVmean = np.array(f["UVMean"])

            # Store the intensities of the data
            self.data = data_[:,::self.PX_S,::self.PX_S,0]
            # Store the lights of the data
            self.lights = data_[:,0,0,1:]
        
        print("Light data shape: ", self.data.shape)
        print("Lights shape: ", self.lights.shape)
        
    def applyPCA(self):
        # The shape (x, -1) becomes (x, n) since -1 represents the unkown dimension and np calculates the value n
        M = np.reshape(self.data, (self.data.shape[0],-1)).T

        # Calculate the mean of the points long the x-axis
        m = np.mean(M, axis=0, keepdims=True)

        # Then for each point subtract the mean
        M = M - m
        
        # Create the PCA (Principal Component Analysis) with N components to keep (8), svd_solver 'auto' choose automatically the method to use
        pca = PCA(n_components = self.pcaNumber, svd_solver='auto')
        
        # The result then is used to transform the matrix M, containing the data
        M_proj = pca.fit_transform(M)
        
        # normalize values
        if self.NORM:
            max_v = np.max(M_proj)
            min_v = np.min(M_proj)
            M_proj = (M_proj-min_v)/(max_v-min_v)
            print("M[0] normalised: ", M_proj[0])
            
        # After that, save the compressed vector containing the PCA results for each pixel
        output = np.reshape(M_proj, (self.data.shape[1], self.data.shape[2], self.pcaNumber))
        
        # Save the reshaped projected pixels
        np.save(self.outputDir + "proj_pixels_pca.npy", output)
        
        # Build the PCA normalised filename as training for the neural network
        filename = self.outputDir + "train_data_pca.h5"
        
        # Create a matrix with shape of the matrix M projected (containing the 8 principal components)
        x = np.zeros((M_proj.shape[0], M_proj.shape[1]+2), dtype=np.float32)
        
        # Then open the HDF5 file, and append the two dataset: one for X (compressed pixels and light direction) axis and the other for Y (Intensity values).
        F = HDF5Appender(filename)
        F.add_dataset("X", shape=( x.shape[1], ), chunk_size=1024 )
        F.add_dataset("Y", shape=( 1, ), chunk_size=1024 )
                
        # Light indices which goes from 0 to light X shape, with 1 step at the time
        all_light_indices = np.arange(0, self.lights.shape[0], self.LIGHT_S)
                
        for ii in tqdm.trange(len(all_light_indices)):    
            # Take current light index
            light_index = all_light_indices[ii]
            
            # Take light
            x_light = np.expand_dims(self.lights[light_index,:], axis=0)

            # Build X: concatenate compressed pixels and light direction
            # Here we have that M_proj on Y shape has dimension N (number of principal components) + 2
            # Here for each sub position, the first 8 items represent the principal components, while the last two represent the light vector
            x[:,:M_proj.shape[1]] = M_proj
            x[:,M_proj.shape[1]:] = x_light.flatten()

            # Build y: Get intensity values
            y = np.expand_dims(self.data[light_index,:].flatten(), axis=1)
            
            # Append to specific dataset
            F.append("X", x)
            F.append("Y", y)
            
        F.close()