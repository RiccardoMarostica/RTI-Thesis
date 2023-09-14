import torch, h5py, numpy as np, cv2 as cv
from constants import *

from classes.video import Video

def getDevice():
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def storeTrainDataset(fileName, lightData, UVMean):
    f = h5py.File(fileName, "w")
    # ... and create datasets
    f.create_dataset("lightdata", lightData.shape, data=lightData)
    f.create_dataset("UVMean", UVMean.shape, data=UVMean)
    # Then stop writing
    f.close()
    
def getUVMean(datafile):
    # Now, read the coin_train file, created in the previous step (which contains the train data and the UVMean)
    with h5py.File(datafile,"r") as f:
        # Array with UV mean for each pixel
        return np.array(f["UVMean"])

def predictRelight(model, L, proj_pixels, device = None):
    if device is None:
        device = getDevice()
        
    # Get width and height of frames
    W = proj_pixels.shape[1]
    H = proj_pixels.shape[0]
    
    # Get the number of lights stored
    n_lights = L.shape[0]

    # Create an output array with dimension (N, H, W)
    out = np.zeros((n_lights, H, W))
    
    # Reshape the projeected pixels
    proj_pixels = np.reshape(proj_pixels, (W*H,-1))
        
    for i in range(n_lights):
        # Now, for each light, project it
        L_proj = np.expand_dims(L[i,:], axis=0)
            
        # np.tile repeats the light along all the projected pixels (they have same shape)
        # Then concatenate the two to obtain an array with shape (N * N, 10)
        XX = np.concatenate([proj_pixels, np.tile(L_proj, (proj_pixels.shape[0],1) )], axis=1).astype(np.float32)
                
        # Get our array and convert it into a tensor, for specific device (CPU or CUDA)
        # A tensor is a multi-dimensional matrix containing elements of a single data type
        # The tensor is (N * N, 10), where the first 8 elements represent the principal components and the last 2 represent the light
        xin = torch.tensor(XX).to(device)
        
        # Take the model and pass the tensor. Then detach the network, store it in the CPU and convert it as a np NDArray
        # Given the tensor, the result is a prediction of the intensity for each light case
        YY_pred = model(xin).detach().cpu().numpy()
        # Convert it as a new W * H array -> The frame intensity
        Y_img = np.reshape(YY_pred, (W,H,-1))
        # Convert values in the range between 0 and 255
        Y_img = np.squeeze(np.clip(Y_img, 0.0, 255.0))
        
        # Then for each position, store the predicted image
        out[i,...] = Y_img        

    return out

def getVideoOrientation(video: Video):
    return 'portrait' if video.getWidth() < video.getHeight() else 'landscape'

def videoNeedsResize(video: Video, defaultSize: (int, int)):
    if video.getWidth() > defaultSize[0] and video.getHeight() > defaultSize[1]:
        return True
    return False

def normaliseCoordinate(value : float, dim: int) -> float:
        # Convert the coordinates to normalized values between -1 and 1
        return (value / dim) * 2 - 1

def getCoordinateFromNormalised(value: float, dim: int) -> int:
    return int(((value + 1) * dim) / 2)


def getRelightingPlot(dim: int):
    
    center_x = center_y = dim // 2
    radius = dim // 2    
    
    # Draw plot image
    relightPlot = np.zeros((dim, dim, 3), dtype=np.uint8)
    
    # Draw the circle border
    cv.circle(relightPlot, (center_x, center_y), radius, (255, 255, 255), 1)
    cv.line(relightPlot, (0, center_y), (dim, center_y), (255, 255, 255), 1)
    cv.line(relightPlot, (center_x, 0), (center_x, dim), (255, 255, 255), 1)
    
    return relightPlot

def getLightDirectionPlot(light, dim):
        """The function shows light direction inside a plot.

        Args:
            light_direction (Any): Light direction in normalised coordinates.

        Returns:
            Any: An image representing a plot of the light direction.
            """
        center_x = center_y = dim // 2
        
        # Create a blank image
        image = getRelightingPlot(dim)
        
        if light is not None:
            x = getCoordinateFromNormalised(light[0][0], dim)
            y = getCoordinateFromNormalised(light[1][0], dim)
            cv.circle(image, (int(x), int(y)), 10, (0, 255, 0), 2)
            cv.line(image, (center_x, center_y), (int(x), int(y)), (0, 255, 0), 2) 
        return image