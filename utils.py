import torch, h5py, numpy as np

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