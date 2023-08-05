import torch
from torch import nn
import numpy as np

class PixelEncoder(nn.Module):
    
    def __init__(self, input_size, B_light ):
        
        super(PixelEncoder, self).__init__()        

        light_freq = B_light.shape[1]
        self.register_buffer('B_light', torch.tensor(B_light.astype(np.float32)), persistent=True)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size-2+light_freq*2, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
        
        
    def forward(self, x):       
        
        x_light = 6.283185 * (x[:,-2:] @ self.B_light)
        x_light = torch.cat( [ torch.cos(x_light), torch.sin(x_light)], dim=-1)
        
        x = torch.cat([x[:,:-2], x_light], dim=-1)
        out = self.linear_relu_stack(x)
        return out