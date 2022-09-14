# Standard libraries
import itertools
from cv2 import dct
import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Local
import khliao_utils


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)
    

class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor =  nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float() )
        
    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result

def zigzag(n:int) -> torch.Tensor:
    """Generates a zigzag position encoding tensor. 
    Source: https://stackoverflow.com/questions/15201395/zig-zag-scan-an-n-x-n-array
    """

    pattern = torch.zeros(n, n)
    triangle = lambda x: (x*(x+1))/2

    # even index sums
    for y in range(0, n):
        for x in range(y%2, n-y, 2):
            pattern[y, x] = triangle(x + y + 1) - x - 1

    # odd index sums
    for y in range(0, n):
        for x in range((y+1)%2, n-y, 2):
            pattern[y, x] = triangle(x + y + 1) - y - 1

    # bottom right triangle
    for y in range(n-1, -1, -1):
        for x in range(n-1, -1+(n-y), -1):
            pattern[y, x] = n*n-1 - pattern[n-y-1, n-x-1]

    return pattern.t().contiguous()

class block_dct(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
    Ouput:
        DCT coefficients : batch x 192 x height/8 x width/8)
    """
    def __init__(self, dataset='MNIST' ,rounding=torch.round, factor=0 ,cut_off = 0):
        super(block_dct, self).__init__()
        
        self.layer = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.N = 8
        zigzag_vector = zigzag(self.N).view(self.N**2).to(torch.long) # N²
        self.register_buffer('zigzag_weight', F.one_hot(zigzag_vector).to(torch.float).inverse()[:,:,None,None])
        if dataset == 'Cifar10': self.height = int(96/8)
        elif dataset == 'ImageNet': self.height = int(256/8) 
        elif dataset == 'Flower102': self.height = int(320/8)  
        elif dataset == 'MNIST': self.height = int(32/8)  


    def forward(self, image):
        # print('len(image.size()):', len(image.size()))
        # print('(image.size()):', (image.size()))
        if len(image.size())!= 4:
            image = image.unsqueeze(0)
            B = 1
        else :
            B,dim,_,_ = image.size()

        img = (image*255).permute(0,2,3,1)

        if(dim == 3):
            components = {'r': img[:,:,:,0], 'g': img[:,:,:,1], 'b': img[:,:,:,2]}
        else:
            components = {'value': img[:,:,:,0]}

        for k in components.keys():
            comp = self.layer(components[k])
            components[k] = comp

        if(dim == 3):
            dct_coff = torch.stack((components['r'],components['g'],components['b']))
            dct_coff = torch.permute(dct_coff , (1,2,3,4,0)) 
            dct_coff = torch.reshape(dct_coff , (B,self.height**2,192)) 
            dct_coff = torch.permute(dct_coff , (0,2,1))
            dct_coff = torch.reshape(dct_coff, (B,192,self.height,self.height))
        else:
            # dct_coff = torch.stack((components['value']))
            dct_coff = torch.permute(components['value'], (1,2,3,0)) 
            dct_coff = torch.reshape(dct_coff , (B,self.height**2,64)) 
            dct_coff = torch.permute(dct_coff , (0,2,1))
            dct_coff = torch.reshape(dct_coff, (B,64,self.height,self.height))

        return dct_coff