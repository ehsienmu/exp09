# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
import utils

class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self):
        super(block_merging, self).__init__()
        
    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height//k, width//k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class block_idct(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """
    def __init__(self, dataset='ImageNet', rounding=torch.round, factor=1):
        super(block_idct, self).__init__()

        self.idct = idct_8x8()
        self.merging = block_merging()
        
        if dataset == 'Cifar10': self.height = self.width = 96
        elif dataset == 'ImageNet': self.height =self.width =256 
        elif dataset == 'Flower102': self.height =self.width =320  
        
    def forward(self, dct_inputs):
        B,_,_,_ = dct_inputs.size()
        dct_inputs = torch.reshape(dct_inputs , (B ,192,int(self.height/8)*int(self.height/8)))
        dct_inputs = torch.permute(dct_inputs , (0,2,1))
        dct_inputs = torch.reshape(dct_inputs , (B ,int(self.height/8)*int(self.height/8),8,8,3))
        dct_inputs = torch.permute(dct_inputs , (4,0,1,2,3))
        components = {'r': dct_inputs[0] , 'g':dct_inputs[1] , 'b':dct_inputs[2]}
        for k in components.keys():
            comp = components[k]  
            height, width = self.height, self.width                
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
        image = torch.stack((components['r'] , components['g'] , components['b']))
        image = torch.permute(image , (1,0,2,3))
        image = torch.min(255*torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image/255