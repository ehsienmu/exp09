import torch
from torchvision.datasets import mnist
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm
import os
from mycnn import CNN

from var_func import cal_cnnlayer_var, cal_dct_var
from khliao_dct import block_dct

import pickle

### load dataset
resize_tfm = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])
train_dataset = CIFAR10(root='./train', train=True, transform=resize_tfm, download=True)
test_dataset = CIFAR10(root='./test', train=False, transform=resize_tfm, download=True)
