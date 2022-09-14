import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from mycnn import CNN

from khliao_dct import block_dct
import random

def cal_cnnlayer_var(dataset, model, lyr, subset_size_x = 1000, subset_size_y = None, flatten = True):
    batch_size = 10000
    dataLoader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if subset_size_x != None:
        ### prepare subset because it need huge time
        subset_index = list(range(subset_size_x))
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_x_dataLoader = DataLoader(subset_dataset, batch_size=batch_size)
    else:
        subset_x_dataLoader = dataLoader
        
    if subset_size_y != None:
        ### prepare subset because it need huge time
        random.seed(1208)
        randomlist = random.sample(range(len(dataset)), subset_size_y)
        subset_index = randomlist
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_y_dataLoader = DataLoader(subset_dataset, batch_size=batch_size)
    else:
        subset_y_dataLoader = dataLoader
        
    dataset_len = 0
    for batch_idx, ( data, label,) in enumerate(tqdm(subset_x_dataLoader)):
        print('data.shape:', data.shape)
        dataset_len += label.shape[0]
        # # print(f'cnn layer: {lyr}', f'batch index: {batch_idx},', f'current ESR = {(dataset_len-non_sep_count)/dataset_len:.4f}' )
        data_device = data.to(device)
        x = model.lout(data_device, lyr)
        print('x.shape:', x.shape)
        flatten_x = x.view(x.shape[0], x.shape[1], 1, -1,)
        permute_x = torch.permute(flatten_x, (1, 0, 2, 3))
        print('permute_x.shape:', permute_x.size())
        var_x = torch.var(permute_x, axis = 1)
        print('torch.var() size:', var_x.size()) # is axis = 1?
        if flatten == False:
            var_x = var_x.view(var_x.shape[0], x.shape[2], x.shape[3]).size()
            print(var_x.size())
        # ssss
    return var_x

def cal_dct_var(dataset, model, dct_ind, subset_size_x = 1000, subset_size_y = None, flatten = True):
    batch_size = 10000
    dataLoader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if subset_size_x != None:
        ### prepare subset because it need huge time
        subset_index = list(range(subset_size_x))
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_x_dataLoader = DataLoader(subset_dataset, batch_size=batch_size)
    else:
        subset_x_dataLoader = dataLoader
        
    if subset_size_y != None:
        ### prepare subset because it need huge time
        random.seed(1208)
        randomlist = random.sample(range(len(dataset)), subset_size_y)
        subset_index = randomlist
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_y_dataLoader = DataLoader(subset_dataset, batch_size=batch_size)
    else:
        subset_y_dataLoader = dataLoader
        
    dataset_len = 0
    for batch_idx, ( data, label,) in enumerate(tqdm(subset_x_dataLoader)):
        # print('data.shape:', data.shape)
        dataset_len += label.shape[0]
        # # print(f'cnn layer: {lyr}', f'batch index: {batch_idx},', f'current ESR = {(dataset_len-non_sep_count)/dataset_len:.4f}' )
        data_device = data.to(device)
        # print('model(data_device).size():', model(data_device).size())
        # x = model(data_device)[:, 3*dct_ind:3*(dct_ind+1)]#,[:, dct_ind]
        # print('x.shape:', x.shape)
        x = model(data_device)[:, dct_ind]
        # print('x.shape:', x.shape)
        flatten_x = x.view(x.shape[0], 1, -1,)
        # print('flatten_x.shape:', flatten_x.size())
        # print(torch.flatten(x, start_dim=1).size())
        permute_x = torch.permute(flatten_x, (1, 0, 2))
        # print('permute_x.shape:', permute_x.size())
        var_x = torch.var(permute_x, axis = 1)
        # print('torch.var() size:', var_x.size()) # is axis = 1?
        # gg
        if flatten == False:
            var_x = var_x.view(x.shape[1], x.shape[2]).size()
            print(var_x.size())
        # asd
    return torch.mean(var_x)