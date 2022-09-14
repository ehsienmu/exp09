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
from dct import block_dct

import pickle

new_img_height = 32
### load dataset
resize_tfm = transforms.Compose([
    transforms.Resize([new_img_height, new_img_height]),
    transforms.ToTensor(),
])
train_dataset = CIFAR10(root='./train', train=True, transform=resize_tfm, download=True)
test_dataset = CIFAR10(root='./test', train=False, transform=resize_tfm, download=True)

# ##### 1. Calculate var (CNN layers) #####

# # ####  load 5 layers cnn model
# model_path = os.path.join('./save_dir', 'cnn_5_layers', 'best_model.pt')
# model = CNN()
# model.load_state_dict(torch.load(model_path))
# # print(model)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# cnn_record = {}
# for i in range(1, 6):
#     cnn_lyr = i
#     print('calculate cnn var', cnn_lyr)
#     cnn_lyr_var = cal_cnnlayer_var(test_dataset, model, lyr = cnn_lyr)
#     cnn_record[cnn_lyr] = cnn_lyr_var
#     with open('./cnn_var_record.pk', 'wb') as f:
#         pickle.dump(cnn_record, f)

#### 3. Calculate var (DCT) #####

model = block_dct(img_height = new_img_height)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
dct_record = {}
for i in range(64):
    dct_index = i
    print('calculate dct sf layer', dct_index)
    dct_var = cal_dct_var(test_dataset, model, dct_ind = dct_index)
    dct_record[dct_index] = dct_var
    with open('./dct_var_record.pk', 'wb') as f:
        pickle.dump(dct_record, f)


for key in dct_record.keys():
    print('freq:',key,dct_record[key])