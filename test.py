import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm
import os
from mycnn import CNN

from sep_func import cal_esr, cal_cnnlayer_esr, cal_dct_esr
from khliao_dct import block_dct

import pickle

### load dataset
resize_tfm = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])
train_dataset = mnist.MNIST(root='./train', train=True, transform=resize_tfm, download=True)
test_dataset = mnist.MNIST(root='./test', train=False, transform=resize_tfm, download=True)

# ### calculate esr, class esr
# all_ESR = cal_esr(test_dataset)
# print('all_ESR:', all_ESR)
# class_ESR = cal_esr(test_dataset, cal_class = True)
# print('class_ESR:', class_ESR)



# ### repeat ESR, classESR measure for each layer

# #####  load 5 layers cnn model
# model_path = os.path.join('./save_dir', 'cnn_5_layers', 'best_model.pt')
# model = CNN()
# model.load_state_dict(torch.load(model_path))
# # print(model)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# sub_size_x = 20
# sub_size_y = 1000

# # lyr = 1
# # cnn_lyr_all_esr = cal_cnnlayer_esr(test_dataset, model, lyr = 1, subset_size_x = sub_size_x, subset_size_y = sub_size_y)
# # print('cnn layer', lyr , 'class_esr:', cnn_lyr_all_esr)

# cnn_record = {}
# for i in range(1, 6):
#     cnn_lyr = i
#     print('calculate cnn layer', cnn_lyr, 'ESR')
#     cnn_lyr_esr = cal_cnnlayer_esr(test_dataset, model, lyr = cnn_lyr, subset_size_x = sub_size_x, subset_size_y = sub_size_y)
#     print('cnn conv', cnn_lyr , 'ESR:', cnn_lyr_esr)
#     cnn_record[cnn_lyr] = cnn_lyr_esr
#     with open('./cnn_esr_record.pk', 'wb') as f:
#         pickle.dump(cnn_record, f)

# # lyr = 1
# # cnn_lyr_class_esr = cal_cnnlayer_esr(test_dataset, model, lyr = lyr, subset_size_x = sub_size_x, subset_size_y = sub_size_y, cal_class = True)
# # print('cnn layer', lyr , 'class_esr:', cnn_lyr_class_esr)
# cnn_class_record = {}
# for i in range(1, 6):
#     cnn_lyr = i
#     print('calculate cnn layer', cnn_lyr, 'class ESR')
#     cnn_lyr_class_esr = cal_cnnlayer_esr(test_dataset, model, lyr = cnn_lyr, subset_size_x = sub_size_x, subset_size_y = sub_size_y, cal_class = True)
#     print('cnn conv', cnn_lyr , 'class ESR:', cnn_lyr_class_esr)
#     cnn_class_record[cnn_lyr] = cnn_lyr_class_esr
#     with open('./cnn_class_esr_record.pk', 'wb') as f:
#         pickle.dump(cnn_class_record, f)


# dct_record = {}
# for i in range(1, 64):
#     dct_index = i
#     print('calculate dct sf layer', dct_index)
#     dct_esr = cal_dct_esr(test_dataset, model, dct_ind = dct_index, subset_size_x = sub_size_x, subset_size_y = sub_size_y)
#     print('dct index', dct_index , 'ESR:', dct_esr)
#     dct_record[dct_index] = dct_esr
#     with open('./dct_esr_record.pk', 'wb') as f:
#         pickle.dump(dct_record, f)

model = block_dct()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
sub_size_x = 20
sub_size_y = 1000
dct_class_record = {}
for i in range(1, 64):
    dct_index = i
    print('calculate class dct sf layer', dct_index)
    dct_class_esr = cal_dct_esr(test_dataset, model, dct_ind = dct_index, subset_size_x = sub_size_x, subset_size_y = sub_size_y, cal_class = True)
    print('dct index', dct_index , 'class_esr:', dct_class_esr)
    dct_class_record[dct_index] = dct_class_esr
    with open('./dct_class_esr_record.pk', 'wb') as f:
        pickle.dump(dct_class_record, f)
dct_class_record = {}
for i in range(1, 64):
    dct_index = i
    print('calculate class dct sf layer', dct_index)
    dct_class_esr = cal_dct_esr(test_dataset, model, dct_ind = dct_index, subset_size_x = sub_size_x, subset_size_y = sub_size_y, cal_class = True)
    print('dct index', dct_index , 'class_esr:', dct_class_esr)
    dct_class_record[dct_index] = dct_class_esr
    # with open('./dct_class_esr_record.pk', 'wb') as f:
    #     pickle.dump(dct_class_record, f)