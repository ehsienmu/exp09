import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from mycnn import CNN

from khliao_dct import block_dct
import random

def cal_esr(dataset, eps = .3, cal_class = False):
    dataLoader = DataLoader(dataset, batch_size=len(dataset))
    count = 0
    dataset_len = 0
    eps = 0.3
    for batch_idx, ( data, label,) in enumerate(dataLoader):
        dataset_len += label.shape[0]
        for i in range(label.shape[0]):
            for j in range(label.shape[0]):
                if i != j:
                    if cal_class == False or (label[i] != label[j]):
                        if(torch.max(torch.abs(data[i][0] - data[j][0])) > eps):
                            count += 1
                            break
    # print(count, dataset_len)
    ESR = count/dataset_len
    return ESR


def cal_cnnlayer_esr(dataset, model, lyr, eps = .3, subset_size_x = None, subset_size_y = None, attack_iter = 100, cal_class = False):
    dataLoader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if subset_size_x != None:
        ### prepare subset because it need huge time
        subset_index = list(range(subset_size_x))
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_x_dataLoader = DataLoader(subset_dataset, batch_size=1)
    else:
        subset_x_dataLoader = dataLoader
        
    if subset_size_y != None:
        ### prepare subset because it need huge time
        random.seed(1208)
        randomlist = random.sample(range(len(dataset)), subset_size_y)
        subset_index = randomlist
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_y_dataLoader = DataLoader(subset_dataset, batch_size=1)
    else:
        subset_y_dataLoader = dataLoader
        

    non_sep_count = 0
    dataset_len = 0
    for batch_idx, ( data, label,) in enumerate(tqdm(subset_x_dataLoader)):
        dataset_len += label.shape[0]
        print(f'cnn layer: {lyr}', f'batch index: {batch_idx},', f'current ESR = {(dataset_len-non_sep_count)/dataset_len:.4f}' )
        for batch_idx_y, ( data_y, label_y,) in enumerate(tqdm(subset_y_dataLoader)):
            if (batch_idx != batch_idx_y):
                if (cal_class == False) or (label != label_y):
                    # print(batch_idx, batch_idx_y)
                    x = data.to(device)
                    y = data_y.to(device)
                    # //gradient descent
                    # noise = eps * torch.randn_like(x) # init_noise
                    adv_img = x.detach().clone()
                    mod_y_lout = model.lout(y, lyr)
                    # // noise has gradient...
                    for k in range(attack_iter):
                        adv_img.requires_grad = True

                        loss = torch.max(torch.abs(mod_y_lout - model.lout(adv_img, lyr)))
                        loss.backward(retain_graph=True)

                        adv_img = adv_img + eps * adv_img.grad.detach().sign()
                        adv_img = adv_img.detach()
                        # noise = adv_img - x
                        # torch.clamp(noise, min = -eps, max = eps)
                    # print(torch.max(torch.abs(model.lout(y, lyr) - model.lout(adv_img, lyr))))
                    if torch.max(torch.abs(mod_y_lout - model.lout(adv_img, lyr))) == 0:
                        non_sep_count += 1
                        break
    print('dataset_len:', dataset_len)
    lyr_ESR = (dataset_len - non_sep_count) / dataset_len
    print(f'ESR of layer {lyr} ', f' : esr = {lyr_ESR:.4f}' )
    return lyr_ESR


def cal_dct_esr(dataset, model, dct_ind, eps = .3, subset_size_x = None, subset_size_y = None, attack_iter = 100, cal_class = False):
    dataLoader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if subset_size_x != None:
        ### prepare subset because it need huge time
        subset_index = list(range(subset_size_x))
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_x_dataLoader = DataLoader(subset_dataset, batch_size=1)
    else:
        subset_x_dataLoader = dataLoader
        
    if subset_size_y != None:
        ### prepare subset because it need huge time
        random.seed(1208)
        randomlist = random.sample(range(len(dataset)), subset_size_y)
        subset_index = randomlist
        subset_dataset = torch.utils.data.Subset(dataset, subset_index)
        subset_y_dataLoader = DataLoader(subset_dataset, batch_size=1)
    else:
        subset_y_dataLoader = dataLoader
        

    non_sep_count = 0
    dataset_len = 0
    for batch_idx, ( data, label,) in enumerate(tqdm(subset_x_dataLoader)):
        dataset_len += label.shape[0]
        print(f'dct[{dct_ind}],', f'batch index: {batch_idx},', f'current ESR = {(dataset_len-non_sep_count)/dataset_len:.4f}' )
        for batch_idx_y, ( data_y, label_y,) in enumerate(tqdm(subset_y_dataLoader)):
            if (batch_idx != batch_idx_y):
                if (cal_class == False) or (label != label_y):
                    # print(batch_idx, batch_idx_y)
                    x = data.to(device)
                    y = data_y.to(device)
                    # //gradient descent
                    # noise = eps * torch.randn_like(x) # init_noise
                    adv_img = x.detach().clone()
                    mod_y_output_dct = model(y)[0, dct_ind]
                    # // noise has gradient...
                    for k in range(attack_iter):
                        adv_img.requires_grad = True

                        loss = torch.max(torch.abs(mod_y_output_dct - model(adv_img)[0, dct_ind]))
                        loss.backward(retain_graph=True)

                        adv_img = adv_img + eps * adv_img.grad.detach().sign()
                        adv_img = adv_img.detach()
                        # noise = adv_img - x
                        # torch.clamp(noise, min = -eps, max = eps)
                    # print(torch.max(torch.abs(model.lout(y, lyr) - model.lout(adv_img, lyr))))
                    if torch.max(torch.abs(mod_y_output_dct - model(adv_img)[0, dct_ind])) == 0:
                        non_sep_count += 1
                        break
    # print('dataset_len:', dataset_len)
    lyr_ESR = (dataset_len - non_sep_count) / dataset_len
    # print(f'ESR of dct[{dct_ind}]', f' : esr = {lyr_ESR:.4f}' )
    return lyr_ESR



# ### load dataset
# resize_tfm = transforms.Compose([
#     transforms.Resize([32, 32]),
#     transforms.ToTensor(),
# ])
# train_dataset = mnist.MNIST(root='./train', train=True, transform=resize_tfm, download=True)
# test_dataset = mnist.MNIST(root='./test', train=False, transform=resize_tfm, download=True)

# # ### calculate esr, class esr
# # all_ESR = cal_esr(test_dataset)
# # print('all_ESR:', all_ESR)
# # class_ESR = cal_esr(test_dataset, cal_class = True)
# # print('class_ESR:', class_ESR)



# # ### repeat ESR, classESR measure for each layer

# #####  load 5 layers cnn model
# model_path = os.path.join('./save_dir', 'cnn_5_layers', 'best_model.pt')
# model = CNN()
# model.load_state_dict(torch.load(model_path))
# # print(model)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)


# # lyr = 1
# # cnn_lyr_all_esr = cal_cnnlayer_esr(test_dataset, model, lyr = 1, subset_size_x = 10)
# # print('cnn layer', lyr , 'class_esr:', cnn_lyr_all_esr)

# lyr = 1
# cnn_lyr_class_esr = cal_cnnlayer_esr(test_dataset, model, lyr = lyr, subset_size_x = 10, cal_class = True)
# print('cnn layer', lyr , 'class_esr:', cnn_lyr_class_esr)

# dct_index = 1
# dct_class_esr = cal_dct_esr(test_dataset, model, dct_ind = dct_index, subset_size_x = 10, cal_class = True)
# print('dct index', dct_index , 'class_。esr:', cnn_lyr1_class_esr)