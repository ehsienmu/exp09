
import torch
import os


from torchvision.datasets import mnist
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim 
import torch.nn as nn

from mycnn import CNN
from model_tool import train, fixed_seed
from torchvision import models

# Modify config if you are conducting different models
# from cfg import LeNet_cfg as cfg
# from cfg import myResnet_cfg as cfg
from model_cfg import mycnn_cfg
import argparse

# from prettytable import PrettyTable

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params+=params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
    

def train_interface():
    
    # parser = argparse.ArgumentParser(description='train_interface of hw2_2 main')
    # parser.add_argument('--model', default='mycnn', type=str, help='training model')
    # args = parser.parse_args()


    cfg = mycnn_cfg
    """ input argumnet """

    # data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    early_stop = cfg['early_stop']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs(os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs(os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)



    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    # do_semi = cfg['do_semi']
    
    ## Modify here if you want to change your model ##

    model = CNN()

    # print model's architecture
    print(model)
    print(cfg)
    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 

    resize_tfm = transforms.Compose([
        transforms.Resize([32, 32]),
        # transforms.RandomResizedCrop((128, 128)),
        # transforms.RandomHorizontalFlip(),
        # transforms.AutoAugment(),
        # # transforms.ColorJitter(0.2, 0.2),
        # transforms.RandomAffine(0, None, (0.7, 1.3)),
        transforms.ToTensor(),
    ])


    train_dataset = mnist.MNIST(root='./train', train=True, transform=resize_tfm, download=True)
    test_dataset = mnist.MNIST(root='./test', train=False, transform=resize_tfm, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

      
    # train_set, val_set = get_cifar10_train_val_set(root=data_root, ratio=split_ratio)    
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


    # define your loss function and optimizer to unpdate the model's parameters.
    optimizer = getattr(torch.optim, cfg['optimizer'])(model.parameters(), **cfg['optim_hparas'])
    ### optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
    ### optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # 沒進步就5個epoch下降 #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, cooldown=3)

    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    # count_parameters(model)
    train(model=model, train_loader=train_loader, val_loader=test_loader, 
          num_epoch=num_epoch, early_stop=early_stop, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, batch_size = batch_size)

    
if __name__ == '__main__':
    train_interface()