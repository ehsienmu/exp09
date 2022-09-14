import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torchvision import transforms
from tqdm import tqdm
from dataset import ImageFolder
from utils import seed_everything
from model.sf_resnet import *
from model.sf_resnet_cnn import *
from model.sf_efficientnet import *
from model.sf_densenet import *
from model.sf_vgg import *
from dct import block_dct
from torchvision.datasets import CIFAR10

def train(freq_num, args = None, cfg = None):
    """
    Steps :
    1. Basic parameters setting
    2. Prepare dataset
    3. Load Model
    4. Training
    5. Evaluation
    6. Save checkpoint
    """
    # Parameters settings
    # if args.resume_ckpt:
    #     have_resume_ckpt = True
    
    assert (args != None or cfg != None), 'Error: train config/args cannot be empty'
    if(args != None):
        config = {
        "dataset": args.dataset,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "epochs": args.epoch,
        "batch_size": args.batch_size,
        "resume_ckpt" : "checkpoint/{}.pth".format(args.resume_ckpt),
        "have_resume_ckpt": (args.resume_ckpt != None),#have_resume_ckpt,
        "ckpt_path" : "checkpoint/{}.pth".format(args.ckpt_path)
        }
    elif(cfg != None):
        config = cfg
        

    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    start_epoch = 0
    best_acc = 0
    best_loss = 0
    seed = 1234
    seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare Dataset & Model
    print('==> Preparing data..')

    ### load dataset
    new_size = 256
    resize_tfm = transforms.Compose([
        transforms.Resize([new_size, new_size]),
        transforms.ToTensor(),
    ])
    if config['dataset'] == "CIFAR10":
        train_dir = '../train'
        test_dir = '../test'
        train_dataset = CIFAR10(root='../train', train=True, transform=resize_tfm, download=True)
        val_dataset = CIFAR10(root='../test', train=False, transform=resize_tfm, download=True)
        num_of_classes = 10
    elif config['dataset'] == "ImageNette":
        train_dir = 'data/ImageNette/train'
        test_dir = 'data/ImageNette/test'
        train_dataset = ImageFolder(train_dir,transform = resize_tfm,backend = 'opencv')
        val_dataset = ImageFolder(test_dir, transform = resize_tfm ,backend = 'opencv')
        num_of_classes = 10
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    
    if config['model'] == "resnet":
        model = ResNet18(num_classes=num_of_classes)
    elif config['model'] == "resnet_freq":
        model = ResNet18_freq(num_classes=num_of_classes)
    elif config['model'] == "efficientnet":
        model = effnetv2(num_classes=num_of_classes)
    elif config['model'] == "densenet":
        model = DenseNet.from_name("densenet121",override_params={"num_classes": num_of_classes})
    elif config['model'] == 'vgg':
        model = vgg11_bn(num_classes = num_of_classes)



    print("Training dataset PATH: {}".format(train_dir))
    print('Number of Training data:{}'.format(len(train_dataset)))
    print("Validation dataset PATH: {}".format(test_dir))
    print('Number of Validation data:{}'.format(len(val_dataset)))

    # Load checkpoint
    if config['have_resume_ckpt']:
    # if args.resume_ckpt:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        ckpt_path = config['resume_ckpt']
        print('Reload the checkpoint from {}'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        print('Current best val loss : {}'.format(best_loss))
        print('Current best val acc : {}'.format(best_acc))
    else:
        best_loss , best_acc = float("inf") , 0
        start_epoch = 0
    model.to(device)
    print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9,weight_decay=4e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode='min')
    cur_lr = lr

    # Traning Process
    """
    RGB images -> DCT with zigzag pattern -> Frequecy info. -> Training 
    """
    block_dct_model = block_dct(img_height=new_size) 
    block_dct_model.to(device)
    print('==> Start Training..')
    for epoch in range(start_epoch,epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        #Freeze SF-layer 
        model.sf_layer.requires_grad = False
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr ,momentum=0.9)
        for data, label in (tqdm(train_loader)):
            data = data.to(device)
            label = label.to(device)
            # print(block_dct_model(data).size())
            freq = block_dct_model(data)[:, 3*freq_num: 3*(freq_num+1)]
            output = model(data, freq)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for batch_i , (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                freq = block_dct_model(data)[:, 3*freq_num: 3*(freq_num+1)]
                val_output = model(data, freq)
                val_loss = criterion(val_output, label)
                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

        # Dynamically modify the learning rate
        scheduler.step(epoch_val_loss)
        if optimizer.param_groups[0]['lr'] != cur_lr:
            print('learning rate changes from {} to {}'.format(cur_lr, optimizer.param_groups[0]['lr']))
            cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < 1e-5:
            print('early stopped due to small learning rate')
            break

        #Save checkpoint
        if epoch_val_accuracy > best_acc :
            best_loss = epoch_val_loss
            best_acc = epoch_val_accuracy
            print('Saving Checkpoint..')
            state = {
                'model': model.state_dict(),
                'acc': best_acc,
                'loss' : best_loss,
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            ckpt_path = config['ckpt_path']
            torch.save(state, ckpt_path)
            print('Saving Checkpoint at {}'.format(ckpt_path))
        
        print(f"Epoch : {epoch+1} - train_loss : {epoch_loss:.4f} - train_acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frequency Training')
    parser.add_argument('--resume_ckpt', type=str,
                        help='resume from the name of checkpoint')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--epoch', default=50, type=int, help='epoch')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--ckpt_path', default='test', type=str, help='name of the checkoint')
    parser.add_argument('--model', default='resnet', type=str, help='model')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
    args = parser.parse_args()
    freq_num = 0
    print(freq_num)
    train(freq_num=freq_num, args=args)