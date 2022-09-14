import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from tqdm.auto  import tqdm
from dataset import ImageFolder
from utils import seed_everything 
from model.sf_resnet import * 
from model.sf_efficientnet import *
from model.sf_densenet import *
from model.sf_vgg import *
from torch.autograd import Variable
from PIL import Image
from collections import defaultdict
from torchvision.datasets import CIFAR10
from model.sf_resnet_cnn import *

def PGD_attack(freq_num, args = None, cfg = None):
    """
    STEPs:
    1. Prepare pre-trained PEGformer checkpoint
    2. Prepare RGB images dataloader
    3. Use dct_pytorch(DCTCompression) to compress images to DCT frequency
    4. Backpropagate gradient to RGB noise (use dct_numpy to decode)
    5. Add noise to RGB images 
    6. Evaluation on PEGformer to check the accuracy after attack 
    """
    # Parameters settings
    
    assert (args != None or cfg != None), 'Error: train config/args cannot be empty'
    if(args != None):
        config = {
        "dataset": args.dataset,
        "model": args.model,
        "batch_size": args.batch_size ,
        "eps" : args.eps,
        "step_size" : args.step_size,
        "iteration" : args.iteration,
        "ckpt_path" : "checkpoint/{}.pth".format(args.path)
        }
    elif(cfg != None):
        config = cfg
        

    batch_size = config['batch_size']
    seed = 1234
    seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    # Prepare dataset
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



    # Load checkpoint.
    print('==> Load checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpt_path = config['ckpt_path']
    print('Reload the checkpoint from {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path,map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    attack_iters = config['iteration']
    eps = config['eps']
    alpha = config['step_size']
    ASR = Tacc = Aacc = 0
    correct = attacked = total = 0
    model.eval()
    
    block_dct_model = block_dct(img_height=new_size) 
    block_dct_model.to(device)

    print('==> Start Attacking..')
    for batch_i, (imgs, targets) in enumerate(tqdm(val_loader)):
        """
        100-PGD attack
        """
        imgs, targets = imgs.to(device), targets.to(device)
        freq = block_dct_model(imgs)[:, 3*freq_num: 3*(freq_num+1)]
        correct_mask = model(imgs, freq).max(1)[1] == targets
        imgs = imgs[correct_mask]
        targets = targets[correct_mask]
        adv_imgs = imgs.clone()   # the adversarial image
        adv_imgs += eps * torch.randn_like(adv_imgs) # init_noise
        adv_imgs.clamp_(0,1)
        
        # Iteration Attack
        for i in (range(attack_iters)):
            img_var = adv_imgs.clone()
            img_var.requires_grad = True
            
            freq = block_dct_model(img_var)[:, 3*freq_num: 3*(freq_num+1)]
            output = model(img_var, freq)
            loss = criterion(output, targets)
            grad = torch.autograd.grad(loss, img_var,
                                       retain_graph=False, create_graph=False)[0]
            adv_imgs = adv_imgs + alpha*grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-eps, max=eps)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1)

        with torch.no_grad():
            freq = block_dct_model(adv_imgs)[:, 3*freq_num: 3*(freq_num+1)]
            adv_good = model(adv_imgs, freq).max(1)[1] == targets
            correct += float(adv_good.sum())
            attacked += float(len(adv_good))
            total += float(len(correct_mask))
    
        ASR = 1 - correct/attacked
        Tacc = attacked/total
        Aacc = correct/total
        print("Attack Success Rate: {:.2f}, Test Accuracy: {:.2f}, Attacked Accuracy: {:.2f}".format(ASR, Tacc, Aacc))
    return ASR, Tacc, Aacc

if __name__ == "__main__":
    # Argument
    parser = argparse.ArgumentParser(description='PGD attack')
    parser.add_argument('--model', default='resnet', type=str, help='SF-model')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--eps', default=0.01, type=float, help='eps')
    parser.add_argument('--step_size', default=0.003, type=float, help='step size')
    parser.add_argument('--iteration', default=100, type=int, help='attack iteration')
    parser.add_argument('--path', type=str, help='name of checkpoint')
    args = parser.parse_args()
    PGD_attack(args)