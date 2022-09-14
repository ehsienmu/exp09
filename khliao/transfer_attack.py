import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from tqdm import tqdm
from dataset import ImageFolder
from utils import seed_everything
import torchvision.models as models
from model.sf_resnet import * 
from model.sf_vgg import * 
from model.sf_efficientnet import *
from model.sf_densenet import *
from model.vgg import vgg11_bn as rgb_vgg11_bn


def PGD_attack(args):
    """
    STEPs:
    1. Prepare pre-trained VGG checkpoint 
    2. Prepare RGB images dataloader
    3. Calaculate sf-CNNs correct mask
    4. Apply 100-PGD attack on VGG model to generate adversarial images
    5. Utilize these adversarial images for evaluation of sfCNNs 
    """
    # Parameters settings
    config = {
    "batch_size": args.batch_size ,
    "eps" : args.eps,
    "step_size" : args.step_size,
    "iteration" : args.iteration,
    "cnn_ckpt" : "checkpoint/{}.pth".format(args.cnn_path),
    "sf_ckpt" : "checkpoint/{}.pth".format(args.sf_path)
    }

    batch_size = config['batch_size']
    seed = 1234
    seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare dataset
    print('==> Preparing data..')
    test_dir = 'data/ImageNette/test' 

    transform_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    val_dataset = ImageFolder(test_dir, transform = transform_test ,backend = 'opencv')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

    print('Number of Validation data:{}'.format(len(val_dataset)))

    #Prepare SF-Model
    if args.model == "resnet":
        sf_model = ResNet18(num_classes=10)
    elif args.model == "efficientnet":
        sf_model = effnetv2(num_classes=10)
    elif args.model == "densenet":
        sf_model = DenseNet.from_name("densenet121",override_params={"num_classes": 10})
    elif args.model == 'vgg':
        sf_model = vgg11_bn(num_classes = 10)
    # Load SF-CNN checkpoint.
    print('==> Load  SF-model checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpt_path = config['sf_ckpt']
    print('Reload the checkpoint from {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    sf_model.load_state_dict(checkpoint['model'])
    sf_model  = sf_model.to(device)
    sf_model.eval()

    # Load CNN model checkpoint.
    cnn_model = rgb_vgg11_bn(num_classes = 10)
    print('==> Load CNN model checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpt_path = config['cnn_ckpt']
    print('Reload the checkpoint from {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    cnn_model.load_state_dict(checkpoint['model'])
    cnn_model.to(device)
    cnn_model.eval()

    # loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluation Metric
    attack_iters = config['iteration']
    eps = config['eps']
    alpha = config['step_size']
    ASR = Tacc = Aacc = 0
    correct = attacked = total = 0
    print('==> Start Attacking..')
    for batch_i, (imgs, targets) in enumerate(tqdm(val_loader)):
        """
        1. sfCNN generate correct mask
        2. 100-PGD attack on Normal VGG to generate adversarial images
        3. sfCNN test on these adversarial images
        """
        imgs, targets = imgs.to(device), targets.to(device)
        correct_mask = sf_model(imgs).max(1)[1] == targets
        imgs = imgs[correct_mask]
        targets = targets[correct_mask]
        adv_imgs = imgs.clone() # the adversarial image
        adv_imgs += eps * torch.randn_like(adv_imgs) # init_noise
        adv_imgs.clamp_(0,1)
        if len(adv_imgs) == 0:
            total += float(len(correct_mask))
            continue
        # 100-PGD Attack
        for i in tqdm(range(attack_iters)):
            img_var = adv_imgs.clone()
            img_var.requires_grad = True
            output = cnn_model(img_var)
            loss = criterion(output, targets)
            cnn_model.zero_grad()
            loss.backward()
            gradient = img_var.grad.data
            adv_imgs += alpha * gradient.sign()
    
            # norm-bounded
            adv_imgs = torch.where(adv_imgs < imgs-eps, imgs-eps, adv_imgs)
            adv_imgs = torch.where(adv_imgs > imgs+eps, imgs+eps, adv_imgs)
            adv_imgs = adv_imgs.clamp(0,1) 
        with torch.no_grad():
            adv_good = sf_model(adv_imgs).max(1)[1] == targets
            correct += float(adv_good.sum())
            attacked += float(len(adv_good))
            total += float(len(correct_mask))

    ASR = 1 - correct/attacked
    Tacc = attacked/total
    Aacc = correct/total
    print("Attack Success Rate: {}, Test Accuracy: {}, Attacked Accuracy: {}".format(ASR, Tacc, Aacc))
        
if __name__ == "__main__":
    # Argument
    parser = argparse.ArgumentParser(description='Transfer attack')
    parser.add_argument('--model', default='resnet', type=str, help='SF-model')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--eps', default=0.01, type=float, help='eps')
    parser.add_argument('--step_size', default=0.003, type=float, help='step size')
    parser.add_argument('--iteration', default=100, type=int, help='attack iteration')
    parser.add_argument('--cnn_path', type=str, help='name of CNN checkpoint')
    parser.add_argument('--sf_path', type=str, help='name of sfCNN checkpoint')
    args = parser.parse_args()
    PGD_attack(args)