import os
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from tqdm import tqdm
from dataset import ImageFolder
from utils import seed_everything
from dct import block_dct
from idct import block_idct
from model.sf_resnet import *
from model.sf_efficientnet import *
from model.sf_densenet import *
from model.sf_vgg import *

def PGD_attack(args):
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
    config = {
    "batch_size": args.batch_size ,
    "eps" : args.eps,
    "step_size" : args.step_size,
    "iteration" : args.iteration,
    "ckpt_path" : "checkpoint/{}.pth".format(args.path)
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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
    print('Number of Validation data:{}'.format(len(val_dataset)))

    #Prepare Model
    if args.model == "resnet":
        model = ResNet18(num_classes=10)
    elif args.model == "efficientnet":
        model = effnetv2(num_classes=10)
    elif args.model == "densenet":
        model = DenseNet.from_name("densenet121",override_params={"num_classes": 10})
    elif args.model == 'vgg':
        model = vgg11_bn(num_classes = 10)

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

    # DCT & iDCT Transform module 
    dct = block_dct()
    dct = dct.to(device)
    idct = block_idct()
    idct = idct.to(device)

    print('==> Start Attacking..')
    attack_iters = config['iteration']
    eps = config['eps']
    alpha = config['step_size']
 
    ASR = Tacc = Aacc = 0
    correct = attacked = total = 0
    model.eval()
    
    for batch_i, (imgs, targets) in enumerate(tqdm(val_loader)):
        """
        1. Utilize iDCT to transform DCT coefficients to pixel values.
        2. Since the range of DCT coefficients is from -1024 to 1023, we would enlarge norm-bound to 2048 times.
        """
        with torch.no_grad():
            # RGB image -> DCT frequency
            imgs, targets = imgs.to(device), targets.to(device)
            dct_coef = dct(imgs)
            correct_mask = model(imgs).max(1)[1] == targets
            imgs = imgs[correct_mask]
            dct_coef = dct_coef[correct_mask]
            targets = targets[correct_mask]
            adv_imgs = dct_coef.clone()     
            if len(adv_imgs) ==0 :
                total += float(len(correct_mask))
                continue
            # init random noise    
            adv_imgs += torch.empty_like(adv_imgs).uniform_(-eps*2048, eps*2048)
            adv_imgs.clamp_(-1024,1023)
            
            
        # PGD Attack on freqeuncy domain
        for i in tqdm(range(attack_iters)):
            img_var = adv_imgs.clone()
            img_var = Variable(img_var.data , requires_grad = True)
            output = model(idct(img_var))
            loss = criterion(output, targets)
    
            model.zero_grad()
            loss.backward()
            grad = img_var.grad.data
            adv_imgs = adv_imgs + alpha*grad.sign()*2048
            delta = torch.clamp(adv_imgs - dct_coef, min=-eps*2048, max=eps*2048)
            adv_imgs = dct_coef + delta
            adv_imgs.clamp_(-1024,1023)
            adv_imgs = idct(adv_imgs).clamp(0,1)
            adv_imgs = dct(adv_imgs)

        with torch.no_grad():
            adv_good = model(idct(adv_imgs)).max(1)[1] == targets
            correct += float(adv_good.sum())
            attacked += float(len(adv_good))
            total += float(len(correct_mask))
    ASR = 1 - correct/attacked
    Tacc = attacked/total
    Aacc = correct/total
    print("Attack Success Rate: {}, Test Accuracy: {}, Attacked Accuracy: {}".format(ASR, Tacc, Aacc))
        
if __name__ == "__main__":
    # Argument
    parser = argparse.ArgumentParser(description='Resnet PGD attack')
    parser.add_argument('--model', default='resnet', type=str, help='SF-model')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--eps', default=0.01, type=float, help='eps')
    parser.add_argument('--step_size', default=0.003, type=float, help='step size')
    parser.add_argument('--iteration', default=100, type=int, help='attack iteration')
    parser.add_argument('--path', type=str, help='name of checkpoint')
    args = parser.parse_args()
    PGD_attack(args)