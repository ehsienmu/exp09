import numpy as np
import random
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import seaborn
import matplotlib.pyplot as plot 
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
# from zigzag import dct_zigzag
from torch.autograd import Variable
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def render(img, path):
    if len(img.shape) == 4:
        img = img.squeeze(0)
    pil = transforms.ToPILImage()(img.cpu())
    pil.save(path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plt_dct(dct_value , attacked_dct ,filename):
    channel_n , h , w = dct_value.shape
    print(channel_n)
    r = torch.empty((channel_n//3,h,w))
    attacked_r = torch.empty((channel_n//3,h,w))
    g = torch.empty((channel_n//3,h,w))
    attacked_g = torch.empty((channel_n//3,h,w))
    b = torch.empty((channel_n//3,h,w))
    attacked_b = torch.empty((channel_n//3,h,w))
    pos = 0
    pos_r = 0
    pos_g = 0
    pos_b = 0
    for i in range(channel_n):
        if pos == 0:
            r[pos_r] = dct_value[i,:,:]
            attacked_r[pos_r] = attacked_dct[i,:,:]
            pos_r += 1
            pos += 1
        elif pos == 1:
            g[pos_g] = dct_value[i,:,:]
            attacked_g[pos_g] = attacked_dct[i,:,:]
            pos_g += 1
            pos += 1
        else:
            b[pos_b] = dct_value[i,:,:]
            attacked_b[pos_b] = attacked_dct[i,:,:]
            pos_b +=1
            pos = 0
    plt_frequency(torch.flatten(r),torch.flatten(attacked_r),"Output/"+filename+"_r.png")
    plt_frequency(torch.flatten(g),torch.flatten(attacked_g),"Output/"+filename+"_g.png")
    plt_frequency(torch.flatten(b),torch.flatten(attacked_b),"Output/"+filename+"_b.png")

def plt_frequency(dct_value,attacked_dct,filename):
    print(filename,stats.wasserstein_distance(dct_value.flatten(),attacked_dct.flatten()))
    plt = seaborn.kdeplot(dct_value, color="b",bw_adjust=0.2)
    plt = seaborn.kdeplot(attacked_dct, color="r",bw_adjust = 0.2)
    fig = plt.get_figure()
    fig.savefig(filename)
    plot.close(fig)
