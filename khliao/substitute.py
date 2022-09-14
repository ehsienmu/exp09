import os
import numpy as np
import random
import pickle
import argparse

from sf_train_sub import *
from attack_sub import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()#description='Frequency Training')
    parser.add_argument('--resume_ckpt', type=str,
                        help='resume from the name of checkpoint')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--epoch', default=50, type=int, help='epoch')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--ckpt_path', default='test', type=str, help='name of the checkoint')
    parser.add_argument('--model', default='resnet', type=str, help='model')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')


    #attack
    parser.add_argument('--attack_batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--attack_eps', default=0.01, type=float, help='eps')
    parser.add_argument('--attack_step_size', default=0.003, type=float, help='step size')
    parser.add_argument('--attack_iteration', default=100, type=int, help='attack iteration')
    parser.add_argument('--attack_checkpoint_path', type=str, help='name of checkpoint')
    args = parser.parse_args()

    train_config = {
    "dataset": args.dataset,
    "model": args.model,
    "learning_rate": args.learning_rate,
    "epochs": args.epoch,
    "batch_size": args.batch_size,
    "resume_ckpt" : "checkpoint/{}.pth".format(args.resume_ckpt),
    "have_resume_ckpt": (args.resume_ckpt != None),#have_resume_ckpt,
    }

    attack_config = {
    "dataset": args.dataset,
    "model": args.model,
    "batch_size": args.attack_batch_size ,
    "eps" : args.attack_eps,
    "step_size" : args.attack_step_size,
    "iteration" : args.attack_iteration,
    }
    attack_record = {}
    for freq in range(64):
        
        train_config["ckpt_path"] = "checkpoint/{}.pth".format(args.ckpt_path+str(freq))
        train_acc = train(freq_num = freq, cfg = train_config)
        attack_config["ckpt_path"] = "checkpoint/{}.pth".format(args.ckpt_path+str(freq))
        ASR, Tacc, Aacc = PGD_attack(freq_num = freq, cfg = attack_config)
        print("Attack Success Rate: {}, Test Accuracy: {}, Attacked Accuracy: {}".format(ASR, Tacc, Aacc))
        attack_record[freq] = (train_acc, ASR, Tacc, Aacc)
        with open('./attack_record.pk', 'wb') as f:
            pickle.dump(attack_record, f)
        