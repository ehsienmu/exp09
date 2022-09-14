mycnn_cfg = {
    'model_type': 'cnn_5_layers',
    # 'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    # 'do_semi': True,
    # training hyperparameters
    'batch_size': 512,
    'lr': 0.001,
    # 'milestones': [20, 40, 60, 80, 100],
    'milestones': [30, 60, 90, 120],
    'num_out': 10,
    'num_epoch': 300,
    'early_stop': 30,
    
    # 'optimizer': 'SGD',             # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.001,              # learning rate of SGD
    #     'momentum': 0.8,              # momentum for SGD
    #     'weight_decay':1e-4,
    # },
    'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,              # learning rate of Adam
        'weight_decay':1e-3,
        # 'betas':(0.4, 0.999)      # Adam才有此參數
    },
}