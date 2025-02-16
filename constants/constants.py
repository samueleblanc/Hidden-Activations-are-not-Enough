"""
    Some constants used in the experiments.
"""

ARCHITECTURES = [
    # 0 -> 0, 1, 2, 3
    (500, 500, 500, 500, 500),
    # 1 -> 4, 5, 6
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    # 2 -> 7, 8
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    # 3 -> 9, 10,
    (10000, 10000),
    # 4 -> 11, 12
    (10000, 10000, 10000, 10000, 10000),
    # 5 -> 13, 14
    (675000, 1500, 1500, 1500, 1500),
    # 6 -> 15, 16
    (2500000, ),
    # 7 -> 8
    (814, 351, 118, 467, 823, 191, 756, 628, 935, 270),
    # CNNs
    ((10,10,10,10), (300,)),
    ((12,10,10,10), (300,))
]

ATTACKS = [
    "GN", 
    "FGSM", 
    "RFGSM", 
    "PGD", 
    "EOTPGD", 
    "FFGSM", 
    # "TPGD",
    "MIFGSM", 
    "UPGD", 
    "DIFGSM", 
    # "NIFGSM",
    "PGDRS", 
    # "VMIFGSM",  
    #"VNIFGSM",
    "CW", 
    # "PGDL2",
    # "PGDRSL2",
    "DeepFool", 
    "SparseFool", 
    "OnePixel", 
    "Pixle",
    "APGD",
    "APGDT",
    "FAB",
    "Square",
    "SPSA",
    "JSMA",
    "EADL1",
    "EADEN"
]

DEFAULT_EXPERIMENTS = {
    'experiment_0': {
        'architecture_index': 0,
        'dataset': 'mnist',
        'optimizer': 'sgd',
        'lr': 0.01,
        'batch_size': 8,
        'epoch': 1,
        'reduce_lr_each': 5,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_1': {
        'architecture_index': 0,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_2': {
        'architecture_index': 0,
        'optimizer': 'adam',
        'dataset': 'fashion',
        'lr': 1e-06,
        'batch_size': 32,
        'epoch': 81, # 21
        'reduce_lr_each': 20,
        'save_every_epochs': 10,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_3': {
        'architecture_index': 0,
        'optimizer': 'sgd',
        'dataset': 'fashion',
        'lr': 0.1,
        'batch_size': 16,
        'epoch': 51, # 35
        'reduce_lr_each': 20,
        'save_every_epochs': 10,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_4': {
        'architecture_index': 1,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 7,
        'reduce_lr_each': 5,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_5': {
        'architecture_index': 1,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_6': {
        'architecture_index': 1,
        'optimizer': 'adam',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 6,
        'reduce_lr_each': 3,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_7': {
        'architecture_index': 2,
        'optimizer': 'adam',
        'dataset': 'fashion',
        'lr': 0.0001,
        'batch_size': 16,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_8': {
        'architecture_index': 7,
        'optimizer': 'adam',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 6,
        'reduce_lr_each': 20,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_9': {
        'architecture_index': 3,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 1,
        'reduce_lr_each': 5,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_10': {
        'architecture_index': 3,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_11': {
        'architecture_index': 4,
        'optimizer': 'sgd',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 64,
        'epoch': 6,
        'reduce_lr_each': 20,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_12': {
        'architecture_index': 4,
        'optimizer': 'sgd',
        'dataset': 'fashion',
        'lr': 0.01,
        'batch_size': 64,
        'epoch': 16,
        'reduce_lr_each': 20,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_13': { #this only trains on 40 GBs GPU
        'architecture_index': 5,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 1e-5,
        'dropout': 0.5,
    },
    'experiment_14': { #this only trains on 40 GBs GPU
        'architecture_index': 5,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 1e-5,
        'dropout': 0.5,
    },
    'experiment_15': { #this only trains on 40 GBs GPU
        'architecture_index': 6,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 256,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_16': { #this only trains on 40 GBs GPU
        'architecture_index': 6,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.001,
        'batch_size': 256,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_17': {
        'architecture_index': 8,
        'optimizer': 'sgd',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_18': {
        'architecture_index': 8,
        'optimizer': 'momentum',
        'dataset': 'cifar10',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 61,
        'reduce_lr_each': 55,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_19': {
        'architecture_index': 9,
        'optimizer': 'adam',
        'dataset': 'cifar10',
        'lr': 0.001,
        'batch_size': 64,
        'epoch': 61,
        'reduce_lr_each': 55,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_20': {
        'architecture_index': -99,  # ResNet
        'optimizer': 'sgd',
        'dataset': 'cifar10',
        'lr': 0.01,
        'batch_size': 16,
        'epoch': 5,
        'reduce_lr_each': 3,
        'save_every_epochs': 1,
        'residual': True,
        'weight_decay': 0,
        'dropout': 0,
    }
}
