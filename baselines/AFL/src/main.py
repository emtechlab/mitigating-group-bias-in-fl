import os, sys, random, time
import torch, numpy as np
from FL.parameters import get_args
from FL.runner import runner_train

# reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# make sure 'data/' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.federated_data_partition import *

def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_start_time = time.time()

    # 1) load raw, no‐transform datasets
    train_ds, test_ds = load_dataset(args.dataset, train_transform=None, test_transform=None)

    # 2) build per‑client transforms
    train_tfs = get_transforms(args.dataset, args.n_clients)
    test_tfs  = get_test_transforms(args.dataset, args.n_clients)

    # 3) Dirichlet partition of train
    targets = getattr(train_ds, "targets", None)
    
    if targets is None:
        
        # for ImageFolder‐style
        targets = [y for _,y in train_ds]

    train_idx = partition_data(targets, args.partition, args.n_clients, alpha=args.dirichlet_alpha)

    # 4) make train loaders
    client_train_loaders = get_client_loaders(train_ds, train_idx, args.batch_size, transforms_list=train_tfs, shuffle=True)

    # 5) equal IID partition of test
    client_test_loaders = get_test_loaders(test_ds, args.batch_size, args.n_clients, transforms_list=test_tfs)

    plot_partition_statistics(train_idx, targets, len(set(targets)), args.dataset, args.partition, args.dirichlet_alpha)

    # Training
    for epoch in range(args.global_epochs):

        runner_train(args, client_train_loaders, client_test_loaders, device, epoch+1)

    train_elapsed_time = time.time() - train_start_time
    
    print('Train time: [{} m {:.2f} s] '.format(train_elapsed_time // 60, train_elapsed_time % 60))

if __name__=="__main__":

    args = get_args()
    
    main(args)
































































'''
import os 
import random
import sys
import torch
import time
import numpy as np
import torchvision

from FL.parameters import get_args
from FL.runner import runner_train

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)
random.seed(seed)

import warnings

# Ignore all warnings (not recommended)
warnings.filterwarnings("ignore")

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.federated_data_partition import (
    get_transforms, 
    load_dataset, 
    partition_data, 
    get_client_loaders, 
    get_test_loader, 
    print_partition_statistics,
    plot_partition_statistics
)

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_start_time = time.time()

    transform = get_transforms(args.dataset)
    train_dataset, test_dataset = load_dataset(args.dataset, transform)
    targets = train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.targets

    client_indices_dict = partition_data(targets, args.partition, args.n_clients, alpha=args.dirichlet_alpha)
    client_loaders = get_client_loaders(train_dataset, client_indices_dict, args.batch_size)
    test_loader = get_test_loader(test_dataset, args.batch_size)

    #plot_partition_statistics(client_indices_dict, targets, len(set(targets)), args.dataset, args.partition, args.dirichlet_alpha)

    # Training
    #for epoch in range(args.global_epochs):

    #    runner_train(args, client_loaders, test_loader, device, epoch+1)

    #train_elapsed_time = time.time() - train_start_time
    
    #print('Train time: [{} m {:.2f} s] '.format(train_elapsed_time // 60, train_elapsed_time % 60))

if __name__ == '__main__':

    args = get_args()
    
    main(args)
'''