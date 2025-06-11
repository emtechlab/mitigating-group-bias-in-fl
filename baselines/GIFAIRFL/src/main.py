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

    initial_loss = random.sample(range(args.n_clients), args.n_clients)
    users_pool = np.random.choice(range(args.n_clients), args.n_clients, replace=False)
    users_pool.sort()
    d_group = args.n_clients
    r_k_values = []

    for i in range(args.n_clients) :
        r_k_values.append(d_group - (2*(i+1)-1))

    loss_dic = {users_pool[i]: initial_loss[i] for i in range(len(users_pool))}
    loss_dic = dict(sorted(loss_dic.items(), key=lambda item: item[1], reverse=True))
    r_k_dic = {list(loss_dic.keys())[i]: r_k_values[i] for i in range(len(list(loss_dic.keys())))}

    weight_coefficient_p = []

    for idx in users_pool:
        
        c = len(client_test_loaders[idx].dataset)
        weight_coefficient_p.append(c)

    total_size = sum(weight_coefficient_p)
    weight_coefficient_p = [number / total_size for number in weight_coefficient_p]

    new_reg = [number / (d_group - 1) for number in weight_coefficient_p]
    lambda_reg = min(new_reg)/2

    # Training
    for epoch in range(args.global_epochs):

        # rank losses
        loss_dic = {users_pool[i]: initial_loss[i] for i in range(len(users_pool))}
        loss_dic = dict(sorted(loss_dic.items(), key=lambda item: item[1], reverse=True))
        r_k_dic = {list(loss_dic.keys())[i]: r_k_values[i] for i in range(len(list(loss_dic.keys())))}

        runner_train(args, client_train_loaders, client_test_loaders, device, epoch+1, r_k_dic, lambda_reg, weight_coefficient_p)

    train_elapsed_time = time.time() - train_start_time
    
    print('Train time: [{} m {:.2f} s] '.format(train_elapsed_time // 60, train_elapsed_time % 60))

if __name__=="__main__":

    args = get_args()
    
    main(args)