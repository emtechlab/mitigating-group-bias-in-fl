#!/bin/bash

cd $HOME/Mitigating-Group-Bias-in-FL/baselines/FedAvg/src

#python main.py --federated_type fedavg --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 5 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0

cd $HOME/Mitigating-Group-Bias-in-FL/baselines/AFL/src

#python main.py --federated_type afl --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 5 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0

cd $HOME/Mitigating-Group-Bias-in-FL/baselines/GIFAIRFL/src

#python main.py --federated_type gifairfl --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 5 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0

cd $HOME/Mitigating-Group-Bias-in-FL/baselines/TERM/src

#python main.py --federated_type term --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 5 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0

cd $HOME/Mitigating-Group-Bias-in-FL/baselines/MWR/src

python main.py --federated_type mwr --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 5 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0 --do_training 1