import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import copy

from FL.utils.utils import weighted_average_weights, euclidean_proj_simplex

from .models import *

class VGGNet(nn.Module):
    
    def __init__(self):
        
        super(VGGNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3), padding=1)
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(7 * 7 * 20, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.dropout_2d(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = self.dropout_2d(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(-1, 7 * 7 * 20)  # flatten / reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

class LeNet(nn.Module):
    
    def __init__(self):
        
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def name(self):
        
        return "LeNet"

class LogisticRegressionModel(nn.Module):
    
    def __init__(self, n_features, n_hidden=32, p_dropout=0.0):
        
        super(LogisticRegressionModel, self).__init__()
        
        # Define a network with multiple hidden layers and dropout regularization
        self.network = nn.Sequential(
            
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),  # Activation function to add non-linearity
            nn.Dropout(p_dropout),  # Dropout to reduce overfitting
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 1),  # Output layer for binary classification
        )

    def forward(self, x):
        
        # Forward pass through the network ending with a sigmoid activation to get the probability
        return torch.sigmoid(self.network(x)).squeeze(1)

class GlobalBase():

   def __init__(self, device, args):

    self.args = args
    self.device = device

    if (self.args.dataset == 'cifar10'):

        self.model = ResNet18_cifar10().to(device)

    if (self.args.dataset == 'mnist'):

        self.model = LeNet().to(device)

    if (self.args.dataset == 'fmnist'):

        self.model = VGGNet().to(self.device)

    if (self.args.dataset == 'utk'):

        self.model = ResNet18_utk().to(device)

   def distribute_weight(self):

    return self.model

class Fedavg_Global(GlobalBase):

    def __init__(self, device, args):

        super().__init__(device, args)

        self.args = args
        self.device = device

    def aggregate(self, local_params, round):

        print("aggregating weights with Term...")

        global_weight = self.model
        local_weights = []

        for client_id ,dataclass in local_params.items():

            local_weights.append(dataclass.weight)

        w_avg = weighted_average_weights(local_weights, global_weight.state_dict())

        self.model.load_state_dict(w_avg)

def define_globalnode(device, args):

    if (args.federated_type == 'term'):

        return Fedavg_Global(device, args) 