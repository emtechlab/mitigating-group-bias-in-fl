import torch
import os
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch import nn
from .utils.validate import *
from .utils.train import *
from .utils.dataset.data_loaders import *
from .utils.dataclass import ClientsParams

class LocalBase():

    def __init__(self, args, device, client_id):

        self.args = args
        self.client_id = client_id
        self.device = device
        
        if (args.dataset in ['acsemployment', 'acsincome', 'acspubliccoverage', 'acsmobility']):
            
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def local_validate(self, test_loader, model, epoch):

        test_acc, test_loss = local_validate(self.args, model, self.device, epoch, test_loader[self.client_id], self.criterion, self.client_id) 
            
        return test_acc, test_loss
    
    def update_weights(self, client_loaders, test_loader, model, global_epoch, r_k_dic, lambda_reg, weight_coefficient_p):

        updated_weight, updated_model, avg_grad_norm = update_weights(self.args, model, self.device, global_epoch, client_loaders[self.client_id], test_loader[self.client_id], self.criterion, self.client_id, r_k_dic, lambda_reg, weight_coefficient_p)

        return updated_weight, updated_model, avg_grad_norm

class Fedavg_Local(LocalBase):

    def __init__(self, args, device, client_id):

        super().__init__(args, device, client_id)

        self.args = args
        self.client_id = client_id

    def localround(self, client_loaders, test_loader, model, global_epoch, r_k_dic, lambda_reg, weight_coefficient_p, validation_only=False):

        test_acc, test_loss = self.local_validate(test_loader, model, global_epoch)
        
        if validation_only:
            
            return 
        
        #update weights
        self.updated_weight, updated_model, avg_grad_norm = self.update_weights(client_loaders, test_loader, model, global_epoch, r_k_dic, lambda_reg, weight_coefficient_p)

        clients_params = ClientsParams(weight=self.updated_weight)

        return clients_params, updated_model, avg_grad_norm

def define_localnode(args, device, client_id):
    
    if (args.federated_type == 'gifairfl') | (args.federated_type == 'fedavg') | (args.federated_type == 'fedprox'):

        return Fedavg_Local(args, device, client_id)