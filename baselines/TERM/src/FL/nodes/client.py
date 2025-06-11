import torch
import os
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch import nn
from .utils.validate import *
from .utils.train import *
from .utils.dataset.data_loaders import *
from .utils.dataclass import ClientsParams

class TERMLoss(nn.Module):
    """
    TERM: Tilted Empirical Risk Minimization loss function.

    Args:
        t (float): tilt parameter. 
                   t > 0 emphasizes large losses (e.g. fairness, robustness).
                   t < 0 suppresses large losses (e.g. outlier-robustness).
                   t = 0 recovers standard ERM.
    """
    def __init__(self, t: float = 0.0):
        
        super(TERMLoss, self).__init__()
        
        self.t = t
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # No reduction for TERM

    def forward(self, logits, labels):
        
        """
        Args:
            logits (Tensor): shape (B, C), raw scores from model
            labels (Tensor): shape (B,), integer class labels
        
        Returns:
            Tensor: scalar TERM loss
        """
        losses = self.ce_loss(logits, labels)  # shape: (B,)

        if self.t == 0:
            return losses.mean()  # ERM

        # TERM: Tilted loss
        tilted = torch.exp(self.t * losses)
        term_loss = (1.0 / self.t) * torch.log(torch.mean(tilted))

        return term_loss

class TERMBCross(nn.Module):
    """
    TERM: Tilted Empirical Risk Minimization loss function.

    Args:
        t (float): tilt parameter. 
                   t > 0 emphasizes large losses (e.g. fairness, robustness).
                   t < 0 suppresses large losses (e.g. outlier-robustness).
                   t = 0 recovers standard ERM.
    """
    def __init__(self, t: float = 0.0):
        
        super(TERMBCross, self).__init__()
        
        self.t = t
        self.ce_loss = nn.BCELoss()  # No reduction for TERM

    def forward(self, logits, labels):
        
        """
        Args:
            logits (Tensor): shape (B, C), raw scores from model
            labels (Tensor): shape (B,), integer class labels
        
        Returns:
            Tensor: scalar TERM loss
        """
        losses = self.ce_loss(logits, labels)  # shape: (B,)

        if self.t == 0:
            return losses.mean()  # ERM

        # TERM: Tilted loss
        tilted = torch.exp(self.t * losses)
        term_loss = (1.0 / self.t) * torch.log(torch.mean(tilted))

        return term_loss

class LocalBase():

    def __init__(self, args, device, client_id):

        self.args = args
        self.client_id = client_id
        self.device = device

        if (args.dataset in ['acsemployment', 'acsincome', 'acspubliccoverage', 'acsmobility']):
            
            self.criterion = TERMBCross(t=-1.0)
        else:
            self.criterion = TERMLoss(t=-1.0)

    def local_validate(self, test_loader, model, epoch):

        test_acc, test_loss = local_validate(self.args, model, self.device, epoch, test_loader[self.client_id], self.criterion, self.client_id) 
            
        return test_acc, test_loss
    
    def update_weights(self, client_loaders, test_loader, model, global_epoch):

        updated_weight, updated_model, avg_grad_norm = update_weights(self.args, model, self.device, global_epoch, client_loaders[self.client_id], test_loader[self.client_id], self.criterion, self.client_id)

        return updated_weight, updated_model, avg_grad_norm

class Fedavg_Local(LocalBase):

    def __init__(self, args, device, client_id):

        super().__init__(args, device, client_id)

        self.args = args
        self.client_id = client_id

    def localround(self, client_loaders, test_loader, model, global_epoch, validation_only=False):

        test_acc, test_loss = self.local_validate(test_loader, model, global_epoch)
        
        if validation_only:
            
            return 
        
        #update weights
        self.updated_weight, updated_model, avg_grad_norm = self.update_weights(client_loaders, test_loader, model, global_epoch)

        clients_params = ClientsParams(weight=self.updated_weight)

        return clients_params, updated_model, avg_grad_norm

def define_localnode(args, device, client_id):
    
    if (args.federated_type == 'term') | (args.federated_type == 'fedavg') | (args.federated_type == 'fedprox'):

        return Fedavg_Local(args, device, client_id)




































