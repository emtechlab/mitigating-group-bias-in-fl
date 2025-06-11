import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.functional as func

# Define the simplified whitening function
def channel_wise_whitening(x):
    
    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    std = torch.std(x, dim=(0, 2, 3), keepdim=True)
    return (x - mean) / (std + 1e-8)  # Adding a small value epsilon for numerical stability
 
 
class VGGNet(nn.Module):
    
    def __init__(self):
        
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(3, 3), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3), padding=1)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(7 * 7 * 20, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.dropout_2d(F.max_pool2d(self.batchnorm1(self.conv1(x)), kernel_size=2))
        x = self.dropout_2d(F.max_pool2d(self.batchnorm2(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 7 * 7 * 20)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
































'''
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

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
    '''
