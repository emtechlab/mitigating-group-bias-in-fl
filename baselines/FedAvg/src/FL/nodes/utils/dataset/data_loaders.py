import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import DataLoader, Subset

import torch.utils.data as data
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


import pandas as pd
import seaborn as sns
import sys

class AddGaussianNoise(object):
    
    def __init__(self, mean=0, std=1):
     
        self.mean = mean
        self.std = std

    def __call__(self, img):
     
        # Convert the image to a PyTorch tensor
        img = transforms.ToTensor()(img)
     
        # Generate Gaussian noise with the same size as the image
        noise = torch.randn(img.size()) * self.std + self.mean
     
        # Add the noise to the image
        noisy_img = img + noise
     
        # Clip the pixel values to be in the range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0, 1)
     
        # Convert the noisy image back to a PIL Image
        noisy_img = transforms.ToPILImage()(noisy_img)
     
        return noisy_img

def get_fmnist_loaders(TRAIN_DATA_PATH, TEST_DATA_PATH, batch_size, client_id):

    transform1 = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(num_output_channels=1),
        #AddGaussianNoise(mean=0, std=1.0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.406], [0.225])
    ])

    trainset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform1) # HERE
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = ImageFolder(root=TEST_DATA_PATH, transform=transform1) # HERE
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_cifar10_loaders(TRAIN_DATA_PATH, TEST_DATA_PATH, batch_size, client_id):
    
    transform1 = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        
    transform_test1 = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform1) # Here
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    #testset = ImageFolder(root=TEST_DATA_PATH, transform=transform_test1) # Here
    #test_loader = DataLoader(testset, batch_size=100, shuffle=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    return train_loader, test_loader

def get_cifar100_loaders(TRAIN_DATA_PATH, TEST_DATA_PATH, batch_size, client_id):
    
    transform1 = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        
    transform_test1 = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform1) # Here
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = ImageFolder(root=TEST_DATA_PATH, transform=transform_test1) # Here
    test_loader = DataLoader(testset, batch_size=100, shuffle=False)
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test1)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_utk_loaders(TRAIN_DATA_PATH, TEST_DATA_PATH, batch_size, client_id):
    
    # Define transformations
    transform_train = transforms.Compose([
      
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    transform_test = transforms.Compose([
      
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    trainset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform_train) # HERE
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = ImageFolder(root=TEST_DATA_PATH, transform=transform_test) # Here
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    return train_loader, test_loader