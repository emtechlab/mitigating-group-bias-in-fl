import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset

import torch.utils.data as data
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

from data_loaders import *

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

transform_mnist = transforms.Compose([

    AddGaussianNoise(mean=0, std=1.2),  # 0.3, 0.6, 0.9, 1.2
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Grayscale(),
])

transform_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    AddGaussianNoise(mean=0, std=0.20),   # 0.05, 0.1, 0.15, 0.2
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_fer = transforms.Compose([
    torchvision.transforms.RandomCrop(44),
    torchvision.transforms.RandomHorizontalFlip(),
    AddGaussianNoise(mean=0, std=0.36), # 0.09, 0.18, 0.27, 0.36
    torchvision.transforms.ToTensor(),
])

transform_utk = transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    AddGaussianNoise(mean=0, std=0.7),               # 0.1, 0.3, 0.5, 0.7
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.49,), (0.23,))
])

# Function to compute variance and skewness
def compute_moments(tensor):
    
    mean = torch.mean(tensor)
    variance = torch.var(tensor)
    skewness = torch.mean(((tensor - mean) / torch.sqrt(variance)) ** 3)
    
    return variance.item(), skewness.item()

def main():

    client_id = 5

    # Initialize arrays to store moments for each class
    num_classes = 2
    class_variances = np.zeros(num_classes)
    class_skewness = np.zeros(num_classes)

    # MNIST DATASET
    #trainData = MNIST(root='.', train=True, transform=transform_mnist, download=True)

    # CIFAR10 DATASET
    #trainData = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_cifar10)

    # FER2013 DATASET
    TRAIN_DATA_PATH = '/home/khotso/FederatedGlobal/data/fer2013'
    #trainData = torchvision.datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform_fer)

    # UTK DATASET
    TRAIN_DATA_PATH = '/home/khotso/FederatedGlobal/data/utk'
    trainData = torchvision.datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform_utk)

    trainDataloader = DataLoader(trainData, batch_size=32, shuffle=True)

    for batch_idx, (images, labels) in enumerate(trainDataloader):

        # Split images by class
        class_images = [images[labels == i] for i in range(num_classes)]

        # Compute moments for each class
        for i in range(num_classes):

            if len(class_images[i]) > 0:

                class_variances[i], class_skewness[i] = compute_moments(class_images[i])

    # Print and/or store the results
    for i in range(num_classes):

        print(f"Class {i + 1} - Variance: {class_variances[i]}")

if __name__ == '__main__':
    
    main()