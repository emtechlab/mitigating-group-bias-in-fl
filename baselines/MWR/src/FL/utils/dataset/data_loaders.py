import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

from .data_utils import DigitsDataset
from .data_utils import OfficeDataset

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

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

def get_cifar10_loaders(TRAIN_DATA_PATH, batch_size, client_id):

    transform1 = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform2 = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        AddGaussianNoise(mean=0, std=0.05),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform3 = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        AddGaussianNoise(mean=0, std=0.10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform4 = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        AddGaussianNoise(mean=0, std=0.15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform5 = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        AddGaussianNoise(mean=0, std=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test1 = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test2 = torchvision.transforms.Compose([
    
        AddGaussianNoise(mean=0, std=0.05),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test3 = torchvision.transforms.Compose([

        AddGaussianNoise(mean=0, std=0.10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test4 = torchvision.transforms.Compose([
    
        AddGaussianNoise(mean=0, std=0.15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test5 = torchvision.transforms.Compose([
    
        AddGaussianNoise(mean=0, std=0.20),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transforms = [transform1, transform2, transform3, transform4, transform5]
    transform_tests = [transform_test1, transform_test2, transform_test3, transform_test4, transform_test5]
      
    trainset = ImageFolder(root=TRAIN_DATA_PATH, transform=transforms[client_id+0]) # Here
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_tests[client_id+0]) # Here
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def get_mnist_loaders(TRAIN_DATA_PATH, batch_size, client_id):
    
    transform1 = torchvision.transforms.Compose([
    
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
    ])
    
    transform2 = torchvision.transforms.Compose([
        
        AddGaussianNoise(mean=0, std=0.03),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
    ])
    
    transform3 = torchvision.transforms.Compose([
    
        AddGaussianNoise(mean=0, std=0.05),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
    ])
    
    transform4 = torchvision.transforms.Compose([

        AddGaussianNoise(mean=0, std=0.5), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
    ])
    
    transform5 = torchvision.transforms.Compose([
    
        AddGaussianNoise(mean=0, std=0.8),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
    ])
    
    transforms = [transform1, transform2, transform3, transform4, transform5]
    
    trainset = ImageFolder(root=TRAIN_DATA_PATH, transform=transforms[client_id+0]) # HERE
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
     
    testset = datasets.MNIST('.', train=False, download=True, transform=transforms[client_id+0])   # HERE
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def get_digits_loaders(TRAIN_DATA_PATH, batch_size, client_id):

    percent = 1.0

    # Prepare data
    transform_mnist_train = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(num_output_channels=3),
        AddGaussianNoise(mean=0, std=0.8),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_svhn_train = torchvision.transforms.Compose([

        torchvision.transforms.Resize([28,28]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_usps_train = torchvision.transforms.Compose([

        torchvision.transforms.Resize([28,28]),
        torchvision.transforms.Grayscale(num_output_channels=3),
        AddGaussianNoise(mean=0, std=1.0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth_train = torchvision.transforms.Compose([

        torchvision.transforms.Resize([28, 28]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm_train = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnist_test = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(num_output_channels=3),
        AddGaussianNoise(mean=0, std=0.8),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_svhn_test = torchvision.transforms.Compose([

        torchvision.transforms.Resize([28,28]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_usps_test = torchvision.transforms.Compose([

        torchvision.transforms.Resize([28,28]),
        torchvision.transforms.Grayscale(num_output_channels=3),
        AddGaussianNoise(mean=0, std=1.0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth_test = torchvision.transforms.Compose([

        torchvision.transforms.Resize([28, 28]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm_test = torchvision.transforms.Compose([
    
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if (client_id == 0):
        
        TRAIN_DATA_PATH = "/home/khotso/FedGlobal/data/MNIST"

        # MNIST
        trainset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=1, percent=percent, train=True,  transform=transform_mnist_train)
        testset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=1, percent=percent, train=False, transform=transform_mnist_test) 

    if (client_id == 1):
        
        TRAIN_DATA_PATH = "/home/khotso/FedGlobal/data/SVHN"

        # SVHN
        trainset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=3, percent=percent,  train=True,  transform=transform_svhn_train)
        testset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=3, percent=percent,  train=False, transform=transform_svhn_test) 

    if (client_id == 2):
        
        TRAIN_DATA_PATH = "/home/khotso/FedGlobal/data/USPS"

        # USPS
        trainset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=1, percent=percent,  train=True,  transform=transform_usps_train) 
        testset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=1, percent=percent,  train=False, transform=transform_usps_test) 

    if (client_id == 3):
        
        TRAIN_DATA_PATH = "/home/khotso/FedGlobal/data/SynthDigits"

        # Synth Digits
        trainset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=3, percent=percent,  train=True,  transform=transform_synth_train) 
        testset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=3, percent=percent,  train=False, transform=transform_synth_test) 

    if (client_id == 4):
        
        TRAIN_DATA_PATH = "/home/khotso/FedGlobal/data/MNIST_M"

        # MNIST-M
        trainset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=3, percent=percent,  train=True,  transform=transform_mnistm_train)
        testset = DigitsDataset(data_path=TRAIN_DATA_PATH, channels=3, percent=percent,  train=False, transform=transform_mnistm_test) 

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    return train_loader, test_loader

def get_fmnist_loaders(TRAIN_DATA_PATH, TEST_DATA_PATH, batch_size, client_id):

    transform1 = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.406], [0.225])
    ])
    
    transform2 = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.406], [0.225])
    ])
    
    transform3 = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(num_output_channels=1),
        AddGaussianNoise(mean=0, std=0.3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.406], [0.225])
    ])
    
    transform4 = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(num_output_channels=1),
        AddGaussianNoise(mean=0, std=0.5), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.406], [0.225])
    ])
    
    transform5 = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(num_output_channels=1),
        AddGaussianNoise(mean=0, std=0.4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.406], [0.225])
    ])
    
    transforms = [transform1, transform4, transform3, transform4, transform5]
    
    trainset = ImageFolder(root=TRAIN_DATA_PATH, transform=transforms[client_id+0]) # HERE
    #trainset = datasets.FashionMNIST(root='.', train=True, download=True, transform=transforms[client_id+0])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = ImageFolder(root=TEST_DATA_PATH, transform=transforms[client_id+0]) # HERE
    #testset = datasets.FashionMNIST('.', train=False, download=True, transform=transforms[client_id+0])
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader

















































def get_office_loaders(TRAIN_DATA_PATH, batch_size, client_id):
    
    data_base_path = TRAIN_DATA_PATH
    
    # Prepare data
    transform_amazon_train = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]),            
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation((-30,30)),
        torchvision.transforms.ToTensor(),
    ])

    transform_caltech_train = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]),            
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation((-30,30)),
        torchvision.transforms.ToTensor(),
    ])

    transform_dslr_train = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]),            
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation((-30,30)),
        AddGaussianNoise(mean=0, std=0.07),
        torchvision.transforms.ToTensor(),
    ])

    transform_webcam_train = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]),            
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation((-30,30)),
        AddGaussianNoise(mean=0, std=0.09),
        torchvision.transforms.ToTensor(),
    ])
    
    transform_amazon_test = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]),            
        torchvision.transforms.ToTensor(),
    ])
    
    transform_caltech_test = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]),            
        torchvision.transforms.ToTensor(),
    ])
    
    transform_dslr_test = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]), 
        AddGaussianNoise(mean=0, std=0.07),           
        torchvision.transforms.ToTensor(),
    ])
    
    transform_webcam_test = torchvision.transforms.Compose([
        
        torchvision.transforms.Resize([256, 256]),
        AddGaussianNoise(mean=0, std=0.09),            
        torchvision.transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_amazon_train)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_amazon_test, train=False)
    
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_caltech_train)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_caltech_test, train=False)
    
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_dslr_train)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_caltech_test, train=False)
    
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_webcam_train)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_webcam_test, train=False)
    
    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)
    
    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=batch_size, shuffle=True)
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=batch_size, shuffle=False)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=batch_size, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=batch_size, shuffle=True)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=batch_size, shuffle=False)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=batch_size, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=batch_size, shuffle=True)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=batch_size, shuffle=False)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=batch_size, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=batch_size, shuffle=True)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=batch_size, shuffle=False)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=batch_size, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    
    return train_loaders[client_id], test_loaders[client_id]
