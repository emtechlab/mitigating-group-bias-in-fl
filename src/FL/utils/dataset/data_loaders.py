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

def get_fer_loaders(args):
    
    data_transforms = {
    
        'client0_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client0_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]),

        'client1_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.09),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client1_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.09),
            torchvision.transforms.ToTensor(),
        ]),

        'client2_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.18),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client2_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.18),
            torchvision.transforms.ToTensor(),
        ]),
    
        'client3_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.27),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client3_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.27),
            torchvision.transforms.ToTensor(),
        ]),

        'client4_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.36),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client4_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.36),
            torchvision.transforms.ToTensor(),
        ]),
















        'client5_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client5_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]),

        'client6_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.09),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client6_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.09),
            torchvision.transforms.ToTensor(),
        ]),

        'client7_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.18),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client7_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.18),
            torchvision.transforms.ToTensor(),
        ]),
    
        'client8_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.27),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client8_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.27),
            torchvision.transforms.ToTensor(),
        ]),

        'client9_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.36),
            torchvision.transforms.ToTensor(),
        ]),
        
        # Define training transformations for other clients similarly
        'client9_test': transforms.Compose([
     
            torchvision.transforms.RandomCrop(44),
            torchvision.transforms.RandomHorizontalFlip(),
            AddGaussianNoise(mean=0, std=0.36),
            torchvision.transforms.ToTensor(),
        ]),
    }
    
    TRAIN_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/fer2013/train'
    TEST_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/fer2013/test'

    clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9']  # Define clients
    
    client_train_datasets = {client: torchvision.datasets.ImageFolder(os.path.join(TRAIN_DATA_PATH, client), transform=data_transforms[f'{client}_train']) for client in clients}
    
    client_test_datasets = {client: torchvision.datasets.ImageFolder(TEST_DATA_PATH, transform=data_transforms[f'{client}_test']) for client in clients}
    
    client_train_data_loaders = {client: DataLoader(client_train_datasets[client], batch_size=128, shuffle=True) for client in clients}
    
    return client_train_data_loaders, client_test_datasets

def get_utk_loaders(args):
    
    # Define transformations and partition CIFAR-10 for different clients
    data_transforms = {
    
        'client0_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        # Define training transformations for other clients similarly
        'client0_test': transforms.Compose([
      
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        'client1_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
    
        # Define training transformations for other clients similarly
        'client1_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        'client2_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
    
        # Define training transformations for other clients similarly
        'client2_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
    
        'client3_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
        
        # Define training transformations for other clients similarly
        'client3_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        'client4_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
        
        # Define training transformations for other clients similarly
        'client4_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.26,))
        ]),















        'client5_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        # Define training transformations for other clients similarly
        'client5_test': transforms.Compose([
      
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        'client6_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
    
        # Define training transformations for other clients similarly
        'client6_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        'client7_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
    
        # Define training transformations for other clients similarly
        'client7_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
    
        'client8_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
        
        # Define training transformations for other clients similarly
        'client8_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),

        'client9_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.23,))
        ]),
        
        # Define training transformations for other clients similarly
        'client9_test': transforms.Compose([
     
            torchvision.transforms.Resize((32, 32)),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.49,), (0.26,))
        ]),
    }

    TRAIN_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/utk/train'
    TEST_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/utk/test'

    clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9']  # Define clients

    client_train_datasets = {client: torchvision.datasets.ImageFolder(os.path.join(TRAIN_DATA_PATH, client), transform=data_transforms[f'{client}_train']) for client in clients}
    client_test_datasets = {client: torchvision.datasets.ImageFolder(TEST_DATA_PATH, transform=data_transforms[f'{client}_test']) for client in clients}

    client_train_data_loaders = {client: DataLoader(client_train_datasets[client], batch_size=64, shuffle=True) for client in clients} # 64

    return client_train_data_loaders, client_test_datasets

def get_mnist_loaders(args):
    
    # Define transformations and partition CIFAR-10 for different clients
    data_transforms = {
    
        'client0_train': transforms.Compose([                                                       # Assign 3 clients here
        
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client0_test': transforms.Compose([
        
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),

        'client1_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=0.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client1_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=0.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),

        'client2_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=0.7), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client2_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=0.7), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        'client3_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=1.0), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client3_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=1.0), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),

        'client4_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=1.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
        
        # Define training transformations for other clients similarly
        'client4_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=1.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
       ]),















        'client5_train': transforms.Compose([                                                       # Assign 3 clients here
        
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client5_test': transforms.Compose([
        
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),

        'client6_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=0.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client6_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=0.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),

        'client7_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=0.7), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client7_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=0.7), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        'client8_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=1.0), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
    
        # Define training transformations for other clients similarly
        'client8_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=1.0), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),

        'client9_train': transforms.Compose([                                                       # Assign 3 clients here
     
            AddGaussianNoise(mean=0, std=1.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]),
        
        # Define training transformations for other clients similarly
        'client9_test': transforms.Compose([
     
            AddGaussianNoise(mean=0, std=1.5), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
       ]),
    }

    TRAIN_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/mnist/train'

    clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9']  # Define clients

    client_train_datasets = {client: torchvision.datasets.ImageFolder(os.path.join(TRAIN_DATA_PATH, client), transform=data_transforms[f'{client}_train']) for client in clients}
    client_test_datasets = {client: torchvision.datasets.MNIST(root='.', train=False, download=True, transform=data_transforms[f'{client}_test']) for client in clients}

    client_train_data_loaders = {client: DataLoader(client_train_datasets[client], batch_size=256, shuffle=True) for client in clients} # 128

    return client_train_data_loaders, client_test_datasets

def get_fmnist_loaders(args):

    # Define transformations and partition CIFAR-10 for different clients
    data_transforms = {
    
        'client0_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client0_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),

        'client1_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client1_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),

        'client2_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client2_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        'client3_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
       ]),
    
       # Define training transformations for other clients similarly
       'client3_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),

        'client4_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client4_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),





        'client5_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client5_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),

        'client6_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client6_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),

        'client7_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client7_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        'client8_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
       ]),
    
       # Define training transformations for other clients similarly
       'client8_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),

        'client9_train': transforms.Compose([                                                       # Assign 3 clients here
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    
        # Define training transformations for other clients similarly
        'client9_test': transforms.Compose([
     
            torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=1.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.406], [0.225])
        ]),
    }

    TRAIN_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/fmnist/train'
    #TEST_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/fmnist/test'
    TEST_DATA_PATH = '/home/' + os.getenv('USER') + '/global_subgroup_fair_fl/train_test_data/fmnist/paper_original/test'

    clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9']  # Define clients

    client_train_datasets = {client: torchvision.datasets.ImageFolder(os.path.join(TRAIN_DATA_PATH, client), transform=data_transforms[f'{client}_train']) for client in clients}
    client_test_datasets = {client: torchvision.datasets.ImageFolder(TEST_DATA_PATH, transform=data_transforms[f'{client}_test']) for client in clients}

    client_train_data_loaders = {client: DataLoader(client_train_datasets[client], batch_size=256, shuffle=True) for client in clients}  # 256

    return client_train_data_loaders, client_test_datasets
