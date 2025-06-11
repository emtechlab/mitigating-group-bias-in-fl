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

from torch.utils.data import DataLoader, Subset

import torch.utils.data as data
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

client_id = 0

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
        
TRAIN_DATA_PATH = '/home/khotso/FederatedGlobal/train_test_data/mnist/entropy'

# Define a function to calculate entropy
def entropy(image):
    # Convert image to probabilities
    probs = F.softmax(image.view(-1), dim=0)
    # Calculate entropy
    entropy_value = -(probs * torch.log2(probs + 1e-10)).sum()  # adding a small value to avoid log(0)
    return entropy_value.item()


# Define a transform to convert images to PyTorch tensors
transform = transforms.Compose([

    AddGaussianNoise(mean=0, std=0.6), 
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(),
])

# Load images from a folder
dataset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Calculate entropy for each image and compute average entropy
total_entropy = 0.0
num_images = len(dataset)
for image, _ in data_loader:
    
    total_entropy += entropy(image)

# Calculate average entropy
average_entropy = total_entropy / num_images

# Calculate maximum possible entropy for 8-bit grayscale images
max_entropy = np.log2(256)

# Rescale average entropy to match the correct range (0 to max_entropy)
scaled_average_entropy = (average_entropy / max_entropy) * 8

print("Average entropy:", scaled_average_entropy)







