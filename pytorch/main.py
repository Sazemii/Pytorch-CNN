import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # converts PIL Images to PyTorchSensors from 0,255 to 0-1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # re-centers each RGB channel to have mean 0.5 and std 0.5, shifting values to
    # roughly [-1.0, 1.0] — this helps training converge faster
])

"""downloads and loads the CIFAR-10 dataset, which contains 60,000 color 
images (32×32 pixels) across 10 classes (airplane, car, bird, cat, etc.)"""


train_data = torchvision.datasets.CIFAR10(
    root='/data', train=True, transform=transform, download=True)
# 50,000 images for training

test_data = torchvision.datasets.CIFAR10(
    root='/data', train=False, transform=transform, download=True)
# 10,000 images for evaluation

# Dataloaders -> wraps the datasets to enable efficient batched iteration

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=2)
# feeds 32 images at a time to the model, randomizes order each epoch, 2 background processes to load data
# in parallel reducing load in gpu

"""Shuffle is important because -> Prevents the model from learning order patterns, Reduces gradient bias (for ex first batch is cats, the weights get pushed to that
, then upon encountering batch of dogs it overcorrects then forgets the cats, -> better balance), """

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=True, num_workers=2)

image, label = train_data[0]
print(image.size())  # prints torch.Size([3,32,32])

class_names = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5)
