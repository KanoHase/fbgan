import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from implementations.data_utils import load_data

data_dir = "./data/"
data_file = "amino_positive_544.txt"


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))]) #make a tensor, and make it 1 dimension

dataset_train = datasets.MNIST( #60000 sets of pixel data and label(index 10)
    root='~/mnist', 
    train=True, 
    download=True, 
    transform=transform)
dataset_valid = datasets.MNIST(
    root='~/mnist', 
    train=False, 
    download=True, 
    transform=transform)

dataloader_train = DataLoader(dataset_train,
                                batch_size=1000,
                                shuffle=True) #len:60 (because batch:1000)
dataloader_valid = DataLoader(dataset_valid,
                                batch_size=1000,
                                shuffle=True)

a = np.array([[1,2],[3,4]])
print(transform(a))

#load_data(data_dir + data_file)