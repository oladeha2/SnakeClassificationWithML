# use RESNET101 as a baseline for the snake classification problem

import torchvision.models as models
import torch.utils.data as torchdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch

import numpy as np

# set random seeds
torch.manual_seed(271828)
np.random.seed(271828)


# load data and generate transforms for the data

train_dir = '../data/snake/train/'
valid_dir = '../data/snake/valid/'

# define the trasforms for each image in the data set

trans = transforms.Compose([
    # image transforms
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    # convert to tensor for input into the model
    transforms.ToTensor(),
    # normalise the images last of all
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# load the sets
train_data = datasets.ImageFolder(train_dir, trans)
valid_data = datasets.ImageFolder(valid_dir, trans)

print('Length of Training set -> ', len(train_data))
print('Length of Validation set -> ', len(valid_data))

print('Classes -> ',  train_data.classes)
number_of_classes = len(train_data.classes)

#load the data using data loader
train_loader = torchdata.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
valid_loader = torchdata.DataLoader(valid_data, batch_size=32, shuffle=True, num_workers=2)

# define the loss function
loss_function = nn.CrossEntropyLoss()

# using resnet101 pretrained to get the baseline 
resnet = models.resnet101(pretrained=True, progress=True)

# construct an optimizer, this must be based on the number of parameters you are attempting to train in the pre-trained model
# try retraining just the final layer for the task, leaving the pretrained network as a feature extractor