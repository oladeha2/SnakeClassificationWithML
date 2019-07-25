# use RESNET101 as a baseline for the snake classification problem

import torchvision.models as models
import torch.utils.data as torchdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
import os
from train import train_model

# set random seeds
torch.manual_seed(271828)
np.random.seed(271828)

device =  'cuda'

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
# load the model and edit for the number of classes required for snake classification (45 classes)
resnet = models.resnet101(pretrained=True, progress=True)
print('Resnet Model Loaded')

# in_features is the number of features for the Linear Layer
num_features = resnet.fc.in_features
# edit the final layer matrix for the new number of classes
resnet.fc = nn.Linear(num_features, number_of_classes)
resnet.to(device)

# construct the optimizer --> optimizer for the gradient descent
optimizer = optim.Adam(resnet.parameters(), lr=0.001) # --> test initial loss rate as 0.01 change subject to results and speed of learning during training

# define a loss rate scheduler that decays that reduces the learning rate as the number of epochs increases, prevents the learning rate fron bouncing up and down as the network trains
lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# train the model end to end as opoosed to just training the final classification layer of the model

# train the model using the folllowing parameters --> model, loss_function, loss rate scheduler, train data and validation data and the number of epochs -> not saving the checkpoints
train_model(model=resnet, loss_function=loss_function, optimiser=optimizer, scheduler=lr_sched, train_data=train_loader, valid_data=valid_loader)