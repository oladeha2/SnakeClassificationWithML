"""
use RESNET101 as the baseline with weighted cross entropy loss and three drop out layers for improved accuracy
The drop out layers are added in the following places:
    1. One after the final set of convolutions
    2. Second after the first drop out layer
    3. A final drop out layer before the final fully connected layer
"""
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
import nonechucks as nc

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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    # convert to tensor for input into the model
    transforms.ToTensor(),
    # normalise image for std of 1 and mean 0 using image net values for mean and std
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# load the sets
train_data = datasets.ImageFolder(train_dir, trans)
valid_data = datasets.ImageFolder(valid_dir, trans)

print('original length of training set ->', len(train_data))

# remove any potential corrupted images
safe_train = nc.SafeDataset(train_data)
print('length after filtering ->', len(safe_train))
print('-' * 30)

print('original length of validatio set ->', len(valid_data))

safe_valid = nc.SafeDataset(valid_data)
print('length after filtering ->', len(safe_valid))
print('-' * 30)

number_of_classes = len(train_data.classes)
print('Number of classes ->', number_of_classes)
print('-' * 30)

#load the data using data loader
train_loader = torchdata.DataLoader(safe_train, batch_size=32, shuffle=True, num_workers=4)
valid_loader = torchdata.DataLoader(safe_valid, batch_size=32, shuffle=True, num_workers=4)


# define the loss function, use weighted croess entropy loss, with weights based on the occurencies of each class in the data set

classes_train = len(train_data.classes)
examples_train = len(safe_train)
print("number of classes in training set ->", classes_train)
print('-' * 30)
class_balance = torch.empty(classes_train)

i = 0
for cl in train_data.classes:
    class_balance[i] = 1/len(os.listdir(os.path.join(train_dir, cl)))/examples_train
    i += 1

normalisation_factor = class_balance.sum()
class_balance /= normalisation_factor
class_balance = class_balance.to(device)

loss_function = nn.CrossEntropyLoss(weight=class_balance)


# using resnet101 pretrained to get the baseline 
# load the model and edit for the number of classes required for snake classification (45 classes) 
resnet = models.resnet101(pretrained=True, progress=True)
print('Resnet Model Loaded')

in_features = resnet.fc.in_features

# add the drop out layers to the model --> three drop out layers are added two before the avg pooling layer and one before the final linear layer for the final classification
# note Sequential layer is a series of layers encapsulated in a single layer
# add the first drop out layer
resnet.layer4[1].relu = nn.Sequential(
    nn.Dropout2d(0.15),
    nn.ReLU(inplace=True)
)

# add the second drop out layer
resnet.layer4[2].relu = nn.Sequential(
    nn.Dropout2d(0.15),
    nn.ReLU(inplace=True)
)

# add the drop out layer and edited final linear layer to allow for classification of the 45 classes
# drop out kills some of the nodes in the layer of the model, allows for better generalisation, should improve the results hopefully
resnet.fc = nn.Sequential(
    nn.Dropout2d(0.5),
    nn.Linear(in_features, number_of_classes)
)

print(resnet)
print('-' * 30 )

# pass the model to the cuda device
resnet.to(device)

# construct the optimizer --> optimizer for the gradient descent
optimizer = optim.Adam(resnet.parameters(), lr=3e-4) # --> test initial loss rate as 0.01 change subject to results and speed of learning during training

# define a loss rate scheduler that decays that reduces the learning rate as the number of epochs increases, prevents the learning rate fron bouncing up and down as the network trains
lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# train the model end to end as opoosed to just training the final classification layer of the model

# train the model using the folllowing parameters --> model, loss_function, loss rate scheduler, train data and validation data and the number of epochs -> not saving the checkpoints
# model is trained for 27 epochs
model = train_model(model=resnet, 
                    loss_function=loss_function,
                    optimiser=optimizer,
                    scheduler=lr_sched,
                    num_epochs=27,
                    data_dict={
                        'train': train_loader,
                        'valid': valid_loader
                    },
                    data_lengths={
                        'train': len(safe_train), 
                        'valid': len(safe_valid)
                        }
                )