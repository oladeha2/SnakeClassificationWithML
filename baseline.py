# use RESNET101 as a baseline for the snake classification problem

import torchvision.models.resnet as RESNET101
import torch.utils.data as torchdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# load data and generate transforms for the data

train_dir = '../data/snake/train/'
valid_dir = '..data/snake/valid'

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
print('Length of Validation set', len(valid_data))

print('Classes -> ',  train_data.classes)

