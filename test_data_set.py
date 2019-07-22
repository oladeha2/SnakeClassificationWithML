import torch
import torchvision
import torch.utils.data as torchdata
import torchvision.datasets as datasets

train_dir = '../data/snake/train/'
valid_dir = '../data/snake/valid/'

# load the data sets
training_set  = datasets.ImageFolder(train_dir)
validation_set = datasets.ImageFolder(valid_dir)
 
print('length of train set', len(training_set))
print('length of validation set', len(validation_set))





