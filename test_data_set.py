import torch
import torchvision
import torch.utils.data as torchdata
import numpy as np
import torchvision.datasets as datasets

train_dir = '../data/snake/train/'
valid_dir = '../data/snake/valid/'

# load the data sets
training_set  = datasets.ImageFolder(train_dir)
validation_set = datasets.ImageFolder(valid_dir)
 
print('length of train set', len(training_set))
print('length of validation set', len(validation_set))

# get a random sample (30 images) of the data set and print the shape and the label, just to ensure correct operation
train_sample = torchdata.RandomSampler(training_set,replacement=True, num_samples=30)
for sample in train_sample:
    print('--------------------------')
    print('image shape --> ', np.array(training_set[sample][0]).shape)
    print('image class -->', training_set[sample][1])
    print('--------------------------')

