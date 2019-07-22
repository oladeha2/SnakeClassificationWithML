import os
import torch
import torchvision
import torch.utils.data as torchdata

# 95% train and 5% validation
split_ratio = 0.95

#training and validation set directories (create and define)
train_dir = '../data/snake/train/'
valid_dir = '../data/snake/valid/'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

#data_set
data = '../data/train/' # --> change path here to point to entire dataset

for class_ in os.listdir(data):
    # get the current class from the main data set and get the number of images and the path to each image
    current = os.path.join(data, class_)
    number_of_images = len(os.listdir(current))
    images = os.listdir(current)
    
    print('current class --> ', class_, 'number of images', number_of_images)

    
    # randomly sample the indices
    image_samples = list(torchdata.SubsetRandomSampler(range(0,number_of_images)))

    thresh = round(len(images)*split_ratio)
    
    class_train = os.path.join(train_dir,class_)
    class_valid  = os.path.join(valid_dir, class_) 
    
    os.makedirs(class_train, exist_ok=True)
    os.makedirs(class_valid, exist_ok=True)
    
    for idx, sample in enumerate(image_samples):
        current_image_path = os.path.join(current, images[sample])
        if idx < thresh:
            train_image_path = os.path.join(class_train, images[sample])
            os.rename(current_image_path, train_image_path)
        else:
            valid_image_path = os.path.join(class_valid, images[sample])
            os.rename(current_image_path, valid_image_path)
            
        
    # create the final path for the training and validation sets

    print('train --> ', class_train, ' number of samples ->', len(os.listdir(class_train)))
    print('valid --> ', class_valid, 'number of samples ->', len(os.listdir(class_valid)))
    print('---------------------------------')