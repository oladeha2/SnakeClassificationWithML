"""
Script for testing the prediction format for the snake classification task
"""

import numpy as np
import os
import glob

import torch
import torchvision.models as models
import torchvision.transforms as transforms#
from torchvision import transforms
from PIL import Image


torch.manual_seed(271828)
np.random.seed(271828)

# use sample  test set, which is the validation set in this case

test_dir = '../data/snake/test'

AICROWD_TEST_IMAGES_PATH = os.getenv('AICROWD_TEST_IMAGES_PATH', test_dir)
AICROWD_PREDICTIONS_OUTPUT_PATH = os.getenv('AICROWD_PREDICTIONS_OUTPUT_PATH', 'random_prediction.csv')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


LINES = []

with open('data/class_idx_mapping.csv') as f:
	classes = ['filename']
	for line in f.readlines()[1:]:
		class_name = line.split(",")[0]
		classes.append(class_name)

LINES.append(','.join(classes))

print('-' * 30)

print(LINES)

print('-' * 30)

# load model for inference
model = models.resnet101()
model.load_state_dict('models/pretrained_resnet')
model.eval()
print(model)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.458, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


images_path = AICROWD_TEST_IMAGES_PATH + '/*.jpg'
for _file_path in glob.glob(images_path):

    img = transform(Image.open(_file_path))
    pred = model(img)
    print('prediction shape -> ', pred.shape)

    probs = softmax(np.random.rand(45))
    print('shape -> ', probs.shape)
    probs = list(map(str, probs))
    LINES.append(",".join([os.path.basename(_file_path)] + probs))



