import torch
import time
import copy

# train model function
def train_model(model, loss_function, optimiser, scheduler, num_epochs, train_data, valid_data):

    # create data dictionary --> using data loaders

    data_dict : { 
        'train': train_data,
        'valid': valid_data        
    }

    start = time.time()

    best_model_weights = copy.deepcopy(model.state_dict)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)


