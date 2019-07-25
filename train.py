
import torch
import time
import copy

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from os import makedirs


# train model function
def train_model(model, loss_function, optimiser, scheduler, num_epochs, data_dict, data_lengths):

    device = 'cuda'
    makedirs('/models/', exist_ok=True)

    # create data dictionary --> using data loaders

    start = time.time()

    # load the model weights and copy in here to be returned at end of code
    best_model_weights = copy.deepcopy(model.state_dict)
    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            f1_scores = []

            # iterate over the data in the data loaders for train and validation for the differnet phases using the data dictionary

            for inputs, labels in data_dict[phase]:
                # move data to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set optimzer gradients to zero
                optimiser.zero_grad()

                # get predictions
                # track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # get the prediction which is the highest value from the output 45 nodes in the final linear layer 
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

            y_true = labels.cpu().numpy()
            predictions = preds.cpu().numpy()
            
            # calculate the F1 score for each batch
            f_score = f1_score(y_true=y_true, y_pred=predictions)
            f1_scores.append(f_score)

            # calculate loss and accuracy for each batch and add to overall
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predictions == labels.data)

        # calculate overall loss accuracy and F1 for current epoch
        epoch_loss = running_loss/data_lengths[phase]
        epoch_acc = running_corrects.double() / data_lengths[phase]
        f1_epoch = np.mean(np.array(f1_scores))

        print('Phase: {} Loss: {:.4f} Acc: {:.4f} F1 Score: {:.4f}'.format(phase, epoch_loss, epoch_acc, f1_epoch))

        # save the model based on the highest validation or f1 score in the validation phase of the trainning
        if phase == 'valid' and f1_epoch > best_f1:
            best_f1  = f1_epoch
            # copy the model weights so they can be returned as part of this function
            best_wts = copy.deepcopy(model.state_dict())
            # save the best model here based on the best F1 score 
            model_save_path = 'models/pre _trained_resnet101_Epoch_{}_F1_{:.2f}.pth'.format(epoch, f1_epoch)
            torch.save(model.state_dict(), model_save_path)
            print('CURRENT BEST MODEL')

        print()

    model.load_state_dict(best_wts)

    train_time = time.time() - start

    print('Training Complete in {:.0f}m {:.0f}s'.format(
        train_time // 60, train_time % 60
    ))

    return model






