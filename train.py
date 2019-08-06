
import torch
import time
import copy
import os

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from progress.bar import IncrementalBar

# train model function
def train_model(model, loss_function, optimiser, scheduler, num_epochs, data_dict, data_lengths):

    device = 'cuda'
    os.makedirs('models/dropout/', exist_ok=True)
    os.makedirs('csvs/', exist_ok=True)

    # create data dictionary --> using data loaders

    start = time.time()

    # load the model weights and copy in here to be returned at end of code
    best_model_weights = copy.deepcopy(model.state_dict)
    best_acc = 0.0
    best_f1 = 0.0

    train_epoch_losees = []
    valid_epoch_losses = []

    train_epoch_accuracy = []
    valid_epoch_accuracy = []

    train_f1  = []
    valid_f1 = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 30)

        epoch_start_time = time.time()

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
                f_score = f1_score(y_true=y_true, y_pred=predictions, average='macro')
                f1_scores.append(f_score)

                # calculate loss and accuracy for each batch and add to overall
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # calculate overall loss accuracy and F1 for current epoch
            epoch_loss = running_loss/data_lengths[phase]
            epoch_acc = running_corrects.double() / data_lengths[phase]
            f1_epoch = np.mean(np.array(f1_scores))

            print('Phase: {} Loss: {:.4f} Acc: {:.4f} F1 Score: {:.4f}'.format(phase, epoch_loss, epoch_acc, f1_epoch)) 

            #create columns for the final datda frame for all the data
            if phase == 'train':
                train_epoch_losees.append(epoch_loss)
                train_epoch_accuracy.append(epoch_acc) 
                train_f1.append(f1_epoch)
            else:
                valid_epoch_losses.append(epoch_loss)
                valid_epoch_accuracy.append(epoch_loss)
                valid_f1.append(f1_epoch)


            # save the model based on the highest validation or f1 score in the validation phase of the trainning
            if phase == 'valid' and f1_epoch > best_f1:
                best_f1  = f1_epoch
                # copy the model weights so they can be returned as part of this function
                best_wts = copy.deepcopy(model.state_dict())
                # save the best model here based on the best F1 score 
                torch.save(model, 'models/dropout/pretrained_resnet')
                print('CURRENT BEST MODEL')

        end_epoch_time = time.time() - epoch_start_time
        print('Epoch Time: {:.0f}m {:.0f}s'.format(
            end_epoch_time // 60, end_epoch_time % 60
        ))

        print()

    # # create and save the data frame with all data
    # cols = ['train_loss', 'train_acc', 'train_f1', 'valid_loss', 'valid_acc', 'valid_f1']
    # data = np.array([train_epoch_losees, train_epoch_accuracy, train_f1, valid_epoch_losses, valid_epoch_accuracy, valid_f1])

    # df = pd.DataFrame(
    #     data, 
    #     columns=cols
    # )
    # df.to_csv(
    #     'csvs/pre_trained_resnet101_baseline.csv',
    #     index=False
    # )

    # load model with the best weights and return
    model.load_state_dict(best_wts)

    train_time = time.time() - start

    # print time taken to train model
    print('Training Complete in {:.0f}m {:.0f}s'.format(
        train_time // 60, train_time % 60
    ))

    return model






