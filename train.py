import torch
import time
import copy

# train model function
def train_model(model, loss_function, optimiser, scheduler, num_epochs, data_dict):

    device = 'cuda'

    # create data dictionary --> using data loaders

    start = time.time()

    best_model_weights = copy.deepcopy(model.state_dict)
    best_acc = 0.0

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

            # iterate over the data in the data loaders for train and validation for the differnet phases using the data dictionary

            for images, labels in data_dict[phase]:
                # move data to GPU
                images = images.to(device)
                labels = labels.to(device)

                # set optimzer gradients to zero
                optimiser.zero_grad()

                # get predictions
                # track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(images)
                    print('prediction shape ', predictions.shape)





