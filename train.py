import cv2
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import meanIoU                  # metric class
from utils import plot_training_results



def evaluate_model(model, dataloader, criterion, metric_class, num_classes, device):
    '''evaluate model on dataset'''
    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric


def train_validate_model(model, num_epochs, model_name, criterion, optimizer,
                         device, dataloader_train, dataloader_valid,
                         metric_class, metric_name, num_classes, lr_scheduler=None,
                         output_path='.'):
    '''training process'''
    # initialize placeholders for running values
    results = []
    min_val_loss = np.Inf
    len_train_loader = len(dataloader_train)

    # move model to device
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting {epoch + 1} epoch ...")

        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(dataloader_train, total=len_train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)
        validation_loss, validation_metric = evaluate_model(
            model, dataloader_valid, criterion, metric_class, num_classes, device)

        print(
            f'Epoch: {epoch + 1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, {metric_name}:{validation_metric: 4.2f}')

        # store results
        results.append({'epoch': epoch,
                        'trainLoss': train_loss,
                        'validationLoss': validation_loss,
                        f'{metric_name}': validation_metric})

        # if validation loss has decreased, save model and reset variable
        if validation_loss <= min_val_loss:
            min_val_loss = validation_loss
            torch.save(model.state_dict(), f"{output_path}/{model_name}.pt")
            # torch.jit.save(torch.jit.script(model), f"{output_path}/{model_name}.pt")

    # plot results
    results = pd.DataFrame(results)
    results.to_csv('results/results_last.csv', index=False)
    plot_training_results(results, model_name)
    return results
