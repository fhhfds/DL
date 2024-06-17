import cv2
import torch

import numpy as np
from tqdm import tqdm
import pandas as pd


def Predict(model, dataloader, device):
    '''预测图片'''
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            y_preds = model(inputs)
            all_preds.append(y_preds)

    return all_preds

def Predict1(model, inputs, device):
    '''预测图片'''
    model.eval()
    all_preds = []

    with torch.no_grad():
        inputs = inputs.to(device)
        y_preds = model(inputs)
        all_preds.append(y_preds)

    return all_preds