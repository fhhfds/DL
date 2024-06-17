# basic imports
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple

# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# for interactive widgets
# import IPython.display as Disp
# from ipywidgets import widgets

###################################
# FILE CONSTANTS
###################################

# Convert to torch tensor and normalize images using Imagenet values
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
])

# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to
# transform to normal RGB format
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))
])

# Constants for Standard color mapping
# reference : https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py

Label = namedtuple("Label", ["name", "train_id", "color"])
drivables = [
    Label("direct", 0, (219, 94, 86)),  # red
    Label("alternative", 1, (86, 211, 219)),  # cyan
    Label("background", 2, (0, 0, 0)),  # black
]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)


###################################
# METRIC CLASS DEFINITION
###################################

class meanIoU:
    """ Class to find the mean IoU using confusion matrix approach """

    def __init__(self, num_classes):
        self.iou_metric = 0.0
        self.num_classes = num_classes
        # placeholder for confusion matrix on entire dataset
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, y_preds, labels):
        """ Function finds the IoU for the input batch
        and add batch metrics to overall metrics """
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix

    def _fast_hist(self, label_true, label_pred):
        """ Function to calculate confusion matrix on single batch """
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def compute(self):
        """ Computes overall meanIoU metric from confusion matrix data """
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        return mean_iu

    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


def plot_training_results(df, model_name):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('validationLoss', color='tab:blue')
    ax2.plot(df['epoch'].values, df['validationLoss'].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.suptitle(f'{model_name} Training, Validation Curves')

    plt.savefig("results/loss_result.png")
    plt.show()

    # 绘制 meanIoU 曲线
    fig2, ax3 = plt.subplots(figsize=(10, 4))
    ax3.set_ylabel('meanIoU')
    ax3.plot(df['epoch'].values, df['meanIoU'].values)
    ax3.tick_params(axis='y')

    plt.suptitle(f'{model_name} meanIoU Curve')

    plt.savefig("results/meanIoU_result.png")
    plt.show()

def visualize_predictions(model: torch.nn.Module, dataSet: Dataset,
                          axes, device: torch.device, numTestSamples: int,
                          id_to_color: np.ndarray = train_id_to_color):
    """Function visualizes predictions of input model on samples from
    cityscapes dataset provided

    Args:
        model (torch.nn.Module): model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        device (torch.device): compute device as in GPU, CPU etc
        numTestSamples (int): number of samples to plot
        id_to_color (np.ndarray) : array to map class to colormap
    """
    model.to(device=device)
    model.eval()

    # predictions on random samples
    testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    # _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 4))

    for i, sampleID in enumerate(testSamples):
        inputImage, gt = dataSet[sampleID]

        # input rgb image
        inputImage = inputImage.to(device)
        landscape = inverse_transform(inputImage).permute(1, 2, 0).cpu().detach().numpy()
        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title("Landscape")

        # groundtruth label image
        label_class = gt.cpu().detach().numpy()
        axes[i, 1].imshow(id_to_color[label_class])
        axes[i, 1].set_title("Groudtruth Label")

        # predicted label image
        y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)
        label_class_predicted = y_pred.cpu().detach().numpy()
        axes[i, 2].imshow(id_to_color[label_class_predicted])
        axes[i, 2].set_title("Predicted Label")

    plt.savefig("results/visualize_prediction.png")
    plt.show()

def koutu(img ,all_pre):
    data = all_pre[0][0].permute(1, 2, 0)
    mask = torch.argmax(data, dim=-1)

    # Move the tensor from GPU to CPU and convert to NumPy array
    mask_np = mask.cpu().numpy()

    # 创建背景图，此处选用白色背景
    bg = np.full([384, 384, 3], 255)

    # 将预测结果的单通道扩展为RGB三通道
    mask_rgb = []
    for i in range(3):
        mask_rgb.append(mask.cpu().numpy())

    photo_mask = np.array(mask_rgb).transpose((1, 2, 0))

    # 抠背景
    photo_bg = bg * (1 - photo_mask)

    # 抠人像
    photo_per = img * photo_mask

    # 将背景和人像结合
    photo = photo_bg + photo_per

    return photo