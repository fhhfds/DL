import os
import io
import cv2
import numpy as np
from collections import namedtuple
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# tensor化和标准化
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
])

# koto数据集
cs_labels = namedtuple('peopleClass', ['name', 'train_id', 'color'])
cs_classes = [
    cs_labels('bg', 0, (128, 64, 128)),
    cs_labels('person', 1, (220, 20, 60))
]

train_id_to_color = [c.color for c in cs_classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)


class peopleDataset(Dataset):
    def __init__(self, rootDir: str, folder: str, tf=None):
        """
        Args:
            rootDir (str): 文件路径
            folder (str) : 'train' or 'val' folder
        """
        self.rootDir = rootDir
        self.folder = folder
        self.transform = tf

        assert self.folder in ['train', 'valid', 'predict'], \
            "mode should be 'train' or 'valid' or 'predict', but got {}".format(self.mode)

        self.train_images = []
        self.label_images = []

        with open('data/koto/{}_list.txt'.format(self.folder), 'r') as f:
            for line in f.readlines():
                image, label = line.strip().split(' ')
                # print(label)
                # image = image.reimageplace('/mnt/d/data/', '')
                # label = label.replace('/mnt/d/data/', '')
                self.train_images.append(image)
                self.label_images.append(label)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        # read source image and convert to RGB, apply transform

        sourceImage = cv2.imread(self.train_images[index], -1)
        # print(sourceImage.shape)
        # 将 cv2 图像转换为 PIL.Image
        sourceImage = Image.fromarray(cv2.cvtColor(sourceImage, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            sourceImage = self.transform(sourceImage)

        # read label image and convert to torch tensor
        labelImage = cv2.imread(self.label_images[index], -1)
        # 将 labelImage 转换为 PIL.Image
        labelImage_pil = Image.fromarray(labelImage)

        # 进行缩放等操作
        labelImage_pil = labelImage_pil.resize((384, 384))

        # 再转换回 numpy.ndarray
        labelImage = np.array(labelImage_pil)

        # 将Label图像转换为标签，0：背景，1：人像
        if labelImage[0][0] == 0:
            # 背景黑色，人像白色
            for i in range(len(labelImage)):
                for j in range(len(labelImage[i])):
                    if labelImage[i][j] > 128:
                        labelImage[i][j] = 1
                    else:
                        labelImage[i][j] = 0
        else:
            # 背景白色，人像黑色
            for i in range(len(labelImage)):
                for j in range(len(labelImage[i])):
                    if labelImage[i][j] < 128:
                        labelImage[i][j] = 1
                    else:
                        labelImage[i][j] = 0

        labelImage = torch.from_numpy(labelImage).long()
        return sourceImage, labelImage


def get_kt_datasets(rootDir):
    data = peopleDataset(rootDir, folder='train', tf=preprocess)
    test_set = peopleDataset(rootDir, folder='valid', tf=preprocess)

    # split train data into train, validation and test sets
    total_count = len(data)
    train_count = int(0.8 * total_count)
    train_set, val_set = torch.utils.data.random_split(data, (train_count, total_count - train_count),
                                                       generator=torch.Generator().manual_seed(1))
    return train_set, val_set, test_set


def get_pre_datasets(rootDir):
    data = peopleDataset(rootDir, folder='predict', tf=preprocess)

    return data


class HkpeopleDataset(Dataset):
    def __init__(self, rootDir: str, folder: str, tf=None):
        """
        Args:
            rootDir (str): 文件路径
            folder (str) : 'train' or 'val' folder
        """
        self.rootDir = rootDir
        self.folder = folder
        self.transform = tf

        assert self.folder in ['training', 'testing', 'predict'], \
            "mode should be 'train' or 'valid' or 'predict', but got {}".format(self.mode)

        self.train_images = []
        self.label_images = []

        folder_path = os.path.join(self.rootDir, self.folder)

        for file in os.listdir(folder_path):
            if file.endswith('_matte.png'):
                self.label_images.append(os.path.join(folder_path, file))
            else:
                self.train_images.append(os.path.join(folder_path, file))
        # print(self.label_images[1])

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        # read source image and convert to RGB, apply transform

        sourceImage = cv2.imread(self.train_images[index], -1)
        # print(sourceImage.shape)
        # 将 cv2 图像转换为 PIL.Image
        sourceImage = Image.fromarray(cv2.cvtColor(sourceImage, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            sourceImage = self.transform(sourceImage)

        # read label image and convert to torch tensor
        labelImage = cv2.imread(self.label_images[index], -1)
        # print(labelImage.shape)
        # 将 labelImage 转换为 PIL.Image
        labelImage_pil = Image.fromarray(labelImage)

        # 进行缩放等操作
        labelImage_pil = labelImage_pil.resize((384, 384))

        # 再转换回 numpy.ndarray
        labelImage = np.array(labelImage_pil)

        # 将Label图像转换为标签，0：背景，1：人像
        if labelImage[0][0] == 0:
            # 背景黑色，人像白色
            for i in range(len(labelImage)):
                for j in range(len(labelImage[i])):
                    if labelImage[i][j] > 128:
                        labelImage[i][j] = 1
                    else:
                        labelImage[i][j] = 0
        else:
            # 背景白色，人像黑色
            for i in range(len(labelImage)):
                for j in range(len(labelImage[i])):
                    if labelImage[i][j] < 128:
                        labelImage[i][j] = 1
                    else:
                        labelImage[i][j] = 0

        labelImage = torch.from_numpy(labelImage).long()
        return sourceImage, labelImage


def get_Hk_datasets(rootDir):
    train_set = HkpeopleDataset(rootDir, folder='training', tf=preprocess)
    test_set = HkpeopleDataset(rootDir, folder='testing', tf=preprocess)

    return train_set, test_set

def get_nor_img(path):
    image = Image.open(path)
    tensor = preprocess(image)
    batch = tensor.unsqueeze(0).repeat(1, 1, 1, 1)  # 添加一个维度并复制
    return batch