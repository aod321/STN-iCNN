import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import io
import cv2


class HelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, parts_root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.mode = 'train'
        if txt_file == "exemplars.txt":
            self.mode = 'train'
        elif txt_file == "testing.txt":
            self.mode = 'test'
        elif txt_file == "tuning.txt":
            self.mode = 'val'
        self.root_dir = root_dir
        self.parts_root_dir = parts_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]

        image = io.imread(img_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(11)]
        labels = np.array(labels)
        # bg = labels[0] + labels[1] + labels[10]
        bg = 255 - labels[2:10].sum(0)
        labels = np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0)

        parts, parts_mask = self.getparts(idx)
        sample = {'image': image, 'labels': labels, 'orig': image, 'orig_label': labels,
                  'parts_gt': parts, 'parts_mask_gt': parts_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def getparts(self, idx):
        name = self.name_list[idx, 1].strip()
        name_list = ['eye1', 'eye2', 'eyebrow1', 'eyebrow2', 'nose', 'mouth']
        path = {x: os.path.join(self.parts_root_dir, x, self.mode)
                for x in name_list}
        parts_path = {x: os.path.join(path[x], name + "_image.png")
                      for x in name_list}
        parts_mask_path = {x: os.path.join(path[x], name + "_label.png")
                           for x in name_list}
        parts = [ io.imread(parts_path[x])
            for x in name_list]

        parts_mask = [cv2.imread(parts_mask_path[x], cv2.IMREAD_GRAYSCALE)
                    for x in name_list]     # (H, W)

        return parts, parts_mask
