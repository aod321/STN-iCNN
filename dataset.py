import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io
import cv2
import torch
from glob import glob
import re
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class HelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, parts_root_dir, stage=None, transform=None):
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
        self.stage = stage

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
        labels = np.uint8(np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0))

        parts, parts_mask = self.getparts(idx)
        orig_size = image.shape

        if self.stage == 'stage1':
            sample = {'image': image, 'labels': labels}
        else:
            sample = {'image': image, 'labels': labels, 'orig': image, 'orig_label': labels, 'orig_size': orig_size,
                      'parts_gt': parts, 'parts_mask_gt': parts_mask, 'name': img_name}

        if self.transform:
            sample = self.transform(sample)
            new_label = sample['labels']
            new_label_fg = torch.sum(new_label[1:], dim=0, keepdim=True)  # 1 x 128 x 128

            new_label[0] = 1. - new_label_fg
            sample['labels'] = new_label
        return sample

    def getparts(self, idx):
        name = self.name_list[idx, 1].strip()
        name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        path = {x: os.path.join(self.parts_root_dir, x, self.mode)
                for x in name_list}
        parts_path = {x: os.path.join(path[x], name + "_image.png")
                      for x in name_list}
        parts_mask_path = {x: os.path.join(path[x], name + "_label.png")
                           for x in name_list}
        parts = [io.imread(parts_path[x])
                 for x in name_list]

        parts_mask = [cv2.imread(parts_mask_path[x], cv2.IMREAD_GRAYSCALE).astype(np.float32())
                      for x in name_list]  # (H, W)

        return parts, parts_mask


class SkinHelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, transform=None):
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
        bg = 255 - labels[1:11].sum(0)
        labels = np.uint8(np.concatenate(([bg.clip(0, 255)], labels[1:11]), axis=0))
        sample = {'image': image, 'labels': labels, 'name': img_name}

        if self.transform:
            sample = self.transform(sample)
            new_label = sample['labels']
            new_label_fg = torch.sum(new_label[1:], dim=0, keepdim=True)  # 1 x 128 x 128
            new_label[0] = 1. - new_label_fg
            sample['labels'] = new_label
        return sample

    def getparts(self, idx):
        name = self.name_list[idx, 1].strip()
        name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        path = {x: os.path.join(self.parts_root_dir, x, self.mode)
                for x in name_list}
        parts_path = {x: os.path.join(path[x], name + "_image.png")
                      for x in name_list}
        parts_mask_path = {x: os.path.join(path[x], name + "_label.png")
                           for x in name_list}
        parts = [io.imread(parts_path[x])
                 for x in name_list]

        parts_mask = [cv2.imread(parts_mask_path[x], cv2.IMREAD_GRAYSCALE).astype(np.float32())
                      for x in name_list]  # (H, W)

        return parts, parts_mask


class PartsDataset(Dataset):

    def __init__(self, txt_file, root_dir, transform=None):
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
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        path = {x: os.path.join(self.root_dir, x, self.mode)
                for x in name_list}
        parts_path = {x: os.path.join(path[x], img_name + "_image.png")
                      for x in name_list}
        parts_mask_path = {x: os.path.join(path[x], img_name + "_label.png")
                           for x in name_list}
        parts = [io.imread(parts_path[x])
                 for x in name_list]

        parts_mask = [cv2.imread(parts_mask_path[x], cv2.IMREAD_GRAYSCALE).astype(np.float32())
                      for x in name_list]  # (H, W)

        sample = {'image': parts, 'labels': parts_mask}

        if self.transform:
            sample = self.transform(sample)
        return sample


class CelebAMask(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.mode = mode
        path = os.path.join(root_dir, 'CelebA-HQ-img')
        self.length = len(glob(os.path.join(path, '*.jpg')))
        self.image_dir = [os.path.join(path, f'{i}.jpg') for i in range(self.length)]
        self.img_name = [str(i) for i in range(self.length)]

        if self.mode == 'train':
            self.image_dir = self.image_dir[:2000]
        elif self.mode == 'val':
            self.image_dir = self.image_dir[2000:2200]
        elif self.mode == 'test':
            self.image_dir = self.image_dir[2200:2400]

        self.transform = transform
        self.length = len(self.image_dir)
        self.label_name = ['r_brow', 'l_brow', 'r_eye', 'l_eye', 'nose', 'u_lip', 'mouth', 'l_lip']

        if self.mode == 'train':
            self.label_name_list = [["%05d" % i + f"_{self.label_name[k]}"
                                     for k in range(len(self.label_name))]
                                    for i in range(self.length)
                                    ]
            self.label_dir = [[os.path.join(root_dir, 'CelebAMask-HQ-mask-anno',
                                            '0',
                                            self.label_name_list[i][k] + '.png')
                               for k in range(len(self.label_name))]
                              for i in range(self.length)
                              ]
        elif self.mode == 'val':
            self.label_name_list = [["%05d" % i + f"_{self.label_name[k]}"
                                     for k in range(len(self.label_name))]
                                    for i in list(np.array(range(self.length)) + 2000)
                                    ]
            self.label_dir = [[os.path.join(root_dir, 'CelebAMask-HQ-mask-anno',
                                            '1',
                                            self.label_name_list[i][k] + '.png')
                               for k in range(len(self.label_name))]
                              for i in list(np.array(range(self.length)))
                              ]

        elif self.mode == 'test':
            self.label_name_list = [["%05d" % i + f"_{self.label_name[k]}"
                                     for k in range(len(self.label_name))]
                                    for i in list(np.array(range(self.length)) + 2200)
                                    ]
            self.label_dir = [[os.path.join(root_dir, 'CelebAMask-HQ-mask-anno',
                                            '1',
                                            self.label_name_list[i][k] + '.png')
                               for k in range(len(self.label_name))]
                              for i in list(np.array(range(self.length)))
                              ]

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        name = self.img_name[idx]
        img_path = self.image_dir[idx]
        labels_path = self.label_dir[idx]
        image = TF.to_tensor(Image.open(img_path))
        image = F.interpolate(image.unsqueeze(0),
                              (512, 512),
                              mode='bilinear',
                              align_corners=True).squeeze(0)
        labels = torch.zeros(8, 512, 512)
        for i in range(len(self.label_name)):
            try:
                labels[i] = TF.to_tensor(Image.open(labels_path[i]))[0]
            except FileNotFoundError:
                pass
                # print(f"WARNNING: {self.label_name[i]} Label Not Found! All zeros for default.")
        labels = torch.cat([1 - torch.sum(labels, dim=0, keepdim=True), labels], dim=0)
        # labels = labels.argmax(dim=0, keepdim=False)
        sample = {'image': image, 'labels': labels, 'name': name, 'orig': image, 'orig_label': labels}

        if self.transform:
            sample = self.transform(sample)
        return sample
