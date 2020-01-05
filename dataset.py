import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io
import cv2
import torch


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
        labels = np.uint8(np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0))

        parts, parts_mask = self.getparts(idx)
        orig_size = image.shape
        sample = {'image': image, 'labels': labels, 'orig': image, 'orig_label': labels, 'orig_size': orig_size,
                  'parts_gt': parts, 'parts_mask_gt': parts_mask}

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


class OldPartsDataset(Dataset):
    #     # HelenDataset
    def __len__(self):
        return len(self.name_list)

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.transform = transform
        self.names = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        self.label_id = {'eyebrow1': [2],
                         'eyebrow2': [3],
                         'eye1': [4],
                         'eye2': [5],
                         'nose': [6],
                         'mouth': [7, 8, 9]
                         }

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        part_path = [os.path.join(self.root_dir, '%s' % x, 'images',
                                  img_name + '.jpg')
                     for x in self.names]
        labels_path = {x: [os.path.join(self.root_dir, '%s' % x,
                                        'labels', img_name,
                                        img_name + "_lbl%.2d.png" % i)
                           for i in self.label_id[x]]
                       for x in self.names}

        parts_image = [io.imread(part_path[i])
                       for i in range(6)]

        labels = {x: np.array([io.imread(labels_path[x][i])
                               for i in range(len(self.label_id[x]))
                               ])
                  for x in self.names
                  }

        for x in self.names:
            bg = 255 - np.sum(labels[x], axis=0, keepdims=True)  # [1, 64, 64]
            labels[x] = np.uint8(np.concatenate([bg, labels[x]], axis=0))  # [L + 1, 64, 64]

        # labels = {'eyebrow1':,
        #           'eyebrow2':,
        #           'eye1':,
        #           'eye2':,
        #           'nose':,
        #           'mouth':}

        sample = {'image': parts_image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
            # 去掉数据增广后的背景黑边
            img, new_label = sample['image'], sample['labels']
            new_label_fg = {x: torch.sum(new_label[x][1:], dim=0, keepdim=True)  # 1 x 64 x 64
                            for x in ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']}
            labels = []
            for x in ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']:
                new_label[x][0] = 1 - new_label_fg[x]
                labels.append(new_label[x].argmax(dim=0, keepdim=True))
            labels = torch.cat(labels, dim=0)
            assert labels.shape == (6, 81, 81)
            sample = {'image': img, 'labels': labels}
        return sample
