import numpy as np
import random
import os
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import math
import cv2
import matplotlib.pyplot as plt
from preprocess import Resize, GaussianNoise, RandomAffine, Blurfilter, \
    ToPILImage, ToTensor, Stage2_ToTensor, Stage2_RandomAffine, Stage2_GaussianNoise, Stage2ToPILImage, OrigPad, \
    Stage2_nose_mouth_RandomAffine


class Stage1Augmentation(object):
    def __init__(self, dataset, txt_file, root_dir, resize):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
        # self.augmentation_name = ['origin', 'choice1']
        self.randomchoice = None
        self.transforms = None
        self.transforms_list = None
        self.dataset = dataset
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.resize = resize
        self.set_choice()
        self.set_transformers()
        self.set_transforms_list()

    def set_choice(self):
        choice = {
            # random_choice 1:  Blur, rotaion, Blur + rotation + scale_translate (random_order)
            self.augmentation_name[1]: [GaussianNoise(),
                                        RandomAffine(degrees=30, translate=(0.5, 0.5),
                                                     scale=(0.8, 2)),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=30, translate=(0.5, 0.5),
                                                                             scale=(0.8, 2))
                                                                ]
                                                               )
                                        ],
            # random_choice 2:  noise, crop, noise + crop + rotation_scale_translate (random_order)
            self.augmentation_name[2]: [Blurfilter(),
                                        RandomAffine(degrees=30, translate=(0.8, 0.8),
                                                     scale=(0.8, 2)),
                                        transforms.RandomOrder([Blurfilter(),
                                                                RandomAffine(degrees=30, translate=(0.5, 0.5),
                                                                             scale=(0.8, 2))
                                                                ]
                                                               )
                                        ],
            # random_choice 3:  noise + blur , noise + rotation ,noise + blur + rotation_scale_translate
            self.augmentation_name[3]: [transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=30, translate=(0.5, 0.5),
                                                                             scale=(0.8, 1.8))
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=15, translate=(0.5, 0.8),
                                                                             scale=(0.8, 1.8), shear=60)
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                Blurfilter(),
                                                                RandomAffine(degrees=15, translate=(0.8, 0.5),
                                                                             scale=(0.8, 1.5))
                                                                ]
                                                               )
                                        ],
            # random_choice 4:  noise + crop , blur + crop ,noise + blur + crop + rotation_scale_translate
            self.augmentation_name[4]: [transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=30, translate=(0.8, 0.5),
                                                                             scale=(0.8, 1.5))]
                                                               ),
                                        transforms.Compose([Blurfilter(),
                                                            RandomAffine(degrees=15, translate=(0.5, 0.8),
                                                                         scale=(0.8, 1.2))
                                                            ]
                                                           ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                Blurfilter(),
                                                                RandomAffine(degrees=30, translate=(0.5, 0.5),
                                                                             scale=(0.8, 1))
                                                                ]
                                                               )
                                        ]
        }
        self.randomchoice = choice

    def set_resize(self, resize):
        self.resize = resize

    def set_transformers(self):
        self.transforms = {
            self.augmentation_name[0]: transforms.Compose([
                ToPILImage(),
                Resize(self.resize),
                ToTensor(),
                OrigPad()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                ToPILImage(),
                # Choose from tranforms_list randomly
                transforms.RandomChoice(self.randomchoice['choice1']),
                Resize(self.resize),
                ToTensor(),
                OrigPad()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice2']),
                Resize(self.resize),
                ToTensor(),
                OrigPad()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice3']),
                Resize(self.resize),
                ToTensor(),
                OrigPad()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice4']),
                Resize(self.resize),
                ToTensor(),
                OrigPad()
            ])
        }

    def set_transforms_list(self):
        self.transforms_list = {
            'train':
                self.transforms,
            'val':
                self.transforms['origin']
        }

    def get_dataset(self):
        datasets = {'train': [self.dataset(txt_file=self.txt_file['train'],
                                           root_dir=self.root_dir,
                                           transform=self.transforms_list['train'][r]
                                           )
                              for r in self.augmentation_name],
                    'val': self.dataset(txt_file=self.txt_file['val'],
                                        root_dir=self.root_dir,
                                        transform=self.transforms_list['val']
                                        )
                    }
        enhaced_datasets = {'train': ConcatDataset(datasets['train']),
                            'val': datasets['val']
                            }

        return enhaced_datasets


class Stage2Augmentation(object):
    def __init__(self, dataset, txt_file, root_dir, resize=None):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
        # self.augmentation_name = ['origin', 'choice1']
        self.randomchoice = None
        self.transforms = None
        self.transforms_list = None
        self.dataset = dataset
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.resize = resize
        self.set_choice()
        self.set_transformers()
        self.set_transforms_list()

    def set_choice(self):
        degree_small = (-15, 15)
        degree_large = (-15, 15)
        None_degree = 0

        translate_small = (0.1, 0.1)
        translate_normal = (0.3, 0.3)

        scale_small = (0.8, 1)
        scale_mouth_translate = (0.8, 1)
        scale_large = (1, 1.5)
        scale_with_translate = (1, 1)

        def rand_affine(degree_eyes, degree_mouth, translate_eyes, translate_mouth, scale_eyes, scale_mouth,
                        noise=False):
            if not noise:
                out = transforms.RandomOrder(
                    [Stage2_RandomAffine(degrees=degree_eyes, translate=translate_eyes,
                                         scale=scale_eyes),
                     Stage2_nose_mouth_RandomAffine(degrees=degree_mouth, translate=translate_mouth,
                                                    scale=scale_mouth)
                     ]
                )
            else:
                out = transforms.Compose([
                    Stage2_GaussianNoise(),
                    transforms.RandomOrder([
                        Stage2_RandomAffine(degrees=degree_eyes, translate=translate_eyes,
                                            scale=scale_eyes),
                        Stage2_nose_mouth_RandomAffine(degrees=degree_mouth,
                                                       translate=translate_mouth,
                                                       scale=scale_mouth)
                    ])
                ]
                )

            return out

        choice = {
            # random_choice 1: 30 rotaion, scale, translate, noise (random_order)
            #  R, S, T, N
            self.augmentation_name[1]: [
                # rotate only
                rand_affine(degree_eyes=degree_large, degree_mouth=degree_large, translate_eyes=None,
                            translate_mouth=None, scale_eyes=None, scale_mouth=None),
                # scale only
                rand_affine(degree_eyes=None_degree, degree_mouth=None_degree, translate_eyes=None,
                            translate_mouth=None, scale_eyes=scale_large, scale_mouth=scale_small),
                # translate only
                rand_affine(degree_eyes=None_degree, degree_mouth=None_degree, translate_eyes=translate_normal,
                            translate_mouth=translate_small, scale_eyes=None, scale_mouth=None),
                # noise only
                Stage2_GaussianNoise()
            ],
            self.augmentation_name[2]: [
                #  RS,RT,RN,ST,SN,TN
                rand_affine(degree_eyes=degree_small, degree_mouth=degree_large, translate_eyes=None,
                            translate_mouth=None, scale_eyes=scale_large, scale_mouth=scale_small),
                rand_affine(degree_eyes=degree_small, degree_mouth=degree_large, translate_eyes=translate_normal,
                            translate_mouth=translate_small, scale_eyes=None, scale_mouth=None),
                rand_affine(degree_eyes=degree_small, degree_mouth=degree_large, translate_eyes=None,
                            translate_mouth=None, scale_eyes=None, scale_mouth=None,
                            noise=True),
                rand_affine(degree_eyes=None_degree, degree_mouth=None_degree, translate_eyes=translate_normal,
                            translate_mouth=translate_normal,
                            scale_eyes=scale_with_translate, scale_mouth=scale_mouth_translate),
                rand_affine(degree_eyes=None_degree, degree_mouth=None_degree, translate_eyes=None,
                            translate_mouth=None, scale_eyes=scale_large, scale_mouth=scale_small,
                            noise=True),
                rand_affine(degree_eyes=None_degree, degree_mouth=None_degree, translate_eyes=translate_normal,
                            translate_mouth=translate_small, scale_eyes=None, scale_mouth=None,
                            noise=True)
            ],
            # RST, RSN, RTN, STN
            self.augmentation_name[3]: [
                transforms.RandomOrder([
                    rand_affine(degree_eyes=degree_small, degree_mouth=degree_large, translate_eyes=translate_normal,
                                translate_mouth=translate_normal, scale_eyes=scale_with_translate,
                                scale_mouth=scale_mouth_translate),
                    rand_affine(degree_eyes=degree_small, degree_mouth=degree_large, translate_eyes=None,
                                translate_mouth=None, scale_eyes=scale_large, scale_mouth=scale_small,
                                noise=True),
                    rand_affine(degree_eyes=degree_small, degree_mouth=degree_large, translate_eyes=translate_normal,
                                translate_mouth=translate_small, scale_eyes=None, scale_mouth=None,
                                noise=True),
                    rand_affine(degree_eyes=None_degree, degree_mouth=None_degree, translate_eyes=translate_normal,
                                translate_mouth=translate_normal,
                                scale_eyes=scale_with_translate, scale_mouth=scale_mouth_translate,
                                noise=True),
                ])
            ],

            #  RSTN
            self.augmentation_name[4]: [
                rand_affine(degree_eyes=degree_small, degree_mouth=degree_large, translate_eyes=translate_normal,
                            translate_mouth=translate_normal,
                            scale_eyes=scale_with_translate, scale_mouth=scale_mouth_translate,
                            noise=True)
            ]
        }
        self.randomchoice = choice

    def set_resize(self, resize):
        self.resize = resize

    def set_transformers(self):
        self.transforms = {
            self.augmentation_name[0]: transforms.Compose([
                Stage2ToPILImage(),
                Stage2_ToTensor()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                Stage2ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice1']),
                Stage2_ToTensor()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                Stage2ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice2']),
                Stage2_ToTensor()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                Stage2ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice3']),
                Stage2_ToTensor()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                Stage2ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice4']),
                Stage2_ToTensor()
            ])
        }

    def set_transforms_list(self):
        self.transforms_list = {
            'train':
                self.transforms,
            'val':
                self.transforms['origin']
        }

    def get_dataset(self):
        datasets = {'train': [self.dataset(txt_file=self.txt_file['train'],
                                           root_dir=self.root_dir,
                                           transform=self.transforms_list['train'][r]
                                           )
                              for r in self.augmentation_name],
                    'val': self.dataset(txt_file=self.txt_file['val'],
                                        root_dir=self.root_dir,
                                        transform=self.transforms_list['val']
                                        )
                    }
        enhaced_datasets = {'train': ConcatDataset(datasets['train']),
                            'val': datasets['val']
                            }

        return enhaced_datasets
