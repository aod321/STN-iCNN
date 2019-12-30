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
from prepcess import GaussianNoise, RandomAffine, Blurfilter, \
    ToPILImage, ToTensor, Stage2_ToTensor, Stage2_ToPILImage, Stage2_RandomAffine, Stage2_GaussianNoise


class Stage1Augmentation(object):
    def __init__(self, dataset, txt_file, root_dir, resize):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
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
                ToTensor()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                ToPILImage(),
                # Choose from tranforms_list randomly
                transforms.RandomChoice(self.randomchoice['choice1']),
                Resize(self.resize),
                ToTensor()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice2']),
                Resize(self.resize),
                ToTensor()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice3']),
                Resize(self.resize),
                ToTensor()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice4']),
                Resize(self.resize),
                ToTensor()
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
    def __init__(self, dataset, txt_file, root_dir, resize):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
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
            self.augmentation_name[1]: [Blurfilter(),
                                        RandomAffine(degrees=15, translate=(0, 0),
                                                     scale=(1, 1)),
                                        transforms.RandomOrder([Stage2_GaussianNoise(),
                                                                Stage2_RandomAffine(degrees=30, translate=(0.3, 0.3),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # random_choice 2:  noise, crop, noise + crop + rotation_scale_translate (random_order)
            self.augmentation_name[2]: [GaussianNoise(),
                                        RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.8, 1.5)),
                                        transforms.RandomOrder([Stage2_GaussianNoise(),
                                                                Stage2_RandomAffine(degrees=30, translate=(0.3, 0.3),
                                                                             scale=(0.8, 1.5))
                                                                ]
                                                               )
                                        ],
            # random_choice 3:  noise + blur , noise + rotation ,noise + blur + rotation_scale_translate
            self.augmentation_name[3]: [transforms.RandomOrder([Stage2_RandomAffine(degrees=30, translate=(0.3, 0.2),
                                                                             scale=(0.8, 1.5)),
                                                                Stage2_RandomAffine(degrees=30, translate=(0.2, 0.3),
                                                                             scale=(1, 1.5))
                                                                ]
                                                               ),
                                        transforms.RandomOrder([
                                                                Stage2_RandomAffine(degrees=30, translate=None,
                                                                             scale=(1, 1)),                                                                
                                                                Stage2_RandomAffine(degrees=0, translate=(0.3, 0.3),
                                                                             scale=(1, 1))
                                                                ]
                                                               
                                                               ),
                                        transforms.RandomOrder([Stage2_GaussianNoise(),
                                                                Stage2_RandomAffine(degrees=30, translate=(0.8, 0.8),
                                                                             scale=(1, 1.5))]
                                                               ),
                                        transforms.RandomOrder([Stage2_RandomAffine(degrees=30, translate=None,
                                                                                    scale=(1, 1))],
                                                               Stage2_RandomAffine(degrees=0, translate=(0.3, 0.3),
                                                                                   scale=(0.5, 1.5))
                                                               )
                                        ],
            # random_choice 4:  noise + crop , blur + crop ,noise + blur + crop + rotation_scale_translate
            self.augmentation_name[4]: [transforms.RandomOrder([RandomAffine(degrees=15, translate=(0.3, 0.3),
                                                                             scale=(0.5, 1)),
                                                                RandomAffine(degrees=30, translate=(0.3, 0.3),
                                                                             scale=(0.5, 1))
                                                                ]
                                                               ),
                                        transforms.Compose([Blurfilter(),
                                                            RandomAffine(degrees=30, translate=(0.3, 0.3),
                                                                         scale=(0.5, 2)),
                                                            ]
                                                           ),
                                        transforms.RandomOrder([
                                                                RandomAffine(degrees=0, translate=(0.3, 0.3),
                                                                  scale=(0.5, 1)),
                                                                RandomAffine(degrees=30, translate=(0.3, 0.3),
                                                                             scale=(1, 1.5))
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
                Stage2_ToPILImage(),
                ToTensor()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                Stage2_ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice1']),
                ToTensor()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                Stage2_ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice2']),
                ToTensor()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                Stage2_ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice3']),
                ToTensor()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                Stage2_ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice4']),
                ToTensor()
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