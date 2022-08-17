from torch.utils.data import ConcatDataset
from torchvision import transforms
from preprocess import Resize, GaussianNoise, RandomAffine, \
    ToPILImage, ToTensor, Stage2_ToTensor, Stage2_RandomAffine, Stage2_GaussianNoise, Stage2ToPILImage, \
    Stage2_nose_mouth_RandomAffine, OrigPad


class Stage1Augmentation(object):
    def __init__(self, dataset, txt_file, root_dir, parts_root_dir, resize, stage="stage1"):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
        self.randomchoice = None
        self.transforms = None
        self.transforms_list = None
        self.dataset = dataset
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.parts_root_dir = parts_root_dir
        self.resize = resize
        self.set_choice()
        self.set_transformers()
        self.set_transforms_list()

    def set_choice(self):
        degree = 15
        translate_range = (0.1, 0.1)
        scale_range = (0.9, 1.2)
        choice = {
            # random_choice 1:
            self.augmentation_name[1]: [GaussianNoise(),
                                        RandomAffine(degrees=degree, translate=translate_range,
                                                     scale=scale_range),
                                        transforms.Compose([GaussianNoise(),
                                                            RandomAffine(degrees=degree, translate=translate_range,
                                                                         scale=scale_range)
                                                            ]
                                                           )
                                        ],
            # random_choice 2: R, S, T
            self.augmentation_name[2]: [
                RandomAffine(degrees=degree, translate=None,
                             scale=None),
                RandomAffine(degrees=0, translate=None,
                             scale=(0.8, 1.5)),
                RandomAffine(degrees=0, translate=(0.3, 0.3),
                             scale=None)
            ],
            # random_choice 3:  RT, RS, ST
            self.augmentation_name[3]: [
                RandomAffine(degrees=degree, translate=translate_range,
                             scale=None),
                RandomAffine(degrees=degree, translate=None,
                             scale=scale_range),
                RandomAffine(degrees=0, translate=translate_range,
                             scale=scale_range),
            ],
            # random_choice 4: RST
            self.augmentation_name[4]: [
                RandomAffine(degrees=degree, translate=translate_range,
                             scale=scale_range),
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
                                           parts_root_dir=self.parts_root_dir,
                                           transform=self.transforms_list['train'][r]
                                           )
                              for r in self.augmentation_name],
                    'val': self.dataset(txt_file=self.txt_file['val'],
                                        root_dir=self.root_dir,
                                        parts_root_dir=self.parts_root_dir,
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

class SkinHairAugmentation(object):
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
        degree = 15
        translate_range = (0.1, 0.1)
        scale_range = (0.9, 1.2)
        choice = {
            # random_choice 1:
            self.augmentation_name[1]: [GaussianNoise(),
                                        RandomAffine(degrees=degree, translate=translate_range,
                                                     scale=scale_range),
                                        transforms.Compose([GaussianNoise(),
                                                            RandomAffine(degrees=degree, translate=translate_range,
                                                                         scale=scale_range)
                                                            ]
                                                           )
                                        ],
            # random_choice 2: R, S, T
            self.augmentation_name[2]: [
                RandomAffine(degrees=degree, translate=None,
                             scale=None),
                RandomAffine(degrees=0, translate=None,
                             scale=(0.8, 1.5)),
                RandomAffine(degrees=0, translate=(0.3, 0.3),
                             scale=None)
            ],
            # random_choice 3:  RT, RS, ST
            self.augmentation_name[3]: [
                RandomAffine(degrees=degree, translate=translate_range,
                             scale=None),
                RandomAffine(degrees=degree, translate=None,
                             scale=scale_range),
                RandomAffine(degrees=0, translate=translate_range,
                             scale=scale_range),
            ],
            # random_choice 4: RST
            self.augmentation_name[4]: [
                RandomAffine(degrees=degree, translate=translate_range,
                             scale=scale_range),
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
