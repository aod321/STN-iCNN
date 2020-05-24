import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from skimage.util import random_noise
from PIL import Image
import torch.nn.functional as F


class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']
        if type(image) is Image.Image:
            image = TF.to_tensor(image)
        if type(labels[0]) is Image.Image:
            labels = [TF.to_tensor(labels[r]).squeeze(0).squeeze(0)
                      for r in range(len(labels))]

        resized_image = F.interpolate(image.unsqueeze(0),
                                      self.size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(0)
        resized_labels = [F.interpolate(labels[r].unsqueeze(0).unsqueeze(0),
                                        self.size, mode='nearest').squeeze(0)
                          for r in range(len(labels))
                          ]
        resized_labels = torch.stack(resized_labels).squeeze(1)
        assert resized_labels.shape == (9, 128, 128), print(resized_labels.shape)

        sample.update({'image': resized_image, 'labels': resized_labels})

        return sample


class OrigPad(object):
    def __init__(self):
        super(OrigPad, self).__init__()

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        orig_label = sample['orig_label']
        orig = sample['orig']
        pad_orig = TF.to_tensor(TF.pad(TF.to_pil_image(orig), (256, 256, 256, 256)))
        pad_label = [TF.to_tensor(TF.pad(TF.to_pil_image(orig_label[r]), (256, 256, 256, 256)))
                     for r in range(len(orig_label))
                     ]
        pad_label = torch.stack(pad_label).squeeze(dim=1)
        assert pad_orig.shape == (3, 1024, 1024), print(pad_orig.shape)
        assert pad_label.shape == (9, 1024, 1024), print(pad_label.shape)

        sample.update({'orig': pad_orig, 'orig_label': pad_label})

        return sample


class ToPILImage(object):
    """Convert a  ``numpy.ndarray`` to ``PIL Image``

    """

    def __call__(self, sample):
        """
                Args:
                    dict of sample (numpy.ndarray): Image and Labels to be converted.

                Returns:
                    dict of sample(PIL,List of PIL): Converted image and Labels.
        """
        image, labels = sample['image'], sample['labels']

        image = TF.to_pil_image(image)
        labels = np.uint8(labels)
        labels = [TF.to_pil_image(labels[i])
                  for i in range(labels.shape[0])]

        sample.update({'image': image, 'labels': labels})
        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.uint8)
        img = np.where(img != 0, random_noise(img), img)
        img = TF.to_pil_image(np.uint8(255 * img))

        sample.update({'image': img})
        return sample


class RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img = TF.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        labels = [TF.affine(labels[r], *ret, resample=self.resample, fillcolor=self.fillcolor)
                  for r in range(len(labels))]

        sample.update({'image': img, 'labels': labels})
        return sample
