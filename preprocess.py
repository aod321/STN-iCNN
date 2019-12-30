import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
import random
from skimage.util import random_noise
from PIL import ImageFilter, Image
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
        parts, parts_mask = sample['parts_gt'], sample['parts_gt']
        resized_image = F.interpolate(image.unsqueeze(0), self.size, mode='bilinear', align_corners=True).squeeze(0)

        resized_labels = torch.cat(
            [F.interpolate(labels[r:r+1].unsqueeze(0), self.size, mode='nearest').squeeze(0)
             for r in range(len(labels))
             ], dim=0)
        assert resized_labels.shape == (9, 128, 128)

        sample = {'image': resized_image, 'labels': resized_labels,
                  'orig': sample['orig'], 'orig_label': sample['orig_label'],
                  'parts_gt': parts, 'parts_mask_gt': parts_mask}

        return sample


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image, labels = sample['image'], sample['labels']
        parts, parts_mask = sample['parts_gt'], sample['parts_mask_gt']

        labels = [TF.to_tensor(labels[r])
                  for r in range(len(labels))
                  ]
        labels = torch.cat(labels, dim=0).float()

        parts = torch.stack([TF.to_tensor(parts[r])
                             for r in range(len(parts))])

        parts_mask = torch.cat([TF.to_tensor(parts_mask[r])
                                  for r in range(len(parts_mask))])

        assert parts.shape == (6, 3, 81, 81)
        assert parts_mask.shape == (6, 81, 81)

        sample = {'image': TF.to_tensor(image), 'labels': labels, 'orig': sample['orig'],
                  'orig_label': sample['orig_label'],
                  'parts_gt': parts, 'parts_mask_gt': parts_mask}

        return sample


class Stage2_ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        parts, parts_mask = sample['image'], sample['labels']

        parts = torch.stack([TF.to_tensor(parts[r])
                             for r in range(len(parts))])

        parts_mask = torch.cat([TF.to_tensor(parts_mask[r])
                                for r in range(len(parts_mask))])

        sample = {'image': parts, 'labels': parts_mask}

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
        image, labels = sample['image'], sample['labels']
        parts, parts_mask = sample['parts_gt'], sample['parts_gt']
        orig_label = sample['orig_label']
        orig = TF.to_pil_image(sample['orig'])

        desired_size = 1024
        delta_width = desired_size - orig.size[0]
        delta_height = desired_size - orig.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = np.array([pad_width, pad_height, delta_width - pad_width, delta_height - pad_height])

        pad_orig = TF.to_tensor(TF.pad(orig, tuple(padding)))

        orig_label = [TF.to_tensor(TF.pad(TF.to_pil_image(orig_label[r]), tuple(padding)))
                      for r in range(len(orig_label))
                      ]
        orig_label = torch.cat(orig_label, dim=0).float()
        orig_label[0] = torch.tensor(1.) - torch.sum(orig_label[1:], dim=0, keepdim=True)

        assert pad_orig.shape == (3, 1024, 1024)
        assert orig_label.shape == (9, 1024, 1024)

        sample = {'image': image, 'labels': labels, 'orig': pad_orig, 'orig_label': orig_label,
                  'parts_gt': parts, 'parts_mask_gt': parts_mask}

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
        sample = {'image': img, 'labels': labels, 'orig': sample['orig'], 'orig_label': sample['orig_label'],
                  'parts_gt': sample['parts_gt'], 'parts_mask_gt': sample['parts_gt']}
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
        parts, parts_mask = sample['parts_gt'], sample['parts_gt']
        orig_label = sample['orig_label']
        orig = TF.to_pil_image(sample['orig'])

        orig_label = np.uint8(orig_label)
        orig_label = [TF.to_pil_image(orig_label[r])
                      for r in range(len(orig_label))]

        image = TF.to_pil_image(image)

        labels = np.uint8(labels)
        labels = [TF.to_pil_image(labels[i])
                  for i in range(labels.shape[0])]

        sample = {'image': image, 'labels': labels, 'orig': orig, 'orig_label': orig_label,
                  'parts_gt': sample['parts_gt'], 'parts_mask_gt': sample['parts_gt']}
        return sample


class Stage2ToPILImage(object):
    """Convert a  ``numpy.ndarray`` to ``PIL Image``

    """

    def __call__(self, sample):
        """
                Args:
                    dict of sample (numpy.ndarray): Image and Labels to be converted.

                Returns:
                    dict of sample(PIL,List of PIL): Converted image and Labels.
        """
        parts, parts_mask = sample['image'], sample['labels']

        parts = [TF.to_pil_image(parts[r])
                 for r in range(len(parts))]

        parts_mask = [TF.to_pil_image(parts_mask[r])
                      for r in range(len(parts_mask))]

        sample = {'image': parts, 'labels': parts_mask}

        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.array(img, np.uint8)
        img = random_noise(img)
        img = TF.to_pil_image(np.uint8(255 * img))

        sample = {'image': img, 'labels': sample['labels'], 'orig': sample['orig'],
                  'orig_label': sample['orig_label'], 'parts_gt': sample['parts_gt'],
                  'parts_mask_gt': sample['parts_mask_gt']
                  }
        return sample


class Blurfilter(object):
    # img: PIL image
    def __call__(self, sample):
        img, labels = sample['image'], sample['labels']
        img = img.filter(ImageFilter.BLUR)
        sample = {'image': img, 'labels': sample['labels'], 'orig': sample['orig'],
                  'orig_label': sample['orig_label'], 'parts_gt': sample['parts_gt'],
                  'parts_mask_gt': sample['parts_mask_gt']
                  }

        return sample


class Stage2_RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img = [TF.affine(img[r], *ret, resample=self.resample, fillcolor=self.fillcolor)
               for r in range(len(img))]
        labels = [TF.affine(labels[r], *ret, resample=self.resample, fillcolor=self.fillcolor)
                  for r in range(len(labels))]
        sample = {'image': img, 'labels': labels}
        return sample


class Stage2_GaussianNoise(object):
    def __call__(self, sample):
        parts = sample['image']
        parts = [np.array(parts[r], np.uint8)
                 for r in range(len(parts))]
        parts = [random_noise(parts[r])
                 for r in range(len(parts))]
        parts = [TF.to_pil_image(np.uint8(255 * parts[r]))
                 for r in range(len(parts))
                 ]
        sample = {'image': parts, 'labels': sample['labels']}
        return sample
