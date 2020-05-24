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


class Stage1ToTensor(transforms.ToTensor):
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

        labels = [TF.to_tensor(labels[r])
                  for r in range(len(labels))
                  ]
        labels = torch.cat(labels, dim=0).float()

        sample.update({'image': TF.to_tensor(image), 'labels': labels})
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
        if type(image) is Image.Image:
            image = TF.to_tensor(image)
        if type(labels[0]) is Image.Image:
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

        sample.update({'image': image, 'labels': labels, 'parts_gt': parts,
                       'parts_mask_gt': parts_mask})
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

        parts = torch.stack([TF.to_tensor(np.array(parts[r]))
                             for r in range(len(parts))])

        parts_mask = torch.cat([TF.to_tensor(np.array(parts_mask[r]))
                                for r in range(len(parts_mask))])
        sample.update({'image': parts, 'labels': parts_mask})

        return sample


class Skin_ToTensor(transforms.ToTensor):
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

        image = TF.to_tensor(image)
        labels = torch.cat([TF.to_tensor(labels[r])
                            for r in range(len(labels))])

        sample = {'image': image, 'labels': labels, 'name': sample['name']}

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
        # parts, parts_mask = sample['parts_gt'], sample['parts_mask_gt']
        orig_label = sample['orig_label']
        orig = sample['orig']
        if type(orig) is not Image.Image:
            orig = TF.to_pil_image(sample['orig'])

        if type(orig_label[0]) is not Image.Image:
            orig_label = [TF.to_pil_image(orig_label[r])
                          for r in range(len(orig_label))]

        desired_size = 1024
        delta_width = desired_size - orig.size[0]
        delta_height = desired_size - orig.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        orig_size = np.array([orig.size[0], orig.size[1]])
        padding = np.array([pad_width, pad_height, delta_width - pad_width, delta_height - pad_height])

        pad_orig = TF.to_tensor(TF.pad(orig, tuple(padding)))

        orig_label = [TF.to_tensor(TF.pad(orig_label[r], tuple(padding)))
                      for r in range(len(orig_label))
                      ]
        orig_label = torch.cat(orig_label, dim=0).float()
        orig_label[0] = torch.tensor(1.) - torch.sum(orig_label[1:], dim=0, keepdim=True)

        assert pad_orig.shape == (3, 1024, 1024)
        assert orig_label.shape == (9, 1024, 1024)

        sample.update({'image': image, 'labels': labels, 'orig': pad_orig, 'orig_label': orig_label,
                       'orig_size': orig_size, 'padding': padding})

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
        sample.update({'image': parts, 'labels': parts_mask})

        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.uint8)
        img = np.where(img != 0, random_noise(img), img)
        img = TF.to_pil_image(np.uint8(255 * img))

        sample.update({'image': img})
        return sample


class Stage2_RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        ret = [self.get_params(self.degrees, self.translate, self.scale, self.shear, img[r].size)
               for r in range(4)]
        new_img = [TF.affine(img[r], *ret[r], resample=self.resample, fillcolor=self.fillcolor)
                   for r in range(4)]
        new_labels = [TF.affine(labels[r], *ret[r], resample=self.resample, fillcolor=self.fillcolor)
                      for r in range(4)]
        for r in range(4):
            img[r] = new_img[r]
            labels[r] = new_labels[r]

        sample.update({'image': img, 'labels': labels})
        return sample


class Stage2_nose_mouth_RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        ret = {r: self.get_params(self.degrees, self.translate, self.scale, self.shear, img[r].size)
               for r in range(4, 6)}
        new_part = [TF.affine(img[r], *ret[r], resample=self.resample, fillcolor=self.fillcolor)
                    for r in range(4, 6)]
        new_labels = [TF.affine(labels[r], *ret[r], resample=self.resample, fillcolor=self.fillcolor)
                      for r in range(4, 6)]
        for r in range(4, 6):
            img[r] = new_part[r - 4]
            labels[r] = new_labels[r - 4]
        sample = {'image': img, 'labels': labels}
        return sample


class Stage2_GaussianNoise(object):
    def __call__(self, sample):
        parts = sample['image']
        parts = [np.array(parts[r], np.uint8)
                 for r in range(len(parts))]
        for r in range(len(parts)):
            parts[r] = np.where(parts[r] != 0, random_noise(parts[r]), parts[r])

        parts = [TF.to_pil_image(np.uint8(255 * parts[r]))
                 for r in range(len(parts))
                 ]
        sample = {'image': parts, 'labels': sample['labels']}
        return sample


name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']


class OldStage2Resize(transforms.Resize):
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
        resized_image = np.array([cv2.resize(image[i], self.size, interpolation=cv2.INTER_AREA)
                                  for i in range(len(image))])
        labels = {x: np.array([np.array(TF.resize(TF.to_pil_image(labels[x][r]), self.size, Image.ANTIALIAS))
                               for r in range(len(labels[x]))])
                  for x in name_list
                  }

        sample = {'image': resized_image,
                  'labels': labels
                  }

        return sample


class OldStage2ToTensor(transforms.ToTensor):
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
        image = sample['image']
        labels = sample['labels']
        image = torch.stack([TF.to_tensor(image[i])
                             for i in range(len(image))])

        labels = {x: torch.cat([TF.to_tensor(labels[x][r])
                                for r in range(len(labels[x]))
                                ])
                  for x in name_list
                  }

        return {'image': image,
                'labels': labels
                }


class OldStage2_ToPILImage(object):
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
        image = [TF.to_pil_image(image[i])
                 for i in range(len(image))]
        labels = {x: [TF.to_pil_image(labels[x][i])
                      for i in range(len(labels[x]))]
                  for x in name_list
                  }

        return {'image': image,
                'labels': labels
                }
