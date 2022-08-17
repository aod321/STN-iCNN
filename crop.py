from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
import os
from dataset import HelenDataset
from tensorboardX import SummaryWriter
from preprocess import ToTensor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="/home/user/new_train/datas/data", type=str, help="Path for HelenDataet")
parser.add_argument("--save_dir", default="/home/user/recroped_parts", type=str, help="Path for save data")
args = parser.parse_args()
print(args)

root_dir = args.root_dir
save_dir = args.save_dir
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

root_dir = {
    'image': root_dir,
    'parts': None
}
txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt",
    'test': "testing.txt"
}

Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir['image'],
                           parts_root_dir=None,
                           transform=ToTensor()
                           )
           for x in ['train', 'val', 'test']
           }

dataloader = {x: DataLoader(Dataset[x], batch_size=1,
                            shuffle=False, num_workers=1)
              for x in ['train', 'val', 'test']
              }

tb_logger = SummaryWriter('tb_logs')


def calc_centroid(tensor):
    # Inputs Shape(N, 9 , 64, 64)
    # Return Shape(N, 9 ,2)
    input = tensor.float() + 1e-10
    n, l, h, w = input.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = input.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    center_x = input.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    output = torch.cat([center_y, center_x], 2)
    # output = torch.cat([center_x, center_y], 2)
    return output


def affine_crop(img, label, points, map_location):
    n, l, h, w = img.shape
    img_in = img.clone()
    label_in = label.clone()
    theta = torch.zeros((n, 6, 2, 3), dtype=torch.float32, device=map_location, requires_grad=False)
    points_in = points.clone()
    points_in = torch.cat([points_in[:, 1:6],
                           points_in[:, 6:9].mean(dim=1, keepdim=True)],
                          dim=1)
    points_in = points_in
    assert points_in.shape == (n, 6, 2)
    for i in range(6):
        theta[:, i, 0, 0] = (81 - 1) / (w - 1)
        theta[:, i, 0, 2] = -1 + (2 * points_in[:, i, 1]) / (w - 1)
        theta[:, i, 1, 1] = (81 - 1) / (h - 1)
        theta[:, i, 1, 2] = -1 + (2 * points_in[:, i, 0]) / (h - 1)

    samples = []
    for i in range(6):
        grid = F.affine_grid(theta[:, i], [n, 3, 81, 81], align_corners=True).to(map_location)
        samples.append(F.grid_sample(input=img_in, grid=grid, align_corners=True,
                                     mode='bilinear', padding_mode='zeros'))
    samples = torch.stack(samples, dim=1)
    temp = []
    labels_sample = []

    # Not-mouth Labels
    for i in range(1, 6):
        grid = F.affine_grid(theta[:, i - 1], [n, 1, 81, 81], align_corners=True).to(map_location)
        temp.append(F.grid_sample(input=label_in[:, i:i + 1], grid=grid,
                                  mode='nearest', padding_mode='zeros', align_corners=True))
    for i in range(5):
        bg = torch.tensor(1.) - temp[i]
        labels_sample.append(torch.cat([bg, temp[i]], dim=1))

    temp = []
    # Mouth Labels
    for i in range(6, 9):
        grid = F.affine_grid(theta[:, 5], [n, 1, 81, 81], align_corners=True).to(map_location)
        temp.append(F.grid_sample(input=label_in[:, i:i + 1], grid=grid, align_corners=True,
                                  mode='nearest', padding_mode='zeros'))
    temp = torch.cat(temp, dim=1)
    assert temp.shape == (n, 3, 81, 81)
    bg = torch.tensor(1.) - temp.sum(dim=1, keepdim=True)
    labels_sample.append(torch.cat([bg, temp], dim=1))
    """
    Shape of Parts
    torch.size(N, 6, 3, 81, 81)
    Shape of Labels
    List: [5x[torch.size(N, 2, 81, 81)], 1x [torch.size(N, 4, 81, 81)]]
    """
    assert samples.shape == (n, 6, 3, 81, 81)
    return samples, labels_sample


def crop(mode='train'):
    step = 0
    for iter, batch in enumerate(dataloader[mode]):
        step += 1
        img = batch['image'].to(device)
        label = batch['labels'].to(device)
        index = batch['index']
        points = calc_centroid(label)
        assert points.shape == (img.shape[0], 9, 2)
        parts, parts_labels = affine_crop(img, label, points, device)

        # Check on the tensorboard
        for i in range(6):
            parts_grid = torchvision.utils.make_grid(parts[:, i].detach().cpu())
            labels_grid = torchvision.utils.make_grid(parts_labels[i].argmax(dim=1, keepdim=True).detach().cpu())
            tb_logger.add_image('croped_parts_%d' % i, parts_grid, step)
            tb_logger.add_image('croped_parts_label_%d' % i, labels_grid[0], step, dataformats='HW')

        name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        for i in range(6):
            path = os.path.join(save_dir, name_list[i], mode)
            os.makedirs(path, exist_ok=True)
            # Save into folder
            parts_labels_gt = parts_labels[i].argmax(dim=1, keepdim=False).detach().cpu().type(
                torch.uint8)  # (N, 81, 81)
            for n in range(img.shape[0]):
                names = Dataset[mode].get_name(index[n])
                img_t = TF.to_pil_image(parts[n, i].detach().cpu())
                label_t = TF.to_pil_image(parts_labels_gt[n])
                final_img_path = os.path.join(path, names + "_image.png")
                final_label_path = os.path.join(path, names + "_label.png")
                print(final_img_path)
                print(final_label_path)
                img_t.save(final_img_path, format="PNG", compress_level=0)  # Save cropped image without any compress
                label_t.save(final_label_path, format="PNG", compress_level=0)
    os.system(f"cp {root_dir}/*.txt {save_dir}")
    print("Crop Data for %s Done! ^_^" % mode)


if __name__ == '__main__':
    modes = ['train', 'val', 'test']
    for mode in modes:
        crop(mode)
