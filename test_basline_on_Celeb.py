from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import CelebAMask
from torchvision import transforms
from celebAMask_preprocess import Resize
from torch.utils.data import DataLoader
from helper_funcs import fast_histogram, calc_centroid, affine_crop, affine_mapback
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms.functional as TF
import torch
import os
import matplotlib.pyplot as plt

writer = SummaryWriter('log')
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
# load state

path_model1 = os.path.join("/home/yinzi/data3/new_train/exp_A/f42800da", "best.pth.tar")
# path_model2_all = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/cbde5caa", "best.pth.tar")
# path_model2_all = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/e5a55dfc", "best.pth.tar")
path_model2_all = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/cec0face", "best.pth.tar")

state1 = torch.load(path_model1, map_location=device)
state2_all = torch.load(path_model2_all, map_location=device)

model1.load_state_dict(state1['model1'])
model2.load_state_dict(state2_all['model2'])

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = "/home/yinzi/data3/CelebAMask-HQ"

# DataLoader
Dataset = CelebAMask(root_dir=root_dir, mode='test', transform=Resize(128, 128))

dataloader = DataLoader(Dataset, batch_size=1,
                        shuffle=False, num_workers=1)

# show predicts
step = 0

for batch in dataloader:
    step += 1
    orig = batch['orig'].to(device)
    orig_label = batch['orig_label'].to(device)
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    names = batch['name']
    N, L, H, W = orig_label.shape

    stage1_pred = F.softmax(model1(image), dim=1)
    big_pred = F.interpolate(stage1_pred,
                             (512, 512),
                             mode='bilinear',
                             align_corners=True
                             )
    # big_pred = F.pad(big_pred, pad=(256, 256, 256, 256), mode='constant', value=0)
    # assert big_pred.shape == (N, 9, 1024, 1024)
    cens = torch.floor(calc_centroid(big_pred))

    # Test B
    parts, parts_labels, theta = affine_crop(orig, orig_label, points=cens, map_location=device, size=127,
                                             mouth_size=255)

    stage2_pred = model2(parts)

    temp = []
    # for i in range(theta.shape[1]):
    #     test = theta[:, i]
    #     grid = F.affine_grid(theta=test, size=[N, 3, 127, 127], align_corners=True)
    #     temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
    # parts = torch.stack(temp, dim=1)
    # for i in range(6):
    #     stage2_arg = stage2_pred[i].argmax(dim=1, keepdim=False)
    #     stage2_np = stage2_arg.detach().cpu().numpy()
    #     print(stage2_np[stage2_np != 0])
    for i in range(6):
        writer.add_image(f"test_baseline_Celeb_parts_{i}", parts[i][0],
                         global_step=step)
    # assert parts.shape == (N, 6, 3, 127, 127)

    final_pred = affine_mapback(stage2_pred, theta, device, size=512)
    for k in range(final_pred.shape[0]):
        final_out = TF.to_pil_image(final_pred.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        # final_out = TF.center_crop(final_out, (512, 512))
        # final_out_grid = torchvision.utils.make_grid(final_out)
        # writer.add_image("test_baseline_Celeb_parts_final", final_out_grid[0],
        #                  global_step=step, dataformats='HW')
        # plt.imshow(TF.to_pil_image(image[0].detach().cpu()))
        # plt.pause(0.1)
        # plt.imshow(final_out)
        # plt.pause(0.1)
        orig_out = TF.to_pil_image(orig_label.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        # orig_out = TF.center_crop(orig_out, (512, 512))
        # plt.imshow(orig_out)
        # plt.pause(0.1)
        os.makedirs("/home/yinzi/data3/celeb_baseline_pred_out", exist_ok=True)
        os.makedirs("/home/yinzi/data3/celeb_baseline_out_gt", exist_ok=True)
        final_out.save("/home/yinzi/data3/celeb_baseline_pred_out/%s.png" % names[k], format="PNG", compress_level=0)
        orig_out.save("/home/yinzi/data3/celeb_baseline_out_gt/%s.png" % names[k], format="PNG", compress_level=0)
