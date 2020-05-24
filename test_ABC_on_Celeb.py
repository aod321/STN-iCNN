from model import Stage2Model, FaceModel, SelectNet_resnet, SelectNet
from tensorboardX import SummaryWriter
from dataset import CelebAMask
from torchvision import transforms
# from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from celebAMask_preprocess import Resize, OrigPad
from torch.utils.data import DataLoader
from helper_funcs import fast_histogram, calc_centroid, affine_crop, affine_mapback, stage2_pred_softmax, \
    stage2_pred_onehot
import torch.nn.functional as F
import torchvision
import torch
import os
import matplotlib.pyplot as plt
import uuid as uid
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
import timeit

uuid = str(uid.uuid1())[0:10]

writer = SummaryWriter('log')
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
select_model = SelectNet().to(device)
select_res_model = SelectNet_resnet().to(device)

# -------- Before End-to-end ------------
# pathA exp_A/f42800da    pretrianed A best_accu 0.851
pathA = os.path.join("/home/yinzi/data3/new_train/exp_A/f42800da", "best.pth.tar")
stateA = torch.load(pathA, map_location=device)

pathB = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/015d1282", "best.pth.tar")
stateB = torch.load(pathB, map_location=device)
# pathC  checkpoints_C/cbde5caa   pretrain C
# mean_error 0.243
# lbrow_accu 0.852
# rbrow_accu 0.834
# leye_accu 0.883
# reye_accu 0.888
# nose_accu 0.943
# mouth_accu 0.922
# mean_accu 0.8869800780312173
# best_accu 0.889
pathC = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/cbde5caa", "best.pth.tar")
stateC = torch.load(pathC, map_location=device)

# -------- After End-to-end ------------
# pathABC = os.path.join("/home/yinzi/data3/new_train/checkpoints_ABC/db5a0098-9", 'best.pth.tar')
pathABC = os.path.join("/home/yinzi/data3/new_train/checkpoints_ABC/d764a684-9", 'best.pth.tar')
stateABC = torch.load(pathABC, map_location=device)

# pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/7a89bbc8", "best.pth.tar")
# pathABC = os.path.join("/home/yinzi/data3/new_train/checkpoints_ABC/3cdb9922-3", "best.pth.tar")
# stateABC = torch.load(pathABC, map_location=device)

model1.load_state_dict(stateABC['model1'])
select_res_model.load_state_dict(stateABC['select_net'])
model2.load_state_dict(stateABC['model2'])

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = "/home/yinzi/data3/CelebAMask-HQ"

transforms_list = {
    'test':
        transforms.Compose([
            Resize((128, 128))
        ])
}
# DataLoader
Dataset = {x: CelebAMask(
    root_dir=root_dir,
    mode=x,
    transform=transforms_list[x]
)
    for x in ['test']
}

dataloader = {x: DataLoader(Dataset[x], batch_size=1,
                            shuffle=False, num_workers=4)
              for x in ['test']
              }

# show predicts
step = 0
time_count = []
for batch in dataloader['test']:
    step += 1
    orig = batch['orig'].to(device)
    orig_label = batch['orig_label'].to(device)
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    names = batch['name']
    N, L, H, W = orig_label.shape

    start = timeit.default_timer()
    stage1_pred = model1(image)
    assert stage1_pred.shape == (N, 9, 128, 128)
    theta = select_res_model(F.softmax(stage1_pred, dim=1))

    # cens = calc_centroid(orig_label)
    # assert cens.shape == (N, 9, 2)
    parts, parts_labels, _ = affine_crop(orig, orig_label, size=127, mouth_size=255, theta_in=theta,
                                         map_location=device)
    # assert parts.shape == (N, 6, 3, 255, 255), print(parts.shape)

    stage2_pred = model2(parts)

    softmax_stage2 = stage2_pred_softmax(stage2_pred)
    # for r in range(len(softmax_stage2)):
    #     stage2_arg = softmax_stage2[r].argmax(dim=1, keepdim=False)
    #     plt.imshow(stage2_arg[0].cpu())
    #     plt.pause(0.01)

    final_pred = affine_mapback(softmax_stage2, theta, device, size=512)
    end = timeit.default_timer()
    # print("Inference Time:\n")
    # print(str(end - start)+"\n")
    time_count.append(end - start)
    hist_list = []
    for k in range(final_pred.shape[0]):
        final_out = TF.to_pil_image(final_pred.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        final_out = TF.center_crop(final_out, (512, 512))
        # plt.imshow(final_out)
        # plt.pause(0.01)
        orig_out = TF.to_pil_image(orig_label.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        orig_out = TF.center_crop(orig_out, (512, 512))
        os.makedirs("/home/yinzi/data3/celeb_ABC_pred_out", exist_ok=True)
        os.makedirs("/home/yinzi/data3/celeb_ABC_out_gt", exist_ok=True)
        final_out.save("/home/yinzi/data3/celeb_ABC_pred_out/%s.png" % names[k], format="PNG", compress_level=0)
        orig_out.save("/home/yinzi/data3/celeb_ABC_out_gt/%s.png" % names[k], format="PNG", compress_level=0)
        # final_np = np.array(final_out)
        # hist_list.append(fast_histogram(TF.to_tensor(np.array(final_out, dtype=np.float32)).long().numpy(),
        #                                 TF.to_tensor(np.array(orig_out, dtype=np.float32)).long().numpy()
        #                                 , 9, 9))

print("time_mean: ", np.mean(time_count))
# name_list = {
#     'brows': [1, 2],
#     'eyes': [3, 4],
#     'nose': [5],
#     'u-lip': [6],
#     'in-mouth': [7],
#     'l-lip': [8],
#     'mouth': [6, 7, 8],
#     'overall': list(range(1, 9))
# }
# F1 = {'brows': 0.0, 'eyes': 0.0, 'nose': 0.0, 'u-lip': 0.0,
#       'in-mouth': 0.0, 'l-lip': 0.0, 'mouth': 0.0, 'overall': 0.0}
# hist_sum = np.sum(np.stack(hist_list, axis=0), axis=0)
# for name, value in name_list.items():
#     A = hist_sum[value, :].sum()
#     B = hist_sum[:, value].sum()
#     intersected = hist_sum[value, :][:, value].sum()
#     F1[name] = 2 * intersected / (A + B)
#
# for key, value in F1.items():
#     print(f'f1_{key}={value}')
