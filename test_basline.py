from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import fast_histogram, calc_centroid, affine_crop, affine_mapback
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms.functional as TF
import torch
import os

writer = SummaryWriter('log')
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
# load state

path_model1 = os.path.join("/home/yinzi/data3/new_train/exp_A/f42800da", "best.pth.tar")
path_model2_all = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/cbde5caa", "best.pth.tar")

state1 = torch.load(path_model1, map_location=device)
state2_all = torch.load(path_model2_all, map_location=device)

model1.load_state_dict(state1['model1'])
model2.load_state_dict(state2_all['model2'])

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"
parts_root_dir = "/home/yinzi/data3/recroped_parts"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt",
    'test': "testing.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            ToTensor(),
            Resize((128, 128)),
            OrigPad()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            ToTensor(),
            Resize((128, 128)),
            OrigPad()
        ]),
    'test':
        transforms.Compose([
            ToPILImage(),
            ToTensor(),
            Resize((128, 128)),
            OrigPad()
        ])
}
# DataLoader
Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir,
                           parts_root_dir=parts_root_dir,
                           transform=transforms_list[x]
                           )
           for x in ['test']
           }

dataloader = {x: DataLoader(Dataset[x], batch_size=1,
                            shuffle=False, num_workers=1)
              for x in ['test']
              }
# show predicts
step = 0

for batch in dataloader['test']:
    step += 1
    orig = batch['orig'].to(device)
    orig_label = batch['orig_label'].to(device)
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    parts_gt = batch['parts_gt'].to(device)
    parts_mask_gt = batch['parts_mask_gt'].to(device)
    orig_size = batch['orig_size']
    padding = batch['padding'].to(device)
    names = batch['name']
    N, L, H, W = orig_label.shape

    stage1_pred = F.softmax(model1(image), dim=1)
    big_pred = []
    hist_list = []
    for w in range(N):
        temp_pred = []
        temp = F.interpolate(stage1_pred[w].unsqueeze(0),
                             (orig_size[w][1].item(), orig_size[w][0].item()),
                             mode='nearest').squeeze(0).detach().cpu()
        for i in range(9):
            temp_pred.append(TF.to_tensor(TF.pad(TF.to_pil_image(temp[i]), tuple(padding[w].tolist()))))
        big_pred.append(torch.cat(temp_pred, dim=0))
    big_pred = torch.stack(big_pred, dim=0)
    assert big_pred.shape == (N, 9, 1024, 1024)

    cens = torch.floor(calc_centroid(big_pred))

    # Test B
    # theta = select_model(label)
    parts, parts_labels, theta = affine_crop(orig, orig_label, points=cens, map_location=device)
    # parts, parts_labels, _ = affine_crop(orig, orig_label, theta_in=theta, map_location=device)

    stage2_pred = model2(parts)

    # imshow predict
    # for i in range(6):
    #     pred_grid = torchvision.utils.make_grid(stage2_pred[i].argmax(dim=1, keepdim=True))
    #     writer.add_image('stage2 predict_%d' % i, pred_grid[0], global_step=step, dataformats='HW')

    temp = []
    for i in range(theta.shape[1]):
        test = theta[:, i]
        grid = F.affine_grid(theta=test, size=[N, 3, 81, 81], align_corners=True)
        temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
    parts = torch.stack(temp, dim=1)
    assert parts.shape == (N, 6, 3, 81, 81)

    # imshow crop
    # for i in range(6):
    #     parts_grid = torchvision.utils.make_grid(
    #         parts[:, i].detach().cpu())
    #     writer.add_image('croped_parts_%s_%d' % ("testC_1-27", i), parts_grid, step)

    final_pred = affine_mapback(stage2_pred, theta, device)
    # pred_arg = final_pred.argmax(dim=1, keepdim=False)
    for k in range(final_pred.shape[0]):
        final_out = TF.to_pil_image(final_pred.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        final_out = TF.center_crop(final_out, orig_size[k].tolist())
        orig_out = TF.to_pil_image(orig_label.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        orig_out = TF.center_crop(orig_out, orig_size[k].tolist())
        final_out.save("/home/yinzi/data3/pred_out/%s.png" % names[k], format="PNG", compress_level=0)
        orig_out.save("/home/yinzi/data3/out_gt/%s.png" % names[k], format="PNG", compress_level=0)

    # hist_list.append(fast_histogram(pred_arg.long().cpu().numpy(),
    #                                 orig_label.argmax(dim=1, keepdim=False).cpu().numpy(),
    #                                 9, 9))

    # final_grid = torchvision.utils.make_grid(pred_arg)
    # writer.add_image("final predict_testC_1-27", final_grid[0], global_step=step, dataformats='HW')

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
#
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
