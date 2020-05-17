from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import calc_centroid, affine_crop, affine_mapback, stage2_pred_softmax
from torchvision import transforms
import torchvision
import torch
import os
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.cm


writer = SummaryWriter('exp')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
select_model = SelectNet().to(device)
# load state
path = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/a0d286ea", "best.pth.tar")
path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/396e4702", "best.pth.tar")
path_select = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6", "best.pth.tar")

state_select = torch.load(path_select, map_location=device)
select_model.load_state_dict(state_select['select_net'])

state = torch.load(path, map_location=device)
model1.load_state_dict(state['model1'])
state2 = torch.load(path_model2, map_location=device)
model2.load_state_dict(state2['model2'])

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
            Resize((128, 128)),
            ToTensor(),
            OrigPad()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
            OrigPad()
        ]),
    'test':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
            OrigPad()
        ])
}
# DataLoader
Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir,
                           parts_root_dir=parts_root_dir,
                           transform=transforms_list[x]
                           )
           for x in ['train', 'val', 'test']
           }

dataloader = {x: DataLoader(Dataset[x], batch_size=10,
                            shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']
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
    padding = batch['padding'].to(device)
    names = batch['name']
    orig_size = batch['orig_size']
    N, L, H, W = orig_label.shape

    stage1_pred = F.softmax(model1(image), dim=1)

    # test no eyebrow1
    # stage1_pred[:, 1] = 0

    # # test no eye2
    # stage1_pred[:, 4] = 0
    #
    # # test no mouth
    stage1_pred[:, 6] = 0
    stage1_pred[:, 7] = 0
    stage1_pred[:, 8] = 0

    #
    assert stage1_pred.shape == (N, 9, 128, 128)

    # pad to orig size
    big_pred = []
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

    # imshow stage1 mask predict
    stage1_pred_grid = torchvision.utils.make_grid(stage1_pred.argmax(dim=1, keepdim=True)).cpu()
    writer.add_image("stage1 predict_%s" % 'no_mouth', stage1_pred_grid[0], step, dataformats='HW')

    cens = torch.floor(calc_centroid(big_pred))
    parts, _, theta = affine_crop(orig, orig_label, points=cens, map_location=device, floor=True)
    assert parts.shape == (N, 6, 3, 81, 81)
    for i in range(6):
        parts_grid = torchvision.utils.make_grid(parts[:, i].detach().cpu())
        writer.add_image('orig_cropped_parts_no_mouth %d' % i, parts_grid, step)

    theta = select_model(stage1_pred)
    parts, _, theta = affine_crop(orig, orig_label, theta_in=theta, map_location=device, floor=True)
    assert parts.shape == (N, 6, 3, 81, 81)
    for i in range(6):
        parts_grid = torchvision.utils.make_grid(parts[:, i].detach().cpu())
        writer.add_image('affine_cropped_parts_no_mouth %d' % i, parts_grid, step)

