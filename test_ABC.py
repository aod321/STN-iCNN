from model import Stage2Model, FaceModel, SelectNet_resnet, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback, stage2_pred_softmax, stage2_pred_onehot
import torch.nn.functional as F
import torchvision
import torch
import os
import uuid as uid
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

uuid = str(uid.uuid1())[0:10]

pred_out = "/home/yinzi/data4/pred_out"
gt_out = "/home/yinzi/data4/out_gt"
os.makedirs(pred_out, exist_ok=True)
os.makedirs(gt_out, exist_ok=True)

writer = SummaryWriter('log')
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
select_model = SelectNet().to(device)
select_res_model = SelectNet_resnet().to(device)
model1.eval()
select_res_model.eval()
model2.eval()

#pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/7a89bbc8", "best.pth.tar")
pathABC = os.path.join("/home/yinzi/data4/STN-iCNN/checkpoints_ABC/ea0ac45c-0", "best.pth.tar")
# pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/3cdb9922-3", "best.pth.tar")
stateABC = torch.load(pathABC, map_location=device)

model1.load_state_dict(stateABC['model1'])
select_res_model.load_state_dict(stateABC['select_net'])
model2.load_state_dict(stateABC['model2'])

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

dataloader = {x: DataLoader(Dataset[x], batch_size=1,
                            shuffle=False, num_workers=10)
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
    names = batch['name']
    orig_size = batch['orig_size']
    N, L, H, W = orig_label.shape

    stage1_pred = model1(image)
    assert stage1_pred.shape == (N, 9, 128, 128)
    theta = select_res_model(F.softmax(stage1_pred, dim=1))

    # cens = calc_centroid(orig_label)
    # assert cens.shape == (N, 9, 2)
    parts, parts_labels, _ = affine_crop(orig, orig_label, theta_in=theta, map_location=device)

    stage2_pred = model2(parts)

    softmax_stage2 = stage2_pred_softmax(stage2_pred)

    final_pred = affine_mapback(softmax_stage2, theta, device)

    for k in range(final_pred.shape[0]):
        final_out = TF.to_pil_image(final_pred.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        final_out = TF.center_crop(final_out, orig_size[k].tolist())
        orig_out = TF.to_pil_image(orig_label.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        orig_out = TF.center_crop(orig_out, orig_size[k].tolist())
        final_out.save("/home/yinzi/data4/pred_out/%s.png" % names[k], format="PNG", compress_level=0)
        orig_out.save("/home/yinzi/data4/out_gt/%s.png" % names[k], format="PNG", compress_level=0)


