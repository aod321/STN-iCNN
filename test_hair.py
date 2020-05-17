from tensorboardX import SummaryWriter
from dataset import SkinHelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize, Skin_ToTensor
from torch.utils.data import DataLoader
from helper_funcs import affine_crop, affine_mapback, stage2_pred_softmax
import torch.nn.functional as F
import numpy as np
import torch
import os
import uuid as uid
import torchvision.transforms.functional as TF
from model import Stage2FaceModel
from tqdm import tqdm

uuid = str(uid.uuid1())[0:10]

writer = SummaryWriter('log')
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model1 = Stage2FaceModel()
model1.set_label_channels(11)
model1 = model1.to(device)
path = os.path.join("/home/yinzi/data4/new_train/exp_A/da56c3a2", "best.pth.tar")
state = torch.load(path, map_location=device)
model1.load_state_dict(state['model1'])

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"
local_dir = "/Users/yinzi/Downloads/helenstar_release"

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
            Skin_ToTensor()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            Skin_ToTensor()
        ]),
    'test':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            Skin_ToTensor()
        ])
}
# DataLoader
Dataset = {x: SkinHelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir,
                           transform=transforms_list[x]
                           )
           for x in ['train', 'val', 'test']
           }

dataloader = {x: DataLoader(Dataset[x], batch_size=10,
                            shuffle=False, num_workers=4)
              for x in ['train', 'val', 'test']
              }
# show predicts
step = 0
for batch in dataloader['test']:
    step += 1
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    names = batch['name']

    N = image.shape[0]
    stage1_pred = model1(image)
    assert stage1_pred.shape == (N, 11, 128, 128)
    pred_out = stage1_pred.argmax(dim=1, keepdim=False).detach().cpu().numpy().astype(np.uint8)
    label_out = label.argmax(dim=1, keepdim=False).detach().cpu().numpy().astype(np.uint8)
    pred_path = "/home/yinzi/data3/predhair_out/"
    label_path = "/home/yinzi/data3/outhair_gt/"
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    for k in range(N):
        save_pred_out = TF.to_pil_image(pred_out[k])
        save_label_out = TF.to_pil_image(label_out[k])
        save_pred_out.save("/home/yinzi/data3/predhair_out/%s.png" % names[k], format="PNG", compress_level=0)
        save_label_out.save("/home/yinzi/data3/outhair_gt/%s.png" % names[k], format="PNG", compress_level=0)

