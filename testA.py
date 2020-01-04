from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback
import torch.nn.functional as F
import torchvision
import torch
import os

writer = SummaryWriter('log')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
pathA = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/8f4afb96", "best.pth.tar")
stateA = torch.load(pathA, map_location=device)
model1.load_state_dict(stateA['model1'])

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
            ToTensor(),
            Resize((64, 64)),
            OrigPad()
        ]),
    'val':
        transforms.Compose([
            ToTensor(),
            Resize((64, 64)),
            OrigPad()
        ]),
    'test':
        transforms.Compose([
            ToTensor(),
            Resize((64, 64)),
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

dataloader = {x: DataLoader(Dataset[x], batch_size=4,
                            shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']
              }
f1_class = F1Score(device)
# show predicts
step = 0
for batch in dataloader['test']:
    step += 1
    orig = batch['orig'].to(device)
    orig_label = batch['orig_label'].to(device)
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    N,L,H,W = orig_label.shape

    stage1_pred = model1(image)
    assert stage1_pred.shape == (N, 9, 64, 64)

    stage1_pred_grid = torchvision.utils.make_grid(stage1_pred.argmax(dim=1, keepdim=True))
    writer.add_image("stage1 predict%s" % "testA", stage1_pred_grid, step)

