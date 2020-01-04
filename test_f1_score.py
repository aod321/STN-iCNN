from model import Stage2Model
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2 = Stage2Model().to(device)

# load state
# path = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/745d57da", "best.pth.tar")
path = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/02a38440", "best.pth.tar")
path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/396e4702", "best.pth.tar")
# path = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/b9d37dbc", "best.pth.tar")
state = torch.load(path_model2, map_location=device)
model2.load_state_dict(state['model2'])

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
            Resize((128, 128)),
            OrigPad()
        ]),
    'val':
        transforms.Compose([
            ToTensor(),
            Resize((128, 128)),
            OrigPad()
        ]),
    'test':
        transforms.Compose([
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
    N,L,H,W = orig_label.shape
    assert orig_label.shape == (N, 9, 1024, 1024)
    cens = calc_centroid(orig_label)
    assert cens.shape == (N, 9, 2)
    parts, parts_labels, theta = affine_crop(img=orig, label=orig_label, points=cens, map_location=device)
    stage2_pred = model2(parts)

    for i in range(6):
        pred_grid = torchvision.utils.make_grid(stage2_pred[i].argmax(dim=1, keepdim=True))
        writer.add_image('stage2 predict_%d' % i, pred_grid[0], global_step=step, dataformats='HW')

    final_pred = affine_mapback(stage2_pred, theta, device)
    final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
    writer.add_image("final predict",final_grid[0], global_step=step, dataformats='HW')
    f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))
f1_class.output_f1_score()

