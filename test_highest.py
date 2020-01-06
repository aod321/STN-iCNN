from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback
import numpy as np
import torch
import torchvision

writer = SummaryWriter('log')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

dataloader = {x: DataLoader(Dataset[x], batch_size=4,
                            shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']
              }
Times = 1
# show predicts
for epoch in range(Times):
    step = 0
    f1_class = F1Score(device)
    f1_all = {x: []
              for x in f1_class.F1_name_list}
    f1_overall_all = []
    for batch in dataloader['test']:
        step += 1
        orig = batch['orig'].to(device)
        orig_label = batch['orig_label'].to(device)
        image = batch['image'].to(device)
        label = batch['labels'].to(device)
        parts_gt = batch['parts_gt'].to(device)
        parts_mask_gt = batch['parts_mask_gt'].to(device)
        N, L, H, W = orig_label.shape

        cens = torch.floor(calc_centroid(orig_label))
        parts, parts_labels, theta = affine_crop(orig, orig_label, points=cens, map_location=device, floor=True)
        # 如果是predict，记得要先softmax
        final_pred = affine_mapback(parts_labels, theta, device)
        final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
        writer.add_image("final_pred_test_MAX", final_grid[0], dataformats='HW')
        f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))
    f1, f1_overall = f1_class.get_f1_score()
    for x in f1_class.F1_name_list:
        f1_all[x].append(f1[x])
    f1_overall_all.append(f1_overall)
    print("Accumulate %d/%d" % (epoch, Times-1))

f1_overall_all = np.mean(f1_overall_all)

for x in f1_class.F1_name_list:
    f1_all[x] = np.mean(f1_all[x])
    print("{}:{}\t".format(x, f1_all[x]))
print("{}:{}\t".format("overall", f1_overall_all))


