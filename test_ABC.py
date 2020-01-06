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
import numpy as np

uuid = str(uid.uuid1())[0:10]

writer = SummaryWriter('log')
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
select_model = SelectNet().to(device)
select_res_model = SelectNet_resnet().to(device)

pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/7a89bbc8", "best.pth.tar")
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

dataloader = {x: DataLoader(Dataset[x], batch_size=10,
                            shuffle=False, num_workers=4)
              for x in ['train', 'val', 'test']
              }
# show predicts
step = 0
epochs = 2
for epoch in range(epochs):
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
        f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))

        for i in range(6):
            parts_grid = torchvision.utils.make_grid(
                parts[:, i].detach().cpu())
            writer.add_image('croped_parts_%s_%d' % (uuid, i), parts_grid, step)
        for i in range(6):
            pred_grid = torchvision.utils.make_grid(stage2_pred[i].argmax(dim=1, keepdim=True))
            writer.add_image('stage2 predict_%s_%d' % (uuid, i), pred_grid[0], global_step=step, dataformats='HW')

        final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
        writer.add_image("final predict_%s" % uuid, final_grid[0], global_step=step, dataformats='HW')
    f1, f1_overall = f1_class.get_f1_score()
    for x in f1_class.F1_name_list:
        f1_all[x].append(f1[x])
    f1_overall_all.append(f1_overall)
    print("Accumulate %d/%d" % (epoch, epochs - 1))

for x in f1_class.F1_name_list:
    f1_all[x] = np.mean(f1_all[x])
    print("{}:{}\t".format(x, f1_all[x]))
print("{}:{}\t".format("overall", f1_overall_all))



