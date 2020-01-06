from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback, stage2_pred_onehot, stage2_pred_softmax
import torchvision.transforms.functional as TF
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import os

writer = SummaryWriter('log')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
select_model = SelectNet().to(device)
# load state
path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/396e4702", "best.pth.tar")
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/0e272174", "best.pth.tar")
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/b33172e4", "best.pth.tar")
path_select = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6", "best.pth.tar")

# 396e4702的嘴得分是0.9166 overall 0.865(0.869)
# 396e4702的单独最佳得分是0.8714
# b9d37dbc overall 0.854
# 49997f1e overall 0.8405

# 1daed2c2 更改了修复后的crop数据集, 无数据增强, 此时单独得分为0.855196

state_select = torch.load(path_select, map_location=device)
select_model.load_state_dict(state_select['select_net'])

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
        names = batch['name']
        orig_size = batch['orig_size']
        N, L, H, W = orig_label.shape
        cens = torch.floor(calc_centroid(orig_label))
        parts, _, theta = affine_crop(orig, orig_label, points=cens, map_location=device, floor=True)

        stage2_pred = model2(parts)
        softmax_stage2 = stage2_pred_softmax(stage2_pred)
        final_pred = affine_mapback(softmax_stage2, theta, device)
        for k in range(final_pred.shape[0]):
            final_out = TF.to_pil_image(final_pred.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
            final_out = TF.center_crop(final_out, orig_size[k].tolist())
            orig_out = TF.to_pil_image(orig_label.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
            orig_out = TF.center_crop(orig_out, orig_size[k].tolist())
            final_out.save("/home/yinzi/data3/pred_out/%s.png" % names[k], format="PNG", compress_level=0)
            orig_out.save("/home/yinzi/data3/out_gt/%s.png" % names[k], format="PNG", compress_level=0)
        # final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
        # writer.add_image("final predict_testC", final_grid[0], global_step=step, dataformats='HW')
        f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))
    f1, f1_overall = f1_class.get_f1_score()
    for x in f1_class.F1_name_list:
        f1_all[x].append(f1[x])
    f1_overall_all.append(f1_overall)
    print("Accumulate %d/%d" % (epoch, Times - 1))

f1_overall_all = np.max(f1_overall_all)

for x in f1_class.F1_name_list:
    f1_all[x] = np.max(f1_all[x])
    print("{}:{}\t".format(x, f1_all[x]))
print("{}:{}\t".format("overall", f1_overall_all))
