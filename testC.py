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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
select_model = SelectNet().to(device)
# load state
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/02a38440", "best.pth.tar")
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/ca8f5c52", "best.pth.tar")
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/b9d37dbc", "best.pth.tar")
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/49997f1e", "best.pth.tar")
path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/396e4702", "best.pth.tar")
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/1daed2c2", "best.pth.tar")
# path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/ea3c3972", "best.pth.tar")
path_select = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6", "best.pth.tar")
#396e4702的嘴得分是0.9166 overall 0.865(0.869)
#396e4702的单独最佳得分是0.8714
#b9d37dbc overall 0.854
#49997f1e overall 0.8405

#1daed2c2 更改了修复后的crop数据集, 无数据增强, 此时单独得分为0.855196

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
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    parts_gt = batch['parts_gt'].to(device)
    parts_mask_gt = batch['parts_mask_gt'].to(device)
    N,L,H,W = orig_label.shape

    cens = torch.floor(calc_centroid(orig_label))

    # Test B
    # theta = select_model(label)

    parts, parts_labels, theta = affine_crop(orig, orig_label, points=cens, map_location=device)
    # parts, parts_labels, _ = affine_crop(orig, orig_label, theta_in=theta, map_location=device)

    stage2_pred = model2(parts_gt)

    # imshow predict
    for i in range(6):
        pred_grid = torchvision.utils.make_grid(stage2_pred[i].argmax(dim=1, keepdim=True))
        writer.add_image('stage2 predict_%d' % i, pred_grid[0], global_step=step, dataformats='HW')

    # imshow crop
    temp = []
    for i in range(theta.shape[1]):
        test = theta[:, i]
        grid = F.affine_grid(theta=test, size=[N, 3, 81, 81], align_corners=True)
        temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
    parts = torch.stack(temp, dim=1)
    assert parts.shape == (N, 6, 3, 81, 81)
    for i in range(6):
        parts_grid = torchvision.utils.make_grid(
            parts[:, i].detach().cpu())
        writer.add_image('croped_parts_%s_%d' % ("testC", i), parts_grid, step)

    final_pred = affine_mapback(stage2_pred, theta, device)
    final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
    writer.add_image("final predict_testC", final_grid[0], global_step=step, dataformats='HW')
    f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))
f1_class.output_f1_score()

