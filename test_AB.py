from model import FaceModel, SelectNet, SelectNet_resnet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToTensor, OrigPad, Resize, ToPILImage
from torch.utils.data import DataLoader
from helper_funcs import F1Score, affine_crop
import torch.nn.functional as F
import torchvision
import torch
import os

writer = SummaryWriter('log')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
# model2 = Stage2Model().to(device)
select_model = SelectNet().to(device)
select_model_res = SelectNet_resnet().to(device)
# load state
# path = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/791288bf4", "best.pth.tar")
path_B = os.path.join("/home/yinzi/data4/new_train/checkpoints_B/9a95687c", "best.pth.tar")
path = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/89ce3b06", "best.pth.tar")
# path_AB = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/89ce3b06", 'best.pth.tar')
path_AB = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB_res/840ea936", 'best.pth.tar')
state = torch.load(path, map_location=device)
state_B = torch.load(path_B, map_location=device)
state_AB = torch.load(path_AB, map_location=device)

model1.load_state_dict(state_AB['model1'])
# select_model.load_state_dict(state_AB['select_net'])
select_model_res.load_state_dict(state_AB['select_net'])

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

    stage1_pred = F.softmax(model1(image), dim=1)
    assert stage1_pred.shape == (N, 9, 128, 128)

    theta = select_model_res(stage1_pred)

    # imshow stage1 mask predict
    # stage1_pred_grid = torchvision.utils.make_grid(stage1_pred.argmax(dim=1, keepdim=True))
    # writer.add_image("stage1 predict%s" % 'a2330644', stage1_pred_grid, step)
    # theta = select_model(F.softmax(stage1_pred, dim=1))

    # cens = calc_centroid(orig_label)
    # assert cens.shape == (N, 9, 2)
    parts, parts_labels, _ = affine_crop(orig, orig_label, theta_in=theta, map_location=device)
    temp = []
    for i in range(theta.shape[1]):
        test = theta[:, i]
        grid = F.affine_grid(theta=test, size=[N, 3, 81, 81], align_corners=True)
        temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
    parts = torch.stack(temp, dim=1)
    assert parts.shape == (N, 6, 3, 81, 81)
    for i in range(6):
        parts_grid = torchvision.utils.make_grid(parts[:, i].detach().cpu())
        writer.add_image('11croped_parts_test2_%d' % i, parts_grid, step)

