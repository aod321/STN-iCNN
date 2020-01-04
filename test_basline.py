from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import torch
import os

writer = SummaryWriter('log')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
# load state

path_model1 = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/dd0a0bf4", "best.pth.tar")
path_model2_all = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/79f85a02", "best.pth.tar")
path_model2_eyebrows = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/396e4702", "best.pth.tar")

state1 = torch.load(path_model1, map_location=device)
state2_all = torch.load(path_model2_all, map_location=device)
state2_brows = torch.load(path_model2_eyebrows, map_location=device)

match_brows = {k: v for k, v in state2_brows['model2'].items() if k.startswith('model.0')}
state2_all['model2'].update(match_brows)

model1.load_state_dict(state1['model1'])
model2.load_state_dict(state2_all['model2'])

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
    orig_size = batch['orig_size']
    padding = batch['padding'].to(device)
    N,L,H,W = orig_label.shape

    stage1_pred = F.softmax(model1(image), dim=1)
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
    assert big_pred.shape == (N,9,1024,1024)

    cens = calc_centroid(big_pred)

    # Test B
    # theta = select_model(label)
    parts, parts_labels, theta = affine_crop(orig, orig_label, points=cens, map_location=device)
    # parts, parts_labels, _ = affine_crop(orig, orig_label, theta_in=theta, map_location=device)

    stage2_pred = model2(parts)

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
        writer.add_image('croped_parts_%s_%d' % ("testC_1-27", i), parts_grid, step)

    final_pred = affine_mapback(stage2_pred, theta, device)
    final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
    writer.add_image("final predict_testC_1-27", final_grid[0], global_step=step, dataformats='HW')
    f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))
f1_class.output_f1_score()

