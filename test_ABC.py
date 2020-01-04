from model import Stage2Model, FaceModel, SelectNet_resnet
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
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
model2 = Stage2Model().to(device)
select_model = SelectNet_resnet().to(device)
# load state
#save model at checkpoints_ABC/00ca488c/25.pth.tar
# epoch 25        error 0.0605    best_error 0.0589


# pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/00ca488c", "best.pth.tar")
# pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/c8c68e16", "best.pth.tar")
# pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/ea3c3972", "best.pth.tar")
pathABC = os.path.join("/home/yinzi/data4/new_train/checkpoints_ABC/09d01660", "best.pth.tar")

pathAB = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6", "best.pth.tar")
pathB = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6", 'best.pth.tar')
# pathC = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/7fc23918", 'best.pth.tar')
# pathC = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/02a38440", 'best.pth.tar')
# pathC = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/02a38440", 'best.pth.tar')
# pathC = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/396e4702", 'best.pth.tar')
pathC = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/1daed2c2", 'best.pth.tar')
# 396e4702 带数据增广,单独F1得分为0.87，AB结果送入，F1得分为0.834

# 1daed2c2 使用了修复的crop数据集，无数据增广训练。单独F1得分为0.8551966, AB结果送入。F1得分为0.650

# state = torch.load(path, map_location=device)
stateAB = torch.load(pathAB)
stageC = torch.load(pathC)
stateABC = torch.load(pathABC)

# model1.load_state_dict(stateAB['model1'])
# select_model.load_state_dict(stateAB['select_net'])
# model2.load_state_dict(stageC['model2'])
model1.load_state_dict(stateAB['model1'])
select_model.load_state_dict(stateAB['select_net'])
model2.load_state_dict(stageC['model2'])
#stateABC 0.8088
#StageC 0.69

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
    N,L,H,W = orig_label.shape

    stage1_pred = model1(image)
    assert stage1_pred.shape == (N, 9, 128, 128)
    theta = select_model(F.softmax(stage1_pred, dim=1))

    # cens = calc_centroid(orig_label)
    # assert cens.shape == (N, 9, 2)
    parts, parts_labels, _ = affine_crop(orig, orig_label, theta_in=theta, map_location=device)

    stage2_pred = model2(parts)
    for i in range(6):
        parts_grid = torchvision.utils.make_grid(
            parts[:, i].detach().cpu())
        writer.add_image('croped_parts_%s_%d' % ("testABC", i), parts_grid, step)
    for i in range(6):
        pred_grid = torchvision.utils.make_grid(stage2_pred[i].argmax(dim=1, keepdim=True))
        writer.add_image('ABC_stage2 predict_%d' % i, pred_grid[0], global_step=step, dataformats='HW')

    final_pred = affine_mapback(stage2_pred, theta, device)
    final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
    writer.add_image("final predict_ABC",final_grid[0], global_step=step, dataformats='HW')
    f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))
f1_class.output_f1_score()

