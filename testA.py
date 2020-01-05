from model import FaceModel, SelectNet
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
import uuid as uid
uuid = str(uid.uuid1())[0:8]
print(uuid)

writer = SummaryWriter('log')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = FaceModel().to(device)
# load state
# path = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/b1d730ea", "best.pth.tar")
path = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/a0d286ea", "best.pth.tar")
state = torch.load(path, map_location=device)

model1.load_state_dict(state['model1'])

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
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    N = image.shape[0]

    stage1_pred = model1(image)
    assert stage1_pred.shape == (N, 9, 128, 128)

    # imshow stage1 mask predict
    stage1_pred_grid = torchvision.utils.make_grid(stage1_pred.argmax(dim=1, keepdim=True))
    writer.add_image("stage1 predict_%s" % uuid, stage1_pred_grid, step)
    f1_class.forward(stage1_pred, label.argmax(dim=1, keepdim=False))

f1_class.output_f1_score()
