from model import Stage2Model, FaceModel, SelectNet
from tensorboardX import SummaryWriter
from dataset import HelenDataset
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback, stage2_pred_softmax
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

path_model1 = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/a0d286ea", "best.pth.tar")
path_model2_396 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/396e4702", "best.pth.tar")

state1 = torch.load(path_model1, map_location=device)
state2_396 = torch.load(path_model2_396, map_location=device)
model1.load_state_dict(state1['model1'])
model2.load_state_dict(state2_396['model2'])

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
    names = batch['name']
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
    cens = torch.floor(calc_centroid(big_pred))

    # Test B
    parts, parts_labels, theta = affine_crop(orig, orig_label, points=cens, map_location=device, floor=True)

    stage2_pred = model2(parts)
    softmax_stage2 = stage2_pred_softmax(stage2_pred)

    final_pred = affine_mapback(softmax_stage2, theta, device)
    path_baseline = "/home/yinzi/data3/baseline_out"
    if not os.path.exists(path_baseline):
        os.makedirs(path_baseline)

    for k in range(final_pred.shape[0]):
        final_out = TF.to_pil_image(final_pred.argmax(dim=1, keepdim=False).detach().cpu().type(torch.uint8)[k])
        final_out = TF.center_crop(final_out, orig_size[k].tolist())
        final_out.save(os.path.join(path_baseline, names[k] + ".png"), format="PNG", compress_level=0)

