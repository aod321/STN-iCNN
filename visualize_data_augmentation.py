from data_augmentation import Stage2Augmentation
from dataset import PartsDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import torchvision
import uuid as uid
uuid = str(uid.uuid1())[0:10]

writer = SummaryWriter('log')

# Dataset Read_in Part
parts_root_dir = "/home/yinzi/data3/recroped_parts"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

stage2_augmentation = Stage2Augmentation(dataset=PartsDataset,
                                         txt_file=txt_file_names,
                                         root_dir=parts_root_dir
                                         )
enhaced_stage2_datasets = stage2_augmentation.get_dataset()

dataloader = {x: DataLoader(enhaced_stage2_datasets[x], batch_size=16,
                            shuffle=True, num_workers=4)
              for x in ['train', 'val']
              }

step = 0
print(uuid)
for batch in dataloader['train']:
    step += 1
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    print("step%d" % step)
    for i in range(6):
        print("imshow %d" % i)
        image_grid = torchvision.utils.make_grid(image[:, i])
        label_grid = torchvision.utils.make_grid(label[:, i:i+1])
        writer.add_image("[Augmentation]Stage2Image%s_%d" % (uuid, i), image_grid, global_step=step)
        writer.add_image("[Augmentation]Stage2Label%s_%d" % (uuid, i), label_grid[0], global_step=step, dataformats='HW')

