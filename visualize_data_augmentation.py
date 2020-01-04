from data_augmentation import Stage2Augmentation
from dataset import PartsDataset, HelenDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import torchvision
from data_augmentation import Stage1Augmentation

writer = SummaryWriter('log')

# Dataset Read_in Part
parts_root_dir = "/home/yinzi/data3/recroped_parts"
root_dir = "/data1/yinzi/datas"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt",
    'test': "testing.txt"
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Augment_Dataset = Stage1Augmentation(
    dataset=HelenDataset,
    txt_file=txt_file_names,
    root_dir=root_dir,
    parts_root_dir=parts_root_dir,
    resize=(64, 64)
)
enhaced_stage1_datasets = Augment_Dataset.get_dataset()


stage2_augmentation = Stage2Augmentation(dataset=PartsDataset,
                                         txt_file=txt_file_names,
                                         root_dir=parts_root_dir
                                         )
enhaced_stage2_datasets = stage2_augmentation.get_dataset()

# dataloader = {x: DataLoader(enhaced_stage2_datasets[x], batch_size=16,
#                             shuffle=True, num_workers=4)
#               for x in ['train', 'val']
#               }
dataloader = {x: DataLoader(enhaced_stage1_datasets[x], batch_size=16,
                            shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']
              }

step = 0
for batch in dataloader['train']:
    step += 1
    image = batch['image'].to(device)
    label = batch['labels'].to(device)
    image_grid = torchvision.utils.make_grid(image)
    writer.add_image("[Augmentation]test_Stage1Image", image_grid, global_step=step)
    label_gt = torchvision.utils.make_grid(label.argmax(dim=1, keepdim=True))
    writer.add_image("[Augmentation]test_Stage1label", label_gt[0], step, dataformats='HW')
    print("step%d" % step)
    # for i in range(6):
    #     print("imshow %d" % i)
    #     label_grid = torchvision.utils.make_grid(label[:, i:i+1])
    #     writer.add_image("[Augmentation]test_Stage2Image_%d" % i, image_grid, global_step=step)
    #     writer.add_image("[Augmentation]test_Stage2Label_%d" % i, label_grid[0], global_step=step, dataformats='HW')
        # writer.add_image("[Augmentation]test_Stage1Image_%d" % i, image_grid, global_step=step)
        # writer.add_image("[Augmentation]test_Stage1Label_%d" % i, label_grid[0], global_step=step, dataformats='HW')


