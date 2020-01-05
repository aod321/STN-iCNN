from time import sleep

from data_augmentation import Stage2Augmentation, Stage1Augmentation
from dataset import PartsDataset, HelenDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import torchvision
import uuid as uid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stage", default=1, type=int, help="1:Check stage1 augmentation, 2: Check stage2 instead")
args = parser.parse_args()
print(args)

uuid = str(uid.uuid1())[0:10]
print(uuid)

writer = SummaryWriter('log')

# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"
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

stage1_augmentation = Stage1Augmentation(dataset=HelenDataset,
                                         txt_file=txt_file_names,
                                         root_dir=root_dir,
                                         parts_root_dir=parts_root_dir,
                                         resize=(128, 128)
                                         )
enhaced_stage1_datasets = stage1_augmentation.get_dataset()

if args.stage == 1:
    dataloader = {x: DataLoader(enhaced_stage1_datasets[x], batch_size=16,
                                shuffle=True, num_workers=4)
                  for x in ['train', 'val']
                  }
elif args.stage == 2:
    dataloader = {x: DataLoader(enhaced_stage2_datasets[x], batch_size=16,
                                shuffle=True, num_workers=4)
                  for x in ['train', 'val']
                  }


step = 0
def show_stage1():
    global step
    for batch in dataloader['train']:
        step += 1
        image = batch['image'].to(device)
        label = batch['labels'].to(device)
        orig = batch['orig'].to(device)
        orig_label = batch['orig_label'].to(device)
        print("step%d" % step)
        image_grid = torchvision.utils.make_grid(image)
        label_grid = torchvision.utils.make_grid(label.argmax(dim=1, keepdim=True))

        orig_grid = torchvision.utils.make_grid(orig)
        orig_label_grid = torchvision.utils.make_grid(orig_label.argmax(dim=1, keepdim=True))


        writer.add_image("[Augmentation]Stage1Image_%s" % uuid, image_grid, global_step=step)
        sleep(0.0001)
        writer.add_image("[Augmentation]Stage1Labels_%s" % uuid, label_grid[0], global_step=step,
                         dataformats='HW')
        sleep(0.0001)
        writer.add_image("[Augmentation]Stage1OrigImage_%s" % uuid, orig_grid, global_step=step)
        sleep(0.0001)
        writer.add_image("[Augmentation]Stage1OrigLabels_%s" % uuid, orig_label_grid[0], global_step=step,
                         dataformats='HW')
        if step == 20:
            break


def show_stage2():
    global step
    for batch in dataloader['train']:
        step += 1
        image = batch['image'].to(device)
        label = batch['labels'].to(device)
        print("step%d" % step)
        for i in range(6):
            print("imshow %d" % i)
            image_grid = torchvision.utils.make_grid(image[:, i])
            label_grid = torchvision.utils.make_grid(label[:, i:i + 1])
            writer.add_image("[Augmentation]Stage2Image%s_%d" % (uuid, i), image_grid, global_step=step)
            writer.add_image("[Augmentation]Stage2Label%s_%d" % (uuid, i), label_grid[0], global_step=step,
                             dataformats='HW')

        if step == 20:
            break


if __name__ == "__main__":
    if args.stage == 1:
        show_stage1()
    elif args.stage == 2:
        show_stage2()
