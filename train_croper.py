from template import TemplateModel
from model import SelectNet, SelectNet_resnet
from preprocess import Resize, ToTensor, OrigPad
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import uuid
import numpy as np
import os
from dataset import HelenDataset
from helper_funcs import calc_centroid

uuid_8 = str(uuid.uuid1())[0:8]
print(uuid_8)
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=0, type=int, help="Which GPU to train.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--select_net", default=0, type=int, help="0 custom, 1 resnet-18")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

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

dataloader = {x: DataLoader(Dataset[x], batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']
              }


class TrainModel(TemplateModel):

    def __init__(self):
        super(TrainModel, self).__init__()
        self.args = args

        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        if self.args.select_net == 0:
            self.model = SelectNet().to(self.device)
        elif self.args.select_net ==1:
            self.model = SelectNet_resnet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.SmoothL1Loss()
        self.metric =  nn.SmoothL1Loss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']
        if self.args.select_net == 0:
            self.ckpt_dir = "checkpoints_B/%s" % uuid_8
        elif self.args.select_net == 1:
            self.ckpt_dir = "checkpoints_B_res/%s" % uuid_8

        self.display_freq = args.display_freq
        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        self.step += 1
        image, labels = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig = batch['orig'].to(self.device)
        orig_label = batch['orig_label'].to(self.device)
        n,l,h,w = orig.shape

        assert labels.shape == (n,9,128,128)
        theta = self.model(labels)

        assert orig_label.shape == (n,9,1024,1024)
        cens = calc_centroid(orig_label)

        assert cens.shape == (n,9,2)
        points = torch.cat([cens[:, 1:6],
                            cens[:, 6:9].mean(dim=1, keepdim=True)],
                           dim=1)
        theta_label = torch.zeros((n, 6, 2, 3), device=self.device, requires_grad=False)
        for i in range(6):
            theta_label[:, i, 0, 0] = (81. - 1.) / (w - 1)
            theta_label[:, i, 0, 2] = -1. + (2. * points[:, i, 1]) / (w - 1)
            theta_label[:, i, 1, 1] = (81. - 1.) / (h - 1)
            theta_label[:, i, 1, 2] = -1. + (2. * points[:, i, 0]) / (h - 1)

        loss = self.criterion(theta, theta_label)

        return loss

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            image, labels = batch['image'].to(
                self.device), batch['labels'].to(self.device)
            orig = batch['orig'].to(self.device)
            orig_label = batch['orig_label'].to(self.device)
            n, l, h, w = orig.shape
            theta = self.model(labels)
            cens = calc_centroid(orig_label)
            assert cens.shape == (n, 9, 2)
            points = torch.cat([cens[:, 1:6],
                                cens[:, 6:9].mean(dim=1, keepdim=True)],
                            dim=1)
            theta_label = torch.zeros(
                (n, 6, 2, 3), device=self.device, requires_grad=False)
            for i in range(6):
                theta_label[:, i, 0, 0] = (81. - 1.) / (w - 1)
                theta_label[:, i, 0, 2] = -1. + (2. * points[:, i, 1]) / (w - 1)
                theta_label[:, i, 1, 1] = (81. - 1.) / (h - 1)
                theta_label[:, i, 1, 2] = -1. + (2. * points[:, i, 0]) / (h - 1)
            loss = self.metric(theta, theta_label)
            loss_list.append(loss.item())

            temp = []
            for i in range(theta.shape[1]):
                test = theta[:, i]
                grid = F.affine_grid(theta=test, size=[n, 3, 81, 81], align_corners=True)
                temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
            parts = torch.stack(temp, dim=1)
            assert parts.shape == (n, 6, 3, 81, 81)
            for i in range(6):
                parts_grid = torchvision.utils.make_grid(
                    parts[:, i].detach().cpu())
                self.writer.add_image('croped_parts_%s_%d' % (uuid_8, i), parts_grid, self.step)
            
        return np.mean(loss_list)

    def train(self):
        self.model.train()
        self.epoch += 1
        for i, batch in enumerate(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = self.train_loss(batch)
            loss.backward()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_train_%s' % uuid_8, loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
        torch.cuda.empty_cache()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            error = self.eval_error()
        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        if error < self.best_error:
            self.best_error = error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)

        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_val%s' % uuid_8, error, self.epoch)

        # self.writer.add_image()
        print('epoch {}\terror {:.3}\tbest_error {:.3}'.format(self.epoch, error, self.best_error))
        torch.cuda.empty_cache()

    def save_state(self, fname, optim=True):
        state = {}
        if isinstance(self.model, torch.nn.DataParallel):
            state['select_net'] = self.model.module.state_dict()
        else:
            state['select_net'] = self.model.state_dict()
        if optim:
            state['optimizer_select'] = self.optimizer.state_dict()

        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))


def start_train():
    train = TrainModel()

    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
