import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorboardX as tb
from torchvision import transforms
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import uuid as uid
from model import Stage2Model
from template import TemplateModel
from preprocess import ToPILImage, ToTensor, Resize

uuid = str(uid.uuid1())[0:8]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr0", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--lr1", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--lr2", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--lr3", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--cuda", default=0, type=int, help="Choose GPU with cuda number")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"
parts_root_dir = "/home/yinzi/data3/recroped_parts"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor()
        ])
}
# DataLoader


 

class TrainModel(TemplateModel):

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2Model().to(self.device)
        lr = [self.args.lr0, self.args.lr1, self.args.lr2, self.args.lr3]
        self.optimizer = [optim.Adam(self.model[i].parameters(), lr[i])
                          for i in range(4)]
        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = [optim.lr_scheduler.StepLR(self.optimizer[i], step_size=5, gamma=0.5)
                          for i in range(4)]

        self.train_loader = stage1_dataloaders['train']
        self.eval_loader = stage1_dataloaders['val']

        self.ckpt_dir = "checkpoints_C/%s" % uuid
        self.display_freq = self.args.display_freq
        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        parts_groundtruth = batch['parts_groundtruth'].to(self.device)
        parts_mask_groundtruth = batch['parts_mask_groundtruth'].to(self.device)
        N = parts_groundtruth.shape[0]
        assert parts_groundtruth.shape == (N, 6, 3, 64, 64)
        assert parts_mask_groundtruth.shape == (N, 6, 64, 64)

        pred = self.model(parts_groundtruth)

        loss = []
        for i in range(6):
            loss.append(self.criterion(pred[i], parts_mask_groundtruth[:, i].long()))
        loss = torch.stack(loss)
        return loss, None

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            parts_groundtruth = batch['parts_groundtruth'].to(self.device)
            parts_mask_groundtruth = batch['parts_mask_groundtruth'].to(self.device)
            N = parts_groundtruth.shape[0]
            assert parts_groundtruth.shape == (N, 6, 3, 64, 64)
            assert parts_mask_groundtruth.shape == (N, 6, 64, 64)

            pred = self.model(parts_groundtruth)

            loss = []
            for i in range(6):
                loss.append(self.criterion(pred[i], parts_mask_groundtruth[:, i].long()))
            loss = torch.stack(loss)
            loss_list.append(torch.mean(loss).item())
        return np.mean(loss_list), None

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            for i in range(4):
                self.optimizer[i].zero_grad()

            loss, others = self.train_loss(batch)
            loss.backward(torch.ones(6, device=self.device, requires_grad=False))
            
            for i in range(4):
                self.optimizer[i].step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss', loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, torch.mean(loss).item()))
                if self.train_logger:
                    self.train_logger(self.writer, others)

    def eval(self):
        self.model.eval()
        error, others = self.eval_error()

        if error < self.best_error:
            self.best_error = error
            self.save_state(osp.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(osp.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error', error, self.epoch)
        print('epoch {}\terror {:.3}\tbest_error {:.3}'.format(self.epoch, error, self.best_error))

        if self.eval_logger:
            self.eval_logger(self.writer, others)

        return error
  


def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.scheduler.step()
        train.train()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
