from utils.template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboardX as tb
from datasets.dataset import Stage1Augmentation
from icnnmodel import FaceModel as Stage1Model
import uuid as uid
import os 
uuid = str(uid.uuid1())[0:8]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--cuda", default=0, type=int, help="Choose GPU with cuda number")
parser.add_argument("--lr", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            Resize((128, 128)),
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            Resize((128, 128)),
            ToTensor()
        ])
}
# DataLoader
stage1_augmentation = Stage1Augmentation(dataset=HelenDataset,
                                         txt_file=txt_file_names,
                                         root_dir=root_dir,
                                         resize=(64, 64)
                                         )
enhaced_stage1_datasets = stage1_augmentation.get_dataset()
stage1_dataloaders = {x: DataLoader(enhaced_stage1_datasets[x], batch_size=args.batch_size,
                                    shuffle=True, num_workers=4)
                      for x in ['train', 'val']}

stage1_dataset_sizes = {x: len(enhaced_stage1_datasets[x]) for x in ['train', 'val']}

 
class TrainModel(TemplateModel):

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.args = argus
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = stage1_dataloaders['train']
        self.eval_loader = stage1_dataloaders['val']

        self.ckpt_dir = "checkpoints_A/%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        x, y = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)

        pred = self.model(x)
        loss = self.criterion(pred, y.argmax(dim=1, keepdim=False))

        return loss, None

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            x, y = batch['image'].to(self.device), batch['labels'].to(self.device)
            pred = self.model(x)
            error = self.metric(pred, y.argmax(dim=1, keepdim=False))
            loss_list.append(error.item())

        return np.mean(loss_list), None

    def eval(self):
        self.model.eval()
        error, others = self.eval_error()

        if error < self.best_error:
            self.best_error = error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_%s' % uuid, error, self.epoch)
        print('epoch {}\terror {:.3}\tbest_error {:.3}'.format(self.epoch, error, self.best_error))

        if self.eval_logger:
            self.eval_logger(self.writer, others)

        return error

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            self.optimizer.zero_grad()

            loss, others = self.train_loss(batch)

            loss.backward()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_%s' % uuid, loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
                if self.train_logger:
                    self.train_logger(self.writer, others)

    def save_state(self, fname, optim=True):
        state = {}
        if isinstance(self.model, torch.nn.DataParallel):
            state['model1'] = self.model.module.state_dict()
        else:
            state['model1'] = self.model.state_dict()

        if optim:
            state['optimizer'] = self.optimizer.state_dict()
        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname)


def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.scheduler.step()
        train.train()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
