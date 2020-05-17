from template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from icnnmodel import FaceModel as Stage1Model
import uuid as uid
import os
from torchvision import transforms
from preprocess import ToTensor, OrigPad, Resize, ToPILImage
from torch.utils.data import DataLoader
from dataset import HelenDataset
from data_augmentation import Stage1Augmentation


uuid = str(uid.uuid1())[0:8]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--size", default=128, type=int, help="Input size")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--pretrainA", default=0, type=int, help="load pretrain")
parser.add_argument("--datamore", default=0, type=int, help="enable data augmentation")
parser.add_argument("--cuda", default=0, type=int, help="Choose GPU with cuda number")
parser.add_argument("--mode", default='resize', type=str, help="orig, resize")
parser.add_argument("--lr", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
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
            ToPILImage(),
            Resize((args.size, args.size)),
            ToTensor(),
            OrigPad()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((args.size, args.size)),
            ToTensor(),
            OrigPad()
        ]),
    'test':
        transforms.Compose([
            ToTensor(),
            Resize((args.size, args.size)),
            OrigPad()
        ])
}



# DataLoader
Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir,
                           parts_root_dir=parts_root_dir,
                           transform=transforms_list[x]
                           )
           for x in ['train', 'val']
           }

stage1_augmentation = Stage1Augmentation(dataset=HelenDataset,
                                         txt_file=txt_file_names,
                                         root_dir=root_dir,
                                         parts_root_dir=parts_root_dir,
                                         resize=(args.size, args.size)
                                         )
enhaced_stage1_datasets = stage1_augmentation.get_dataset()

if args.datamore == 0:
    dataloader = {x: DataLoader(Dataset[x], batch_size=args.batch_size,
                                shuffle=True, num_workers=4)
                  for x in ['train', 'val']
                  }

elif args.datamore == 1:
    dataloader = {x: DataLoader(enhaced_stage1_datasets[x], batch_size=args.batch_size,
                                shuffle=True, num_workers=4)
                  for x in ['train', 'val']
                  }


class TrainModel(TemplateModel):

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.args = argus
        self.writer = SummaryWriter('exp')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        if self.args.pretrainA == 1:
            path_A = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/88736bbe", 'best.pth.tar')
            state_A = torch.load(path_A, map_location=self.device)
            self.model.load_state_dict(state_A['model1'])
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']

        self.ckpt_dir = "exp_A/%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        x, y = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)
        pred = self.model(x)
        loss = self.criterion(pred, y.argmax(dim=1, keepdim=False))
        if self.args.mode == 'orig':
            orig = batch['orig'].to(self.device)
            orig_label = batch['orig_label'].to(self.device)
            pred = self.model(orig)
            loss = self.criterion(pred, orig_label.argmax(dim=1, keepdim=False))

        return loss, None

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            x, y = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y.argmax(dim=1, keepdim=False))
            if self.args.mode == 'orig':
                orig = batch['orig'].to(self.device)
                orig_label = batch['orig_label'].to(self.device)
                pred = self.model(orig)
                loss = self.criterion(pred, orig_label.argmax(dim=1, keepdim=False))
            loss_list.append(loss.item())
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
        print('save model at {}'.format(fname))

    def load_state(self, fname, optim=True, map_location=None):
        state = torch.load(fname, map_location=map_location)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state['model1'])
        else:
            self.model.load_state_dict(state['model1'])

        if optim and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.epoch = state['epoch']
        self.best_error = state['best_error']
        print('load model from {}'.format(fname))


def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
