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
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from dataset import HelenDataset
from model import Stage2Model, SelectNet
from helper_funcs import affine_crop

uuid = str(uid.uuid1())[0:8]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0, type=float, help="Learning rate for optimizer1")
parser.add_argument("--lr2", default=1e-5, type=float, help="Learning rate for optimizer2")
parser.add_argument("--lr_s", default=1e-5, type=float, help="Learning rate for optimizer_select")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--cuda", default=0, type=int, help="Choose GPU with cuda number")
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

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.args = argus
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device(
            "cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        # self.load_pretrained("model1")
        self.model2 = Stage2Model().to(self.device)
        self.load_pretrained("model2")
        self.select_net = SelectNet().to(self.device)
        self.load_pretrained("select_net")
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.optimizer2 = optim.Adam(self.model2.parameters(), self.args.lr2)
        self.optimizer_select = optim.Adam(
            self.select_net.parameters(), self.args.lr_s)

        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5)
        self.scheduler2 = optim.lr_scheduler.StepLR(
            self.optimizer2, step_size=5, gamma=0.5)
        self.scheduler3 = optim.lr_scheduler.StepLR(
            self.optimizer_select, step_size=5, gamma=0.5)

        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']

        self.ckpt_dir = "checkpoints_BC/%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
        parts_mask = batch['parts_mask_gt'].to(self.device)

        N, L, H, W = orig_label.shape
        assert label.shape == (N, 9, 128, 128)

        theta = self.select_net(label)
        assert theta.shape == (N, 6, 2, 3)

        parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, map_location=self.device)
        assert parts.grad_fn is not None
        assert parts.shape == (N, 6, 3, 81, 81)

        stage2_pred = self.model2(parts)
        assert len(stage2_pred) == 6

        loss = []
        for i in range(6):
            loss.append(
                self.criterion(stage2_pred[i], parts_mask[:, i].long()))
        loss = torch.stack(loss)
        return loss

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
            parts_mask = batch['parts_mask_gt'].to(self.device)

            N, L, H, W = orig_label.shape
            assert label.shape == (N, 9, 128, 128)

            theta = self.select_net(label)
            assert theta.shape == (N, 6, 2, 3)

            parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, map_location=self.device)
            assert parts.grad_fn is not None
            assert parts.shape == (N, 6, 3, 81, 81)

            stage2_pred = self.model2(parts)
            assert len(stage2_pred) == 6

            loss = []
            for i in range(6):
                loss.append(
                    self.criterion(stage2_pred[i], parts_mask[:, i].long()).item())
            loss_list.append(np.mean(loss))
        return np.mean(loss_list)

    def train(self):
        # self.model.train()
        self.model2.train()
        self.select_net.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            # self.optimizer.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer_select.zero_grad()
            loss = self.train_loss(batch)
            # [1,1,1,1,1,1] weight for 6 parts loss
            loss.backward(torch.ones(6, device=self.device, requires_grad=False))
            self.optimizer2.step()
            self.optimizer_select.step()
            # self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_%s' % uuid, torch.mean(loss).item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(
                    self.epoch, self.step, torch.mean(loss).item()))

    def eval(self):
        # self.model.eval()
        self.model2.eval()
        self.select_net.eval()
        error = self.eval_error()
        if error < self.best_error:
            self.best_error = error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(
            os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_%s' % uuid, error, self.epoch)
        print('epoch {}\terror {:.3}\tbest_error {:.3}'.format(
            self.epoch, error, self.best_error))

        return error

    def save_state(self, fname, optim=True):
        state = {}

        if isinstance(self.model, torch.nn.DataParallel):
            state['model1'] = self.model.module.state_dict()
            state['model2'] = self.model2.module.state_dict()
            state['select_net'] = self.select_net.module.state_dict()
        else:
            state['model1'] = self.model.state_dict()
            state['model2'] = self.model2.state_dict()
            state['select_net'] = self.select_net.state_dict()

        if optim:
            state['optimizer'] = self.optimizer.state_dict()
            state['optimizer2'] = self.optimizer2.state_dict()
            state['optimizer_select'] = self.optimizer_select.state_dict()

        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def load_pretrained(self, model):
        if model == 'model1':
            fname = "a"
            state = torch.load(fname, map_location=self.device)
            self.model1.load_state_dict(state['model1'])
        elif model == 'model2':
            fname = "/home/yinzi/data4/new_train/checkpoints_C/02a38440/best.pth.tar"
            state = torch.load(fname, map_location=self.device)
            self.model2.load_state_dict(state['model2'])
        elif model == 'select_net':
            fname = "/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6/best.pth.tar"
            state = torch.load(fname, map_location=self.device)
            self.select_net.load_state_dict(state['select_net'])


def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.train()
        train.scheduler2.step(epoch)
        train.scheduler3.step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
