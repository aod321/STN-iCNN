from template import TemplateModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from icnnmodel import FaceModel as Stage1Model
import uuid as uid
import os
from torchvision import transforms
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback
from preprocess import ToPILImage, ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from dataset import HelenDataset
from model import Stage2Model, SelectNet
from helper_funcs import affine_crop, F1Score, affine_mapback
import torchvision

uuid = str(uid.uuid1())[0:8]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for optimizer1")
parser.add_argument("--lr2", default=1e-3, type=float, help="Learning rate for optimizer2")
parser.add_argument("--lr_s", default=1e-3, type=float, help="Learning rate for optimizer_select")
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

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        self.load_pretrained("model1")
        self.model2 = Stage2Model().to(self.device)
        # self.load_pretrained("model2")
        self.select_net = SelectNet().to(self.device)
        self.load_pretrained("select_net")
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.optimizer2 = optim.Adam(self.model2.parameters(), self.args.lr2)
        self.optimizer_select = optim.Adam(self.select_net.parameters(), self.args.lr_s)

        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer2, step_size=5, gamma=0.5)
        self.scheduler3 = optim.lr_scheduler.StepLR(self.optimizer_select, step_size=5, gamma=0.5)

        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']

        self.ckpt_dir = "checkpoints_ABC/%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig, orig_label = batch['orig'].to(self.device), batch['orig_label']
        parts_mask_gt = batch['parts_mask_gt'].to(self.device)
        N, L, H, W = orig_label.shape

        assert parts_mask_gt.shape == (N, 6, 81, 81)

        stage1_pred = self.model(image)
        assert stage1_pred.shape == (N, 9, 128, 128)

        theta = self.select_net(F.softmax(stage1_pred, dim=1))
        assert theta.shape == (N, 6, 2, 3)

        parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, map_location=self.device)
        assert parts.shape == (N, 6, 3, 81, 81)

        assert parts.grad_fn is not None

        stage2_pred = self.model2(parts)
        assert len(stage2_pred) == 6
        loss = []
        for i in range(6):
            # loss.append(self.criterion(stage2_pred[i], parts_label[i].argmax(dim=1, keepdim=False)))
            loss.append(self.criterion(stage2_pred[i], parts_mask_gt[:, i].long()))
        loss = torch.stack(loss)
        return loss

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label']
            parts_mask_gt = batch['parts_mask_gt'].to(self.device)
            N, L, H, W = orig_label.shape

            stage1_pred = self.model(image)
            assert stage1_pred.shape == (N, 9, 128, 128)

            theta = self.select_net(F.softmax(stage1_pred, dim=1))
            assert theta.shape == (N, 6, 2, 3)
            parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, map_location=self.device)
            assert parts.shape == (N, 6, 3, 81, 81)

            stage2_pred = self.model2(parts)
            assert len(stage2_pred) == 6
            loss = []
            for i in range(6):
                # loss.append(self.criterion(stage2_pred[i], parts_label[i].argmax(dim=1, keepdim=False)).item())
                loss.append(self.criterion(stage2_pred[i], parts_mask_gt[:, i].long()).item())
            loss_list.append(np.mean(loss))
        return np.mean(loss_list)

    def train(self):
        self.model.train()
        self.model2.train()
        self.select_net.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer_select.zero_grad()
            loss = self.train_loss(batch)
            loss.backward(
                torch.ones(6, device=self.device, requires_grad=False))  # [1,1,1,1,1,1] weight for 6 parts loss
            self.optimizer2.step()
            self.optimizer_select.step()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_%s' % uuid, torch.mean(loss).item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, torch.mean(loss).item()))

    def eval(self):
        self.model.eval()
        self.model2.eval()
        self.select_net.eval()
        error = self.eval_error()

        if error < self.best_error:
            self.best_error = error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_%s' % uuid, error, self.epoch)
        print('epoch {}\terror {:.3}\tbest_error {:.3}'.format(self.epoch, error, self.best_error))

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
        path_model1 = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6", 'best.pth.tar')
        path_select = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB/6b4324c6", 'best.pth.tar')
        # path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_C/49997f1e", "best.pth.tar")
        path_model2 = os.path.join("/home/yinzi/data4/new_train/checkpoints_BC/b29dbe7e", "best.pth.tar")
        # checkpoints_BC / b29dbe7e
        if model == 'model1':
            fname = path_model1
            state = torch.load(fname, map_location=self.device)
            self.model.load_state_dict(state['model1'])
            print("load from" + fname)
        elif model == 'model2':
            fname = path_model2
            state = torch.load(fname, map_location=self.device)
            self.model2.load_state_dict(state['model2'])
            print("load from" + fname)
        elif model == 'select_net':
            fname = path_select
            state = torch.load(fname, map_location=self.device)
            self.select_net.load_state_dict(state['select_net'])
            print("load from" + fname)


class TrainModel_F1val(TrainModel):
    def __init__(self):
        super(TrainModel_F1val, self).__init__()
        self.f1_class = F1Score(self.device)
        self.best_F1 = float('-Inf')

    def eval(self):
        self.model.eval()
        self.model2.eval()
        self.select_net.eval()
        F1, error = self.eval_F1()
        if F1 > self.best_F1:
            self.best_F1 = F1
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('F1_overall_%s' % uuid, F1, self.epoch)
        print('epoch {}\tF1 {:.3}\terror {:.3}\tbest_F1 {:.3}'.format(self.epoch, F1, error, self.best_F1))

    def eval_F1(self):
        self.f1_class = F1Score(self.device)
        for batch in self.eval_loader:
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            parts_mask_gt = batch['parts_mask_gt'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label']
            N, L, H, W = orig_label.shape

            stage1_pred = self.model(image)
            assert stage1_pred.shape == (N, 9, 128, 128)

            theta = self.select_net(F.softmax(stage1_pred, dim=1))
            assert theta.shape == (N, 6, 2, 3)
            parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, map_location=self.device)
            assert parts.shape == (N, 6, 3, 81, 81)

            stage2_pred = self.model2(parts)

            error = []
            for i in range(6):
                error.append(self.criterion(stage2_pred[i], parts_mask_gt[:, i].long()))
            final_pred = affine_mapback(stage2_pred, theta, self.device)
            final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
            self.writer.add_image("final predict_%s" % uuid, final_grid[0], global_step=self.step, dataformats='HW')
            self.f1_class.forward(final_pred, orig_label.argmax(dim=1, keepdim=False))
        _, F1_overall = self.f1_class.get_f1_score()
        return F1_overall, error


def start_train():
    # train = TrainModel(args)
    train = TrainModel_F1val(args)

    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step(epoch)
        train.scheduler2.step(epoch)
        train.scheduler3.step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
