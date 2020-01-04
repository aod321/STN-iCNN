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
from preprocess import ToTensor, OrigPad, Resize
from torch.utils.data import DataLoader
from dataset import HelenDataset
from model import SelectNet, SelectNet_resnet
import torchvision
from helper_funcs import F1Score, calc_centroid, affine_crop, affine_mapback

uuid = str(uid.uuid1())[0:8]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--select_net", default=1, type=int, help="Choose B structure, 0: custom 16 layer, 1: Res-18")
parser.add_argument("--pretrainA", default=0, type=int, help="Load pretrainA")
parser.add_argument("--pretrainB", default=0, type=int, help="Load pretrainB")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer1")
parser.add_argument("--lr_s", default=0.001, type=float, help="Learning rate for optimizer_select")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--cuda", default=1, type=int, help="Choose GPU with cuda number")
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
        self.writer = SummaryWriter('log_new')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        if self.args.pretrainA:
            self.load_pretrained("model1")
        if self.args.select_net == 1:
            self.select_net = SelectNet_resnet().to(self.device)
        elif self.args.select_net == 0:
            self.select_net = SelectNet().to(self.device)
        if self.args.pretrainB:
            self.load_pretrained("select_net", self.args.select_net)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.optimizer_select = optim.Adam(self.select_net.parameters(), self.args.lr_s)

        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.scheduler3 = optim.lr_scheduler.StepLR(self.optimizer_select, step_size=5, gamma=0.5)

        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']
        if self.args.select_net == 1:
            self.ckpt_dir = "checkpoints_AB_res/%s" % uuid
        else:
            self.ckpt_dir = "checkpoints_AB_custom/%s" % uuid
        self.display_freq = args.display_freq
        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
        N, L, H, W = orig_label.shape

        # Get stage1 predict mask (corase mask)
        stage1_pred = F.softmax(self.model(image), dim=1)
        assert stage1_pred.shape == (N, 9, 128, 128)

        # Mask2Theta
        theta = self.select_net(stage1_pred)
        assert theta.shape == (N, 6, 2, 3)

        """""
            Using original mask groundtruth to calc theta_groundtruth
        """""
        assert orig_label.shape == (N, 9, 1024, 1024)
        cens = torch.floor(calc_centroid(orig_label))
        points = torch.floor(torch.cat([cens[:, 1:6],
                                        cens[:, 6:9].mean(dim=1, keepdim=True)],
                                       dim=1))
        theta_label = torch.zeros((N, 6, 2, 3), device=self.device, requires_grad=False)
        for i in range(6):
            theta_label[:, i, 0, 0] = (81. - 1.) / (W - 1)
            theta_label[:, i, 0, 2] = -1. + (2. * points[:, i, 1]) / (W - 1)
            theta_label[:, i, 1, 1] = (81. - 1.) / (H - 1)
            theta_label[:, i, 1, 2] = -1. + (2. * points[:, i, 0]) / (H - 1)

        # Calc Regression loss, Loss func: Smooth L1 loss
        loss = self.regress_loss(theta, theta_label)
        return loss

    def eval_error(self):
        loss_list = []
        step = 0
        for batch in self.eval_loader:
            step += 1
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
            N, L, H, W = orig_label.shape
            # Get Stage1 mask predict
            stage1_pred = F.softmax(self.model(image), dim=1)
            assert stage1_pred.shape == (N, 9, 128, 128)

            # imshow stage1 mask predict
            stage1_pred_grid = torchvision.utils.make_grid(stage1_pred.argmax(dim=1, keepdim=True))
            self.writer.add_image("stage1 predict%s" % uuid, stage1_pred_grid, step)

            # Stage1Mask to Affine Theta
            theta = self.select_net(stage1_pred)
            assert theta.shape == (N, 6, 2, 3)

            # Calculate Affine theta ground truth
            assert orig_label.shape == (N, 9, 1024, 1024)
            cens = torch.floor(calc_centroid(orig_label))
            assert cens.shape == (N, 9, 2)
            points = torch.floor(torch.cat([cens[:, 1:6],
                                            cens[:, 6:9].mean(dim=1, keepdim=True)],
                                           dim=1))
            theta_label = torch.zeros((N, 6, 2, 3), device=self.device, requires_grad=False)
            for i in range(6):
                theta_label[:, i, 0, 0] = (81. - 1.) / (W - 1)
                theta_label[:, i, 0, 2] = -1. + (2. * points[:, i, 1]) / (W - 1)
                theta_label[:, i, 1, 1] = (81. - 1.) / (H - 1)
                theta_label[:, i, 1, 2] = -1. + (2. * points[:, i, 0]) / (H - 1)

            # calc regression loss
            loss = self.regress_loss(theta, theta_label)
            loss_list.append(loss.item())

        # imshow cropped parts
        temp = []
        for i in range(theta.shape[1]):
            test = theta[:, i]
            grid = F.affine_grid(theta=test, size=[N, 3, 81, 81], align_corners=True)
            temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
        parts = torch.stack(temp, dim=1)
        assert parts.shape == (N, 6, 3, 81, 81)
        for i in range(6):
            parts_grid = torchvision.utils.make_grid(
                parts[:, i].detach().cpu())
            self.writer.add_image('croped_parts_%s_%d' % (uuid, i), parts_grid, self.step)
        return np.mean(loss_list)

    def train(self):
        self.model.train()
        self.select_net.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            self.optimizer.zero_grad()
            self.optimizer_select.zero_grad()
            loss = self.train_loss(batch)
            loss.backward()
            self.optimizer_select.step()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_%s' % uuid, torch.mean(loss).item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, torch.mean(loss).item()))

    def eval(self):
        self.model.eval()
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
            state['select_net'] = self.select_net.module.state_dict()
        else:
            state['model1'] = self.model.state_dict()
            state['select_net'] = self.select_net.state_dict()

        if optim:
            state['optimizer'] = self.optimizer.state_dict()
            state['optimizer_select'] = self.optimizer_select.state_dict()

        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def load_pretrained(self, model, mode=None):
        path_modelA = os.path.join("/home/yinzi/data4/new_train/checkpoints_A/88736bbe", 'best.pth.tar')
        if mode == 0:
            path_modelB_select_net = os.path.join("/home/yinzi/data4/new_train/checkpoints_B_selectnet/cab2d814",
                                                  'best.pth.tar')
        elif mode == 1:
            path_modelB_select_net = os.path.join("/home/yinzi/data4/new_train/checkpoints_B_resnet/2a8e078e",
                                                  'best.pth.tar')

        if model == 'model1':
            fname = path_modelA
            state = torch.load(fname, map_location=self.device)
            self.model.load_state_dict(state['model1'])
            print("load from" + fname)
        elif model == 'select_net':
            fname = path_modelB_select_net
            state = torch.load(fname, map_location=self.device)
            self.select_net.load_state_dict(state['select_net'])
            print("load from" + fname)


class TrainModel_eval(TrainModel):

    def __init__(self, argus=args):
        super(TrainModel_eval, self).__init__(argus)
        self.best_f1 = float('-Inf')

    def eval_F1(self):
        step = 0
        f1_class = F1Score(self.device)  # reload f1 calc class
        loss_list = []
        for batch in self.eval_loader:
            step += 1
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
            N, L, H, W = orig_label.shape

            # Get stage1 predict mask (corase mask)
            stage1_pred = self.model(image)
            assert stage1_pred.shape == (N, 9, 128, 128)

            # Mask2Theta
            theta = self.select_net(F.softmax(stage1_pred, dim=1))
            test_pred = F.softmax(stage1_pred, dim=1).argmax(dim=1, keepdim=True)
            test_pred_grid = torchvision.utils.make_grid(test_pred)
            self.writer.add_image("stage1_pred_%s" % uuid, test_pred_grid[0], step, dataformats='HW')
            assert theta.shape == (N, 6, 2, 3)

            """""
                Using original mask groundtruth to calc theta_groundtruth
            """""
            assert orig_label.shape == (N, 9, 1024, 1024)
            cens = torch.floor(calc_centroid(orig_label))

            assert cens.shape == (N, 9, 2)
            points = torch.floor(torch.cat([cens[:, 1:6],
                                            cens[:, 6:9].mean(dim=1, keepdim=True)],
                                           dim=1))
            theta_label = torch.zeros((N, 6, 2, 3), device=self.device, requires_grad=False)
            for i in range(6):
                theta_label[:, i, 0, 0] = (81. - 1.) / (W - 1)
                theta_label[:, i, 0, 2] = -1. + (2. * points[:, i, 1]) / (W - 1)
                theta_label[:, i, 1, 1] = (81. - 1.) / (H - 1)
                theta_label[:, i, 1, 2] = -1. + (2. * points[:, i, 0]) / (H - 1)
            """""
                 Calc Regression loss, Loss func: Smooth L1 loss
            """""
            loss = self.regress_loss(theta, theta_label)
            loss_list.append(loss.item())

            """""
                 Imshow parts cropped by Affine cropper(using predicted theta) for debug convinience
            """""
            temp = []
            for i in range(theta.shape[1]):
                test = theta[:, i]
                grid = F.affine_grid(theta=test, size=[N, 3, 81, 81], align_corners=True)
                temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
            parts = torch.stack(temp, dim=1)
            assert parts.shape == (N, 6, 3, 81, 81)
            for i in range(6):
                parts_grid = torchvision.utils.make_grid(
                    parts[:, i].detach().cpu())
                self.writer.add_image('croped_parts_%s_%d' % (uuid, i), parts_grid, step)

            """""
                 Imshow final predict mask for debug convinience
            """""
            parts, parts_labels, _ = affine_crop(orig, orig_label, theta_in=theta, map_location=self.device)
            final_parts_mask = affine_mapback(parts_labels, theta, self.device)
            final_grid = torchvision.utils.make_grid(
                final_parts_mask.argmax(dim=1, keepdim=True))
            self.writer.add_image(
                "final_parts_mask %s" % uuid, final_grid[0], global_step=step, dataformats='HW')

            # Accumulate F1
            f1_class.forward(final_parts_mask, orig_label.argmax(dim=1, keepdim=False))

        # Calc F1 overall
        _, F1_overall = f1_class.get_f1_score()
        return F1_overall, np.mean(loss_list)

    def eval(self):
        self.model.eval()
        self.select_net.eval()
        f1_overall, error = self.eval_F1()
        if error < self.best_error:
            self.best_error = error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)

        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('f1overall_%s' % uuid, f1_overall, self.epoch)
        self.writer.add_scalar('val_error_%s' % uuid, error, self.epoch)
        print('epoch {}\terror {:.3}\tf1_overall {:.3}\tbest_error {:.3}'.format(self.epoch, error,
                                                                                 f1_overall, self.best_error))


def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step(epoch)
        # train.scheduler2.step(epoch)
        train.scheduler3.step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
