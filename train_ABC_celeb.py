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
from celebAMask_preprocess import ToPILImage, Resize
from torch.utils.data import DataLoader
from dataset import CelebAMask
from data_augmentation import Stage1Augmentation
from data_augmentation_celeb import Stage1Augmentation
from model import Stage2Model, SelectNet, SelectNet_resnet
from helper_funcs import affine_crop, affine_mapback, stage2_pred_softmax, calc_centroid
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

import torchvision

uuid = str(uid.uuid1())[0:10]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--select_net", default=1, type=int, help="Choose B structure, 0: custom 16 layer, 1: Res-18")
parser.add_argument("--datamore", default=0, type=int, help="enable data augmentation")
parser.add_argument("--pretrainA", default=1, type=int, help="Load ModelA pretrain")
parser.add_argument("--pretrainB", default=1, type=int, help="Load ModelB pretrain")
parser.add_argument("--pretrainC", default=1, type=int, help="Load ModelC pretrain")
parser.add_argument("--lr", default=0, type=float, help="Learning rate for optimizer1")
parser.add_argument("--lr2", default=1e-4, type=float, help="Learning rate for optimizer2")
parser.add_argument("--lr_s", default=0, type=float, help="Learning rate for optimizer_select")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--cuda", default=9, type=int, help="Choose GPU with cuda number")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = "/home/yinzi/data3/CelebAMask-HQ"

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128))
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128))
        ]),
    'test':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128))
        ])
}

# DataLoader
Dataset = {x: CelebAMask(
    root_dir=root_dir,
    mode=x,
    transform=transforms_list[x]
)
    for x in ['train', 'val', 'test']
}

stage1_augmentation = Stage1Augmentation(dataset=CelebAMask,
                                         root_dir=root_dir,
                                         resize=(128, 128)
                                         )
enhaced_stage1_datasets = stage1_augmentation.get_dataset()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


shuffles = {'train': True,
            'val': False}

batch_sizes = {'train': args.batch_size,
               'val': 1}

if args.datamore == 0:
    dataloader = {x: DataLoaderX(Dataset[x], batch_size=batch_sizes[x],
                                 shuffle=shuffles[x], num_workers=4)
                  for x in ['train', 'val']
                  }

elif args.datamore == 1:
    dataloader = {x: DataLoaderX(enhaced_stage1_datasets[x], batch_size=batch_sizes[x],
                                 shuffle=shuffles[x], num_workers=4)
                  for x in ['train', 'val']
                  }


class TrainModel(TemplateModel):

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.args = argus
        self.writer = SummaryWriter('log')
        self.step = 0
        self.step_eval = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        if self.args.pretrainA:
            self.load_pretrained("model1")

        if self.args.select_net == 0:
            self.select_net = SelectNet().to(self.device)
        elif self.args.select_net == 1:
            self.select_net = SelectNet_resnet().to(self.device)

        if self.args.pretrainB:
            self.load_pretrained("select_net", mode=self.args.select_net)

        self.model2 = Stage2Model().to(self.device)
        if self.args.pretrainC:
            self.load_pretrained("model2")

        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        # self.optimizer2 = optim.Adam(self.model2.parameters(), self.args.lr2)
        self.optimizer2 = [optim.Adam(self.model2.model[i].parameters(), self.args.lr2)
                           for i in range(4)]
        self.optimizer_select = optim.Adam(self.select_net.parameters(), self.args.lr_s)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_regress = nn.SmoothL1Loss()
        self.metric = nn.CrossEntropyLoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        # self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer2, step_size=5, gamma=0.5)
        self.scheduler2 = [optim.lr_scheduler.StepLR(self.optimizer2[i], step_size=5, gamma=0.5)
                           for i in range(4)]
        self.scheduler3 = optim.lr_scheduler.StepLR(self.optimizer_select, step_size=5, gamma=0.5)

        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']

        self.ckpt_dir = "checkpoints_ABC/%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
        N, L, H, W = orig_label.shape

        # Get Stage1 coarse mask
        pred = self.model(image)
        # stage1_pred = F.softmax(F.interpolate(pred, scale_factor=4, mode='nearest'), dim=1)
        stage1_pred = F.softmax(pred, dim=1)
        assert stage1_pred.shape == (N, 9, 128, 128), print(stage1_pred.shape)
        # lossA = self.criterion(pred, label.argmax(dim=1, keepdim=False))
        # Mask2Theta
        theta = self.select_net(stage1_pred)
        assert theta.shape == (N, 6, 2, 3)

        # Get ThetaLabel

        # Affine_cropper
        parts, parts_label, _ = affine_crop(img=orig, label=orig_label, size=127, mouth_size=255,
                                            theta_in=theta, map_location=self.device)
        # assert parts.shape == (N, 6, 3, 255, 255)
        # lossB = self.criterion_regress(theta, theta_gt)

        # Make sure the backward grad stream unblocked
        # assert parts.grad_fn is not None

        # Get Stage2 Mask Predict
        stage2_pred = self.model2(parts)

        # Calc Stage2 CrossEntropy Loss
        lossC = []
        for i in range(6):
            lossC.append(self.criterion(stage2_pred[i], parts_label[i].argmax(dim=1, keepdim=False)))
        lossC = torch.stack(lossC)
        return lossC

    def train(self):
        self.eval()
        self.model.train()
        self.model2.train()
        self.select_net.train()
        if self.args.lr == 0:
            self.model.eval()
        if self.args.lr_s == 0:
            self.select_net.eval()

        self.epoch += 1
        for batch in tqdm(self.train_loader):
            self.step += 1
            # self.optimizer.zero_grad()
            for i in range(4):
                self.optimizer2[i].zero_grad()
            # self.optimizer_select.zero_grad()
            lossC = self.train_loss(batch)
            lossC.backward(
                torch.ones(6, device=self.device, requires_grad=False))  # [1,1,1,1,1,1] weight for 6 parts loss
            mean_loss = torch.mean(lossC).item()
            for i in range(4):
                self.optimizer2[i].step()
            # self.optimizer_select.step()
            # self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalars('loss/train', {
                                                       'mean_loss': mean_loss
                                                       }, global_step=self.step)
                self.writer.close()
                print('epoch {}\tstep {}\t'
                      ' mean_loss {:.3}\t'.format(self.epoch, self.step,
                                                  mean_loss)
                      )

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
            for i in range(4):
                state['optimizer2_%d' % i] = self.optimizer2[i].state_dict()
            state['optimizer_select'] = self.optimizer_select.state_dict()

        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def load_pretrained(self, model, mode=None):
        # path_model1 = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB/89ce3b06", 'best.pth.tar')
        # path_model1 = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/840ea936", 'best.pth.tar')
        # checkpoints_AB_res/e3b6ee4a
        # path_model1 = os.path.join("/home/yinzi/data3/new_train/exp_A/f42800da", 'best.pth.tar')
        # path_model1 = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/6573baaa", 'best.pth.tar')
        # path_model1 = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/582423b2", 'best.pth.tar')
        # path_model1 = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/015d1282", 'best.pth.tar')
        path_model1 = os.path.join("/home/yinzi/data3/new_train/exp_A/f42800da", 'best.pth.tar')
        # path_model1 = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/54b10b10", 'best.pth.tar')

        if mode == 0:
            path_select = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_custom/122c2032", 'best.pth.tar')
        elif mode == 1:
            # path_select = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/6573baaa", 'best.pth.tar')
            # path_select = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/582423b2", 'best.pth.tar')
            path_select = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/a1cb96bc", 'best.pth.tar')
            # path_select = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/e3b6ee4a", 'best.pth.tar')
            # path_select = os.path.join("/home/yinzi/data3/new_train/checkpoints_AB_res/8e723d0a", 'best.pth.tar')
        # path_model2 = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/396e4702", "best.pth.tar")
        # path_model2 = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/cbde5caa", "best.pth.tar")
        path_model2 = os.path.join("/home/yinzi/data3/new_train/checkpoints_C/cec0face", "best.pth.tar")
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
    def __init__(self, arugs):
        super(TrainModel_F1val, self).__init__(arugs)
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
        print('epoch {}\tF1 {:.3}\terror {:.3}\tbest_F1 {:.3}'.format(self.epoch, F1, error, self.best_F1))
        self.writer.add_scalar('F1_overall_%s' % uuid, F1, self.epoch)

    def eval_F1(self):
        # Reset f1 calc class
        hist_list = []
        for batch in tqdm(self.eval_loader):
            self.step_eval += 1
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
            N, L, H, W = orig_label.shape

            # Get Stage1 coarse mask
            pred = self.model(image)
            stage1_pred = F.softmax(pred, dim=1)
            assert stage1_pred.shape == (N, 9, 128, 128)

            # Imshow stage1_pred on Tensorborad
            stage1_pred_grid = torchvision.utils.make_grid(stage1_pred.argmax(dim=1, keepdim=True))
            self.writer.add_image("stage1 predict%s" % uuid, stage1_pred_grid[0], self.step_eval, dataformats="HW")

            # Mask2Theta using ModelB
            theta = self.select_net(stage1_pred)
            assert theta.shape == (N, 6, 2, 3)
            # AffineCrop
            parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, size=127, mouth_size=255,
                                                map_location=self.device)
            # assert parts.shape == (N, 6, 3, 255, 255)
            for i in range(6):
                parts_grid = torchvision.utils.make_grid(parts[i].detach().cpu())
                parts_label_grid = torchvision.utils.make_grid(parts_label[i].argmax(dim=1, keepdim=False).unsqueeze(1))
                self.writer.add_image('croped_parts_%s_%d' % (uuid, i), parts_grid, self.step_eval)
                self.writer.add_image('croped_partslabel_%s_%d' % (uuid, i), parts_label_grid[0], self.step_eval,
                                      dataformats='HW')

            # Predict Cropped Parts
            stage2_pred = self.model2(parts)
            softmax_stage2_pred = stage2_pred_softmax(stage2_pred)
            # Calc stage2 CrossEntropy Loss
            error = []
            for i in range(6):
                error.append(self.criterion(stage2_pred[i], parts_label[i].argmax(dim=1, keepdim=False)).item())
                stage2_grid = torchvision.utils.make_grid(softmax_stage2_pred[i].argmax(dim=1, keepdim=True))
                self.writer.add_image('stage2_predict__%s_%d' % (uuid, i), stage2_grid[0], global_step=self.step_eval,
                                      dataformats='HW')

            # Imshow final predict mask
            final_pred = affine_mapback(softmax_stage2_pred, theta, self.device, size=512)
            final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
            self.writer.add_image("final predict_%s" % uuid, final_grid[0], global_step=self.step_eval,
                                  dataformats='HW')

            # Accumulate F1
            hist = self.fast_histogram(final_pred.argmax(dim=1, keepdim=False).cpu().numpy(),
                                       orig_label.argmax(dim=1, keepdim=False).cpu().numpy(),
                                       9, 9)
            hist_list.append(hist)

        # Calc F1 overall
        hist_sum = np.sum(np.stack(hist_list, axis=0), axis=0)
        A = hist_sum[1:9, :].sum()
        B = hist_sum[:, 1:9].sum()
        intersected = hist_sum[1:9, :][:, 1:9].sum()
        F1_overall = 2 * intersected / (A + B)
        return F1_overall, np.mean(error)

    def fast_histogram(self, a, b, na, nb):
        '''
        fast histogram calculation
        ---
        * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
        '''
        assert a.shape == b.shape, print(a.shape, b.shape)
        assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
        # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
        hist = np.bincount(
            nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
            minlength=na * nb).reshape(na, nb)
        assert np.sum(hist) == a.size
        return hist


def start_train():
    # train = TrainModel(args)
    train = TrainModel_F1val(args)

    for epoch in range(args.epochs):
        train.train()
        # train.scheduler.step()
        for i in range(4):
            train.scheduler2[i].step()
        # train.scheduler3.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
