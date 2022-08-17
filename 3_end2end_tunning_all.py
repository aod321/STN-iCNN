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
from preprocess import ToTensor, OrigPad, Resize, ToPILImage
from torch.utils.data import DataLoader
from dataset import HelenDataset
from data_augmentation import Stage1Augmentation
from model import Stage2Model, SelectNet, SelectNet_resnet
from helper_funcs import affine_crop, F1Accuracy, affine_mapback, stage2_pred_softmax
import torchvision
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import torchvision.transforms.functional as TF

uuid = str(uid.uuid1())[0:10]
print(uuid)
parser = argparse.ArgumentParser()
parser.add_argument("--stage1_dataset", type=str, help="Path for stage1 dataset")
parser.add_argument("--stage2_dataset", type=str, help="Path for stage2 dataset")
parser.add_argument("--pathmodel1",  type=str, help="Path for model A")
parser.add_argument("--pathmodel_select",  type=str, help="Path for cropper")
parser.add_argument("--pathmodel2",  type=str, help="Path for model B")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--select_net", default=1, type=int, help="Choose B structure, 0: custom 16 layer, 1: Res-18")
parser.add_argument("--f1_eval", default=1, type=int, help="using f1_score as eval")
parser.add_argument("--datamore", default=1, type=int, help="enable data augmentation")
parser.add_argument("--pretrainA", default=1, type=int, help="Load ModelA pretrain")
parser.add_argument("--pretrainB", default=1, type=int, help="Load ModelB pretrain")
parser.add_argument("--pretrainC", default=0, type=int, help="Load ModelC pretrain")
parser.add_argument("--lr", default=0, type=float, help="Learning rate for optimizer1")
parser.add_argument("--lr2", default=1e-3, type=float, help="Learning rate for optimizer2")
parser.add_argument("--lr_s", default=0, type=float, help="Learning rate for optimizer_select")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--cuda", default=0, type=int, help="Choose GPU with cuda number")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = args.stage1_dataset
parts_root_dir = args.stage2_dataset

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt",
    'test': "testing.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
            OrigPad()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
            OrigPad()
        ]),
    'test':
        transforms.Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
            OrigPad()
        ])
}

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

batch_size = {
    'train': args.batch_size,
    'val': 1,
    'test': 1
}

# DataLoader
Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir,
                           parts_root_dir=parts_root_dir,
                           transform=transforms_list[x]
                           )
           for x in ['train', 'val', 'test']
           }

stage1_augmentation = Stage1Augmentation(dataset=HelenDataset,
                                         txt_file=txt_file_names,
                                         root_dir=root_dir,
                                         parts_root_dir=parts_root_dir,
                                         resize=(128, 128)
                                         )
enhaced_stage1_datasets = stage1_augmentation.get_dataset()


if args.datamore == 0:
    dataloader = {x: DataLoaderX(Dataset[x], batch_size=batch_size[x],
                                shuffle=True, num_workers=4)
                  for x in ['train', 'val']
                  }

elif args.datamore == 1:
    dataloader = {x: DataLoaderX(enhaced_stage1_datasets[x], batch_size=batch_size[x],
                                shuffle=True, num_workers=4)
                  for x in ['train', 'val']
                  }

test_loader = DataLoaderX(Dataset['test'], batch_size=1,
                                shuffle=False, num_workers=4)


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
            self.load_pretrained("select_net", mode = self.args.select_net)

        self.model2 = Stage2Model().to(self.device)
        if self.args.pretrainC:
            self.load_pretrained("model2")

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
        self.test_loader = test_loader

        self.ckpt_dir = "checkpoints_ABC/%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
        parts_mask_gt = batch['parts_mask_gt'].to(self.device)
        N, L, H, W = orig_label.shape

        assert parts_mask_gt.shape == (N, 6, 81, 81)
        # Get Stage1 coarse mask
        stage1_pred = F.softmax(self.model(image), dim=1)
        assert stage1_pred.shape == (N, 9, 128, 128)

        # Mask2Theta
        theta = self.select_net(stage1_pred)
        assert theta.shape == (N, 6, 2, 3)

        # Affine_cropper
        parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, map_location=self.device)
        assert parts.shape == (N, 6, 3, 81, 81)

        # Make sure the backward grad stream unblocked
        assert parts.grad_fn is not None

        # Get Stage2 Mask Predict
        stage2_pred = self.model2(parts)

        # Calc Stage2 CrossEntropy Loss
        loss = []
        for i in range(6):
            loss.append(self.criterion(stage2_pred[i], parts_label[i].argmax(dim=1, keepdim=False)))
            # loss.append(self.criterion(stage2_pred[i], parts_mask_gt[:, i].long()))
        loss = torch.stack(loss)
        return loss

    def eval_error(self):
        loss_list = []
        for batch in tqdm(self.eval_loader):
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
            # parts_mask_gt = batch['parts_mask_gt'].to(self.device)
            N, L, H, W = orig_label.shape

            stage1_pred = F.softmax(self.model(image), dim=1)
            assert stage1_pred.shape == (N, 9, 128, 128)

            theta = self.select_net(stage1_pred)
            assert theta.shape == (N, 6, 2, 3)
            parts, parts_label, _ = affine_crop(img=orig, label=orig_label, theta_in=theta, map_location=self.device)
            assert parts.shape == (N, 6, 3, 81, 81)

            stage2_pred = self.model2(parts)
            assert len(stage2_pred) == 6
            loss = []
            for i in range(6):
                loss.append(self.criterion(stage2_pred[i], parts_label[i].argmax(dim=1, keepdim=False)).item())
                # loss.append(self.criterion(stage2_pred[i], parts_mask_gt[:, i].long()).item())
            loss_list.append(np.mean(loss))
        return np.mean(loss_list)

    def train(self):
        self.model.train()
        self.model2.train()
        self.select_net.train()
        self.epoch += 1
        for batch in tqdm(self.train_loader):
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

    def load_pretrained(self, model, mode=None):
        # path_model1 = os.path.join("/home/yinzi/data4/STN-iCNN/checkpoints_AB_res/e0de5954", 'best.pth.tar')
        path_model1 = args.pathmodel1
        # if mode == 0:
            # path_select = os.path.join("/home/yinzi/data4/new_train/checkpoints_AB_custom/122c2032", 'best.pth.tar')
        # elif mode == 1:
        # path_select = os.path.join("/home/yinzi/data4/STN-iCNN/checkpoints_AB_res/e0de5954", 'best.pth.tar')
        path_select = args.pathmodel_select
        # path_model2 = os.path.join("/home/yinzi/data4/STN-iCNN/checkpoints_C/c1f2ab1a", "best.pth.tar")
        path_model2 = args.pathmodel2
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
        self.f1_class = F1Accuracy(9)
        self.best_F1 = float('-Inf')
        self.test_best_f1 = float('-Inf')
        self.test_f1 = F1Accuracy(9)

    def eval(self):
        self.model.eval()
        self.model2.eval()
        self.select_net.eval()
        f1_accu, error = self.eval_F1()
        
        if f1_accu > self.best_F1:
            self.best_F1 = f1_accu
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        print('epoch {}\tF1 {:.3}\terror {:.3}\tbest_F1 {:.3}'.format(self.epoch, f1_accu, error, self.best_F1))
        self.writer.add_scalar('F1_overall_%s' % uuid, f1_accu, self.epoch)
        torch.cuda.empty_cache()
    
    def test(self):
        self.model.eval()
        self.model2.eval()
        self.select_net.eval()
        f1_accu, error = self.eval_F1(mode='test')
        if f1_accu > self.test_best_f1:
            self.test_best_f1 = f1_accu
        print('epoch {}\ttest_F1 {:.3}\terror {:.3}\ttest_best_F1 {:.3}'.format(self.epoch, f1_accu, error, self.test_best_f1))
        self.writer.add_scalar('test_F1_overall_%s' % uuid, f1_accu, self.epoch)
        torch.cuda.empty_cache()


    def eval_F1(self, mode='eval'):
        data_loader = self.eval_loader
        if mode == 'test':
            data_loader = self.test_loader
        for batch in tqdm(data_loader):
            names = batch['name']
            self.step_eval += 1
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            parts_mask_gt = batch['parts_mask_gt'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label'].to(self.device)
            N, L, H, W = orig_label.shape

            stage1_pred = F.softmax(self.model(image), dim=1)
            assert stage1_pred.shape == (N, 9, 128, 128)

            # Imshow stage1_pred on Tensorborad
            # stage1_pred_grid = torchvision.utils.make_grid(stage1_pred.argmax(dim=1, keepdim=True))
            # self.writer.add_image(f"{mode}_stage1 predict %s" % uuid, stage1_pred_grid[0], self.step_eval, dataformats="HW")

            # Mask2Theta using ModelB
            theta = self.select_net(stage1_pred)
            assert theta.shape == (N, 6, 2, 3)
            # AffineCrop
            parts, _ = affine_crop(img=orig, label=None, theta_in=theta, map_location=self.device)
            assert parts.shape == (N, 6, 3, 81, 81)

            # # imshow cropped parts
            # temp = []
            # for i in range(theta.shape[1]):
            #     test = theta[:, i]
            #     grid = F.affine_grid(theta=test, size=[N, 3, 81, 81], align_corners=True)
            #     temp.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
            # parts = torch.stack(temp, dim=1)
            # assert parts.shape == (N, 6, 3, 81, 81)
            # for i in range(6):
            #     parts_grid = torchvision.utils.make_grid(parts[:, i].detach().cpu())
            #     self.writer.add_image(f'{mode}_croped_parts_%s_%d' % (uuid, i), parts_grid, self.step_eval)

            # Predict Cropped Parts
            stage2_pred = self.model2(parts)
            softmax_stage2_pred = stage2_pred_softmax(stage2_pred)
            # Calc stage2 CrossEntropy Loss
            error = []
            for i in range(6):
                error.append(self.criterion(stage2_pred[i], parts_mask_gt[:, i].long()).item())

            # Imshow final predict mask
            final_pred = affine_mapback(softmax_stage2_pred, theta, self.device)
            # final_grid = torchvision.utils.make_grid(final_pred.argmax(dim=1, keepdim=True))
            # self.writer.add_image(f"{mode}_final predict_%s" % uuid, final_grid[0], global_step=self.step_eval, dataformats='HW')
            # Accumulate F1
            self.f1_class.collect(final_pred.argmax(dim=1, keepdim=False).detach(), orig_label.argmax(dim=1, keepdim=False).detach())
        f1_accu = self.f1_class.calc()
        return f1_accu, np.mean(error)


def start_train():
    if args.f1_eval !=0:
        train = TrainModel_F1val(args)
    else:
        train = TrainModel(args)
    for epoch in range(args.epochs):
        if epoch == 0:
            train.eval()
            try:
                train.test()
            except:
                pass
        train.train()
        train.scheduler.step(epoch)
        train.scheduler2.step(epoch)
        train.scheduler3.step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()
        try:
            train.test()
        except:
            pass

    print('Done!!!')


start_train()
