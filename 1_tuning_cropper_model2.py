from utils.template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.dataset import HelenDataset
from torchvision import transforms
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboardX as tb
import uuid as uid
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

# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor()
        ])
}
# DataLoader


 

class TrainModel(TemplateModel):

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.args = argus
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" %args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        self.load_pretrained("model1")
        self.model2 = Stage2Model().to(self.device)
        self.load_pretrained("model2")
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

        self.train_loader = stage1_dataloaders['train']
        self.eval_loader = stage1_dataloaders['val']

        self.ckpt_dir = "checkpoints_BC/%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig, orig_label = batch['orig'].to(self.device), batch['orig_label']

        parts_groundtruth = batch['parts'].to(self.device)
        parts_mask_groundtruth = batch['parts_mask_groundtruth'].to(self.device)

        N, L, H, W = orig_label.shape
        assert label.shape == (N, 9, 128, 128)

        theta = self.select_net(label)
        assert theta.shape == (N, 6, 2, 3)

        parts, _ = self.affine_cropper(orig, orig_label, theta)

        assert parts.grad_fn is not None
        
        assert parts.shape == (N, 6, 3, 81, 81)
        stage2_pred = self.model2(parts)
        assert len(stage2_pred) == 6

        loss = []
        for i in range(6):
            loss.append(self.criterion(stage2_pred[i], parts_mask_groundtruth[i].long()))
        loss = torch.stack(loss)
        return loss

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig, orig_label = batch['orig'].to(self.device), batch['orig_label']
            N, L, H, W = orig_label.shape

            assert label.shape == (N, 9, 128, 128)

            theta = self.select_net(label)
            assert theta.shape == (N, 6, 2, 3)

            parts, parts_label = affine_cropper(orig, orig_label, theta)
            assert parts.shape == (N, 6, 3, 81, 81)
            assert parts_label.shape == (N, 6, 81, 81) 

            stage2_pred = self.model2(parts)
            assert len(stage2_pred) == 6
            loss = []
            for i in range(6):
                loss.append(self.criterion(stage2_pred[i], parts_label[i].long()).item())
            loss_list.append(np.mean(loss))
        return np.mean(loss_list)

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            # self.optimizer.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer_select.zero_grad()
            loss = self.train_loss(batch)
            assert loss.shape == 6
            loss.backward(torch.ones(6, device=self.device, requires_grad=False))   #[1,1,1,1,1,1] weight for 6 parts loss
            self.optimizer2.step()
            self.optimizer_select.step()
            # self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss', loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, torch.mean(loss).item()))

    def eval(self):
        self.model.eval()
        error = self.eval_error()

        if error < self.best_error:
            self.best_error = error
            self.save_state(osp.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(osp.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error', error, self.epoch)
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

def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.train()
        # train.scheduler.step()
        train.scheduler2.step()
        train.scheduler3.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
