import torch
import torch.nn as nn
import torch.nn.functional as F
from icnnmodel import FaceModel, Stage2FaceModel
import torchvision


class Stage1Model(nn.Module):
    def __init__(self):
        super(Stage1Model, self).__init__()
        self.model = FaceModel()

    def forward(self, x):
        y = self.model(x)
        return y


class Stage2Model(nn.Module):
    def __init__(self):
        super(Stage2Model, self).__init__()
        self.model = nn.ModuleList([Stage2FaceModel()
                                    for _ in range(4)])
        for i in range(3):
            self.model[i].set_label_channels(2)
        self.model[3].set_label_channels(4)

    def forward(self, parts):
        eyebrow1_pred = self.model[0](parts[:, 0])
        eyebrow2_pred = torch.flip(self.model[0](torch.flip(parts[:, 1], [3])), [3])
        eye1_pred = self.model[1](parts[:, 2])
        eye2_pred = torch.flip(self.model[1](torch.flip(parts[:, 3], [3])), [3])
        nose_pred = self.model[2](parts[:, 4])
        mouth_pred = self.model[3](parts[:, 5])
        predict = [eyebrow1_pred, eyebrow2_pred,
                   eye1_pred, eye2_pred, nose_pred, mouth_pred]
        return predict


class SelectNet(nn.Module):
    def __init__(self):
        super(SelectNet, self).__init__()
        self.localize_net = nn.Sequential(
            nn.Conv2d(9, 6, kernel_size=3, stride=2, padding=1),  # 6 x 64 x 64
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 64 x 64
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 32 x 32
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 32 x 32
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 16 x 16
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 16 x 16
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 8 x 8
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 8 x 8
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 4 x 4
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 4 x 4
            nn.BatchNorm2d(6),

            nn.Conv2d(6, 6, kernel_size=[3, 2], stride=2, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.Tanh()
        )

    def forward(self, x):
        out = self.localize_net(x)
        assert out.shape == (x.shape[0], 6, 2, 3)
        activate_tensor = torch.tensor([[[1., 0., 1.],
                                         [0., 1., 1.]]], device=x.device,
                                       requires_grad=False).repeat((out.shape[0], out.shape[1], 1, 1))
        theta = out * activate_tensor
        return theta


class SelectNet_resnet(nn.Module):
    def __init__(self):
        super(SelectNet_resnet, self).__init__()
        self.model_res = torchvision.models.resnet18(pretrained=False)
        self.model_res.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.model_res.fc.in_features
        self.model_res.fc = nn.Linear(num_ftrs, 36)  # 6 x 2 x 3

    def forward(self, x):
        out = self.model_res(x).view(-1, 6, 2, 3)
        assert out.shape == (x.shape[0], 6, 2, 3)
        activate_tensor = torch.tensor([[[1., 0., 1.],
                                         [0., 1., 1.]]], device=x.device,
                                       requires_grad=False).repeat((out.shape[0], out.shape[1], 1, 1))
        theta = out * activate_tensor
        return theta


class SelectNet_dw(nn.Module):
    def __init__(self):
        super(SelectNet_dw, self).__init__()

        # 标准卷积
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        # 深度卷积
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        self.localize_net = nn.Sequential(
            conv_dw(9, 6, 2),  # 6 x 64 x 64
            conv_dw(6, 6, 2),  # 6 x 32 x 32
            conv_dw(6, 6, 2),  # 6 x 16 x 16
            conv_dw(6, 6, 2),  # 6 x 4 x 4
            nn.AdaptiveAvgPool2d((2, 3)),  # 6 x 2 x 3
            conv_dw(6, 6, 1),  # 6 x 2 x 3
            nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0),  # 6 x 2 x 3
            nn.Tanh()
        )

    def forward(self, x):
        out = self.localize_net(x)
        assert out.shape == (x.shape[0], 6, 2, 3)
        activate_tensor = torch.tensor([[[1., 0., 1.],
                                         [0., 1., 1.]]], device=x.device,
                                       requires_grad=False).repeat((out.shape[0], out.shape[1], 1, 1))
        theta = out * activate_tensor
        return theta


class BasicBlock(nn.Module):
    def __init__(self, channel_num):
        super(BasicBlock, self).__init__()
        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out


class SelectNet_dw_resblock(nn.Module):
    def __init__(self):
        super(SelectNet_dw_resblock, self).__init__()
        # 标准卷积
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))
        self.localize_net = nn.Sequential(
            conv_bn(9, 6, 2),               # 6 x 64 x 64
            BasicBlock(6),                  # res block 6 x 64 x 64
            conv_bn(6, 6, 2),               # 6 x 32 x 32
            BasicBlock(6),                  # res block 6 x 32 x 32
            conv_bn(6, 6, 2),               # 6 x 16 x 16
            BasicBlock(6),                  # res block 6 x 16 x 16
            conv_bn(6, 6, 2),               # 6 x 8 x 8
            BasicBlock(6),                  # res block 6 x 8 x 8
            conv_bn(6, 6, 2),               # 6 x 4 x 4
            BasicBlock(6),                  # res block 6 x 4 x 4
            nn.Conv2d(6, 6, kernel_size=[3, 2], stride=2, padding=1),   # 6 x 2 x 3
            BasicBlock(6),                  # res block 6 x 2 x 3
            nn.Tanh()                       # Tanh activation
        )

    def forward(self, x):
        out = self.localize_net(x)
        assert out.shape == (x.shape[0], 6, 2, 3)
        activate_tensor = torch.tensor([[[1., 0., 1.],
                                         [0., 1., 1.]]], device=x.device,
                                       requires_grad=False).repeat((out.shape[0], out.shape[1], 1, 1))
        theta = out * activate_tensor
        return theta
