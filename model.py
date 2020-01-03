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
            nn.Conv2d(9, 6, kernel_size=3, stride=2, padding=1),  # 6 x 32 x 32
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

class SelectNet_new(nn.Module):
    def __init__(self):
        super(SelectNet_new, self).__init__()
        self.localize_net = nn.Sequential(
            nn.Conv2d(9, 6, kernel_size=16, stride=16, padding=0),       # 6x 8 x 8
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),       # 6x 4 x 4
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=[3, 2], stride=2, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(36, 36),
                                nn.Tanh())

    def forward(self, x):
        out = self.localize_net(x).view(-1, 36)
        out = self.fc(out).view(-1, 6, 2, 3)
        assert out.shape == (x.shape[0], 6, 2, 3)
        activate_tensor = torch.tensor([[[1., 0., 1.],
                                         [0., 1., 1.]]], device=x.device,
                                       requires_grad=False).repeat((out.shape[0], out.shape[1], 1, 1))
        theta = out * activate_tensor
        return theta


class SelectNet_new2(nn.Module):
    def __init__(self):
        super(SelectNet_new2, self).__init__()
        self.localize_net = nn.Sequential(
            nn.Conv2d(9, 6, kernel_size=8, stride=8, padding=0),  # 6x 8 x 8
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6x 8 x 8
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6x 4 x 4
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6x 4 x 4
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=[3, 2], stride=2, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU()
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
        self.model_res.fc = nn.Linear(num_ftrs, 36)      # 6 x 2 x 3

    def forward(self, x):
        out = self.model_res(x).view(-1, 6, 2, 3)
        assert out.shape == (x.shape[0], 6, 2, 3)
        activate_tensor = torch.tensor([[[1., 0., 1.],
                                         [0., 1., 1.]]], device=x.device,
                                       requires_grad=False).repeat((out.shape[0], out.shape[1], 1, 1))
        theta = out * activate_tensor
        return theta
