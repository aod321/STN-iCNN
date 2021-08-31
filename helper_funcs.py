import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def calc_centroid(tensor):
    # Inputs Shape(N, 9 , 128, 128)
    # Return Shape(N, 9 ,2)
    input = tensor.float() + 1e-10
    n, l, h, w = input.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = input.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    center_x = input.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    output = torch.cat([center_y, center_x], 2)
    # output = torch.cat([center_x, center_y], 2)
    return output


class F1Score(torch.nn.CrossEntropyLoss):
    def __init__(self, device):
        super(F1Score, self).__init__()
        self.device = device
        self.name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        self.F1_name_list = ['eyebrows', 'eyes', 'nose', 'u_lip', 'i_mouth', 'l_lip', 'mouth_all']

        self.TP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.TN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.recall = {x: 0.0 + 1e-20
                       for x in self.F1_name_list}
        self.precision = {x: 0.0 + 1e-20
                          for x in self.F1_name_list}
        self.F1_list = {x: []
                        for x in self.F1_name_list}
        self.F1 = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}

        self.recall_overall_list = {x: []
                                    for x in self.F1_name_list}
        self.precision_overall_list = {x: []
                                       for x in self.F1_name_list}
        self.recall_overall = 0.0
        self.precision_overall = 0.0
        self.F1_overall = 0.0

    def forward(self, predict, labels):
        part_name_list = {1: 'eyebrow1', 2: 'eyebrow2', 3: 'eye1', 4: 'eye2',
                          5: 'nose', 6: 'u_lip', 7: 'i_mouth', 8: 'l_lip'}
        F1_name_list_parts = ['eyebrow1', 'eyebrow2',
                              'eye1', 'eye2',
                              'nose', 'u_lip', 'i_mouth', 'l_lip']
        TP = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        FP = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        TN = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        FN = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        pred = predict.argmax(dim=1, keepdim=False)
        # ground = labels.argmax(dim=1, keepdim=False)
        ground = labels.long()
        assert ground.shape == pred.shape
        for i in range(1, 9):
            TP[part_name_list[i]] += ((pred == i) * (ground == i)).sum().tolist()
            TN[part_name_list[i]] += ((pred != i) * (ground != i)).sum().tolist()
            FP[part_name_list[i]] += ((pred == i) * (ground != i)).sum().tolist()
            FN[part_name_list[i]] += ((pred != i) * (ground == i)).sum().tolist()

        self.TP['mouth_all'] += (((pred == 6) + (pred == 7) + (pred == 8)) *
                                 ((ground == 6) + (ground == 7) + (ground == 8))
                                 ).sum().tolist()
        self.TN['mouth_all'] += (
                (1 - ((pred == 6) + (pred == 7) + (pred == 8)).float()) *
                (1 - ((ground == 6) + (ground == 7) + (ground == 8)).float())
        ).sum().tolist()
        self.FP['mouth_all'] += (((pred == 6) + (pred == 7) + (pred == 8)) *
                                 (1 - ((ground == 6) + (ground == 7) + (ground == 8)).float())
                                 ).sum().tolist()
        self.FN['mouth_all'] += ((1 - ((pred == 6) + (pred == 7) + (pred == 8)).float()) *
                                 ((ground == 6) + (ground == 7) + (ground == 8))
                                 ).sum().tolist()

        for r in ['eyebrow1', 'eyebrow2']:
            self.TP['eyebrows'] += TP[r]
            self.TN['eyebrows'] += TN[r]
            self.FP['eyebrows'] += FP[r]
            self.FN['eyebrows'] += FN[r]

        for r in ['eye1', 'eye2']:
            self.TP['eyes'] += TP[r]
            self.TN['eyes'] += TN[r]
            self.FP['eyes'] += FP[r]
            self.FN['eyes'] += FN[r]

        for r in ['u_lip', 'i_mouth', 'l_lip']:
            self.TP[r] += TP[r]
            self.TN[r] += TN[r]
            self.FP[r] += FP[r]
            self.FN[r] += FN[r]

        for r in ['nose']:
            self.TP[r] += TP[r]
            self.TN[r] += TN[r]
            self.FP[r] += FP[r]
            self.FN[r] += FN[r]

        for r in self.F1_name_list:
            self.recall[r] = self.TP[r] / (
                    self.TP[r] + self.FP[r])
            self.precision[r] = self.TP[r] / (
                    self.TP[r] + self.FN[r])
            self.recall_overall_list[r].append(self.recall[r])
            self.precision_overall_list[r].append(self.precision[r])
            self.F1_list[r].append((2 * self.precision[r] * self.recall[r]) /
                                   (self.precision[r] + self.recall[r]))
        return self.F1_list, self.recall_overall_list, self.precision_overall_list

    def output_f1_score(self):
        # print("All F1_scores:")
        for x in self.F1_name_list:
            self.recall_overall_list[x] = np.array(self.recall_overall_list[x]).mean()
            self.precision_overall_list[x] = np.array(self.precision_overall_list[x]).mean()
            self.F1[x] = np.array(self.F1_list[x]).mean()
            print("{}:{}\t".format(x, self.F1[x]))
        for x in self.F1_name_list:
            self.recall_overall += self.recall_overall_list[x]
            self.precision_overall += self.precision_overall_list[x]
        self.recall_overall /= len(self.F1_name_list)
        self.precision_overall /= len(self.F1_name_list)
        self.F1_overall = (2 * self.precision_overall * self.recall_overall) / \
                          (self.precision_overall + self.recall_overall)
        print("{}:{}\t".format("overall", self.F1_overall))
        return self.F1, self.F1_overall

    def get_f1_score(self):
        # print("All F1_scores:")
        for x in self.F1_name_list:
            self.recall_overall_list[x] = np.array(self.recall_overall_list[x]).mean()
            self.precision_overall_list[x] = np.array(self.precision_overall_list[x]).mean()
            self.F1[x] = np.array(self.F1_list[x]).mean()
        for x in self.F1_name_list:
            self.recall_overall += self.recall_overall_list[x]
            self.precision_overall += self.precision_overall_list[x]
        self.recall_overall /= len(self.F1_name_list)
        self.precision_overall /= len(self.F1_name_list)
        self.F1_overall = (2 * self.precision_overall * self.recall_overall) / \
                          (self.precision_overall + self.recall_overall)
        return self.F1, self.F1_overall


def affine_crop(img, label=None, pred=None, points=None, theta_in=None, map_location=None, floor=False):
    n, l, h, w = img.shape
    img_in = img.to(map_location)
    if pred is not None:
        pred_in = F.interpolate(pred.type_as(img), scale_factor=1024 / 128, mode='bilinear', align_corners=True)
        assert pred_in.shape == (n, 9, 1024, 1024)

    if points is not None:
        theta = torch.zeros((n, 6, 2, 3), dtype=torch.float32, device=map_location, requires_grad=False)
        points_in = points.to(map_location)
        points_in = torch.cat([points_in[:, 1:6],
                               points_in[:, 6:9].mean(dim=1, keepdim=True)],
                              dim=1)
        if floor:
            points_in = torch.floor(points_in)
        assert points_in.shape == (n, 6, 2)
        for i in range(6):
            theta[:, i, 0, 0] = (81 - 1) / (w - 1)
            theta[:, i, 0, 2] = -1 + (2 * points_in[:, i, 1]) / (w - 1)
            theta[:, i, 1, 1] = (81 - 1) / (h - 1)
            theta[:, i, 1, 2] = -1 + (2 * points_in[:, i, 0]) / (h - 1)
    elif theta_in is not None:
        theta = theta_in
    assert theta.shape == (n, 6, 2, 3)
    samples = []
    for i in range(6):
        grid = F.affine_grid(theta[:, i], [n, 3, 81, 81], align_corners=True).type_as(theta)
        samples.append(F.grid_sample(input=img_in, grid=grid, align_corners=True,
                                     mode='bilinear', padding_mode='zeros'))
    samples = torch.stack(samples, dim=1)
    assert samples.shape == (n, 6, 3, 81, 81)
    if label is not None:
        label_in = label.to(map_location)
        labels_sample = crop_labels(label_in, theta)
        """
        Shape of Parts
        torch.size(N, 6, 3, 81, 81)
        Shape of Labels
        List: [5x[torch.size(N, 2, 81, 81)], 1x [torch.size(N, 4, 81, 81)]]
        """
        if pred is not None:
            pred_sample = crop_labels(pred_in, theta)
            return samples, labels_sample, pred_sample, theta
        else:
            return samples, labels_sample, theta
    else:
        return samples, theta
    
    
def crop_labels(label_in, theta):
    n = theta.shape[0]
    temp = []
    labels_sample = []
    # Not-mouth Labels
    for i in range(1, 6):
        grid = F.affine_grid(theta[:, i - 1], [n, 1, 81, 81], align_corners=True).type_as(theta)
        temp.append(F.grid_sample(input=label_in[:, i:i + 1], grid=grid,
                                  mode='nearest', padding_mode='zeros', align_corners=True))
    for i in range(5):
        bg = torch.tensor(1., device=theta.device, requires_grad=False) - temp[i]
        labels_sample.append(torch.cat([bg, temp[i]], dim=1))

    temp = []
    # Mouth Labels
    for i in range(6, 9):
        grid = F.affine_grid(theta[:, 5], [n, 1, 81, 81], align_corners=True).type_as(theta)
        temp.append(F.grid_sample(input=label_in[:, i:i + 1], grid=grid, align_corners=True,
                                  mode='nearest', padding_mode='zeros'))
    temp = torch.cat(temp, dim=1)
    assert temp.shape == (n, 3, 81, 81)
    bg = torch.tensor(1., device=theta.device, requires_grad=False) - temp.sum(dim=1, keepdim=True)

    labels_sample.append(torch.cat([bg, temp], dim=1))

    return labels_sample


def affine_mapback(preds, theta, device):
    N = theta.shape[0]
    ones = torch.tensor([[0., 0., 1.]]).repeat(N, 6, 1, 1).to(device)
    rtheta = torch.cat([theta, ones], dim=2).to(device)
    rtheta = torch.inverse(rtheta)
    rtheta = rtheta[:, :, 0:2]
    assert rtheta.shape == (N, 6, 2, 3)
    del ones
    # Parts_pred argmax Shape(N, 128, 128)
    fg = []
    bg = []
    for i in range(6):
        all_pred = preds[i]
        grid = F.affine_grid(theta=rtheta[:, i], size=[N, preds[i].shape[1], 1024, 1024],
                             align_corners=True).to(device)
        bg_grid = F.affine_grid(theta=rtheta[:, i], size=[N, 1, 1024, 1024], align_corners=True).to(device)
        temp = F.grid_sample(input=all_pred, grid=grid, mode='nearest', padding_mode='zeros',
                             align_corners=True)
        temp2 = 1 - F.grid_sample(input=1 - all_pred[:, 0:1], grid=bg_grid, mode='nearest', padding_mode='zeros',
                              align_corners=True)
        bg.append(temp2)
        fg.append(temp[:, 1:])
        # del temp, temp2
    bg = torch.cat(bg, dim=1)
    bg = (bg[:, 0:1] * bg[:, 1:2] * bg[:, 2:3] * bg[:, 3:4] *
          bg[:, 4:5] * bg[:, 5:6])
    fg = torch.cat(fg, dim=1)  # Shape(N, 8, 512 ,512)
    sample = torch.cat([bg, fg], dim=1)
    assert sample.shape == (N, 9, 1024, 1024)
    return sample


def stage2_pred_onehot(stage2_predicts):
    out = [(F.softmax(stage2_predicts[i], dim=1) > 0.5).float()
           for i in range(6)]
    return out


def stage2_pred_softmax(stage2_predicts):
    out = [F.softmax(stage2_predicts[i], dim=1)
           for i in range(6)]
    return out


class F1Accuracy(object):
    def __init__(self, num=2):
        super(F1Accuracy, self).__init__()
        self.hist_list = []
        self.num = num

    def fast_histogram(self, a, b, na, nb):
        '''
        fast histogram calculation
        ---
        * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
        '''
        assert a.shape == b.shape, (a.shape, b.shape)
        assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
        # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
        hist = np.bincount(
            nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
            minlength=na * nb).reshape(na, nb)
        assert np.sum(hist) == a.size
        return hist

    def collect(self, input, target):
        hist = self.fast_histogram(input.cpu().numpy(), target.cpu().numpy(),
                                   self.num, self.num)
        self.hist_list.append(hist)

    def calc(self):
        if self.hist_list:
            hist_sum = np.sum(np.stack(self.hist_list, axis=0), axis=0)
            A = hist_sum[1:self.num, :].sum()
            B = hist_sum[:, 1:self.num].sum()
            intersected = hist_sum[1:self.num, :][:, 1:self.num].sum()
            F1 = 2 * intersected / (A + B)
            self.hist_list.clear()
            return F1
        else:
            raise RuntimeError('No datas')