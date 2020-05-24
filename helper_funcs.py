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


def affine_crop(img, label, size=81, mouth_size=81, points=None, theta_in=None, map_location=None, floor=False):
    n, l, h, w = img.shape
    img_in = img.to(map_location)
    label_in = label.to(map_location)
    if points is not None:
        theta = torch.zeros((n, 6, 2, 3), dtype=torch.float32, device=map_location, requires_grad=False)
        points_in = points.to(map_location)
        points_in = torch.cat([points_in[:, 1:6],
                               points_in[:, 6:9].mean(dim=1, keepdim=True)],
                              dim=1)
        if floor:
            points_in = torch.floor(points_in)
        assert points_in.shape == (n, 6, 2)
        for i in range(5):
            theta[:, i, 0, 0] = (size - 1) / (w - 1)
            theta[:, i, 0, 2] = -1 + (2 * points_in[:, i, 1]) / (w - 1)
            theta[:, i, 1, 1] = (size - 1) / (h - 1)
            theta[:, i, 1, 2] = -1 + (2 * points_in[:, i, 0]) / (h - 1)

        theta[:, 5, 0, 0] = (mouth_size - 1) / (w - 1)
        theta[:, 5, 0, 2] = -1 + (2 * points_in[:, 5, 1]) / (w - 1)
        theta[:, 5, 1, 1] = (mouth_size - 1) / (h - 1)
        theta[:, 5, 1, 2] = -1 + (2 * points_in[:, 5, 0]) / (h - 1)

    elif theta_in is not None:
        theta = theta_in
    assert theta.shape == (n, 6, 2, 3)

    samples = []
    # Not-mouth samples
    for i in range(5):
        grid = F.affine_grid(theta[:, i], [n, 3, size, size], align_corners=True).to(map_location)
        samples.append(F.grid_sample(input=img_in, grid=grid, align_corners=True,
                                     mode='bilinear', padding_mode='zeros'))
    # Mouth samples
    grid = F.affine_grid(theta[:, 5], [n, 3, mouth_size, mouth_size], align_corners=True).to(map_location)
    samples.append(F.grid_sample(input=img_in, grid=grid, align_corners=True,
                                 mode='bilinear', padding_mode='zeros'))
    # samples = torch.stack(samples, dim=1)
    temp = []
    labels_sample = []

    # Not-mouth Labels
    for i in range(1, 6):
        grid = F.affine_grid(theta[:, i - 1], [n, 1, size, size], align_corners=True).to(map_location)
        temp.append(F.grid_sample(input=label_in[:, i:i + 1], grid=grid,
                                  mode='nearest', padding_mode='zeros', align_corners=True))
    for i in range(5):
        bg = torch.tensor(1., device=map_location, requires_grad=False) - temp[i]
        labels_sample.append(torch.cat([bg, temp[i]], dim=1))

    temp = []
    # Mouth Labels
    for i in range(6, 9):
        grid = F.affine_grid(theta[:, 5], [n, 1, mouth_size, mouth_size], align_corners=True).to(map_location)
        temp.append(F.grid_sample(input=label_in[:, i:i + 1], grid=grid, align_corners=True,
                                  mode='nearest', padding_mode='zeros'))
    temp = torch.cat(temp, dim=1)
    assert temp.shape == (n, 3, mouth_size, mouth_size), print(temp.shape)
    bg = torch.tensor(1., device=map_location, requires_grad=False) - temp.sum(dim=1, keepdim=True)
    labels_sample.append(torch.cat([bg, temp], dim=1))
    """
    Shape of Parts
    torch.size(N, 6, 3, size, size)
    Shape of Labels
    List: [5x[torch.size(N, 2, size, size)], 1x [torch.size(N, 4, size, size)]]
    """
    # assert samples.shape == (n, 6, 3, size, size)
    return samples, labels_sample, theta


def affine_mapback(preds, theta, device, size=512):
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
        grid = F.affine_grid(theta=rtheta[:, i], size=[N, preds[i].shape[1], size, size],
                             align_corners=True).to(device)
        bg_grid = F.affine_grid(theta=rtheta[:, i], size=[N, 1, size, size], align_corners=True).to(device)
        temp = F.grid_sample(input=all_pred, grid=grid, mode='nearest', padding_mode='zeros',
                             align_corners=True)
        temp2 = F.grid_sample(input=all_pred[:, 0:1], grid=bg_grid, mode='nearest', padding_mode='border',
                              align_corners=True)
        bg.append(temp2)
        fg.append(temp[:, 1:])
        # del temp, temp2
    fg = torch.cat(fg, dim=1)   # Shape(N, 8, 512 ,512)
    bg = torch.cat(bg, dim=1)
    bg = (bg[:, 0:1] * bg[:, 1:2] * bg[:, 2:3] * bg[:, 3:4] *
          bg[:, 4:5] * bg[:, 5:6])
    sample = torch.cat([bg, fg], dim=1)
    # assert sample.shape == (N, 9, 5, 1024)
    return sample


def stage2_pred_onehot(stage2_predicts):
    out = [(F.softmax(stage2_predicts[i], dim=1) > 0.5).float()
           for i in range(6)]
    return out


def stage2_pred_softmax(stage2_predicts):
    out = [F.softmax(stage2_predicts[i], dim=1)
           for i in range(6)]
    return out


def fast_histogram(a, b, na, nb):
    '''
    fast histogram calculation
    ---
    * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
    '''
    assert a.shape == b.shape
    assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
    # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
    hist = np.bincount(
        nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
        minlength=na * nb).reshape(na, nb)
    assert np.sum(hist) == a.size
    return hist
