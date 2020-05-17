import os
import numpy as np
import cv2
from tqdm import tqdm


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


def _read_names(file_name):
    label_names = []
    for name in open(file_name, 'r'):
        name = name.strip()
        if len(name) > 0:
            label_names.append(name)
    return label_names


def _merge(*list_pairs):
    a = []
    b = []
    for al, bl in list_pairs:
        a += al
        b += bl
    return a, b


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-gt_dir', help='the directory containing the groundtruth labels (in .png)')
    parser.add_argument(
        '-pred_dir', help='the directory containing the prediction labels (names should be the same with groundtruth files)')
    args = parser.parse_args()

    label_names_file = './label_names.txt'
    gt_label_names = pred_label_names = _read_names(label_names_file)

    assert gt_label_names[0] == pred_label_names[0] == 'bg'

    hists = []
    for name in tqdm(os.listdir(args.gt_dir)):
        if not name.endswith('.png'):
            continue

        gt_labels = cv2.imread(os.path.join(
            args.gt_dir, name), cv2.IMREAD_GRAYSCALE)

        pred_labels = cv2.imread(os.path.join(
            args.pred_dir, name), cv2.IMREAD_GRAYSCALE)
        hist = fast_histogram(gt_labels, pred_labels,
                              len(gt_label_names), len(pred_label_names))
        hists.append(hist)

    hist_sum = np.sum(np.stack(hists, axis=0), axis=0)

    eval_names = dict()
    for label_name in gt_label_names:
        gt_ind = gt_label_names.index(label_name)
        pred_ind = pred_label_names.index(label_name)
        eval_names[label_name] = ([gt_ind], [pred_ind])
    if 'le' in eval_names and 're' in eval_names:
        eval_names['eyes'] = _merge(eval_names['le'], eval_names['re'])
    if 'lb' in eval_names and 'rb' in eval_names:
        eval_names['brows'] = _merge(eval_names['lb'], eval_names['rb'])
    if 'ulip' in eval_names and 'imouth' in eval_names and 'llip' in eval_names:
        eval_names['mouth'] = _merge(
            eval_names['ulip'], eval_names['imouth'], eval_names['llip'])
    if 'eyes' in eval_names and 'brows' in eval_names and 'nose' in eval_names and 'mouth' in eval_names:
        eval_names['overall'] = _merge(
            eval_names['eyes'], eval_names['brows'], eval_names['nose'], eval_names['mouth'])
    print(eval_names)

    for eval_name, (gt_inds, pred_inds) in eval_names.items():
        A = hist_sum[gt_inds, :].sum()
        B = hist_sum[:, pred_inds].sum()
        intersected = hist_sum[gt_inds, :][:, pred_inds].sum()
        f1 = 2 * intersected / (A + B)
        print(f'f1_{eval_name}={f1}')
