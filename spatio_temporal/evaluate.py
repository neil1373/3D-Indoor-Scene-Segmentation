import argparse
import os
import warnings

import numpy as np

from lib.pc_utils import read_plyfile
from lib.utils import (
    read_txt, AverageMeter, fast_hist, per_class_iu,
)


def get_args():
    parser = argparse.ArgumentParser(
        description='evaluate validation prediction')
    parser.add_argument('--dataset_dir', type=str,
                        help='training and testing dataset root dir')
    parser.add_argument('--val_txt', type=str,
                        help='full path name to val txt file')
    parser.add_argument('--pred', type=str,
                        help='prediction folders')
    parser.add_argument('--show_ious', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = get_args()
    dataset_dir = args.dataset_dir
    val_txt = args.val_txt
    pred_dir = args.pred
    show_ious = args.show_ious

    val_fnames = sorted(read_txt(val_txt))
    val_fnames = [os.path.join(dataset_dir, fn) for fn in val_fnames]

    pred_fnames = get_txts(pred_dir)
    if os.path.isdir(pred_fnames[0]):
        pred_sub_dir = pred_fnames.copy()
        for one_sub in pred_sub_dir:
            pred_fnames = get_txts(one_sub)
            print(f"==> evaluate {one_sub}")
            validate(val_fnames, pred_fnames, show_ious)
            print()
    else:
        validate(val_fnames, pred_fnames, show_ious)


def get_txts(pred_dir):
    pred_fnames = sorted(os.listdir(pred_dir))
    pred_fnames = [os.path.join(pred_dir, fn) for fn in pred_fnames]
    return pred_fnames


def validate(val_fnames, pred_fnames, show_ious=False):
    assert len(val_fnames) == len(pred_fnames)
    num_labels = 200
    ious = AverageMeter()
    hist = np.zeros((num_labels, num_labels))

    for val_name, pred_name in zip(val_fnames, pred_fnames):
        pred = read_pred(pred_name)
        target = read_plyfile(val_name)[:, -1]
        assert pred.shape[0] == target.shape[0]

        hist += fast_hist(pred.flatten(), target.flatten(), num_labels)
        ious = per_class_iu(hist) * 100

    if show_ious:
        iou_message = ""
        for idx, one_iou in enumerate(ious, 1):
            iou_message += f"{one_iou:.1f}\t"
            if idx % 20 == 0:
                print(iou_message)
                iou_message = ""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        acc = hist.diagonal() / hist.sum(1) * 100
    mAcc = np.nanmean(acc)
    mIOU = np.nanmean(ious)
    message = f"mIOU: {mIOU:.3f}, mAcc: {mAcc:.3f}"
    print(message)


def read_pred(pred_name):
    """read labels from txt"""
    with open(pred_name, 'r') as file:
        lines = file.readlines()
    labels = [int(ln.strip()) for ln in lines]
    return np.array(labels)


if __name__ == "__main__":
    main()
