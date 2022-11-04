import argparse
import os
import random
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch

from train_utils.losses import DiceLoss, FocalLoss

focal_criterion = FocalLoss()
dice_criterion = DiceLoss()

category_names = [
    "Rock",
    "Joint"
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'T'):
        return True
    elif v.lower() in ('false', 'F'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def createDirectory(save_dir):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print("Error: Failed to create the directory.")


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(
        int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result
    - [acc]: overall accuracy
    - [acc_cls]: mean accuracy
    - [mean_iou]: mean IoU
    - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) +
                               hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
    return acc, acc_cls, mean_iou, fwavacc, iou


def train_step(training, batch_item, model, optimizer, criterion, device, beta=0.0):
    imgs = batch_item[0].to(device)
    masks = batch_item[1].to(device)

    if training is True:
        model.train()
        optimizer.zero_grad()
        hist = np.zeros((2, 2))

        if beta > 0:
            rand_idx = torch.randperm(imgs.size()[0])
            prob = np.random.random()
            if prob >= 0.7:
                # 가로
                imgs[:, :, imgs.shape[2]//2:,
                    :] = imgs[rand_idx, :, imgs.shape[2]//2:, :]
                masks[:, masks.shape[1]//2, :] = masks[rand_idx, masks.shape[1]//2, :]
            elif 0.4 <= prob < 0.7:
                imgs[:, :, :, imgs.shape[2] //
                    2:] = imgs[rand_idx, :, :, imgs.shape[2]//2:]
                masks[:, :, masks.shape[1] //
                    2:] = masks[rand_idx, :, masks.shape[1]//2:]

        with torch.cuda.amp.autocast():
            outputs = model(imgs)

        # calculate loss
        loss = criterion(outputs, masks.long())

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        hist = add_hist(hist, masks, outputs, n_class=2)
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        return loss, acc, acc_cls, mIoU, fwavacc, IoU

    else:
        model.eval()
        hist = np.zeros((2, 2))

        with torch.no_grad():
            outputs = model(imgs)

        # calculate loss
        loss = criterion(outputs, masks.long())

        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        hist = add_hist(hist, masks, outputs, n_class=2)
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        return batch_item[2], masks, outputs, loss, acc, acc_cls, mIoU, fwavacc, IoU


def two_loss_step(training, batch_item, model, optimizer, device, beta=0.0):
    imgs = batch_item[0].to(device)
    masks = batch_item[1].to(device)

    if training is True:
        model.train()
        optimizer.zero_grad()
        hist = np.zeros((2, 2))

        if beta > 0.0:
            rand_idx = torch.randperm(imgs.size()[0])
            prob = np.random.random()
            if prob >= 0.7:
                # 가로
                imgs[:, :, imgs.shape[2]//2:,
                    :] = imgs[rand_idx, :, imgs.shape[2]//2:, :]
                masks[:, masks.shape[1]//2, :] = masks[rand_idx, masks.shape[1]//2, :]
            elif 0.4 <= prob < 0.7:
                imgs[:, :, :, imgs.shape[2] //
                    2:] = imgs[rand_idx, :, :, imgs.shape[2]//2:]
                masks[:, :, masks.shape[1] //
                    2:] = masks[rand_idx, :, masks.shape[1]//2:]

        with torch.cuda.amp.autocast():
            outputs = model(imgs)

        # calculate loss
        focal_loss = focal_criterion(outputs, masks.long())
        dice_loss = dice_criterion(outputs, masks, softmax=True)
        loss = 0.5 * focal_loss + 0.5 * dice_loss

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        hist = add_hist(hist, masks, outputs, n_class=2)
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        return loss, acc, acc_cls, mIoU, fwavacc, IoU

    else:
        model.eval()
        hist = np.zeros((2, 2))

        with torch.no_grad():
            outputs = model(imgs)

        # calculate loss
        focal_loss = focal_criterion(outputs, masks.long())
        dice_loss = dice_criterion(outputs, masks, softmax=True)
        loss = 0.5 * focal_loss + 0.5 * dice_loss

        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        hist = add_hist(hist, masks, outputs, n_class=2)
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        return batch_item[2], masks, outputs, loss, acc, acc_cls, mIoU, fwavacc, IoU
