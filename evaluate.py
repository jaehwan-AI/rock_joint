import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import ROCKDataset
from dataset.transforms import test_transforms
from models.model import DeepCrack


def eval(args):
    # device
    if args.device == 'gpu0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda")
    elif args.device == 'gpu1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    transform = test_transforms()
    test_dataset = ROCKDataset(args, root=args.data_path, mode='val', transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model weights
    ckpt_dict = {'fold1': 'weights/DeepCrack_fold1_weight.pt',
                 'fold2': 'weights/DeepCrack_fold2_weight.pt',
                 'fold3': 'weights/DeepCrack_fold3_weight.pt',
                 'fold4': 'weights/DeepCrack_fold4_weight.pt',
                 'fold5': 'weights/DeepCrack_fold5_weight.pt'}

    total_mIoU = 0
    for ckpt in ckpt_dict.values():
        model = DeepCrack(pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)
        model.eval()

        hist = np.zeros((2, 2))
        fold_mIoU = 0
        for batch_item in test_loader:
            img = batch_item[0].to(device)
            mask = batch_item[1].to(device)

            with torch.no_grad():
                output = model(img)
            
            output = torch.argmax(output, dim=1).detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            hist = add_hist(hist, mask, output, n_class=2)
            _, _, mIoU, _, _ = label_accuracy_score(hist)
            fold_mIoU += mIoU
        
        fold_mIoU /= len(test_dataset)
        total_mIoU += fold_mIoU

    return total_mIoU / len(ckpt_dict.keys())


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default='cpu', help="inference device")
    parser.add_argument(
        "--data_path", type=str, required=True, help="data path for evaluate")
    parser.add_argument(
        "--img_size", type=int, default=1024)

    args = parser.parse_args()
    print(args)

    mIoU_result = eval(args)
    print(mIoU_result)
