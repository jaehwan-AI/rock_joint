import os
from collections import namedtuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

RockJointClass = namedtuple('RockJointClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
CLASSES = [
        RockJointClass('rock', 0, 0, 'rock', 0, False, False, (0, 0, 0)),
        RockJointClass('joint', 1, 1, 'joint', 1, False, False, (255, 255, 255)),
    ]

class ROCKDataset(Dataset):
    # convert ids to train_ids
    id2trainid = np.array([label.train_id for label in CLASSES if label.train_id >= 0], dtype='uint8')
    
    # convert train_ids to colors
    mask_colors = [list(label.color) for label in CLASSES if label.train_id >= 0 and label.train_id <= 1]
    mask_colors.append([0, 0, 0])
    mask_colors = np.array(mask_colors)

    # convert train_ids to ids
    trainid2id = np.zeros((2), dtype='uint8')
    for label in CLASSES:
        if label.train_id >= 0 and label.train_id < 2:
            trainid2id[label.train_id] = label.id

    # Create list of class names
    classLabels = [label.name for label in CLASSES if not (label.ignore_in_eval or label.id < 0)]
    
    def __init__(self, args, root, mode='train', transforms=None):
        super(ROCKDataset, self).__init__()
        self.root = root
        self.args = args
        self.img_dir = os.path.join(self.root, mode, 'images')
        self.mask_dir = os.path.join(self.root, mode, 'Label')
        self.mode = mode
        self.transforms = transforms
        self.images = []
        self.targets = []

        assert mode in ['train', 'val', 'test'], 'Unknown value {} for argument mode'.format(mode)

        for file_name in os.listdir(self.img_dir):
            self.images.append(os.path.join(self.img_dir, file_name))
            if mode != 'test':
                self.targets.append(os.path.join(self.mask_dir, file_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]
        img = Image.open(file_path).convert('RGB')
        img = np.array(img)
        img = cv2.resize(img, (self.args.img_size, self.args.img_size))
        raw_img = img.copy()

        if self.mode != 'test':
            target = Image.open(self.targets[idx])
            target = np.array(target)
            target = cv2.resize(target, (self.args.img_size, self.args.img_size))
            target = cv2.normalize(target, None, 0, 1, cv2.NORM_MINMAX)

        if self.transforms is not None:
            if self.mode != 'test':
                transformed = self.transforms(image=img, mask=target)
                img = transformed['image']
                target = transformed['mask']
                target = self.id2trainid[target]
                return img, target, raw_img
            else:
                img = self.transforms(image=img)['image']
                return img, file_path
        else:
            if self.mode != 'test':
                target = self.id2trainid[target]
                return img, target, file_path
            else:
                return img, file_path


def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    target = torch.LongTensor([item[1] for item in batch])
    return data, target
