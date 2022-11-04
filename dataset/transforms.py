# https://albumentations.ai/docs/getting_started/mask_augmentation/
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms():
    return A.Compose([
        A.Rotate(limit=180, p=0.3),
        A.Flip(p=0.3),
        A.CLAHE(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(),
        A.CoarseDropout(max_holes=20, max_height=100, max_width=100, min_holes=10, min_height=50, min_width=50, fill_value=(0, 0, 0), mask_fill_value=None),
        ToTensorV2()
    ])


def test_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
