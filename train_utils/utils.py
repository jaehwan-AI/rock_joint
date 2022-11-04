import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class_colormap = pd.read_csv('/home/petro/joint/train_utils/class_color_dict.csv') #./train_utils/class_color_dict.csv'  

def create_label_colormap():
    global class_colormap

    colormap = np.zeros((2, 3), dtype=np.uint8)
    for idx, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[idx] = [r, g, b]
    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")
    
    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large!")

    return colormap[label]


def grid_image(images, masks, preds, n=4, shuffle=False):
    batch_size = masks.shape[0]
    if n > batch_size:
        n = batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 16)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다.
    gs = figure.add_gridspec(4, 3)
    ax = [None for _ in range(12)]

    for idx, choice in enumerate(choices):
        image = images[choice]
        mask = masks[choice]
        pred = preds[choice]

        ax[idx * 3] = figure.add_subplot(gs[idx, 0])
        ax[idx * 3].imshow(image)
        ax[idx * 3].grid(False)

        ax[idx * 3 + 1] = figure.add_subplot(gs[idx, 1])
        ax[idx * 3 + 1].imshow(label_to_color_image(mask))
        ax[idx * 3 + 1].grid(False)

        ax[idx * 3 + 2] = figure.add_subplot(gs[idx, 2])
        ax[idx * 3 + 2].imshow(label_to_color_image(pred))
        ax[idx * 3 + 2].grid(False)
        # 나중에 확률 값으로 얼마나 틀렸는지 시각화 해주는 열을 추가하면 더 좋을듯?

    figure.suptitle("image / GT / pred", fontsize=16)

    return figure
