import shutil
from glob import glob

img_lst = glob('data/raw/images/*.png')
label_lst = glob('data/raw/Label/*.png')

for i in range(len(img_lst)):
    rock_num = img_lst[i].split('/')[-1].split(' ')[1][1:-5]
    joint_num = label_lst[i].split('/')[-1].split(' ')[1][1:-5]

    shutil.copy(img_lst[i], f'data/images/joint_{rock_num}.png')
    shutil.copy(label_lst[i], f'data/Label/joint_{joint_num}.png')

