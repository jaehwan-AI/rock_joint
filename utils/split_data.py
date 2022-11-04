import shutil
from glob import glob

img_lst = sorted(glob('/home/petro/joint/data/crop_data2/pick_Label/images/*.png'))

img_test = set(img_lst[:200])
img_train = set(img_lst) - img_test

for i in range(5):
    img_test = set(img_lst[30*i:30*(i+1)])
    img_train = set(img_lst) - img_test

    for img in img_train:
        num = img.split('_')[-1].split('.')[0]
        label = '/home/petro/joint/data/Label/joint_' + num + '.png'
        shutil.copy(img, f'/home/petro/joint/data/fold{i+1}/train/images/joint_{num}.png')
        shutil.copy(label, f'/home/petro/joint/data/fold{i+1}/train/Label/joint_{num}.png')

    for img in img_test:
        num = img.split('_')[-1].split('.')[0]
        label = '/home/petro/joint/data/Label/joint_' + num + '.png'
        shutil.copy(img, f'/home/petro/joint/data/fold{i+1}/val/images/joint_{num}.png')
        shutil.copy(label, f'/home/petro/joint/data/fold{i+1}/val/Label/joint_{num}.png')

fold1_lst = glob('/home/petro/joint/data/fold1/train/images/*.png')
fold2_lst = glob('/home/petro/joint/data/fold2/train/images/*.png')
fold3_lst = glob('/home/petro/joint/data/fold3/train/images/*.png')
fold4_lst = glob('/home/petro/joint/data/fold4/train/images/*.png')
fold5_lst = glob('/home/petro/joint/data/fold5/train/images/*.png')
print(len(fold1_lst), len(fold2_lst), len(fold3_lst), len(fold4_lst), len(fold5_lst))
