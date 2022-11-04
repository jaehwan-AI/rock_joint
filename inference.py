import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset.transforms import test_transforms
from models.model import DeepCrack


def inference(args):
    # device
    if args.device == 'gpu0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda")
    elif args.device == 'gpu1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = test_transforms()
    model = DeepCrack(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    img_lst = glob(args.data_path+'/*')

    for img_path in tqdm(img_lst):
        file_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        img = np.array(img)
        h, w = img.shape[:2]
        img = cv2.resize(img, (1024, 1024))

        transformed_img = transform(image=img)['image'].to(device)
        with torch.no_grad():
            output = model(transformed_img.unsqueeze(0))
        result = torch.argmax(output, dim=1).squeeze(0).detach().cpu().numpy()
        
        # 원본 사이즈 복원
        result = cv2.resize(result.astype('float32'), (h, w))
        _, result = cv2.threshold(result, args.threshold, 1, cv2.THRESH_BINARY)
        result = result * 255

        # 추론 결과 저장
        cv2.imwrite(args.save_path+'/'+str(args.threshold)+'_'+file_name, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default='cpu', help="inference device")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="model checkpoint path")
    parser.add_argument(
        "--data_path", type=str, required=True, help="data path for inference")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="binary threshold")
    parser.add_argument(
        "--save_path", type=str, required=True, help="save path for save inference images")

    args = parser.parse_args()
    print(args)

    inference(args)
