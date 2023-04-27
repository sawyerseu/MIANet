import torch.utils.data as data
import os
from glob import glob
import torch
from torchvision import transforms
import random
import numpy as np
import cv2
import json


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size=256,
                 downsample_ratio=8,
                 method='train'):
        # dark_bright
        import json
        darkjson = '/home/sawyer/jupyter/CrowdCountingRGBT/BL+IADM for RGBT Crowd Counting/data/RGBT-CC-CVPR2021/dark_list.json'
        brightjson = '/home/sawyer/jupyter/CrowdCountingRGBT/BL+IADM for RGBT Crowd Counting/data/RGBT-CC-CVPR2021/bright_list.json'
        json_f = open(brightjson)
        darklist = json.load(json_f)
        json_f.close()


        self.root_path = root_path
        self.gt_list = sorted(glob(os.path.join(self.root_path, '*.json')))  # change to npy for gt_list
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        all_json = []
        for json in self.gt_list:
            temp = json.split('/')[-1]
            all_json.append(temp)
        share = list(set(darklist).intersection(set(all_json)))

        self.dark_gt_list = []
        for i in range(len(share)):
            temp = os.path.join(self.root_path, str(share[i]))
            self.dark_gt_list.append(temp)



        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.RGB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.407, 0.389, 0.396],
                std=[0.241, 0.246, 0.242]),
        ])
        self.T_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.492, 0.168, 0.430],
                std=[0.317, 0.174, 0.191]),
        ])

    def __len__(self):
        return len(self.dark_gt_list)

    def __getitem__(self, item):



        gt_path = self.dark_gt_list[item]
        rgb_path = gt_path.replace('GT', 'RGB').replace('json', 'jpg')
        t_path = gt_path.replace('GT', 'T').replace('json', 'jpg')

        RGB = cv2.imread(rgb_path)[..., ::-1].copy()
        T = cv2.imread(t_path)[..., ::-1].copy()

        if self.method == 'train':
            #keypoints = np.load(gt_path)
            keypoints = []
            f = open(gt_path)
            dict_keypoints = json.load(f)
            for i in dict_keypoints['points']:    #keypoints['count']
                keypoints.append(i)
            #print(keypoints)

            f.close()
            return self.train_transform(RGB, T, keypoints)

        elif self.method == 'val' or self.method == 'test':  # TODO
            #keypoints = np.load(gt_path)
            gt = []
            f = open(gt_path)
            dict_keypoints = json.load(f)
            for i in dict_keypoints['points']:
                keypoints = i
                gt.append(keypoints)
            f.close()
            k = np.zeros((T.shape[0], T.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < T.shape[0] and int(gt[i][0]) < T.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            target = k

            RGB = self.RGB_transform(RGB)
            T = self.T_transform(T)
            name = os.path.basename(gt_path).split('.')[0]

            input = [RGB, T]
            return input, target, name

        else:
            raise Exception("Not implement")

    def train_transform(self, RGB, T, keypoints):
        ht, wd, _ = RGB.shape
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        RGB = RGB[i:i+h, j:j+w, :]
        T = T[i:i+h, j:j+w, :]
        keypoints = np.array(keypoints) - [j, i]
        #print(keypoints)
        idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                   (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
        keypoints = keypoints[idx_mask]

        RGB = self.RGB_transform(RGB)
        T = self.T_transform(T)
        input = [RGB, T]
        return input, torch.from_numpy(keypoints.copy()).float(), st_size

