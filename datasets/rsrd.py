import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform
import torchvision.transforms as transforms
import torch


class RSRDDataset(Dataset):
    def __init__(self, max_depth, root='/dataset/RSRD-dense/', training=True):
        self.max_depth = max_depth
        self.image_list = []
        image1_list = []
        image2_list = []
        depth_left_list = []
        disp_list = []

        if training:
            root = root + 'train/'
            folders = os.listdir(root)
            for folder in folders:
                imgs = os.listdir(root + folder + '/left_half/')
                image1_list += [root + folder + '/left_half/' + i for i in imgs]
                image2_list += [root + folder + '/right_half/' + i for i in imgs]
                disp_list += [root + folder + '/disparity_half/' + i.replace('jpg', 'png') for i in imgs]
                depth_left_list += [root + folder + '/depth_half/' + i.replace('jpg', 'png') for i in imgs]
        else:
            root = root + 'test/'
            imgs = os.listdir(root + 'left_half/')
            image1_list += [root + 'left_half/' + i for i in imgs]
            image2_list += [root + 'right_half/' + i for i in imgs]
            disp_list += [root + 'disparity_half/' + i.replace('jpg', 'png') for i in imgs]
            depth_left_list += [root + 'depth_half/' + i.replace('jpg', 'png') for i in imgs]

        for img1, img2, dep_left in zip(image1_list, image2_list, depth_left_list):
            self.image_list += [[img1, img2, dep_left]]
        self.disparity_list = disp_list

        self.transform = get_transform()

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def __len__(self):
        return len(self.disparity_list)

    def __getitem__(self, index):
        left_img = Image.open(self.image_list[index][0])
        right_img = Image.open(self.image_list[index][1])
        disp_left = cv2.imread(self.disparity_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        depth_left = cv2.imread(self.image_list[index][2], cv2.IMREAD_ANYDEPTH) / 256.0

        ## the H&W of images should be multiples of 32, the images are cropped to 512*960
        # for fair comparison, please do not change the crop parameter
        left_img = self.transform(left_img)[:, 28:, :]
        right_img = self.transform(right_img)[:, 28:, :]
        disp_left = np.array(disp_left).astype(np.float32)[28:, :]
        depth_left = np.array(depth_left).astype(np.float32)[28:, :]
        depth_left_normalized = depth_left / self.max_depth

        return {
            "left": left_img,
            "right": right_img,
            "disparity": disp_left,
            "depth_left": depth_left_normalized,
            "name": self.image_list[index][0]
        }