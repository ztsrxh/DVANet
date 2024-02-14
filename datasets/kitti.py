import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch

class KITTIDataset(Dataset):
    def __init__(self, kitti15_datapath, kitti12_datapath, list_filename: list, training):
        self.datapath_15 = kitti15_datapath
        self.datapath_12 = kitti12_datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.processed = get_transform()
        self.min_disp = 3.8
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filenames: list):
        left_images = []
        right_images = []
        disp_images = []

        for list_filename in list_filenames:
            lines = read_all_lines(list_filename)
            splits = [line.split() for line in lines]

            left_name = splits[0][0].split('/')[1]
            if left_name.startswith('image'):
                datapath = self.datapath_15
            else:
                datapath = self.datapath_12

            left_images += [os.path.join(datapath, x[0]) for x in splits]
            right_images += [os.path.join(datapath, x[1]) for x in splits]

            if len(splits[0]) == 2:  # ground truth available
                disp_images = None
            else:
                disp_images += [os.path.join(datapath, x[2]) for x in splits]

        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(self.left_filenames[index])
        right_img = self.load_image(self.right_filenames[index])

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(self.disp_filenames[index])
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 1216, 352

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            # to tensor, normalize
            left_img = self.processed(left_img)
            right_img = self.processed(right_img)

            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            # generate the normalized depth map from the disparity map
            depth_norm_left = np.zeros(disparity.shape, dtype=np.float32)
            valid = disparity > 0.1
            depth_norm_left[valid] = self.min_disp / disparity[valid]

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "depth_left": depth_norm_left
                    }

        else:
            w, h = left_img.size

            # normalize
            left_img = self.processed(left_img).numpy()
            right_img = self.processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=-1)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=-1)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                depth_norm_left = np.zeros(disparity.shape, dtype=np.float32)
                valid = disparity > 0.1
                depth_norm_left[valid] = self.min_disp / disparity[valid]

                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "depth_left": depth_norm_left,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}