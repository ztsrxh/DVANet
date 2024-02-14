import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import pickle

class SceneFlowDatset(Dataset):
    def __init__(self, training):
        self.datapath= 'XXX'
        if training:
            self.files_path = './filenames/sceneflow_train.txt'
            self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(self.files_path)
        else:
            self.files_path = './filenames/sceneflow_test.txt'
            self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(self.files_path)

            # self.left_filenames, self.right_filenames, self.disp_filenames = \
            #     self.left_filenames[::1000], self.right_filenames[::1000], self.disp_filenames[::1000]

        self.training = training
        self.processed = get_transform()
        self.min_disp = 1

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            # to tensor, normalize
            left_img = self.processed(left_img)
            right_img = self.processed(right_img)

            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            depth_norm_left = np.zeros(disparity.shape, dtype=np.float32)
            valid = disparity > 0.01
            depth_norm_left[valid] = self.min_disp / disparity[valid]

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "depth_left": depth_norm_left}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]
            depth_norm_left = np.zeros(disparity.shape, dtype=np.float32)
            valid = disparity > 0.01
            depth_norm_left[valid] = self.min_disp / disparity[valid]

            left_img = self.processed(left_img)
            right_img = self.processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "depth_left": depth_norm_left}
