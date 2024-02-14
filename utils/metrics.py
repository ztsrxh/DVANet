import torch
import torch.nn.functional as F
from utils.experiment import make_nograd_func
from torch.autograd import Variable
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import pickle

def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()

# a wrapper to compute metrics for each image individually
def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.1:
                print("masks[idx].float().mean() too small, skip")
            else:
                ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
                results.append(ret)
        if len(results) == 0:
            print("masks[idx].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

@make_nograd_func
def get_depth_metric(depth_est, depth_gt, max_depth):
    depth_est = depth_est * max_depth
    depth_gt = depth_gt * max_depth
    mask = (depth_gt > 0.01) & (depth_gt < max_depth)
    valid_est = depth_est[mask]
    valid_gt = depth_gt[mask]

    metric_dict = {}
    thresh = torch.max((valid_gt / valid_est), (valid_est / valid_gt))
    metric_dict['a1'] = (thresh < 1.25).float().mean().data.item()
    metric_dict['a2'] = (thresh < 1.25 ** 2).float().mean().data.item()
    metric_dict['a3'] = (thresh < 1.25 ** 3).float().mean().data.item()
    rmse = (valid_gt - valid_est) ** 2
    metric_dict['rmse'] = torch.sqrt(torch.mean(rmse)).data.item()
    rmse_log = (torch.log(valid_gt) - torch.log(valid_est)) ** 2
    metric_dict['rmse_log'] = torch.sqrt(torch.mean(rmse_log)).data.item()
    metric_dict['abs_diff'] = torch.mean(torch.abs(valid_gt - valid_est)).data.item()
    metric_dict['abs_rel'] = torch.mean(torch.abs(valid_gt - valid_est) / valid_gt).data.item()
    metric_dict['sq_rel'] = torch.mean(((valid_gt - valid_est) ** 2) / valid_gt).data.item()

    return metric_dict

@make_nograd_func
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, reduction='mean')

class WRDE():
    def __init__(self, max_depth, min_depth, resolution, focal, baseline):
        '''
            max_depth, min_depth and resolution are all in meter
        '''
        self.max_depth, self.min_depth, self.resolution, self.focal, self.baseline = max_depth, min_depth, resolution, focal, baseline
        self.bf = self.baseline * self.focal

        self.num_intervals = int((self.max_depth - self.min_depth) / self.resolution)
        self.max_depth = self.min_depth + self.num_intervals * self.resolution
        self.depths = np.linspace(self.min_depth, self.max_depth, self.num_intervals+1)
        # self.disparities = self.bf / self.depths

        self.rel_err_sum = np.zeros(self.num_intervals)
        self.sample_count = np.zeros(self.num_intervals)

    @make_nograd_func
    def update_one(self, dis_est, dis_gt):
        B, H, W = dis_gt.shape
        border_width = int(0.05 * W)
        border_height = int(0.05 * H)
        dis_est = dis_est[:, border_height:-border_height, border_width:-border_width]
        dis_gt = dis_gt[:, border_height:-border_height, border_width:-border_width]

        mask = dis_gt > 0.1
        depth_gt = dis_gt.new_zeros(dis_gt.shape)
        depth_gt[mask] = self.bf / dis_gt[mask]
        depth_est = self.bf / dis_est
        depth_abs_error = torch.abs(depth_est - depth_gt)
        for i in range(self.num_intervals):
            mask = (depth_gt > self.depths[i]) & (depth_gt <= self.depths[i+1])
            if mask.float().sum() > 10:  # to reduce randomness, keep the intervals with more than 10 points
                self.rel_err_sum[i] += torch.mean(depth_abs_error[mask]/depth_gt[mask]).data.item()
                self.sample_count[i] += 1

    def clear(self):
        self.rel_err_sum *= 0
        self.sample_count *= 0

    def get_wrde_metric(self, weighted=True, segments=3):
        index_vali = self.sample_count >= 10
        sample_count = self.sample_count[index_vali]
        rel_err_avg = 100 * self.rel_err_sum[index_vali] / sample_count

        if weighted:
            count_per_seg = len(sample_count) // segments
            weights_all = np.zeros(len(sample_count))
            for i in range(1, segments):
                weights_all[(i - 1) * count_per_seg:i * count_per_seg] = i
            weights_all[i * count_per_seg:] = i + 1
            weights_all = weights_all / np.sum(weights_all)
            wrde = np.sum(weights_all * rel_err_avg)
        else:
            wrde = np.mean(rel_err_avg)

        # plt.plot(self.depths[:-1][index_vali], rel_err_avg, marker='*')
        # plt.ylabel('Relative error (%)', fontsize=16)
        # plt.xlabel('Depth (m)', fontsize=16)
        # plt.title(wrde, fontsize=16)
        # plt.show()

        return wrde