import torch.nn.functional as F
import torch
from torch import nn

class MyLoss(nn.Module):
    def __init__(self, max_disp):
        super(MyLoss, self).__init__()
        self.max_disp = max_disp
        self.disp_values = torch.arange(0, max_disp, dtype=torch.float32).view(1, self.max_disp, 1, 1).cuda()

    def forward(self, depth_ests, volume_ests: list, depth_gt, disp_gt):
        valid_dis = (disp_gt < self.max_disp) & (disp_gt > 0.1)
        disp_gt_valid = disp_gt[valid_dis]

        #######  depth loss #######
        valid_dep = depth_gt > 1e-5
        loss_depth = F.smooth_l1_loss(depth_ests[valid_dep], depth_gt[valid_dep], reduction='mean')

        #######  disparity loss #######
        disp_est = torch.sum(volume_ests[0] * self.disp_values, 1, keepdim=False)
        loss_disp1 = F.smooth_l1_loss(disp_est[valid_dis], disp_gt_valid, reduction='mean')
        disp_est = torch.sum(volume_ests[1] * self.disp_values, 1, keepdim=False)
        loss_disp2 = F.smooth_l1_loss(disp_est[valid_dis], disp_gt_valid, reduction='mean')

        return loss_depth/loss_depth.detach() + loss_disp1/loss_disp1.detach() + loss_disp2/loss_disp2.detach(), [loss_depth.data.item(), loss_disp1.data.item(), loss_disp2.data.item()]