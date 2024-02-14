from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from .efficientnet import efficientnet

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


class DVANet(nn.Module):
    def __init__(self, maxdisp):
        super(DVANet, self).__init__()
        self.maxdisp = maxdisp
        self.gw_channels = 32
        self.feature_extraction = efficientnet()

        self.dres_att_dis = nn.Sequential(convbn_3d(self.gw_channels, self.gw_channels, 3, 1, 1),
                                          nn.LeakyReLU(inplace=True),
                                          hourglass(32),
                                          convbn_3d(self.gw_channels, self.gw_channels, 3, 1, pad=1),
                                          nn.LeakyReLU(inplace=True))

        self.classif_att_dis = nn.Sequential(convbn_3d(self.gw_channels, self.gw_channels, 3, 1, 1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv3d(self.gw_channels, 1, kernel_size=3, padding=1, stride=1,
                                                       bias=False))

        self.volume_agg = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        hourglass(32),
                                        convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False)
                                        )

        self.dres0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True)
                                   )
        self.dres1 = hourglass(32)
        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        features_left, chann_atten, depth_norm_left = self.feature_extraction(left, left=True)
        features_right = self.feature_extraction(right, left=False)

        gwdiff_volume = build_gw_diff_volume(features_left, features_right, self.maxdisp // 4, self.gw_channels)
        gwdiff_volume = gwdiff_volume * chann_atten.unsqueeze(2)

        # backbone 3D agg
        gwdiff_volume_agg = self.volume_agg(gwdiff_volume)
        dis_atten = self.dres_att_dis(gwdiff_volume_agg)
        dis_atten = self.classif_att_dis(dis_atten)
        gwdiff_volume_agg = gwdiff_volume_agg * F.softmax(dis_atten, dim=2)

        gwdiff_volume_agg = self.dres0(gwdiff_volume_agg)
        gwdiff_volume_agg = self.dres1(gwdiff_volume_agg)
        gwdiff_volume_agg = self.classif0(gwdiff_volume_agg)

        volume_pred = F.interpolate(gwdiff_volume_agg, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                    align_corners=True)
        volume_pred = torch.squeeze(volume_pred, 1)
        volume_pred = F.softmax(volume_pred, dim=1)

        if self.training:
            volume_pred_atten = F.interpolate(dis_atten, [self.maxdisp, left.size()[2], left.size()[3]],
                                              mode='trilinear', align_corners=True)
            volume_pred_atten = torch.squeeze(volume_pred_atten, 1)
            volume_pred_atten = F.softmax(volume_pred_atten, dim=1)
            return depth_norm_left.squeeze(dim=1), [volume_pred_atten, volume_pred]
        else:
            return depth_norm_left.squeeze(dim=1), volume_pred


