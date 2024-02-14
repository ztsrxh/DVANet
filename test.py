import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from datasets import KITTIDataset, SceneFlowDatset, RSRDDataset
from utils import *
from torch.utils.data import DataLoader
from models.dvanet import DVANet
import os

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Depth-aware volume attention for texture-less stereo matching (DVANet)')
parser.add_argument('--dataset', choices=["kitti", "rsrd", "sceneflow"], default='rsrd', help='dataset name')
parser.add_argument('--loadckpt', default='./checkpoints/20240214030111/checkpoint_epoch02_002400.ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--seed', type=int, default=3407, metavar='S', help='random seed')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# dataset, dataloader
args.maxdisp = 192
if args.dataset == 'kitti':
    kitti15_datapath = '/dataset/stereo_dataset/kitti15/stereo/'
    kitti12_datapath = '/dataset/stereo_dataset/kitti12/stereo/'
    test_dataset_12 = KITTIDataset(kitti15_datapath, kitti12_datapath, ['filenames/kitti12_val.txt'], training=False)
    # test_dataset_15 = KITTIDataset(kitti15_datapath, kitti12_datapath, ['filenames/kitti15_val.txt'], training=False)
    args.use_wrde = True
    wrde_metric = WRDE(max_depth=50, min_depth=7, resolution=0.15, focal=719, baseline=0.537)
    args.max_depth = 60
    TestImgLoader = [DataLoader(test_dataset_12, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)]
elif args.dataset == 'sceneflow':
    args.use_wrde = False
    test_dataset = SceneFlowDatset(training=False)
    TestImgLoader = [DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)]
elif args.dataset == 'rsrd':
    args.maxdisp = 64
    args.max_depth = 13
    args.use_wrde = True
    wrde_metric = WRDE(max_depth=8, min_depth=2, resolution=0.15, focal=1001.46, baseline=0.1188)
    test_dataset = RSRDDataset(args.max_depth, training=False)
    TestImgLoader = [DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)]

# model, optimizer
model = DVANet(args.maxdisp)
print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
model.cuda()

# load the checkpoint file specified by args.loadckpt
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict, strict=True)


@make_nograd_func
def test_sample(test_loader):
    dis_metric_avg = AvgMteric()
    dep_metric_avg = AvgMteric()
    for batch_idx, sample in enumerate(test_loader):
        imgL, imgR, depth_left, disp_gt = sample['left'], sample['right'], sample["depth_left"], sample['disparity']
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        depth_left = depth_left.cuda()
        disp_gt = disp_gt.cuda()

        depth_norm_est, volume_ests = model(imgL, imgR)
        disp_est = disparity_regression(volume_ests, args.maxdisp)

        mask = (disp_gt < args.maxdisp) & (disp_gt > 0.1)
        dis_metric = {}
        dis_metric["EPE"] = EPE_metric(disp_est, disp_gt, mask).data.item()
        dis_metric["D1"] = D1_metric(disp_est, disp_gt, mask).data.item()
        dis_metric["Thres1"] = Thres_metric(disp_est, disp_gt, mask, 1.0).data.item()
        dis_metric["Thres2"] = Thres_metric(disp_est, disp_gt, mask, 2.0).data.item()
        dis_metric["Thres3"] = Thres_metric(disp_est, disp_gt, mask, 3.0).data.item()

        if not np.isnan(dis_metric["EPE"]):
            dis_metric_avg.update(dis_metric)
            if args.dataset in ['rsrd', 'kitti']:
                dep_metric = get_depth_metric(depth_norm_est, depth_left, args.max_depth)
                dep_metric_avg.update(dep_metric)
                wrde_metric.update_one(disp_est, disp_gt)

    if args.dataset in ['rsrd', 'kitti']:
        dep_avg_metrics = dep_metric_avg.get_metric()
    else:
        dep_avg_metrics = {'rmse': 0, 'a1': 0, 'abs_rel': 0}

    return dis_metric_avg.get_metric(), dep_avg_metrics

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


if __name__ == '__main__':
    model.eval()
    disp_avg_metrics, dep_avg_metrics = test_sample(TestImgLoader[0])

    if args.use_wrde:
        wrde = wrde_metric.get_wrde_metric()
        print('vali: EPE:%.3f, D1:%.3f,  >1px:%.3f, >3px:%.3f | wrde:%.3f | rmse:%.3f, a1:%.3f, rel:%.3f'% (
                disp_avg_metrics['EPE'], disp_avg_metrics['D1'] * 100, disp_avg_metrics['Thres1'] * 100, disp_avg_metrics['Thres3'] * 100,
                wrde, dep_avg_metrics['rmse'], dep_avg_metrics['a1'], dep_avg_metrics['abs_rel']))
    else:
        print('vali:  EPE:%.3f, D1:%.3f, >1px:%.3f, >3px:%.3f | rmse:%.3f, a1:%.3f, rel:%.3f' % (
                disp_avg_metrics['EPE'], disp_avg_metrics['D1'] * 100, disp_avg_metrics['Thres1'] * 100,disp_avg_metrics['Thres3'] * 100,
                dep_avg_metrics['rmse'], dep_avg_metrics['a1'], dep_avg_metrics['abs_rel']))

