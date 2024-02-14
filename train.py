import argparse
import os
import shutil
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from datasets import KITTIDataset, SceneFlowDatset, RSRDDataset
from models import MyLoss
from utils import *
from torch.utils.data import DataLoader
from models.dvanet import DVANet
import pickle
import os
from datetime import datetime

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Depth-aware volume attention for texture-less stereo matching (DVANet)')
parser.add_argument('--dataset', choices=["kitti", "rsrd", "sceneflow"], default='rsrd', help='dataset name')
parser.add_argument('--lr', type=float, default=0.0014, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=5, help='training batch size')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train')
parser.add_argument('--logdir', default='./checkpoints/', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
parser.add_argument('--seed', type=int, default=3407, metavar='S', help='random seed')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# dataset, dataloader
args.maxdisp = 192
if args.dataset == 'kitti':
    args.summary_freq = 10
    kitti15_datapath = '/dataset/stereo_dataset/kitti15/stereo/'
    kitti12_datapath = '/dataset/stereo_dataset/kitti12/stereo/'
    train_dataset = KITTIDataset(kitti15_datapath, kitti12_datapath, ['filenames/kitti12_train.txt'], training=True)
    test_dataset_12 = KITTIDataset(kitti15_datapath, kitti12_datapath, ['filenames/kitti12_val.txt'], training=False)
    # test_dataset_15 = KITTIDataset(kitti15_datapath, kitti12_datapath, ['filenames/kitti15_val.txt'], training=False)
    args.use_wrde = True
    wrde_metric = WRDE(max_depth=50, min_depth=7, resolution=0.15, focal=719, baseline=0.537)
    args.max_depth = 60
    TestImgLoader = [DataLoader(test_dataset_12, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True),]
                     # DataLoader(vali_dataset_15, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)]
elif args.dataset == 'sceneflow':
    args.summary_freq = 600
    args.use_wrde = False
    train_dataset = SceneFlowDatset(training=True)
    test_dataset = SceneFlowDatset(training=False)
    TestImgLoader = [DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)]
elif args.dataset == 'rsrd':
    args.summary_freq = 50
    args.maxdisp = 64
    args.max_depth = 13
    args.use_wrde = True
    wrde_metric = WRDE(max_depth=8, min_depth=2, resolution=0.15, focal=1001.46, baseline=0.1188)
    train_dataset = RSRDDataset(args.max_depth, training=True)
    test_dataset = RSRDDataset(args.max_depth, training=False)
    TestImgLoader = [DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)]

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

# model, optimizer
model = DVANet(args.maxdisp)
print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
model.cuda()
model.train()
if args.loadckpt is not None:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict, strict=True)
loss_func = MyLoss(args.maxdisp).cuda()

from torch.cuda.amp import GradScaler
scaler = GradScaler()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, pct_start=0.01, three_phase=False,
                                            div_factor=8, anneal_strategy='linear', steps_per_epoch=len(TrainImgLoader))

# logging
args.logdir = os.path.join(args.logdir, datetime.utcnow().strftime('%Y%m%d%H%M%S'))
os.makedirs(args.logdir, exist_ok=True)
shutil.copy('./models/dvanet.py', os.path.join(args.logdir, 'dvanet.py'))
shutil.copy('train.py', os.path.join(args.logdir, 'train.py'))
shutil.copy('./models/loss.py', os.path.join(args.logdir, 'loss.py'))
shutil.copy('./models/efficientnet.py', os.path.join(args.logdir, 'efficientnet.py'))
log_file = open(os.path.join(args.logdir, 'log.txt'), 'a')

def train():
    global_step = 0
    for epoch_idx in tqdm(range(args.epochs)):
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step += 1
            imgL, imgR, depL_gt, disp_gt = sample['left'], sample['right'], sample["depth_left"], sample['disparity']
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            depL_gt = depL_gt.cuda()
            disp_gt = disp_gt.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                depth_ests, volume_ests = model(imgL, imgR)
                loss_all, [loss1, loss2, loss3] = loss_func(depth_ests, volume_ests, depL_gt, disp_gt)

            if np.isnan(loss_all.data.item()):
                print('nan loss!')
                exit()
            scaler.scale(loss_all).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if global_step % args.summary_freq == 0:
                info = 'train--> epoch%2d, lr:%.6f, loss_depth:%.4f, loss_disp1:%.4f, loss_disp2:%.4f' % (
                    epoch_idx+1, optimizer.param_groups[0]['lr'], loss1, loss2, loss3)
                log_file.write(info + '\n')
                log_file.flush()
                print(info)

                if global_step % (3 * args.summary_freq) == 0:
                    torch.save(model.state_dict(), "{}/checkpoint_epoch{:0>2}_{:0>6}.ckpt".format(args.logdir, epoch_idx+1, global_step))

                model.eval()
                for test_loader in TestImgLoader:
                    disp_avg_metrics, dep_avg_metrics = test_sample(test_loader)
                    epe = disp_avg_metrics['EPE']
                    if np.isnan(epe):
                        print('nan happened!')
                        exit()
                    if args.use_wrde:
                        wrde = wrde_metric.get_wrde_metric()
                        wrde_metric.clear()
                        info = 'vali:    EPE:%.3f, D1:%.3f,  >1px:%.3f, >3px:%.3f | wrde:%.3f | rmse:%.3f, a1:%.3f, rel:%.3f'% (
                            epe, disp_avg_metrics['D1'] * 100, disp_avg_metrics['Thres1'] * 100, disp_avg_metrics['Thres3'] * 100,
                            wrde, dep_avg_metrics['rmse'], dep_avg_metrics['a1'], dep_avg_metrics['abs_rel'])
                    else:
                        info = 'vali:    EPE:%.3f, D1:%.3f, >1px:%.3f, >3px:%.3f | rmse:%.3f, a1:%.3f, rel:%.3f'% (
                            epe, disp_avg_metrics['D1'] * 100, disp_avg_metrics['Thres1'] * 100,disp_avg_metrics['Thres3'] * 100,
                            dep_avg_metrics['rmse'], dep_avg_metrics['a1'], dep_avg_metrics['abs_rel'])
                    log_file.write(info + '\n')
                    log_file.flush()
                    print(info)
                model.train()


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
    train()
