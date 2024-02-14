import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from datasets import KITTIDataset
from utils import *
from torch.utils.data import DataLoader
from models.dvanet import DVANet
import os
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Depth-aware volume attention for texture-less stereo matching (DVANet)')
parser.add_argument('--loadckpt', default='./checkpoints/XXX', help='load the weights from a specific checkpoint')
parser.add_argument('--save_dir', default='submit', help='directory for saving the inferred disparity maps')
parser.add_argument('--seed', type=int, default=3407, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.makedirs(args.save_dir, exist_ok=True)

args.maxdisp = 192
kitti15_datapath = '/dataset/stereo_dataset/kitti15/stereo/'
kitti12_datapath = '/dataset/stereo_dataset/kitti12/stereo/'
test_dataset = KITTIDataset(kitti15_datapath, kitti12_datapath, ['filenames/kitti12_test.txt'], training=False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

# model, optimizer
model = DVANet(args.maxdisp)
print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
model.cuda()

# load the checkpoint file specified by args.loadckpt
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict, strict=True)

@make_nograd_func
def test_sample():
    for batch_idx, sample in enumerate(TestImgLoader):
        imgL, imgR, top_pad, right_pad, filename = sample['left'], sample['right'], sample['top_pad'], sample['right_pad'], sample['left_filename']

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        _, volume_ests = model(imgL, imgR)
        disp_est = disparity_regression(volume_ests, args.maxdisp)
        w = disp_est.shape[-1] - right_pad
        disp_est = disp_est[0, top_pad:, :w]
        disp_est = disp_est.cpu().data.numpy()
        disp_est = np.round(disp_est*256).astype(np.uint16)
        path = os.path.join(args.save_dir, filename[0].split('/')[-1])
        cv2.imwrite(path, disp_est)
        # cv2.imwrite(path, cv2.applyColorMap(cv2.convertScaleAbs(disp_est, alpha=0.01), cv2.COLORMAP_JET))

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


if __name__ == '__main__':
    # save the inference results and submit to the website
    model.eval()
    test_sample()

