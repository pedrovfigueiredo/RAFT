import torch
import torch.nn as nn
import os
import glob
import cv2
import argparse
import numpy as np

import torchvision.transforms as transforms
from core.utils import flow_viz

from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="set gpu device")
parser.add_argument('--small',  action='store_true')
parser.add_argument("--raft_pretrained_ckpt", type=str, default="RAFT/pretrained_weights/raft-kitti.pth", help='path for pre-trained raft ckpt')

args = parser.parse_args()

folder = '/mnt/g/Projects/TAMU/Research/data/viewsynthesisnima_v2_allframes/test/Cars/00_01_02_03_04_05_06'
files = sorted(glob.glob(os.path.join(folder, '*')))

model = torch.nn.DataParallel(RAFT(args)).to(args.device)
model.load_state_dict(torch.load(args.raft_pretrained_ckpt))
model = model.module
model.eval()
print('Loaded pre-trained flow net weights: {}'.format(os.path.basename(args.raft_pretrained_ckpt)))

for i in range(len(files) - 1):
    img1 = cv2.imread(files[i])
    img1 = cv2.resize(img1, (224, 128), interpolation = cv2.INTER_AREA)
    img1 = np.float32(img1)
    img1 = img1[:, :, [2, 1, 0]]
    img1 = torch.from_numpy(np.transpose(img1, (2, 0, 1))).float().unsqueeze(dim=0).to(args.device)

    img2 = cv2.imread(files[i+1])
    img2 = cv2.resize(img2, (224, 128), interpolation = cv2.INTER_AREA)
    img2 = np.float32(img2)
    img2 = img2[:, :, [2, 1, 0]]
    img2 = torch.from_numpy(np.transpose(img2, (2, 0, 1))).float().unsqueeze(dim=0).to(args.device)

    padder = InputPadder(img1.shape)
    flow_net_img1, flow_net_img2 = padder.pad(img1, img2)
    flow_low, flow_up = model(flow_net_img1, flow_net_img2, iters=32, test_mode=True)
    # aux_flow = padder.unpad(flow_up)

    aux_flow = flow_up[0].permute(1,2,0).cpu().detach().numpy()

    print('flow {}. max: {}, min:{}'.format(i, np.max(aux_flow), np.min(aux_flow)))

    Flow = flow_viz.flow_to_image(aux_flow)

    if not os.path.exists('RAFT/res_test'):
        os.makedirs('RAFT/res_test')

    cv2.imwrite(os.path.join('RAFT/res_test', '{}-{}.png'.format(i, i+1)), np.uint8(Flow))

