import numpy as np
import argparse

import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torch
import rlkit.torch.pytorch_util as ptu
import seaborn as sns

matplotlib.use('Agg')
ptu.set_gpu_mode(True)

parser = argparse.ArgumentParser()
parser.add_argument('--traj_num', type=int, default=0)
parser.add_argument('--traj_path', type=str, default='/home/ashvin/ros_ws/evaluation.npy')
parser.add_argument('--save_path', type=str, default='/home/ashvin/ros_ws/src/railrl-private_anikait/images')
args = parser.parse_args()

def plot_img(obs_img):
    plt.figure()
    if type(obs_img) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(obs_img)
    else:
        im_new = obs_img
    plt.imshow(im_new)
    plt.show()

def save_img(obs_img, traj_num, img_num, prefix = 'img'):
    plt.figure()
    if type(obs_img) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(obs_img)
    else:
        im_new = obs_img
    import os
    f_name = os.path.join(args.save_path, prefix + '_' + str(traj_num)+ '_' + str(img_num))
    plt.imshow(im_new)
    plt.savefig(f_name)
    plt.show()

path = args.traj_path
trajs = np.load(path, allow_pickle=True)

for i in range(len(trajs)):
    traj = trajs[i]['cropped_images']
    for j in range(len(traj)):
        save_img(traj[j], i, j)



