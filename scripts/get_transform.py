from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.misc import eval_util
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
import argparse
import pickle
import uuid
from rlkit.core import logger
import torch
from sawyer_control.envs.sawyer_grip import SawyerGripEnv
import matplotlib.pyplot as plt
filename = str(uuid.uuid4())
import numpy as np

import ipdb
import pandas as pd
import numpy as np
import argparse
# from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, VQVAEEncoderCNN

import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torch
import rlkit.torch.pytorch_util as ptu
import seaborn as sns
import os
import pickle
from rlkit.data_management.load_buffer_real import *
ptu.set_gpu_mode(True)

parser = argparse.ArgumentParser()
parser.add_argument('--buffer', default=1)
parser.add_argument('--local', action='store_false', default=True)
args = parser.parse_args()

if not args.local:
    data_path = '/nfs/kun1/users/ashvin/data/val_data'
    observation_key = 'image'
    paths = []
    args.buffer = int(args.buffer)
    if args.buffer == 0:
        print('lid on')
        paths.append((os.path.join(data_path, 'fixed_pot_demos.npy'), os.path.join(data_path, 'fixed_pot_demos_putlidon_rew.pkl')))
    elif args.buffer == 1:
        print('lid off')
        paths.append((os.path.join(data_path, 'fixed_pot_demos.npy'), os.path.join(data_path, 'fixed_pot_demos_takeofflid_rew.pkl')))
    elif args.buffer == 2:
        print('tray')
        paths.append((os.path.join(data_path, 'fixed_tray_demos.npy'), os.path.join(data_path, 'fixed_tray_demos_rew.pkl')))
    elif args.buffer == 3:
        print('drawer')
        paths.append((os.path.join(data_path, 'fixed_drawer_demos.npy'), os.path.join(data_path, 'fixed_drawer_demos_rew.pkl')))
    elif args.buffer == 4:
        print('Stephen Tool Use')
        path = '/nfs/kun1/users/stephentian/on_policy_longer_1_26_buffers/move_tool_obj_together_fixed_6_2_train.pkl'
    elif args.buffer == 5:
        print('General Demos')
        paths.append((os.path.join(data_path, 'general_demos.npy'), None))
    else:
        assert False

    if args.buffer in [4]:
        replay_buffer = pickle.load(open(path, 'rb'))
    else:
        replay_buffer = get_buffer(observation_key=observation_key, image_shape=(64, 64, 3))
        for path, rew_path in paths:
            load_path(path, rew_path, replay_buffer, bc=False)
else:
    img = Image.open('/home/ashvin/ros_ws/src/sawyer_control/src/frame.png')
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    obs_local = trans1(img.convert("RGB"))

import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from PIL import Image

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

plt.clf()
plt.setp(plt.gca(), autoscale_on=True)

from sawyer_control.envs.sawyer_grip import SawyerGripEnv
env = SawyerGripEnv(action_mode='position',
            config_name='ashvin_config',
            reset_free=False,
            position_action_scale=0.05,
            max_speed=0.4,
            step_sleep_time=0.2,
            crop_version_str="crop_val_torch")

def plot(obs_img):
    if type(obs_img) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(obs_img.cpu())
    else:
        im_new = obs_img
    plt.imshow(im_new)

def crop(img, img_dim = (64,64)):
    from matplotlib import cm
    img = img.astype(float)
    img /= 255.
    img = img[:, 50:530, :] 
    img = Image.fromarray(np.uint8(img*255))
    
    img = F.resize(img, img_dim, Image.ANTIALIAS)
    img = np.array(img)

    img = img*1.0/255
    img = img.transpose([2,0,1]) #.flatten()
    return torch.from_numpy(img).float()

while True:
    pts = []
    while len(pts) < 4:
        observation = env._get_obs()
        if args.local:
            obs_img_curr = np.flip(observation['hires_image_observation'], axis=-1)
        else:
            obs_img_curr = crop(np.flip(observation['hires_image_observation'], axis=-1), img_dim=(64,64))
        plot(obs_img_curr)
        tellme('Select 4 corners with mouse')
        pts = np.asarray(plt.ginput(4, timeout=-1))
        if len(pts) < 4:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second
 
    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

    tellme('Happy? Key click for yes, mouse click for no')

    if plt.waitforbuttonpress():
        for p in ph:
            p.remove()
        break

    for p in ph:
        p.remove()

src = pts

while True:
    pts = []
    while len(pts) < 4:
        observation = obs_local if args.local else torch.from_numpy(replay_buffer._obs['image'][0].reshape(3,64,64))
        obs_img = observation if args.local else torch.from_numpy(observation.numpy().swapaxes(-2,-1))
        plot(obs_img)
        tellme('Select 4 corners with mouse')
        pts = np.asarray(plt.ginput(4, timeout=-1))
        if len(pts) < 4:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second
 
    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

    tellme('Happy? Key click for yes, mouse click for no')

    if plt.waitforbuttonpress():
        break

    plt.clf()

dest = pts

plt.close()

import cv2 
# from torchgeometry.core.imgwarp import warp_perspective
matrix = cv2.getPerspectiveTransform(src.astype(np.float32),dest.astype(np.float32)) #Try this SWAP
obsnp = obs_img_curr if args.local else obs_img_curr.permute(1,2,0).cpu().numpy()
# cv2.imshow('original', obsnp)
# cv2.waitKey(0)
warped = cv2.warpPerspective(obsnp, matrix, (obsnp.shape[1], obsnp.shape[0])).squeeze()
# cv2.imshow('warped', warped)
# cv2.waitKey(0)

def plot_two_img(obs_img, obs_img2):
    plt.figure()
    plt.subplot(1,2,1)
    if type(obs_img) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(obs_img)
    else:
        im_new = obs_img
    plt.imshow(im_new)

    plt.subplot(1,2,2)
    if type(obs_img2) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(obs_img2)
    else:
        im_new = obs_img2
    plt.imshow(im_new)

    plt.show()

obs = env._get_obs()['hires_image_observation']
warped = cv2.warpPerspective(obs, matrix, (obs.shape[1], obs.shape[0])).squeeze()

obs = np.flip(obs, axis=-1)
img_curr = np.flip(warped, axis=-1)

plot_two_img(obs, img_curr)

if args.local:
    img1, img2 = crop(obs,img_dim=(64,64)), crop(img_curr, img_dim=(64,64))

plot_two_img(img1, img2)

np.save('/home/ashvin/ros_ws/src/railrl-private_anikait/scripts/matrix.npy', matrix)
import ipdb; ipdb.set_trace()