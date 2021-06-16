import ipdb
import pandas as pd
import numpy as np
import argparse
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, VQVAEEncoderCNN

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
parser.add_argument('--traj_num', type=int, default=1)
parser.add_argument('--qfunc_path', type=str,
                    default='/home/ashvin/pickle_eval/guassian-policy-pot-off-minq1/guassian_policy_pot_off_minq1_2021_06_05_01_06_04_0000--s-0/model_pkl/500.pt')
parser.add_argument('--action_dim', type=int, default=4)
parser.add_argument('--policy', action='store_true')
parser.add_argument('--deeper_net', action='store_true')
parser.add_argument('--vqvae_enc', action='store_true')
parser.add_argument('--smimg', action='store_true')
parser.add_argument('--bottleneck', action='store_true')
parser.add_argument('--bc', action='store_true')
parser.add_argument('--buffer', default=1)
args = parser.parse_args()

data_path = '/nfs/kun1/users/ashvin/data/val_data'
observation_key = 'image'
paths = []
args.buffer = int(args.buffer)
if args.buffer == 0:
    print('lid on')
    paths.append((os.path.join(data_path, 'fixed_pot_demos.npy'),
                 os.path.join(data_path, 'fixed_pot_demos_putlidon_rew.pkl')))
elif args.buffer == 1:
    print('lid off')
    paths.append((os.path.join(data_path, 'fixed_pot_demos.npy'),
                 os.path.join(data_path, 'fixed_pot_demos_takeofflid_rew.pkl')))
elif args.buffer == 2:
    print('tray')
    paths.append((os.path.join(data_path, 'fixed_tray_demos.npy'),
                 os.path.join(data_path, 'fixed_tray_demos_rew.pkl')))
elif args.buffer == 3:
    print('drawer')
    paths.append((os.path.join(data_path, 'fixed_drawer_demos.npy'),
                 os.path.join(data_path, 'fixed_drawer_demos_rew.pkl')))
elif args.buffer == 4:
    print('Stephen Tool Use')
    path = '/nfs/kun1/users/stephentian/on_policy_longer_1_26_buffers/move_tool_obj_together_fixed_6_2_train.pkl'
else:
    assert False
if args.buffer in [4]:
    replay_buffer = pickle.load(open(path, 'rb'))
else:
    replay_buffer = get_buffer(observation_key=observation_key, image_shape=(
        48, 48, 3) if args.smimg else (64, 64, 3))
    for path, rew_path in paths:
        load_path(path, rew_path, replay_buffer,
                  small_img=args.smimg, bc=args.bc)


def plot_img(obs_img):
    plt.figure()
    if type(obs_img) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(obs_img.cpu())
    else:
        im_new = obs_img
    plt.imshow(im_new)
    plt.show()


action_dim = args.action_dim

cnn_params = dict(
    kernel_sizes=[3, 3, 3],
    n_channels=[16, 16, 16],
    strides=[1, 1, 1],
    hidden_sizes=[1024, 512, 256],
    paddings=[1, 1, 1],
    pool_type='max2d',
    pool_sizes=[2, 2, 1],  # the one at the end means no pool
    pool_strides=[2, 2, 1],
    pool_paddings=[0, 0, 0],
    image_augmentation=True,
    image_augmentation_padding=4)

cnn_params.update(
    input_width=64,
    input_height=64,
    input_channels=3,
    output_size=1,
    added_fc_input_size=action_dim,
)


if args.policy:
    if args.deeper_net:
        print('deeper conv net')
        cnn_params.update(
            kernel_sizes=[3, 3, 3, 3, 3],
            n_channels=[32, 32, 32, 32, 32],
            strides=[1, 1, 1, 1, 1],
            paddings=[1, 1, 1, 1, 1],
            pool_sizes=[2, 2, 1, 1, 1],
            pool_strides=[2, 2, 1, 1, 1],
            pool_paddings=[0, 0, 0, 0, 0]
        )

    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )

    cnn_params.update(
        output_size=256,
        added_fc_input_size=0,
        hidden_sizes=[1024, 512],
    )
    if args.vqvae_enc:
        policy_obs_processor = VQVAEEncoderCNN(**cnn_params)
    else:
        policy_obs_processor = CNN(**cnn_params)
    from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )
else:
    if args.bottleneck:
        qf1 = ConcatBottleneckCNN(
            action_dim, bottleneck_dim=16, deterministic=False, width=64, height=64)
    else:
        qf1 = ConcatCNN(**cnn_params)

parameters = torch.load(args.qfunc_path)
# import ipdb; ipdb.set_trace()
if args.policy:
    policy.load_state_dict(parameters['policy_state_dict'])
    policy = policy.to(ptu.device)
else:
    qf1.load_state_dict(parameters['qf1_state_dict'])
    qf1 = qf1.to(ptu.device)

def resize_small(img):
    from skimage.transform import resize
    flag = type(img) == torch.Tensor
    if flag:
        img = img.cpu().numpy()
        img = img.transpose(1, 2, 0)
    img = resize(img, (48, 48), anti_aliasing=True)
    if flag:
        img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = torch.from_numpy(img.numpy().swapaxes(-2, -1))
    return img

qvs = []
for i in range(args.traj_num):
    obs = replay_buffer._obs['image'][i]
    obs = ptu.from_numpy(obs)

    action = ptu.from_numpy(replay_buffer._actions[i])[None]    
    obs_tens = obs[None].to(ptu.device)
    
    if args.policy:
        qvals = policy.log_prob(obs_tens, action)
    else:
        qvals = qf1(obs_tens, action)
    
    qval = qvals.detach().cpu().numpy().flatten().item()
    qvs.append(qval)
plt.plot(qvs)
plt.show()
