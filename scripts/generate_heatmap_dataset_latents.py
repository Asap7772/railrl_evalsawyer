import ipdb
import pandas as pd
import numpy as np
import argparse
from rlkit.torch.conv_networks import ConcatMlp
from rlkit.torch.sac.policies_v2 import TanhGaussianPolicy, GaussianPolicy, MakeDeterministic

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

from rlkit.misc.asset_loader import load_local_or_remote_file
vqvae_path = '/nfs/kun1/users/asap7772/best_vqvae.pt'
vqvae = load_local_or_remote_file(vqvae_path)

parser = argparse.ArgumentParser()
parser.add_argument('--traj_num', type=int, default=1)
parser.add_argument('--qfunc_path', type=str,
                    default='/home/ashvin/pickle_eval/guassian-policy-pot-off-minq1/guassian_policy_pot_off_minq1_2021_06_05_01_06_04_0000--s-0/model_pkl/500.pt')
parser.add_argument('--action_dim', type=int, default=4)
parser.add_argument('--policy', action='store_true')
parser.add_argument('--smimg', action='store_false', default=True)
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

if args.policy:
    policy = TanhGaussianPolicy(
        obs_dim=args.state_dim,
        action_dim=action_dim,
        hidden_sizes=[512]*4,
        obs_processor=None,
    )
else:
    M = 512
    mlp_params = dict(
        input_size=args.state_dim + action_dim,
        output_size=1,
        hidden_sizes=[M]*3,
    )
    qf1 = ConcatMlp(**mlp_params)

parameters = torch.load(args.qfunc_path)
# import ipdb; ipdb.set_trace()
if policy:
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

for i in range(args.traj_num):
    batch = replay_buffer.random_batch(1)
    obs = ptu.from_numpy(batch['observations']).squeeze()
    obs = obs.reshape(3,48,48)
    plot_img(obs)
    if args.smimg and replay_buffer is None:
        obs = resize_small(obs)
        obs = torch.from_numpy(ptu.get_numpy(obs).swapaxes(-2, -1))
    plot_img(obs)

    x = np.linspace(-0.8, 0.8)
    y = np.flip(np.linspace(-0.8, 0.8))
    actions = torch.from_numpy(np.array(np.meshgrid(x, y)))
    actions = actions.flatten(1).T

    actions_close = torch.cat(
        (actions, torch.zeros_like(actions)), axis=1).float().to(ptu.device)
    actions_open = torch.cat((actions, torch.zeros_like(actions)[
                             :, :1], torch.ones_like(actions)[:, :1]), axis=1).float().to(ptu.device)

    obs_tens = obs.repeat(actions.shape[0], 1, 1, 1).flatten(1).to(ptu.device)
    # qf1 = lambda x, y: y.sum(axis=1, keepdim=True)

    columns = np.around(x, decimals=2)
    index = np.around(y, decimals=2)
    columns = [str(x) for x in columns]
    index = [str(x) for x in index]

    print(actions)
    plt.plot(actions[:, 0], actions[:, 1])
    plt.show()
    if args.policy:
        qvals = policy.log_prob(obs_tens, actions_close)
    else:
        qvals = qf1(obs_tens, actions_close)
    plt.plot(qvals.detach().cpu().numpy().flatten())
    plt.show()
    qvals = qvals.detach().cpu().numpy().flatten().reshape(50, 50)
    df = pd.DataFrame(qvals, columns=columns, index=index)
    ax = sns.heatmap(df)
    plt.show()

    if args.policy:
        qvals = policy.log_prob(obs_tens, actions_open)
    else:
        qvals = qf1(obs_tens, actions_open)
    qvals = qvals.detach().cpu().numpy().flatten().reshape(50, 50)
    df = pd.DataFrame(qvals, columns=columns, index=index)
    ax = sns.heatmap(df)
    plt.show()
