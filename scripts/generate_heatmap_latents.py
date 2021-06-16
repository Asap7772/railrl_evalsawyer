import ipdb
import numpy as np
import argparse
from rlkit.torch.conv_networks import ConcatMlp
from rlkit.torch.sac.policies_v2 import TanhGaussianPolicy

import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torch
import rlkit.torch.pytorch_util as ptu
import seaborn as sns

ptu.set_gpu_mode(True)

parser = argparse.ArgumentParser()
parser.add_argument('--traj_num', type=int, default=0)
parser.add_argument('--traj_path', type=str, default='/home/ashvin/ros_ws/evaluation.npy')
parser.add_argument('--path', type=str, default='/home/ashvin/pickle_eval/guassian-policy-pot-off-minq1/guassian_policy_pot_off_minq1_2021_06_05_01_06_04_0000--s-0/model_pkl/500.pt')
parser.add_argument('--action_dim', type=int, default=4)
parser.add_argument('--state_dim', default=720, type=int)
parser.add_argument('--policy', action='store_true')
parser.add_argument('--smimg', action='store_false', default=True)
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

action_dim = args.action_dim
    
from rlkit.misc.asset_loader import load_local_or_remote_file
vqvae_path = '/nfs/kun1/users/asap7772/best_vqvae.pt'
vqvae = load_local_or_remote_file(vqvae_path)

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

parameters = torch.load(args.path)
if policy:
    policy.load_state_dict(parameters['policy_state_dict'])
    policy = policy.to(ptu.device)
else:
    qf1.load_state_dict(parameters['qf1_state_dict'])
    qf1 = qf1.to(ptu.device)

path = args.traj_path
trajs = np.load(path, allow_pickle=True)
traj = trajs[args.traj_num]['cropped_images']
import pandas as pd

def resize_small(img):
    from skimage.transform import resize
    flag = type(img) == torch.Tensor
    if flag:
        img = img.cpu().numpy() 
        img = img.transpose(1,2,0)
    img = resize(img, (48, 48), anti_aliasing=True)
    if flag:
        img = img.transpose(2,0,1)
    img = torch.from_numpy(img)
    img = torch.from_numpy(img.numpy().swapaxes(-2,-1))
    return img

for i in range(len(traj)):
    obs = traj[i]
    state = trajs[args.traj_num]['observations'][i]['state_observation']
    
    plot_img(obs)
    if args.smimg:
        obs = resize_small(obs)
        # obs = torch.from_numpy(obs.numpy().swapaxes(-2,-1))
        plot_img(obs)

    vqvae = vqvae.cpu()
    plot_img(vqvae.decode(vqvae.encode(obs)).squeeze())
    obs = vqvae.encode(obs)

    x = np.linspace(-0.8,0.8)
    y = np.flip(np.linspace(-0.8,0.8))
    actions = torch.from_numpy(np.array(np.meshgrid(x,y)))
    actions = actions.flatten(1).T

    actions_close = torch.cat((actions, torch.zeros_like(actions)), axis=1).float().to(ptu.device)
    actions_open = torch.cat((actions, torch.zeros_like(actions)[:,:1], torch.ones_like(actions)[:, :1]), axis=1).float().to(ptu.device)
    
    obs_tens = obs.flatten(1).repeat(actions.shape[0], 1).to(ptu.device)
    # qf1 = lambda x, y: y.sum(axis=1, keepdim=True)
    
    columns=np.around(x,decimals=2)
    index=np.around(y,decimals=2)
    columns = [str(x) for x in columns]
    index = [str(x) for x in index]

    print(actions)
    plt.plot(actions[:,0],actions[:,1])
    plt.show()

    state = torch.from_numpy(state).float()
    state_tens = state.repeat(actions.shape[0],1).to(ptu.device)
    if args.policy:
        qvals = policy.log_prob(obs_tens, actions_close, extra_fc_input = None, unsumed=True)
    else:
        qvals = qf1(obs_tens, actions_close)

    plt.plot(qvals.detach().cpu().numpy().flatten())
    plt.show()
    qvals = qvals.detach().cpu().numpy().flatten().reshape(50,50)
    df = pd.DataFrame(qvals, columns=columns, index=index)
    ax = sns.heatmap(df)
    plt.show()

    if args.policy:
        qvals = policy.log_prob(obs_tens, actions_open, extra_fc_input = None, unsumed=True)
    else:
        qvals = qf1(obs_tens, actions_open)
    qvals = qvals.detach().cpu().numpy().flatten().reshape(50,50)
    df = pd.DataFrame(qvals, columns=columns, index=index)
    ax = sns.heatmap(df)
    plt.show()

