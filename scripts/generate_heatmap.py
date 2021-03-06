import ipdb
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

ptu.set_gpu_mode(True)

parser = argparse.ArgumentParser()
parser.add_argument('--traj_num', type=int, default=0)
parser.add_argument('--traj_path', type=str, default='/home/ashvin/ros_ws/evaluation.npy')
parser.add_argument('--path', type=str, default='/home/ashvin/pickle_eval/guassian-policy-pot-off-minq1/guassian_policy_pot_off_minq1_2021_06_05_01_06_04_0000--s-0/model_pkl/500.pt')
parser.add_argument('--action_dim', type=int, default=4)
parser.add_argument('--state_dim', default=3, type=int)
parser.add_argument('--policy', action='store_true')
parser.add_argument('--deeper_net', action='store_true')
parser.add_argument('--vqvae_enc', action='store_true')
parser.add_argument('--imgstate', action='store_true')
parser.add_argument('--smimg', action='store_true')
parser.add_argument('--bottleneck', action='store_true')
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
    
cnn_params=dict(
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
        added_fc_input_size=args.state_dim if args.imgstate else 0,
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
        qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=16,deterministic=False, width=64, height=64)
    else:
        qf1 = ConcatCNN(**cnn_params)

parameters = torch.load(args.path)
# import ipdb; ipdb.set_trace()
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
        obs = torch.from_numpy(obs.numpy().swapaxes(-2,-1))
    plot_img(obs)

    x = np.linspace(-0.8,0.8)
    y = np.flip(np.linspace(-0.8,0.8))
    actions = torch.from_numpy(np.array(np.meshgrid(x,y)))
    actions = actions.flatten(1).T

    actions_close = torch.cat((actions, torch.zeros_like(actions)), axis=1).float().to(ptu.device)
    actions_open = torch.cat((actions, torch.zeros_like(actions)[:,:1], torch.ones_like(actions)[:, :1]), axis=1).float().to(ptu.device)
    
    obs_tens = obs.repeat(actions.shape[0], 1, 1, 1).flatten(1).to(ptu.device)
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
        qvals = policy.log_prob(obs_tens, actions_close, extra_fc_input = state_tens if args.imgstate else None)
    else:
        qvals = qf1(obs_tens, actions_close)

    plt.plot(qvals.detach().cpu().numpy().flatten())
    plt.show()
    qvals = qvals.detach().cpu().numpy().flatten().reshape(50,50)
    df = pd.DataFrame(qvals, columns=columns, index=index)
    ax = sns.heatmap(df)
    plt.show()

    if args.policy:
        qvals = policy.log_prob(obs_tens, actions_open,extra_fc_input = state_tens if args.imgstate else None)
    else:
        qvals = qf1(obs_tens, actions_open)
    qvals = qvals.detach().cpu().numpy().flatten().reshape(50,50)
    df = pd.DataFrame(qvals, columns=columns, index=index)
    ax = sns.heatmap(df)
    plt.show()

 