import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file
from copy import deepcopy
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize, RandomAffine
import torchvision.transforms.functional as F
import rlkit.torch.pytorch_util as ptu
import pickle as pkl
import random
import sys
import cv2
import torch

path_func = lambda name: '/media/ashvin/data2/data/pot_data/' + name

# TASKS: PNP, TRAY, OPEN POT, CLOSE POT, OPEN DRAWER, CLOSE DRAWER

# MAKE THIS THE SAME AS THE ENV: split, resize
SIZE = 48
def crop(x):
    x1 = F.resize(x, (SIZE, SIZE), Image.ANTIALIAS)
    x1 = np.array(x1)
    x1 = x1 / 255
    img1 = x1.transpose([2, 1, 0]).flatten()
    return img1
#pretrained_vae_path = "/home/ashvin/data/real_world_val/best_vqvae.pt"
#pretrained_vae_path = "/home/ashvin/data/sasha/awac-exps/real-world/vqvae/run6/id0/best_vqvae.pt"
# pretrained_vae_path = "/home/ashvin/data/sasha/awac-exps/real-world/vqvae/run5/id0/best_vqvae.pt"
# pretrained_vae_path = path_func("best_vqvae.pt")
pretrained_vae_path = '/media/ashvin/data2/data/baseline/vae.pt'
model = load_local_or_remote_file(pretrained_vae_path)
ptu.set_gpu_mode(True)

DEMO_NAME = 'obj_ccvae_pot1' # 'obj_fixed_drawer_distractor10'
SAVE_NAME = 'ccvae_pot1'
data = np.load('/home/ashvin/data/s3doodad/demos/icra2021/dataset_v3/' + DEMO_NAME + '.npy', allow_pickle=True)
cond_frac = 0.3 # Use FIRST cond_frac of traj for conditioning
samples_per_img = 20 # Number of goals sampled per image

# NOTE: This should depend on task!
goal_frac = 0.1 # Use LAST goal_frac of traj for goals

expl_goals = []
eval_goals = {'image_desired_goal': [], 'initial_image_observation': [], 'state_desired_goal': []}
expl_size, eval_size = 0, 0
    
for traj_i in range(len(data)):
    print('Num Exploration Goals:', expl_size)
    print('Num Evaluation Goals:', eval_size)

    D = deepcopy(data[traj_i])
    traj = D["observations"]

    # Prepare Images
    for t in range(len(traj)):
        if not traj[t]:
            print(traj_i, t)
            continue
        img = traj[t]["image_observation"]
        img = img[:, 50:530, ::-1]
        img = Image.fromarray(img, mode='RGB')
        y = crop(img)
        traj[t]["image_observation"] = y
        
    num_images = len(traj)
    images = np.stack([traj[i]['image_observation'] for i in range(num_images)])
    x0 = np.tile(images[0, :], (len(images), 1))
    latents = model.encode_np(images, x0)

    # Add Expl Goals
    if samples_per_img > 0:
        cond_timesteps = int(len(traj) * cond_frac)
        expl_size += (samples_per_img * cond_timesteps)
        for t in range(cond_timesteps):
            sampled_z = model.sample_prior(samples_per_img, cond=latents[t].reshape(1,-1))
            expl_goals.append(sampled_z)

    # Add Eval Goals
    start_ind = int((1 - goal_frac) * len(traj))
    eval_size += (len(traj) - start_ind)
    for t in range(start_ind, len(traj)):
        eval_goals['image_desired_goal'].append(traj[t]["image_observation"])
        eval_goals['initial_image_observation'].append(traj[0]["image_observation"])
        eval_goals['state_desired_goal'].append(traj[t]["state_achieved_goal"])


# Save Expl Goals
if samples_per_img > 0:
    save_file = '/media/ashvin/data2/data/val/v1/' + SAVE_NAME + '_expl_goals.npy'
    expl_goals = np.concatenate(expl_goals, axis=0)
    np.save(save_file, expl_goals)
    print("Num Exploration Goals:", expl_goals.shape[0])


# Save Eval Goals
save_file = '/media/ashvin/data2/data/val/v1/' + SAVE_NAME + '_eval_goals.pkl'
for key in eval_goals.keys():
    eval_goals[key] = np.stack(eval_goals[key])

file = open(save_file, 'wb')
pkl.dump(eval_goals, file)
file.close()
