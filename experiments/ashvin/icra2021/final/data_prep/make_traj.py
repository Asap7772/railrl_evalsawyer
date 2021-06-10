import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file
from copy import deepcopy
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize, RandomAffine
import torchvision.transforms.functional as F
import rlkit.torch.pytorch_util as ptu
from torchvision.utils import save_image
import random
import math
import sys
import cv2

AUGMENT = 3
SIZE = 48

crop_prob = 0.95
cond_frac = 0.15 # Use FIRST cond_frac of traj for conditioning
samples_per_traj = 25 # Number of goals sampled per traj
samples_per_trans = 1 # Number of goals samples per trans
fixed_data_only = True # Only keep fixed script trajectories

total_size = 0
total_traj = 0
total_samples = 0

jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
cropper = RandomResizedCrop((SIZE, SIZE), (0.9, 1.0), (0.9, 1.1))

def filter_files(all_files):
    if not fixed_data_only:
        return all_files
    filtered_files = []

    for f in all_files:
        if 'fixed' in f:
            filtered_files.append(f)

    return filtered_files

def aug(x, j, c, do_c, do_f1, do_f2):
    if do_c: x = F.resized_crop(x, c[0], c[1], c[2], c[3], (SIZE, SIZE), Image.ANTIALIAS)
    else: x = F.resize(x, (SIZE, SIZE), Image.ANTIALIAS)
    #if do_f1: x = F.hflip(x) # Don't horizonal flip it might confuse robot
    #if do_f2: x = F.vflip(x)
    x = j(x)
    x = np.array(x) / 255
    img = x.transpose([2, 1, 0]).flatten()
    return img


def filter_keys(dictionary, keep=['latent', 'state']):
    all_keys = list(dictionary.keys())
    for key in all_keys:
        delete = not any([word in key for word in keep])
        if delete: del dictionary[key]

pretrained_vae_path = "/home/ashvin/data/sasha/awac-exps/real-world/vqvae/run4/id0/best_vqvae.pt"
model = load_local_or_remote_file(pretrained_vae_path)
ptu.set_gpu_mode(True)

catagorized_data = {'fixed_pnp': [], 'fixed_tray': [], 'fixed_pot': [], 'fixed_drawer': [], 'general': []}

all_files = glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/dataset_v3/*")
all_files = filter_files(all_files)
random.shuffle(all_files)
for filename in all_files:
    print(filename)
    try:
        data = np.load(filename, allow_pickle=True)
    except:
        print("COULDNT LOAD ABOVE FILE")
        continue

    data_list = None

    # Check if traj is in specific catagory
    for key in catagorized_data.keys():
        if key in filename:
            data_list = catagorized_data[key]

    # Check not, assign to general
    if data_list is None:
        data_list = catagorized_data['general']
    
    for traj_i in range(len(data)):
        for _ in range(AUGMENT):

            # Prepare augmentation
            D = deepcopy(data[traj_i])
            traj = D["observations"]
            img = traj[0]["image_observation"]
            img = img[:, 50:530, ::-1]
            img = Image.fromarray(img, mode='RGB')
            c = cropper.get_params(img, (0.75, 1.0), (0.75, 1.25))
            j = jitter.get_params((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
            do_c = np.random.uniform() < crop_prob
            do_f1 = np.random.uniform() < 0.5
            do_f2 = np.random.uniform() < 0.5


            # Process images 
            for t in range(len(traj)):
                if not traj[t]:
                    print(traj_i, t)
                    continue
                img = traj[t]["image_observation"]
                img = img[:, 50:530, ::-1]
                img = Image.fromarray(img, mode='RGB')
                y = aug(img, j, c, do_c, do_f1, do_f2)
                traj[t]["image_observation"] = y
            

            # Encode images
            num_images = len(traj)
            images = np.stack([traj[i]['image_observation'] for i in range(num_images)])
            latents = model.encode_np(images)


            # Sample goals
            if samples_per_traj > 0:
                cond_timesteps = int(len(traj) * cond_frac)
                num_repeat = math.ceil(samples_per_traj / cond_timesteps)
                goal_context = np.repeat(latents[:cond_timesteps], num_repeat, axis=0)[:samples_per_traj]
                sampled_goals = model.sample_prior(samples_per_traj, cond=goal_context)


            # Add latent observations
            for i in range(num_images):
                if samples_per_traj > 0:
                    traj[i]["presampled_latent_goals"] = sampled_goals[i % samples_per_traj]
                
                traj[i]["initial_latent_state"] = latents[0]
                traj[i]["latent_observation"] = latents[i]
                traj[i]["latent_achieved_goal"] = latents[i]
                traj[i]["latent_desired_goal"] = latents[-1]
                filter_keys(traj[i]) # Delete unnecesary keys


            decoded_samples = model.decode(ptu.from_numpy(sampled_goals))
            decoded_traj = model.decode(ptu.from_numpy(latents))
            save_image(decoded_samples.data.view(-1, 3, 48, 48).transpose(2, 3),"/home/ashvin/data/sample_testing/decoded_samples.png")
            save_image(decoded_traj.data.view(-1, 3, 48, 48).transpose(2, 3),"/home/ashvin/data/sample_testing/decoded_traj.png")
            import pdb; pdb.set_trace()

            # Update
            data_list.append(D)
            total_size += num_images
            total_samples += samples_per_traj
            total_traj += 1

        print("Trajectories:", total_traj)
        print("Datapoints:", total_size)
        print("Samples:", total_samples)

# SAVE TRAJECTORIES FOR REINFORCEMENT LEARNING #
for key in catagorized_data.keys():
    data_list = catagorized_data[key]
    np.save('/media/ashvin/data2/data/val/v2/' + key + '_demos.npy', data_list)
