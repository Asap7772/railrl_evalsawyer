import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file

import rlkit.torch.pytorch_util as ptu
from torch.distributions import kl_divergence
import rlkit.pythonplusplus as ppp
ptu.set_gpu_mode(True)

from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
import torchvision.transforms.functional as F
from copy import deepcopy
from torchvision.utils import save_image

# def crop(img):
#     import cv2
#     img = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
#     Cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = Cimg.transpose(2,1,0) / 256 # .reshape(1, -1)
#     return img
AUGMENT = 10
SIZE = 48

jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
cropper = RandomResizedCrop((SIZE, SIZE), (0.9, 1.0), (0.9, 1.1))

def crop(x, c, j):
    x = x[:, 50:530, ::-1]
    # img = img[:, :, ::-1]
    x = Image.fromarray(x, mode='RGB')
    x1 = F.resized_crop(x, c[0], c[1], c[2], c[3], (SIZE, SIZE), Image.ANTIALIAS)
    x1 = np.array(j(x1)) / 255
    # img1 = x1.transpose([2, 1, 0]) # .flatten()
    img1 = x1.transpose([2, 0, 1]) # .flatten()

    # z = img1.reshape(3, 48, 48).transpose()[:, :, ::-1]
    # cv2.imshow('x_t', z)
    # cv2.waitKey(20)
    return img1

# pretrained_model_path="/home/ashvin/ros_ws/src/sawyer_control/src/raw_vae.pt"
# pretrained_model_path="/home/ashvin/ros_ws/src/sawyer_control/src/raw_vae_aug.pt"
# pretrained_model_path="/home/ashvin/ros_ws/src/sawyer_control/src/raw_vae_data_aug_3.pt"
pretrained_model_path="/home/ashvin/raw_non_set_vae_for_ashvin2.pt" # encoder
model = load_local_or_remote_file(pretrained_model_path)
model.to(ptu.device)
model.eval()

# REWARD CODE
base_path = '/home/ashvin/ros_ws/src/sawyer_control/src/'
# vae = model
pretrained_vae_path="/home/ashvin/raw_vae_for_ashvin.pt" # encoder
# pretrained_vae_path="/home/ashvin/ros_ws/src/sawyer_control/src/raw_vae_data_aug_3.pt"
vae = load_local_or_remote_file(pretrained_vae_path)
vae.to(ptu.device)
vae.eval()
vae.to(ptu.device)
train_sets = [ptu.from_numpy(t) for t in np.load(base_path + 'train_sets.npy')]
eval_sets = [ptu.from_numpy(t) for t in np.load(base_path + 'eval_sets.npy')]
set_images = train_sets[0]  # 0 = closed door, 1 = open door
prior_c = vae.encoder_c(set_images)
c = prior_c.mean
prior = vae.prior_z_given_c(c)
print(prior_c.mean)
print(prior.mean)

# import ipdb; ipdb.set_trace()

set_images = train_sets[0]  # 0 = closed door, 1 = open door
prior_c = vae.encoder_c(set_images)
c = prior_c.mean
prior = vae.prior_z_given_c(c)
reward_fn = lambda q_z: -kl_divergence(q_z, prior)
x = eval_sets[0]
q_z = vae.q_zs_given_independent_xs(x)
reward = ptu.get_numpy(reward_fn(q_z))



all_data = []

# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj*.npy"):
for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/dataset_v3/obj_close_drawer*.npy"):
    print(filename)
    data = np.load(filename, allow_pickle=True)

    for traj_i in range(len(data)):
        for _ in range(AUGMENT):
            D = deepcopy(data[traj_i])
            traj = D["observations"]
            # traj = data[traj_i]["observations"]
            print(traj_i, len(traj))

            img = traj[0]["image_observation"]
            img = Image.fromarray(img, mode='RGB')

            c = cropper.get_params(img, (0.9, 1.0), (0.9, 1.1))
            j = jitter.get_params((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
            for t in range(len(traj)):
                # print("frame", t)
                if not traj[t]:
                    print("skipping", traj_i, t)
                    continue

                img = traj[t]["image_observation"]
                img = crop(img, c, j)
                traj[t]["image_observation"] = img

            images = np.stack([traj[i]['image_observation'] for i in range(len(traj))])
            images_pt = ptu.from_numpy(images)
            latents = model.encode_np(images)

            q_z = vae.q_zs_given_independent_xs(images_pt)
            reward = ptu.get_numpy(reward_fn(q_z))
            print(reward)
            # latents = ptu.get_numpy(q_z.mean)

            # save_image(images_pt[:, :, :, :], "compile_test.png")
            # import ipdb; ipdb.set_trace()

            for i in range(len(traj)):
                traj[i]["initial_latent_state"] = latents[0]
                traj[i]["latent_observation"] = latents[i]
                traj[i]["latent_achieved_goal"] = np.zeros((0, ))
                traj[i]["latent_desired_goal"] = np.zeros((0, ))
                del traj[i]['image_observation']
                del traj[i]['hires_image_observation']
                D['rewards'][i] = reward[i]
            del D['next_observations']
    
            all_data.append(D)
        print("trajectories:", len(all_data))

# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_data.npy"
new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/disco_close_drawer_aug2.npy"
np.save(new_filename, all_data)