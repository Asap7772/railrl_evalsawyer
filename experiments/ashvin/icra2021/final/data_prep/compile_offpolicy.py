import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file
from copy import deepcopy

from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
import torchvision.transforms.functional as F

import rlkit.torch.pytorch_util as ptu

import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# x = np.load("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_horseindrawer.npy", allow_pickle=True)
AUGMENT = 10
SIZE = 48

jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
cropper = RandomResizedCrop((SIZE, SIZE), (0.9, 1.0), (0.9, 1.1))

def crop(x, c, j):
    x1 = F.resized_crop(x, c[0], c[1], c[2], c[3], (SIZE, SIZE), Image.ANTIALIAS)
    x1 = np.array(j(x1)) / 255
    img1 = x1.transpose([2, 1, 0]).flatten()

    # z = img1.reshape(3, 48, 48).transpose()[:, :, ::-1]
    # cv2.imshow('x_t', z)
    # cv2.waitKey(20)
    return img1

pretrained_vae_path="/home/ashvin/data/sasha/awac-exps/real-world/vqvae/run2/id0/best_vqvae.pt"
# pretrained_vae_path="/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp-augment1/run9/id0/best_vqvae.pt"
model = load_local_or_remote_file(pretrained_vae_path)
ptu.set_gpu_mode(True)

all_data = []

# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj*.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_*_grasp*.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_*.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_drawerclose_*.npy"):
for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/dataset_v3/obj_close_drawer6.npy"):
    print(filename)
    data = np.load(filename, allow_pickle=True)

    for traj_i in range(len(data)):
        for _ in range(AUGMENT):
            D = deepcopy(data[traj_i])
            traj = D["observations"]
            print(traj_i, len(traj))

            img = traj[0]["image_observation"]
            img = img[:, 50:530, ::-1]
            img = Image.fromarray(img, mode='RGB')

            c = cropper.get_params(img, (0.9, 1.0), (0.9, 1.1))
            j = jitter.get_params((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))

            for t in range(len(traj)):
                # print("frame", t)
                if not traj[t]:
                    print(traj_i, t)
                    continue

                img = traj[t]["image_observation"]
                img = img[:, 50:530, ::-1]
                img = Image.fromarray(img, mode='RGB')

                y = crop(img, c, j)
                traj[t]["image_observation"] = y

            num_images = len(traj)
            images = np.stack([traj[i]['image_observation'] for i in range(num_images)])
            latents = model.encode_np(images)
            for i in range(num_images):
                traj[i]["initial_latent_state"] = latents[0]
                traj[i]["latent_observation"] = latents[i]
                traj[i]["latent_achieved_goal"] = latents[i]
                traj[i]["latent_desired_goal"] = latents[-1]
                del traj[i]['image_observation']

                latent_distance = -np.linalg.norm(latents[i] - latents[-1])
                print(latent_distance)

            # reconstructions = model.decode_np(latents)
            # for i in range(len(reconstructions)):
            #     img = reconstructions[i, ::-1, :, :].transpose()
            #     cv2.imshow('x_t', img)
            #     cv2.waitKey(20)
            # import ipdb; ipdb.set_trace()
        
            all_data.append(D)
        print("trajectories:", len(all_data))

# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_data.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_grasp_augment_data.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_mixed_v1.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_drawerclose_v1.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_draweropen_v1.npy"
new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/goals_v3/offpolicy_close1.npy"
np.save(new_filename, all_data)