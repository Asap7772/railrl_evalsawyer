import numpy as np
import pickle

import skvideo.io
import sys

from pathlib import Path
import glob

import json
import torch
from rlkit.torch import pytorch_util as ptu
from torchvision.utils import save_image
import scipy.misc
import scipy.ndimage
import imageio
import cv2

ZOOM = 3

dirname = sys.argv[1] #"/home/ashvin/data/ashvin/icra2021/final/new/pickup-shoe1/run0/id0/video_0_env.p"
if dirname[-1] != "/":
    dirname = dirname + "/"

(Path(dirname) / "media").mkdir(parents=True, exist_ok=True)

# for fname in glob.glob(dirname + "video_*_env.p"):
#     x = pickle.load(open(fname, "rb"))
#     print(fname)

#     name = fname[len(dirname):-2]

#     (Path(dirname) / "media" / name).mkdir(parents=True, exist_ok=True)

#     # # ipdb> x[0]['observations'][0]['hires_image_observation'].shape
#     # # (480, 640, 3)

#     edited_imgs = []
#     imgs = []
#     for i, traj in enumerate(x):
#         (Path(dirname) / "media" / name / str(i)).mkdir(parents=True, exist_ok=True)
#         traj_dir = str(Path(dirname) / "media" / name / str(i))
#         goal_image = traj['observations'][0]['image_desired_goal'].reshape((3, 48, 48))
#         goal_img = np.uint8(255 * goal_image).transpose()
#         imageio.imwrite(traj_dir + "/goal.png", goal_img)

#         zoomed_goal_img = cv2.resize(goal_img,(ZOOM * 48, ZOOM * 48),interpolation=cv2.INTER_NEAREST)

#         for j, obs in enumerate(traj['observations']):
#             img = obs['hires_image_observation'][:, :, ::-1]
#             imgs.append(img)

#             if j % 10 == 0:
#                 imageio.imwrite(traj_dir + "/img%d.png" % j, img)
#                 imageio.imwrite(traj_dir + "/img_square%d.png" % j, img[:, 50:530, :])

#             edited_img = img.copy()
#             edited_img[480-160:, 640-160:, :] = 0 # black border
#             edited_img[328:328+48*ZOOM, 488:488+48*ZOOM, :] = zoomed_goal_img
#             edited_imgs.append(edited_img)

#     imgs = np.array(imgs)
#     print(imgs.shape)
#     skvideo.io.vwrite(dirname + "/media/" + name + ".mp4", imgs)

#     edited_imgs = np.array(edited_imgs)
#     skvideo.io.vwrite(dirname + "/media/" + name + "_with_goal.mp4", edited_imgs)

ptu.set_gpu_mode(True)

variant_fname = dirname + "variant.json"
variant = json.load(open(variant_fname, "r"))
vqvae_path = variant['pretrained_vae_path']
model = torch.load(vqvae_path)

for fname in glob.glob(dirname + "video_*_vae.p"):
    x = pickle.load(open(fname, "rb"))
    print(fname)

    name = fname[len(dirname):-2]

    (Path(dirname) / "media" / name).mkdir(parents=True, exist_ok=True)

    edited_imgs = []
    imgs = []
    for i, traj in enumerate(x):
        (Path(dirname) / "media" / name / str(i)).mkdir(parents=True, exist_ok=True)
        traj_dir = str(Path(dirname) / "media" / name / str(i))
        obs = traj['observations'][0]
        latent_goal = obs['latent_desired_goal']
        goal_image = model.decode_one_np(latent_goal)
        goal_img = np.uint8(255 * goal_image).transpose()
        imageio.imwrite(traj_dir + "/goal_%d.png" % i, goal_img)

        zoomed_goal_img = cv2.resize(goal_img,(ZOOM * 48, ZOOM * 48),interpolation=cv2.INTER_NEAREST)

        for j, obs in enumerate(traj['observations']):
            img = obs['hires_image_observation'][:, :, ::-1]
            imgs.append(img)

            if j % 10 == 0:
                imageio.imwrite(traj_dir + "/img%d.png" % j, img)
                imageio.imwrite(traj_dir + "/img_square%d.png" % j, img[:, 50:530, :])

            edited_img = img.copy()
            edited_img[480-160:, 640-160:, :] = 0 # black border
            edited_img[328:328+48*ZOOM, 488:488+48*ZOOM, :] = zoomed_goal_img
            edited_imgs.append(edited_img)

    imgs = np.array(imgs)
    print(imgs.shape)
    skvideo.io.vwrite(dirname + "/media/" + name + ".mp4", imgs)

    edited_imgs = np.array(edited_imgs)
    skvideo.io.vwrite(dirname + "/media/" + name + "_with_goal.mp4", edited_imgs)
