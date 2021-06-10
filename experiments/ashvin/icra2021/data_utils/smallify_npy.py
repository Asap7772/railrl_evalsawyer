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

import copy
from skimage.transform import rescale, resize, downscale_local_mean

def crop(img):
    img = resize(img[:, 50:530, ::-1], (48, 48), anti_aliasing=True) * 255
    return img.astype(np.uint8).transpose([2, 1, 0]).flatten()

prefix = "/media/ashvin/data2/data/s3doodad/demos/icra2021/release/"
# output_dir = "/media/ashvin/data2/data/s3doodad/demos/icra2021/release/onpolicy_eval/"
# pickle_format = "**/video_0_env.p"
output_dir = "/media/ashvin/data2/data/s3doodad/demos/icra2021/release_rl/"
pickle_format = "**/*.npy"

names = {}

num_trajs = 0

files = sorted(glob.glob(prefix + pickle_format, recursive=True))
print(len(files), "files")
for i, filename in enumerate(files):
    exp_name = filename.split("/")[9:]
    output_name = "/".join(exp_name)
    print(i, output_name)

    try:
        p = np.load(filename, allow_pickle=True)
    except:
        print("failed to load", exp_name)
        continue

    for traj in p:
        for k in ["observations", "next_observations"]:
            for t, obs in enumerate(traj[k]):
                if 'hires_image_observation' in obs:
                    del obs['hires_image_observation']
                if obs['image_observation'].shape == (480, 640, 3):
                    obs['image_observation'] = crop(obs['image_observation'])
                if obs['image_observation'].dtype is not np.uint8:
                    obs['image_observation'] = (obs['image_observation'] * 255).astype(np.uint8)

    num_trajs += len(p)
    print("num_trajs", num_trajs)

    output_filename = output_dir + output_name
    np.save(output_filename, p, )
