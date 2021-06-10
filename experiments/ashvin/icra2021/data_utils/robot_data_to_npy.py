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

prefix = "/media/ashvin/data2/data/ashvin/icra2021/final/"
# output_dir = "/media/ashvin/data2/data/s3doodad/demos/icra2021/release/onpolicy_eval/"
# pickle_format = "**/video_0_env.p"
output_dir = "/media/ashvin/data2/data/s3doodad/demos/icra2021/release/onpolicy_expl/"
pickle_format = "**/video_0_vae.p"

names = {}

num_trajs = 0

files = sorted(glob.glob(prefix + pickle_format, recursive=True))
for i, filename in enumerate(files):
    exp_name = filename.split("/")[8:-3]
    output_name = "-".join(exp_name)

    n = names.setdefault(output_name, 0)
    names[output_name] += 1

    output_name = "%s-%d" % (output_name, n)

    print(i, output_name)

    try:
        p = pickle.load(open(filename, "rb"))
    except:
        print("failed to load", exp_name)
        continue

    hires = "hires_image_observation" in p[0]['observations'][0].keys()

    if hires:
        for traj in p:
            del traj['full_observations']
            del traj['full_next_observations']
        num_trajs += len(p)
        print("# trajectories", num_trajs)

        output_filename = output_dir + output_name + ".npy"

        np.save(output_filename, p, )
