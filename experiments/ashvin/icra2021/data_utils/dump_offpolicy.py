import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import pickle

import skvideo.io
import sys

from pathlib import Path

import json
import torch
from rlkit.torch import pytorch_util as ptu
from torchvision.utils import save_image
import scipy.misc
import scipy.ndimage
import imageio
import cv2

INPUT_DIR = "/media/ashvin/data2/data/s3doodad/demos/icra2021/dataset_v3/"
OUTPUT_DIR = "/media/ashvin/data2/data/s3doodad/demos/icra2021/dataset_v3_imgs/"

# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj*.npy"):
for filename in glob.glob(INPUT_DIR + "obj_*.npy"):
    print(filename)
    name = filename[len(INPUT_DIR):-4]
    (Path(OUTPUT_DIR) / name).mkdir(parents=True, exist_ok=True)

    data = np.load(filename, allow_pickle=True)

    for traj_i in range(len(data)):
        traj_dir = "%s/%s/%d" % (OUTPUT_DIR, name, traj_i)
        Path(traj_dir).mkdir(parents=True, exist_ok=True)
        traj = data[traj_i]["observations"]
        print(traj_i, len(traj))
        for t in range(0, len(traj), 10):
            # print("frame", t)
            if not traj[t]:
                print(traj_i, t)
                continue

            img = traj[t]["image_observation"]
            imageio.imwrite(traj_dir + "/img%d.png" % t, img[:, :, ::-1])
            imageio.imwrite(traj_dir + "/img_square%d.png" % t, img[:, 50:530, ::-1])
