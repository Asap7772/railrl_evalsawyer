import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

def crop(img):
    img = resize(img[:, 50:530, ::-1], (48, 48), anti_aliasing=True) * 255
    img = img.astype(np.uint8)
    return img.transpose([2, 1, 0]).flatten()
    # return img[::10, 50:530:10, ::-1].transpose([2, 1, 0]).flatten()
    # return img.reshape((3, 64, 64))[::-1, 8:56, 8:56].transpose([0, 2, 1]).flatten()
    # return img.reshape((3, 64, 64))[:, 8:56, 8:56].flatten()

pretrained_vae_path="/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp/run0/id0/itr_1500.pt"
# load_local_or_remote_file(pretrained_vae_path)

all_data = []

# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj*.npy"):
for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_*_grasp*.npy"):
    print(filename)
    data = np.load(filename, allow_pickle=True)

    for traj_i in range(len(data)):
        traj = data[traj_i]["observations"]
        print(traj_i, len(traj))
        for t in range(len(traj)):
            # print("frame", t)
            if not traj[t]:
                print(traj_i, t)
                continue

            img = crop(traj[t]["image_observation"])
            traj[t]["image_observation"] = img
    
    all_data.extend(data)
    print("trajectories:", len(all_data))

# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_data.npy"
new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/offpolicy_grasp_data.npy"
np.save(new_filename, all_data)