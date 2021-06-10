import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import rlkit.torch.pytorch_util as ptu

def crop(img):
    img = resize(img[:, 50:530, ::-1], (48, 48), anti_aliasing=True) #  * 255
    # img = img.astype(np.uint8)
    img = img.transpose([2, 1, 0]).flatten()

    z = img.reshape(3, 48, 48).transpose()[:, :, ::-1]
    cv2.imshow('x_t', z)
    cv2.waitKey(100)
    return img

# pretrained_vae_path="/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp-augment1/run3/id0/itr_1500.pt"
pretrained_vae_path="/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp-augment1/run9/id0/best_vqvae.pt"
model = load_local_or_remote_file(pretrained_vae_path)
ptu.set_gpu_mode(True)

x = []
x0 = []
    
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_yellow_goals.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj2.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_giraffe_grasp*.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_dice_grasp*.npy"):
for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/grasp/obj_drill_grasp*.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_drawerclose_*.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_draweropen*.npy"):
    print(filename)
    data = np.load(filename, allow_pickle=True)


    for traj_i in range(len(data)):
        traj = data[traj_i]["observations"]
        print(traj_i, len(traj))
        img0 = crop(traj[0]["image_observation"])
        # for t in range(len(traj)):
        for t in [len(traj)-1]:
            # print("frame", t)
            if not traj[t]:
                print(traj_i, t)
                continue

            img = crop(traj[t]["image_observation"])
            x.append(img)
            x0.append(img0)

x = np.array(x)
x0 = np.array(x0)

goals = {'image_desired_goal': x, 'initial_image_observation': x0}
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals_obj2.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals_grasp.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals_giraffe_grasp_final.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals_dice_grasp_final.npy"
new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals_drill_grasp_final.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals_drawerclose_final.npy"
# new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/datasets/goals_draweropen_final.npy"
np.save(new_filename, goals)

z = model.encode_np(np.array(x))
reconstructions = model.decode_np(z)
for i in range(len(reconstructions)):
    img = reconstructions[i, ::-1, :, :].transpose()
    cv2.imshow('x_t', img)
    cv2.waitKey(20)