import numpy as np
import pickle

import skvideo.io
import sys

fname = sys.argv[1] #"/home/ashvin/data/ashvin/icra2021/final/new/pickup-shoe1/run0/id0/video_0_env.p"

x = pickle.load(open(fname, "rb"))

# ipdb> x[0]['observations'][0]['hires_image_observation'].shape
# (480, 640, 3)

imgs = []
for traj in x:
	for obs in traj['observations']:
		img = obs['hires_image_observation'][:, :, ::-1]
		imgs.append(img)

imgs = np.array(imgs)
print(imgs.shape)

skvideo.io.vwrite(fname + ".mp4", imgs)
