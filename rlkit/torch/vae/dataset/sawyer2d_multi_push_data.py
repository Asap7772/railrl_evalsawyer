import time

import numpy as np
import os.path as osp

from rlkit.envs.mujoco.sawyer_push_and_reach_env import SawyerMultiPushAndReachEasyEnv
from rlkit.envs.wrappers import ImageMujocoEnv
from rlkit.images.camera import sawyer_init_camera, sawyer_init_camera_zoomed_in
import cv2

from rlkit.exploration_strategies.ou_strategy import OUStrategy

def generate_vae_dataset(N = 10000, test_p = 0.9, use_cached=True, imsize=84, show=False):
    filename = "/tmp/sawyer2d_multi_push_" + str(N) + ".npy"
    info = {}
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        env = SawyerMultiPushAndReachEasyEnv(hide_goal=True)
        env = ImageMujocoEnv(
            env, imsize,
            transpose=True,
            init_camera=sawyer_init_camera_zoomed_in,
            normalize=True,
        )
        info['env'] = env
        policy = OUStrategy(env.action_space)

        dataset = np.zeros((N, imsize*imsize*3))
        for i in range(N):
            # env.reset()
            if i % 100 == 0:
                g = env.sample_goal_for_rollout()
                env.set_goal(g)
                policy.reset()
            u = policy.get_action_from_raw_action(env.action_space.sample())
            img = env.step(u)[0]
            dataset[i, :] = img
            if show:
                # env.render()
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info

if __name__ == "__main__":
    generate_vae_dataset(10000, use_cached=False, show=False)
