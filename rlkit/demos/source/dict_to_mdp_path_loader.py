from collections import OrderedDict
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import copy
import rlkit.torch.pytorch_util as ptu
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from rlkit.misc.asset_loader import (
    load_local_or_remote_file, sync_down_folder, get_absolute_path, sync_down
)

import random
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.data_management.path_builder import PathBuilder

from rlkit.launchers.config import LOCAL_LOG_DIR, AWS_S3_PATH

from rlkit.core import logger

import glob


class DictToMDPPathLoader:
    """
    Path loader for that loads obs-dict demonstrations
    into a Trainer with EnvReplayBuffer
    """

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_paths=None, # list of dicts
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,

            **kwargs
    ):
        self.trainer = trainer

        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
        self.demo_data_split = demo_data_split
        self.replay_buffer = replay_buffer
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer

        self.demo_paths = [] if demo_paths is None else demo_paths

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.demo_trajectory_rewards = []

        self.env_info_key = env_info_key
        self.obs_key = obs_key
        self.recompute_reward = recompute_reward
        self.load_terminals = load_terminals

        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.demo_train_buffer = self.demo_train_buffer
        self.trainer.demo_test_buffer = self.demo_test_buffer

    def load_path(self, path, replay_buffer, obs_dict=None):
        rewards = []
        path_builder = PathBuilder()

        print("loading path, length", len(path["observations"]), len(path["actions"]))
        H = min(len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))

        for i in range(H):
            if obs_dict:
                ob = path["observations"][i][self.obs_key]
                next_ob = path["next_observations"][i][self.obs_key]
            else:
                ob = path["observations"][i]
                next_ob = path["next_observations"][i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            terminal = path["terminals"][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]

            if self.recompute_reward:
                reward = self.env.compute_reward(
                    action,
                    next_ob,
                )

            reward = np.array([reward]).flatten()
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1, ))
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print("path sum rewards", sum(rewards), len(rewards))

    def load_demos(self):
        # Off policy
        for demo_path in self.demo_paths:
            self.load_demo_path(**demo_path)

    # Parameterize which demo is being tested (and all jitter variants)
    # If is_demo is False, we only add the demos to the
    # replay buffer, and not to the demo_test or demo_train buffers
    def load_demo_path(self, path, is_demo, obs_dict, train_split=None, data_split=None, sync_dir=None):
        print("loading off-policy path", path)

        if sync_dir is not None:
            sync_down_folder(sync_dir)
            paths = glob.glob(get_absolute_path(path))
        else:
            paths = [path]

        data = []

        for filename in paths:
            data.extend(list(load_local_or_remote_file(filename)))

        # if not is_demo:
            # data = [data]
        print('WARNING!!!!')
        print('SHUFFLING DEMOS BEFORE SPLITTING!!!!')
        random.shuffle(data)

        if train_split is None:
            train_split = self.demo_train_split

        if data_split is None:
            data_split = self.demo_data_split

        M = int(len(data) * train_split * data_split) // 10 * 10
        N = int(len(data) * data_split)
        print("using", M, "paths for training")
        print("using", N, "paths for training+testing")

        if self.add_demos_to_replay_buffer:
            for path in data[:M]:
                self.load_path(path, self.replay_buffer, obs_dict=obs_dict)

        if is_demo:
            for path in data[:M]:
                self.load_path(path, self.demo_train_buffer, obs_dict=obs_dict)
            for path in data[M:N]:
                self.load_path(path, self.demo_test_buffer, obs_dict=obs_dict)

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        # obs = batch['observations']
        # next_obs = batch['next_observations']
        # goals = batch['resampled_goals']
        # import ipdb; ipdb.set_trace()
        # batch['observations'] = torch.cat((
        #     obs,
        #     goals
        # ), dim=1)
        # batch['next_observations'] = torch.cat((
        #     next_obs,
        #     goals
        # ), dim=1)
        return batch

class EncoderDictToMDPPathLoader(DictToMDPPathLoader):

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            model=None,
            model_path=None,
            reward_fn=None,
            env=None,
            demo_paths=[], # list of dicts
            normalize=False,
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            condition_encoding=False,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            object_list=None,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,
            **kwargs
    ):
        super().__init__(trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_paths,
            demo_train_split,
            demo_data_split,
            add_demos_to_replay_buffer,
            bc_num_pretrain_steps,
            bc_batch_size,
            bc_weight,
            rl_weight,
            q_num_pretrain_steps,
            weight_decay,
            eval_policy,
            recompute_reward,
            env_info_key,
            obs_key,
            load_terminals,
            **kwargs)
       
        if model is None:
            self.model = load_local_or_remote_file(model_path)
        else:
            self.model = model
        self.condition_encoding = condition_encoding
        self.reward_fn = reward_fn
        self.normalize = normalize
        self.object_list = object_list
        self.env = env

        print("Using temp preprocessing fix")

    def preprocess(self, observation, is_next_obs=False):
        num_images = len(observation) - 1
        observation = copy.deepcopy(observation)
        if is_next_obs:
            images = np.stack([observation[i]['image_observation'] for i in range(1, len(observation))])
        else:
            images = np.stack([observation[i]['image_observation'] for i in range(len(observation) - 1)])

        if self.normalize:
            images = images / 255.0

        if self.condition_encoding:
            cond = images[0].repeat(num_images, axis=0)
            latents = self.model.encode_np(images, cond)
        else:
            latents = self.model.encode_np(images)

        for i in range(num_images):
            observation[i]["initial_latent_state"] = latents[0]
            observation[i]["latent_observation"] = latents[i]
            observation[i]["latent_achieved_goal"] = latents[i]
            observation[i]["latent_desired_goal"] = latents[-1]
            del observation[i]['image_observation']

        return observation

    # def preprocess(self, observation):
    #     observation = copy.deepcopy(observation)
    #     images = np.stack([observation[i]['image_observation'] for i in range(len(observation))])
    #     #goals = np.stack([np.zeros_like(observation[i]['image_observation']) for i in range(len(observation))])

    #     if self.normalize:
    #         images = images / 255.0

    #     if self.condition_encoding:
    #         cond = images[0].repeat(len(observation), axis=0)
    #         latents = self.model.encode_np(images, cond)
    #     else:
    #         latents = self.model.encode_np(images)
    #         #latents = ptu.get_numpy(self.model.encode(ptu.from_numpy(images)))
        
    #     #goals = ptu.get_numpy(self.model.encode(ptu.from_numpy(goals)))

    #     for i in range(len(observation)):
    #         observation[i]["initial_latent_state"] = latents[0]
    #         observation[i]["latent_observation"] = latents[i]
    #         observation[i]["latent_achieved_goal"] = latents[i]
    #         observation[i]["latent_desired_goal"] = latents[-1]
    #         #observation[i]["latent_desired_goal"] = goals[-1]
    #         del observation[i]['image_observation']

    #     return observation

    def preprocess_array_obs(self, observation):
        new_observations = []
        for i in range(len(observation)):
            new_observations.append(dict(observation=observation[i]))
            # observation[i]["no_goal"] = np.zeros((0, ))
        return new_observations

    def encode(self, obs):
        if self.normalize:
            return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs) / 255.0))
        return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs)))


    def load_path(self, path, replay_buffer, obs_dict=None):
        # Ignore objects not in set
        # filter_objects = self.object_list is not None
        # if filter_objects and (path["object_name"] not in self.object_list):
        #     return

        rewards = []
        path_builder = PathBuilder()


        # H = min(len(path["observations"]), len(path["actions"]))
        # if obs_dict:
        #     traj_obs = self.preprocess(path["observations"])
        #     next_traj_obs = self.preprocess(path["next_observations"])
        H = min(len(path["observations"]), len(path["actions"])) - 1
        print("valid steps", H)
        if H <= 0:
            return

        if obs_dict:
            # traj_obs = self.preprocess(path["observations"], is_next_obs=False)
            # next_traj_obs = self.preprocess(path["observations"], is_next_obs=True)
            traj_obs = path["observations"][:H]
            next_traj_obs = path["observations"][1:H+1]
        else:
            traj_obs = self.preprocess_array_obs(path["observations"])
            next_traj_obs = self.preprocess_array_obs(path["next_observations"])

        for i in range(H):
            ob = traj_obs[i]
            next_ob = next_traj_obs[i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            terminal = path["terminals"][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]
            if self.recompute_reward:
                reward = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1,))
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print("rewards", np.min(rewards), np.max(rewards))
        print("loading path, length", len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))
        print("path sum rewards", sum(rewards), len(rewards))
