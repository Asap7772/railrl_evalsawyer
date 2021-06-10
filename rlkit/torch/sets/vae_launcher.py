from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch

import rlkit.torch.sets.rewards
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.launchers.contextual.util import get_gym_env
from torch.utils import data

from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.core import logger
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.envs.pygame import pnp_util
from rlkit.misc import ml_util
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.sets import debug
from rlkit.torch.sets import set_creation
from rlkit.torch.sets.set_vae_trainer import (
    SetVAETrainer, PriorModel,
    CustomDictLoader,
)
from rlkit.torch.sets.batch_algorithm import (
    DictLoader,
    BatchTorchAlgorithm,
)
from rlkit.torch.sets import models
from rlkit.torch.vae.vae_torch_trainer import VAE
import rlkit.pythonplusplus as ppp


def generate_images(
        env,
        env_renderer,
        num_images=32,
        set=None,
):
    for state in pnp_util.generate_goals(env, num_images):
        if set:
            state = set.description.project(state)
        env._set_positions(state)
        img = env_renderer(env)
        yield img


def create_pygame_env(num_objects):
    return PickAndPlaceEnv(
        # Environment dynamics
        action_scale=1.0,
        ball_radius=0.75,  # 1.
        boundary_dist=4,
        object_radius=0.50,
        min_grab_distance=0.5,
        walls=None,
        # Rewards
        action_l2norm_penalty=0,
        reward_type="dense",  # dense_l1
        success_threshold=0.60,
        # Reset settings
        fixed_goal=None,
        # Visualization settings
        images_are_rgb=True,
        render_dt_msec=0,
        render_onscreen=False,
        render_size=84,
        show_goal=False,
        # get_image_base_render_size=(48, 48),
        # Goal sampling
        goal_samplers=None,
        goal_sampling_mode='random',
        num_presampled_goals=10000,
        object_reward_only=True,
        init_position_strategy='random',
        num_objects=num_objects,
    )


def create_pybullet_env(num_objects):
    from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC
    env = SawyerLiftEnvGC(
        action_scale=.06,
        action_repeat=10, #5
        timestep=1./120, #1./240
        solver_iterations=500, #150
        max_force=1000,
        gui=False,
        num_obj=num_objects,
        pos_init=[.75, -.3, 0],
        pos_high=[.75, .4, .3],
        pos_low=[.75, -.4, -.36],
        reset_obj_in_hand_rate=0.0,
        goal_sampling_mode='ground',
        random_init_bowl_pos=False,
        sliding_bowl=False,
        heavy_bowl=False,
        bowl_bounds=[-0.40, 0.40],
        reward_type='obj_dist',
        use_rotated_gripper=True,  # False
        use_wide_gripper=True,  # False
        soft_clip=True,
        obj_urdf='spam',
        max_joint_velocity=None,
    )
    env.num_objects = num_objects
    return env


def create_env(version='pygame', num_objects=4):
    if version == 'pygame':
        return create_pygame_env(num_objects=num_objects)
    elif version == 'pybullet':
        return create_pybullet_env(num_objects=num_objects)
    else:
        raise NotImplementedError()


def save_images(images):
    from moviepy import editor as mpy
    def create_video(imgs):
        imgs = np.array(imgs).transpose([0, 2, 3, 1])
        imgs = (255 * imgs).astype(np.uint8)
        return mpy.ImageSequenceClip(list(imgs), fps=5)

    def concatenate_imgs_into_video(images_list):
        subclips = [create_video(imgs) for imgs in images_list]
        together = mpy.clips_array([subclips])
        together.write_videofile('/home/vitchyr/tmp.mp4')

    concatenate_imgs_into_video(images)


def infinite(iterator):
    while True:
        for x in iterator:
            yield x


def load_path_or_paths(p):
    if isinstance(p, str):
        new_d = np.load(p, allow_pickle=True).item()
    else:
        dicts = [
            np.load(path, allow_pickle=True).item()
            for path in p
        ]
        keys = dicts[0].keys()
        new_d = {}
        for k in keys:
            new_d[k] = np.concatenate(
                [
                    d[k] for d in dicts
                ],
                axis=0,
            )
    return new_d


def normalize_if_needed(img):
    if img.dtype == np.uint8:
        img = img / 255.
    return img


def load_images_and_sets(
        ungrouped_images_dataset_path=None,
        train_set_images_dataset_paths=None,
        eval_set_images_dataset_paths=None,
        image_key='image_desired_goal'
) -> Tuple[np.ndarray, List, List, dict, dict]:
    """
    Each path must be a numpy file containing a dictionary:

    {
        image_key: list or numpy array of images, shape [3, H, W]
    }
    :param ungrouped_images_dataset_path:
    :param train_set_images_dataset_paths:
    :param eval_set_images_dataset_paths:
    :param image_key: key to look up in the dictionary
    :return:
    """
    train_dicts = [
        load_path_or_paths(p)
        for p in train_set_images_dataset_paths
    ]
    train_dicts = ppp.treemap(normalize_if_needed, train_dicts, atomic_type=np.ndarray)
    train_images = [d[image_key] for d in train_dicts]
    if eval_set_images_dataset_paths is None:
        eval_images = train_images
        eval_dicts = train_dicts
    else:
        eval_dicts = [
            load_path_or_paths(p)
            for p in eval_set_images_dataset_paths
        ]
        eval_dicts = ppp.treemap(normalize_if_needed, eval_dicts, atomic_type=np.ndarray)
        eval_images = [d[image_key] for d in eval_dicts]
    if ungrouped_images_dataset_path:
        ungrouped = np.load(ungrouped_images_dataset_path, allow_pickle=True).item()
        ungrouped_images = ungrouped[image_key]
    else:
        ungrouped_images = np.zeros((0, *train_images[0][0].shape))
    ungrouped_images = ppp.treemap(
        normalize_if_needed, ungrouped_images, atomic_type=np.ndarray)

    return ungrouped_images, train_images, eval_images, train_dicts, eval_dicts


def train_set_vae(
        create_vae_kwargs,
        vae_trainer_kwargs,
        vae_algo_kwargs,
        data_loader_kwargs,
        num_ungrouped_images=0,
        load_images_kwargs=None,
        generate_test_set_kwargs=None,
        generate_train_set_kwargs=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        renderer_kwargs=None,
        example_state_key="example_state",
        example_image_key="example_image",
        latent_observation_key='latent_observation',
        mean_key='latent_mean',
        covariance_key='latent_covariance',
        image_observation_key='image_observation',
        include_env_debug=True,
        reward_visualization_period=None,
        # Allow re-use of this method with predefined variables
        env=None,
        renderer=None,
        train_sets=None,
        eval_sets=None,
        set_dict_loader_kwargs=None,
) -> VAE:
    if set_dict_loader_kwargs is None:
        set_dict_loader_kwargs = {}
    if load_images_kwargs is None:
        load_images_kwargs = {}
    if generate_train_set_kwargs is None:
        generate_train_set_kwargs = {}
    if generate_test_set_kwargs is None:
        generate_test_set_kwargs = {}
    if renderer_kwargs is None:
        renderer_kwargs = {}
    def create_set_imgs(gen_kwargs, sets):
        sets = sets or set_creation.create_sets(
            env,
            renderer,
            example_state_key=example_state_key,
            example_image_key=example_image_key,
            **gen_kwargs,
        )
        set_imgs = np.array([
            set_.example_dict[example_image_key] for set_ in sets
        ])
        return ptu.from_numpy(np.array(set_imgs)), sets

    if load_images_kwargs:
        (
            ungrouped_imgs,
            train_set_imgs,
            eval_set_imgs,
            train_sets,
            eval_sets,
        ) = (
            load_images_and_sets(**load_images_kwargs)
        )
        train_set_imgs = [ptu.from_numpy(s) for s in train_set_imgs]
        eval_set_imgs = [ptu.from_numpy(s) for s in eval_set_imgs]
        image_chw = train_set_imgs[0][0].shape
        train_sets = [
            set_creation.create_debug_set({example_image_key: set_})
            for set_ in train_set_imgs
        ]
    else:
        renderer = EnvRenderer(**renderer_kwargs)
        env = env or get_gym_env(env_id, env_class, env_kwargs)
        ungrouped_imgs = generate_images(
            env, renderer, num_images=num_ungrouped_images)
        train_set_imgs, train_sets = create_set_imgs(generate_train_set_kwargs, train_sets)
        eval_set_imgs, test_sets = create_set_imgs(generate_test_set_kwargs, eval_sets)
        image_chw = renderer.image_chw
    ungrouped_imgs = ptu.from_numpy(ungrouped_imgs)

    if len(train_set_imgs):
        all_imgs = torch.cat([ungrouped_imgs] + train_set_imgs, dim=0)
    else:
        all_imgs = ungrouped_imgs
    all_imgs_iterator = data.DataLoader(all_imgs, **data_loader_kwargs)

    vae = models.create_image_set_vae(img_chw=image_chw, **create_vae_kwargs)

    set_key = 'set'
    data_key = 'data'
    # dict_loader = DictLoader({
    #     data_key: all_imgs_iterator,
    #     set_key: infinite(train_set_imgs),
    # })
    dict_loader = CustomDictLoader(
        all_imgs_iterator,
        train_set_imgs,
        data_key=data_key,
        set_key=set_key,
        **set_dict_loader_kwargs
    )
    if include_env_debug:
        reward_fn, _ = rlkit.torch.sets.rewards.create_normal_likelihood_reward_fns(
            latent_observation_key=latent_observation_key,
            mean_key=mean_key,
            covariance_key=covariance_key,
            reward_fn_kwargs=dict(
                drop_log_det_term=True,
                sqrt_reward=True,
            ),
        )
        # vae.to(ptu.device)
        # n_obs = 1024
        # renderer = EnvRenderer(**renderer_kwargs)
        # img_env = InsertImageEnv(env, renderer=renderer)
        # encoder_env = EncoderWrappedEnv(
        #     img_env,
        #     vae,
        #     step_keys_map={image_observation_key: latent_observation_key},
        # )
        # states = debug.sample_states(encoder_env, n_obs)
        states = eval_sets[-1]
        states = {
            'state_observation': states['state_desired_goal'],
            'image_observation': states['image_desired_goal'],
            # 'latent_observation': vae.encode_np(states['image_desired_goal']),
        }
        state_observation_key = 'state_observation'

        def extra_debug_fn(trainer: SetVAETrainer):
            if 'latent_observation' not in states:
                states['latent_observation'] = vae.encode_np(
                    states['image_observation'])
            stats = OrderedDict()
            correlations = debug.compute_reward_correlations(
                reward_fn,
                train_sets,
                states,
                vae,
            )
            for set, cor in zip(train_sets, correlations):
                stats[
                    '{}/reward_correlation'.format(set.description.describe())
                ] = cor
            state_obs = states[state_observation_key]
            latent_obs = states[latent_observation_key]
            y = np.array(state_obs)
            x = np.array(latent_obs)
            A = np.linalg.inv(x.T @ x) @ x.T @ y
            y_hat = x @ A

            squared_errors = (y - y_hat) ** 2
            for i in range(squared_errors.shape[1]):
                mse_i = np.mean(squared_errors[:, i])
                variance_i = y[:, i].std() ** 2
                stats[
                    'normalized_mse/dim_{}'.format(i)
                ] = mse_i / variance_i
            return stats
    else:
        extra_debug_fn = None
    vae_trainer = SetVAETrainer(
        vae=vae,
        set_key=set_key,
        data_key=data_key,
        train_sets=train_set_imgs,
        eval_sets=eval_set_imgs,
        extra_log_fn=extra_debug_fn,
        **vae_trainer_kwargs)
    algorithm = BatchTorchAlgorithm(
        vae_trainer,
        dict_loader,
        **vae_algo_kwargs,
    )
    algorithm.to(ptu.device)

    if (
            reward_visualization_period is not None
            and reward_visualization_period > 0
    ):
        def add_visualization(trainer, epoch):
            if (
                    epoch % reward_visualization_period == 0
                    or epoch >= algorithm.num_iters - 1
            ):
                debug.save_reward_visualizations(
                    train_sets,
                    vae,
                    env,
                    renderer,
                    save_dir=logger.get_snapshot_dir(),
                    tag='_obj1_epoch{}'.format(epoch),
                    x_i=2,
                    y_i=3,
                )
                debug.save_reward_visualizations(
                    train_sets,
                    vae,
                    env,
                    renderer,
                    save_dir=logger.get_snapshot_dir(),
                    tag='_obj0_epoch{}'.format(epoch),
                    x_i=0,
                    y_i=1,
                )
        algorithm.post_epoch_funcs.append(add_visualization)

    algorithm.run()
    print(logger.get_snapshot_dir())
    return vae
