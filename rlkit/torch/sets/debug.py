import time
from collections import OrderedDict, namedtuple
from os import path as osp
from typing import MutableMapping
import typing

import cv2
import gym
import numpy as np
from torch import optim

from rlkit import pythonplusplus as ppp
from rlkit.core import logger
from rlkit.envs.encoder_wrappers import Encoder, DictEncoderWrappedEnv
from rlkit.envs.images import EnvRenderer, InsertImagesEnv, InsertImageEnv
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.disentanglement.networks import DisentangledMlpQf
from rlkit.torch.networks import Mlp
from rlkit.torch.sets import set_vae_trainer, rewards
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.visualization.image import combine_images_into_grid



class DebugTrainer(TorchTrainer):
    def __init__(self, observation_space, encoder, encoder_output_dim):
        super().__init__()
        self._ob_space = observation_space
        self._encoder = encoder
        self._latent_dim = encoder_output_dim

    def train_from_torch(self, batch):
        pass

    @property
    def networks(self):
        return []

    def get_diagnostics(self):
        start_time = time.time()
        linear_loss = get_linear_loss(self._ob_space, self._encoder)
        linear_time = time.time() - start_time

        start_time = time.time()
        non_linear_results = get_non_linear_results(
            self._ob_space, self._encoder, self._latent_dim)
        non_linear_time = time.time() - start_time

        stats = OrderedDict([
            ('debug/reconstruction/linear/loss', linear_loss),
            ('debug/reconstruction/linear/train_time (s)', linear_time),
            ('debug/reconstruction/non_linear/loss', non_linear_results.loss),
            ('debug/reconstruction/non_linear/initial_loss',
             non_linear_results.initial_loss),
            ('debug/reconstruction/non_linear/last_10_percent_contribution',
             non_linear_results.last_10_percent_contribution),
            ('debug/reconstruction/non_linear/train_time (s)', non_linear_time),
        ])
        return stats


def get_linear_loss(ob_space, encoder):
    x = get_batch(ob_space, batch_size=2 ** 15)
    z_np = eval_np(encoder, x)
    results = np.linalg.lstsq(z_np, x, rcond=None)
    matrix = results[0]

    eval_states = get_batch(ob_space, batch_size=2 ** 15)
    z_np = eval_np(encoder, eval_states)
    x_hat = z_np.dot(matrix)
    return ((eval_states - x_hat) ** 2).mean()


NonLinearResults = namedtuple(
    'NonLinearResults',
    [
        'loss',
        'last_10_percent_contribution',
        'initial_loss',
    ],
)


def get_non_linear_results(
        ob_space, encoder, latent_dim,
        batch_size=128,
        num_batches=10000,
) -> NonLinearResults:
    state_dim = ob_space.low.size

    decoder = Mlp(
        hidden_sizes=[64, 64],
        output_size=state_dim,
        input_size=latent_dim,
    )
    decoder.to(ptu.device)
    optimizer = optim.Adam(decoder.parameters())

    initial_loss = last_10_percent_loss = 0
    for i in range(num_batches):
        states = get_batch(ob_space, batch_size)
        x = ptu.from_numpy(states)
        z = encoder(x)
        x_hat = decoder(z)

        loss = ((x - x_hat) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 0:
            initial_loss = ptu.get_numpy(loss)
        if i == int(num_batches * 0.9):
            last_10_percent_loss = ptu.get_numpy(loss)

    eval_states = get_batch(ob_space, batch_size=2 ** 15)
    x = ptu.from_numpy(eval_states)
    z = encoder(x)
    x_hat = decoder(z)
    reconstruction = ptu.get_numpy(x_hat)
    loss = ((eval_states - reconstruction) ** 2).mean()
    last_10_percent_contribution = (
                                       (last_10_percent_loss - loss)
                                   ) / (initial_loss - loss)
    del decoder, optimizer
    return NonLinearResults(
        loss=loss,
        initial_loss=initial_loss,
        last_10_percent_contribution=last_10_percent_contribution,
    )


def get_batch(ob_space, batch_size):
    noise = np.random.randn(batch_size, *ob_space.low.shape)
    return noise * (ob_space.high - ob_space.low) + ob_space.low


class DebugEnvRenderer(EnvRenderer):
    def __init__(
            self,
            encoder: Encoder,
            head_index,
            **kwargs
    ):
        super().__init__(**kwargs)
        """Render an image."""
        self.channels = 3
        self.encoder = encoder
        self.head_idx = head_index

    def _create_image(self, env, encoded):
        values = encoded[:, self.head_idx]
        value_image = values.reshape(self.image_shape[:2])
        value_img_rgb = np.repeat(
            value_image[:, :, None],
            3,
            axis=2
        )
        value_img_rgb = (
                (value_img_rgb - value_img_rgb.min()) /
                (value_img_rgb.max() - value_img_rgb.min())
        )
        return value_img_rgb

class InsertDebugImagesEnv(InsertImagesEnv):
    def __init__(
            self,
            wrapped_env: gym.Env,
            renderers: typing.Dict[str, DebugEnvRenderer],
            compute_shared_data=None,
    ):
        super().__init__(wrapped_env, renderers)
        self.compute_shared_data = compute_shared_data

    def _update_obs(self, obs):
        shared_data = self.compute_shared_data(obs, self.env)
        for image_key, renderer in self.renderers.items():
            obs[image_key] = renderer(self.env, shared_data)



Slice2d = namedtuple(
    'Slice2D',
    'template xs ys xi yi'
)


def generate_slice_heat_map(batch_function, slice: Slice2d):
    slice_batch = generate_slice_batch(slice)
    values = batch_function(slice_batch)
    value_image = values.reshape((slice.xs.size, slice.ys.size))
    return value_image


def generate_slice_batch(slice: Slice2d):
    """
    Given some state, x values, and y values

    state = [a, b, c, d, e]
    x = [1, 2]
    y = [3, 4]
    x_i = 0
    y_i = 2

    generate all the Cartesian product with the x- and y- value replaced
    [
        [1, b, 3, d, e]
        [1, b, 4, d, e]
        [2, b, 3, d, e]
        [2, b, 4, d, e]
    ]

    :param anchor_state:
    :param x:
    :param y:
    :param x_i:
    :param y_i:
    :return:
    """
    all_xy = np.transpose([
        np.repeat(slice.ys, len(slice.xs)),
        np.tile(slice.xs, len(slice.ys)),
    ])
    repeated_state = slice.template[None].repeat(all_xy.shape[0], axis=0)
    repeated_state[:, slice.xi] = all_xy[:, 0]
    repeated_state[:, slice.yi] = all_xy[:, 1]
    return repeated_state


def create_visualize_representation(
        encoder,
        obj_to_sweep,
        env,
        renderer,
        start_states,
        save_period=50,
        env_renderer=None,
        initial_save_period=None,
        state_to_encoder_input=None,
):
    if initial_save_period is None:
        initial_save_period = save_period
    env_renderer = env_renderer or renderer
    state_space = env.env.observation_space['state_observation']
    low = state_space.low.min()
    high = state_space.high.max()
    y = np.linspace(low, high, num=env_renderer.image_chw[1])
    x = np.linspace(low, high, num=env_renderer.image_chw[2])
    all_xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

    goal_dicts = []
    for start_state in start_states:
        new_states = np.repeat(start_state[None], all_xy.shape[0], axis=0)
        start_i = obj_to_sweep * 2
        end_i = start_i + 2
        new_states[:, start_i:end_i] = all_xy
        if state_to_encoder_input:
            new_states = np.concatenate(
                [
                    state_to_encoder_input(state)[None, ...] for state in new_states
                ],
                axis=0,
            )
        goal_dict = {
            'state_desired_goal': start_state,
        }
        env_state = env.get_env_state()
        env.set_to_goal(goal_dict)
        start_img = renderer(env)
        env.set_env_state(env_state)
        goal_dict['image_observation'] = start_img
        goal_dict['new_states'] = new_states
        goal_dicts.append(goal_dict)

    def visualize_representation(epoch):
        # logdir = logger.get_snapshot_dir()
        logdir = '/tmp/'
        filename = osp.join(
            logdir,
            'obj{obj_id}_sweep_visualization_{epoch}.png'.format(
                obj_id=obj_to_sweep,
                epoch=epoch),
        )
        visualizations = []
        for goal_dict in goal_dicts:
            start_img = goal_dict['image_observation']
            new_states = goal_dict['new_states']

            encoded = encoder.encode_np(new_states)
            # img_format = renderer.output_image_format
            images_to_stack = [start_img]
            for i in range(encoded.shape[1]):
                values = encoded[:, i]
                value_image = values.reshape(env_renderer.image_chw[1:])
                # TODO: fix hardcoding of CHW
                value_img_rgb = np.repeat(
                    value_image[None, :, :],
                    3,
                    axis=0
                )
                value_img_rgb = (
                        (value_img_rgb - value_img_rgb.min()) /
                        (value_img_rgb.max() - value_img_rgb.min() + 1e-9)
                )
                images_to_stack.append(value_img_rgb)

            combined_img = combine_images_into_grid(
                images_to_stack,
                imwidth=renderer.image_chw[2],
                imheight=renderer.image_chw[1],
                max_num_cols=len(start_states),
                pad_length=1,
                pad_color=0,
                subpad_length=1,
                subpad_color=128,
                image_format=renderer.output_image_format,
                unnormalize=True,
            )

            visualizations.append(combined_img)

        final_image = combine_images_into_grid(
            visualizations,
            imwidth=visualizations[0].shape[1],
            imheight=visualizations[0].shape[0],
            max_num_cols=3,
            image_format='HWC',
            pad_length=0,
            subpad_length=0,
        )
        cv2.imwrite(filename, final_image)

        print("Saved visualization image to to ", filename)

    return visualize_representation


def correlation(x, y):
    return np.mean(
        (x - x.mean()) * (y - y.mean()) /
        (x.std() * y.std())
    )


def get_learned_rewards(
        set,
        vae,
        reward_fn,
        states,
        mean_key='latent_mean', covariance_key='latent_covariance'
):
    example_imgs = set.example_dict['example_image']
    # posteriors = vae.encode_np(ptu.from_numpy(example_imgs))
    # learned_prior = set_vae_trainer.compute_prior(posteriors)
    # context = {
    #     mean_key: ptu.get_numpy(learned_prior.mean)[0],
    #     covariance_key: ptu.get_numpy(learned_prior.variance)[0],
    # }
    # import ipdb; ipdb.set_trace()
    p_z = vae.approx_p_z_given_set(example_imgs)
    context = {
        mean_key: ptu.get_numpy(p_z.mean)[0],
        covariance_key: ptu.get_numpy(p_z.variance)[0],
    }
    learned_rewards = reward_fn(None, None, states, context)
    return np.array(learned_rewards)


def get_true_reward(set, states):
    true_rewards = - set.description.distance_to_set(
        states['state_observation']
    )
    return np.array(true_rewards)


def compute_reward_correlations(reward_fn, sets, states, vae):
    learned_rewards = []
    true_rewards = []
    correlations = []
    for set in sets:
        learned = get_learned_rewards(set, vae, reward_fn, states)
        true = get_true_reward(set, states)
        correlations.append(correlation(learned, true))
        learned_rewards.append(learned)
        true_rewards.append(true)
    return correlations


def sample_states(env, n_obs):
    states = []
    for n in range(n_obs):
        state = env.reset()
        states.append({
            k: v for k, v in state.items() if isinstance(v, np.ndarray)
        })
    states = ppp.treemap(
        lambda x: x[None], states, atomic_type=np.ndarray)
    states = ppp.treemap(
        lambda *x: np.concatenate(x, axis=0), *tuple(states),
        atomic_type=np.ndarray)
    return states


def save_reward_visualizations(
        sets,
        vae,
        state_env,
        renderer,
        save_dir=None,
        tag='',
        x_i=2,
        y_i=3,
):
    reward_fns = dict(
        mahalanobis_distance=rewards.NormalLikelihoodRewardFn(
            observation_key='latent_observation',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            drop_log_det_term=True,
            sqrt_reward=True,
        ),
        proper_likelihood=rewards.NormalLikelihoodRewardFn(
            observation_key='latent_observation',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            drop_log_det_term=False,
            use_proper_scale_diag=True,
            sqrt_reward=False,
        ),
        cross_entropy_prior_to_obs=rewards.LatentRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            obs_to_prior_direction=False,
        ),
        kl_prior_to_obs=rewards.LatentRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            use_kl_not_ce=True,
            obs_to_prior_direction=False,
        ),
        cross_entropy_obs_to_prior=rewards.LatentRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            obs_to_prior_direction=True,
        ),
        kl_obs_to_prior=rewards.LatentRewardFn(
            observation_mean_key='posterior_mean',
            observation_covariance_key='posterior_covariance',
            mean_key='latent_mean',
            covariance_key='latent_covariance',
            use_kl_not_ce=True,
            obs_to_prior_direction=True,
        ),
    )

    img_env = InsertImageEnv(state_env, renderer=renderer)
    env = DictEncoderWrappedEnv(
        img_env,
        vae,
        encoder_input_key='image_observation',
        encoder_output_remapping={'posterior_mean': 'latent_observation'},
    )

    def to_image(state, renderer):
        state_env._set_positions(state)
        return renderer(state_env)

    _, height, width = renderer.image_chw
    height = width = 28
    vis_renderer = EnvRenderer(
        width=width,
        height=height,
        output_image_format='CWH',
    )
    xs = np.linspace(-4, 4, width)
    ys = np.linspace(-4, 4, height)
    for name, reward_fn in reward_fns.items():
        imgs = []
        for _ in range(5):
            state = env.reset()['state_observation']
            anchor_img = to_image(state, vis_renderer)
            state_slice = Slice2d(state, xs, ys, x_i, y_i)
            state_slice_batch = generate_slice_batch(state_slice)
            images = np.array([to_image(s, renderer) for s in state_slice_batch])
            states = vae.encode_to_dict_np(images)
            states['latent_observation'] = states['posterior_mean']
            states['state_observation'] = state_slice_batch
            imgs.append(anchor_img)
            for set in sets:
                learned_rewards_slice = get_learned_rewards(set, vae, reward_fn, states)
                img = learned_rewards_slice.reshape(width, height)
                rgb_img = np.stack([img, img, img], axis=0)
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                imgs.append(rgb_img)
            imgs.append(anchor_img)
            for set in sets:
                true_rewards_slice = get_true_reward(set, states)
                img = true_rewards_slice.reshape(width, height)
                rgb_img = np.stack([img, img, img], axis=0)
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                imgs.append(rgb_img)
        num_cols = len(sets) + 1
        imgs = transpose_list(imgs, num_cols)
        num_cols = len(imgs) // num_cols
        combined = combine_images_into_grid(
            imgs=imgs,
            imwidth=width,
            imheight=height,
            image_format='CWH',
            unnormalize=True,
            max_num_cols=num_cols,
        )
        if tag:
            name = 'reward_heatmap_{}_{}.png'.format(name, tag)
        else:
            name = 'reward_heatmap_{}.png'.format(name)
        img_path = osp.join(str(save_dir), name)
        cv2.imwrite(
            img_path,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
        )
        # print("saved image to", img_path)


def transpose_list(x, num_cols):
    """
    re-order a list from row-scan to column-scan, where k is the number of columns
    :param x:
    :param k:
    :return:
    """
    num_elems = len(x)
    num_rows = num_elems // num_cols
    outputs = []
    for i in range(num_cols):
        for j in range(num_rows):
            outputs.append(x[j * num_cols + i])
    return outputs