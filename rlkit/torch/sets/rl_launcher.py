import os.path as osp
import typing
from collections import OrderedDict
from functools import partial

import numpy as np
from matplotlib.ticker import ScalarFormatter

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.distribution import DictDistribution
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.envs.contextual import (
    ContextualEnv,
    delete_info,
)
from rlkit.envs.contextual.contextual_env import batchify
from rlkit.envs.contextual.set_distributions import (
    LatentGoalDictDistributionFromSet,
    SetDiagnostics,
    OracleRIGMeanSetter,
    GoalDictDistributionFromSets,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv, AutoEncoder, DictEncoderWrappedEnv
from rlkit.envs.images import EnvRenderer, InsertImageEnv, InsertImagesEnv

from rlkit.launchers.experiments.vitchyr.disco.visualize import (
    DynamicsModelEnvRenderer, InsertDebugImagesEnv,
    DynamicNumberEnvRenderer,
)
from rlkit.envs.images.env_renderer import DummyRenderer
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector,
)
from rlkit.samplers.rollout_functions import contextual_rollout
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sets import set_vae_trainer
from rlkit.torch.sets import models
from rlkit.torch.sets.rewards import (
    create_ground_truth_set_rewards_fns,
    create_normal_likelihood_reward_fns,
    create_latent_reward_fn,
)
from rlkit.torch.sets.set_creation import create_sets
from rlkit.torch.sets.set import Set
from rlkit.torch.sets.vae_launcher import train_set_vae
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.visualization import video


class StateImageGoalDiagnosticsFn:
    def __init__(self, state_to_goal_keys_map):
        self.state_to_goal_keys_map = state_to_goal_keys_map

    def __call__(self, paths, contexts):
        diagnostics = OrderedDict()
        for state_key in self.state_to_goal_keys_map:
            goal_key = self.state_to_goal_keys_map[state_key]
            values = []
            for i in range(len(paths)):
                state = paths[i]["observations"][-1][state_key]
                goal = contexts[i][goal_key]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
            diagnostics_key = goal_key + "/final/distance"
            diagnostics.update(
                create_stats_ordered_dict(diagnostics_key, values,)
            )
        return diagnostics


class FilterKeys(DictDistribution):
    def __init__(self, distribution: DictDistribution, keys_to_keep):
        self.keys_to_keep = keys_to_keep
        self.distribution = distribution
        self._spaces = {
            k: v for k, v in distribution.spaces.items() if k in keys_to_keep
        }

    def sample(self, batch_size: int):
        batch = self.distribution.sample(batch_size)
        return {k: v for k, v in batch.items() if k in self.keys_to_keep}

    @property
    def spaces(self):
        return self._spaces


class OracleMeanSettingPathCollector(ContextualPathCollector):
    def __init__(
            self,
            env: ContextualEnv,
            policy,
            max_num_epoch_paths_saved=None,
            observation_key='observation',
            context_keys_for_policy='context',
            render=False,
            render_kwargs=None,
            **kwargs
    ):
        rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=context_keys_for_policy,
            observation_key=observation_key,
        )
        super().__init__(
            env, policy, max_num_epoch_paths_saved, render, render_kwargs,
            rollout_fn=rollout_fn,
            **kwargs
        )
        self._observation_key = observation_key
        self._context_keys_for_policy = context_keys_for_policy

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            context_keys_for_policy=self._context_keys_for_policy,
        )
        return snapshot


class InitStateConditionedContextualEnv(ContextualEnv):
    def reset(self):
        obs = self.env.reset()
        self._rollout_context_batch = self.context_distribution.sample(
            1, init_obs=obs
        )
        self._update_obs(obs)
        self._last_obs = obs
        return obs


class ContextualRewardFnToRewardFn(object):
    def __init__(self, reward_fn):
        self.reward_fn = reward_fn

    def __call__(self, obs_dict, action, next_obs_dict):
        return self.reward_fn(obs_dict, action, next_obs_dict, next_obs_dict)


def disco_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        create_exploration_sets_kwargs=None,
        create_vae_training_sets_kwargs=None,
        create_evaluation_sets_kwargs=None,
        # VAE parameters
        create_vae_kwargs=None,
        load_pretrained_vae_kwargs=None,
        vae_env_settings=None,
        train_set_vae_kwargs=None,
        # Oracle settings
        use_ground_truth_reward=False,
        ground_truth_reward_kwargs=None,
        use_onehot_set_embedding=False,
        # use_dummy_model=False,
        pure_state_based=False,
        init_policy_path=None,
        # observation_key="latent_observation",
        # RIG comparison
        rig_goal_setter_kwargs=None,
        rig=False,
        # Env / Reward parmaeters
        reward_fn_kwargs=None,
        latent_goal_distribution_kwargs=None,
        # None-VAE Params
        env_id=None,
        env_class=None,
        env_kwargs=None,
        eval_env_kwargs=None,
        # latent_observation_key="latent_observation",
        latent_obs_mean_key='latent_mean_obs',
        latent_obs_covariance_key='latent_covariance_obs',
        observation_keys=('latent_mean_obs', 'latent_covariance_obs'),
        latent_goal_mean_key='latent_mean_goal',
        latent_goal_covariance_key='latent_covariance_goal',
        state_observation_key="state_observation",
        image_observation_key="image_observation",
        set_description_key="set_description",
        example_state_key="example_state",
        example_image_key="example_image",
        # Exploration
        exploration_policy_kwargs=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
        renderer_class=None,
        plot_renderer_kwargs=None,
        hd_renderer_class=None,
        hd_renderer_kwargs=None,
        tags=None,
):
    if eval_env_kwargs is None:
        eval_env_kwargs = env_kwargs
    if latent_goal_distribution_kwargs is None:
        latent_goal_distribution_kwargs = {}
    if create_exploration_sets_kwargs is None:
        create_exploration_sets_kwargs = {}
    if create_vae_training_sets_kwargs is None:
        create_vae_training_sets_kwargs = create_exploration_sets_kwargs
    if create_evaluation_sets_kwargs is None:
        create_evaluation_sets_kwargs = create_exploration_sets_kwargs
    if train_set_vae_kwargs is None:
        train_set_vae_kwargs = {}
    if vae_env_settings is None:
        vae_env_settings = {}
    if create_vae_kwargs is None:
        create_vae_kwargs = {}
    if load_pretrained_vae_kwargs is None:
        load_pretrained_vae_kwargs = {}
    if rig_goal_setter_kwargs is None:
        rig_goal_setter_kwargs = {}
    if reward_fn_kwargs is None:
        reward_fn_kwargs = {}
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if ground_truth_reward_kwargs is None:
        ground_truth_reward_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    if renderer_class:
        renderer = renderer_class(**renderer_kwargs)
    else:
        renderer = EnvRenderer(**renderer_kwargs)
    env = get_gym_env(env_id, env_class, env_kwargs)

    def _create_sets(kwargs):
        return create_sets(
            env,
            renderer,
            example_state_key=example_state_key,
            example_image_key=example_image_key,
            **kwargs,
        )

    if pure_state_based:
        model = models.create_dummy_image_vae(
            img_chw=renderer.image_chw,
            **create_vae_kwargs
        )
    elif load_pretrained_vae_kwargs:
        model = models.load_pretrained_vae(**load_pretrained_vae_kwargs)
    else:
        if vae_env_settings:
            vae_env = get_gym_env(**vae_env_settings)
        else:
            vae_env = env
        vae_train_sets = _create_sets(create_vae_training_sets_kwargs)
        model = train_set_vae(
            create_vae_kwargs,
            env=vae_env,
            train_sets=vae_train_sets,
            renderer=renderer,
            **train_set_vae_kwargs,
        )
    expl_sets = _create_sets(create_exploration_sets_kwargs)

    def _make_env(sets, env_kwargs_, **kwargs):
        return contextual_env_distrib_and_reward(
            vae=model,
            sets=sets,
            state_env=get_gym_env(
                env_id, env_class=env_class, env_kwargs=env_kwargs_,
            ),
            renderer=renderer,
            reward_fn_kwargs=reward_fn_kwargs,
            use_ground_truth_reward=use_ground_truth_reward,
            ground_truth_reward_kwargs=ground_truth_reward_kwargs,
            state_observation_key=state_observation_key,
            latent_obs_mean_key=latent_obs_mean_key,
            latent_obs_covariance_key=latent_obs_covariance_key,
            latent_goal_mean_key=latent_goal_mean_key,
            latent_goal_covariance_key=latent_goal_covariance_key,
            example_image_key=example_image_key,
            set_description_key=set_description_key,
            observation_keys=observation_keys,
            image_observation_key=image_observation_key,
            rig_goal_setter_kwargs=rig_goal_setter_kwargs,
            latent_goal_distribution_kwargs=latent_goal_distribution_kwargs,
            pure_state_based=pure_state_based,
            plot_renderer_kwargs=plot_renderer_kwargs,
            hd_renderer_class=hd_renderer_class,
            hd_renderer_kwargs=hd_renderer_kwargs,
            **kwargs
        )
    expl_env, expl_context_distrib, expl_reward, expl_env_video = _make_env(
        expl_sets,
        env_kwargs,
    )
    eval_sets = _create_sets(create_evaluation_sets_kwargs)
    eval_env, eval_context_distrib, eval_reward, eval_env_video = _make_env(
        eval_sets,
        eval_env_kwargs,
        rig_goal_setter=rig,
    )

    if pure_state_based:
        context_keys = [
            expl_context_distrib.set_index_key,
            expl_context_distrib.set_embedding_key,
        ]
    else:
        context_keys = [
            expl_context_distrib.mean_key,
            expl_context_distrib.covariance_key,
            expl_context_distrib.set_index_key,
            expl_context_distrib.set_embedding_key,
        ]
    if rig:
        context_keys_for_rl = [
            expl_context_distrib.mean_key,
        ]
    else:
        if use_onehot_set_embedding:
            context_keys_for_rl = [
                expl_context_distrib.set_embedding_key,
            ]
        else:
            context_keys_for_rl = [
                expl_context_distrib.mean_key,
                expl_context_distrib.covariance_key,
            ]

    obs_dim = sum(
        np.prod(expl_env.observation_space.spaces[k].shape)
        for k in observation_keys
    )

    obs_dim += sum(
        [np.prod(expl_env.observation_space.spaces[k].shape)
         for k in context_keys_for_rl]
    )
    action_dim = np.prod(expl_env.action_space.shape)

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim, output_size=1, **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    if init_policy_path:
        import torch
        data = torch.load(init_policy_path)
        policy = data['trainer/policy']
        del data
    else:
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim, action_dim=action_dim, **policy_kwargs
        )

    def concat_context_to_obs(batch, *args, **kwargs):
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        contexts = [batch[k] for k in context_keys_for_rl]
        batch["observations"] = np.concatenate((*obs, *contexts), axis=1)
        batch["next_observations"] = np.concatenate(
            (*next_obs, *contexts), axis=1,
        )
        return batch

    if pure_state_based:
        context_from_obs = RemapKeyFn({
            expl_context_distrib.set_index_key:
                expl_context_distrib.set_index_key,
            expl_context_distrib.set_embedding_key:
                expl_context_distrib.set_embedding_key,
        })
        ob_keys_to_save = list({
            state_observation_key,
        })
    else:
        ob_keys_to_save =list({
            state_observation_key,
            latent_obs_mean_key,
            latent_obs_covariance_key,
        })
        if rig:
            context_from_obs = RemapKeyFn({
                expl_context_distrib.mean_key: latent_obs_mean_key,
                expl_context_distrib.covariance_key:
                    expl_context_distrib.covariance_key,
                expl_context_distrib.set_index_key:
                    expl_context_distrib.set_index_key,
                expl_context_distrib.set_embedding_key:
                    expl_context_distrib.set_embedding_key,
            })
        else:
            context_from_obs = RemapKeyFn({
                expl_context_distrib.mean_key: latent_obs_mean_key,
                expl_context_distrib.covariance_key:
                    latent_obs_covariance_key,
                expl_context_distrib.set_index_key:
                    expl_context_distrib.set_index_key,
                expl_context_distrib.set_embedding_key:
                    expl_context_distrib.set_embedding_key,
            })

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=context_keys,
        observation_keys_to_save=ob_keys_to_save,
        observation_keys=observation_keys,
        context_distribution=FilterKeys(expl_context_distrib, context_keys,),
        sample_context_from_obs_dict_fn=context_from_obs,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs,
    )
    trainer = SACTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs,
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_keys=observation_keys,
        context_keys_for_policy=context_keys_for_rl,
    )
    eval_path_collector_video = ContextualPathCollector(
        eval_env_video,
        MakeDeterministic(policy),
        observation_keys=observation_keys,
        context_keys_for_policy=context_keys_for_rl,
    )
    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs
    )
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_keys=observation_keys,
        context_keys_for_policy=context_keys_for_rl,
    )
    expl_path_collector_video = ContextualPathCollector(
        expl_env_video,
        exploration_policy,
        observation_keys=observation_keys,
        context_keys_for_policy=context_keys_for_rl,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs,
    )
    algorithm.to(ptu.device)

    if save_video:
        set_index_key = eval_context_distrib.set_index_key

        extra_keys_to_save = [] if pure_state_based else ['reward', 'hd_image']

        def _create_video_func(sets, path_collector, tag):
            return DisCoVideoSaveFunction(
                model,
                sets,
                path_collector,
                tag=tag,
                example_image_key=example_image_key,
                set_index_key=set_index_key,
                reconstruction_key=None if pure_state_based else "image_reconstruction",
                set_visualization_key=None if pure_state_based else "set_visualization",
                latent_obs_mean_key=None if pure_state_based else latent_obs_mean_key,
                columns=len(expl_sets),
                unnormalize=True,
                image_format=renderer.output_image_format,
                extra_keys_to_save=extra_keys_to_save,
                **save_video_kwargs,
            )
        expl_video_func = _create_video_func(
            expl_sets, expl_path_collector_video, tag='xplor')
        algorithm.post_train_funcs.append(expl_video_func)
        eval_video_func = _create_video_func(
            eval_sets, eval_path_collector_video, tag='eval')
        algorithm.post_train_funcs.append(eval_video_func)

    algorithm.train()


def create_modify_fn(title, set_params=None, scientific=True,):
    def modify(ax):
        ax.set_title(title)
        if set_params:
            ax.set(**set_params)
        if scientific:
            scaler = ScalarFormatter(useOffset=True)
            scaler.set_powerlimits((1, 1))
            ax.yaxis.set_major_formatter(scaler)
            ax.ticklabel_format(axis='y', style='sci')
    return modify


def add_left_margin(fig):
    fig.subplots_adjust(left=0.2)


def contextual_env_distrib_and_reward(
        vae,
        sets: typing.List[Set],
        state_env,
        renderer,
        state_observation_key,
        latent_obs_mean_key,
        latent_obs_covariance_key,
        latent_goal_mean_key,
        latent_goal_covariance_key,
        example_image_key,
        set_description_key,
        observation_keys,
        image_observation_key,
        ground_truth_reward_kwargs,
        example_state_key=None,
        reward_fn_kwargs=None,
        use_ground_truth_reward=False,
        rig_goal_setter_kwargs=None,
        latent_goal_distribution_kwargs=None,
        rig_goal_setter=False,
        pure_state_based=False,
        # for visualization
        plot_renderer_kwargs=None,
        hd_renderer_class=None,
        hd_renderer_kwargs=None,
):
    if reward_fn_kwargs is None:
        reward_fn_kwargs = {}
    if rig_goal_setter_kwargs is None:
        rig_goal_setter_kwargs = {}
    if latent_goal_distribution_kwargs is None:
        latent_goal_distribution_kwargs = {}
    plot_renderer_kwargs = plot_renderer_kwargs or {}
    if example_state_key is None:
        example_state_key = example_image_key.replace('image', 'state')

    if pure_state_based:
        context_env_class = ContextualEnv
        goal_distribution_params_distribution = GoalDictDistributionFromSets(
            sets,
            desired_goal_key=example_state_key,
        )
    elif rig_goal_setter:
        context_env_class = InitStateConditionedContextualEnv
        goal_distribution_params_distribution = (
            OracleRIGMeanSetter(
                sets, vae, example_image_key,
                env=state_env,
                renderer=renderer,
                cycle_for_batch_size_1=True,
                **rig_goal_setter_kwargs
            )
        )
    else:
        context_env_class = ContextualEnv
        goal_distribution_params_distribution = (
            LatentGoalDictDistributionFromSet(
                sets, vae, example_image_key, cycle_for_batch_size_1=True,
                mean_key=latent_goal_mean_key,
                covariance_key=latent_goal_covariance_key,
                **latent_goal_distribution_kwargs
            )
        )
    if use_ground_truth_reward:
        reward_fn, unbatched_reward_fn = create_ground_truth_set_rewards_fns(
            sets,
            goal_distribution_params_distribution.set_index_key,
            state_observation_key,
            **ground_truth_reward_kwargs
        )
    else:
        reward_fn, unbatched_reward_fn = create_latent_reward_fn(
            observation_mean_key=latent_obs_mean_key,
            observation_covariance_key=latent_obs_covariance_key,
            goal_mean_key=latent_goal_mean_key,
            goal_covariance_key=latent_goal_covariance_key,
            reward_fn_kwargs=reward_fn_kwargs,
        )
    set_diagnostics = SetDiagnostics(
        set_description_key=set_description_key,
        set_index_key=goal_distribution_params_distribution.set_index_key,
        observation_key=state_observation_key,
    )
    img_env = InsertImageEnv(state_env, renderer=renderer)

    if pure_state_based:
        env = context_env_class(
            state_env,
            context_distribution=goal_distribution_params_distribution,
            reward_fn=reward_fn,
            unbatched_reward_fn=unbatched_reward_fn,
            observation_keys=observation_keys,
            contextual_diagnostics_fns=[
                set_diagnostics,
            ],
            update_env_info_fn=delete_info,
        )
        env_for_videos = context_env_class(
            img_env,
            context_distribution=goal_distribution_params_distribution,
            reward_fn=reward_fn,
            unbatched_reward_fn=unbatched_reward_fn,
            observation_keys=observation_keys,
            contextual_diagnostics_fns=[
                set_diagnostics,
            ],
            update_env_info_fn=delete_info,
        )
    else:
        encoded_env = DictEncoderWrappedEnv(
            img_env,
            vae,
            encoder_input_key=image_observation_key,
            encoder_output_remapping={
                latent_obs_mean_key: 'p_z_mean',
                latent_obs_covariance_key: 'p_z_covariance',
            },
        )
        env = context_env_class(
            encoded_env,
            context_distribution=goal_distribution_params_distribution,
            reward_fn=reward_fn,
            unbatched_reward_fn=unbatched_reward_fn,
            observation_keys=observation_keys,
            contextual_diagnostics_fns=[
                # goal_diagnostics,
                set_diagnostics,
            ],
            update_env_info_fn=delete_info,
        )

        # only insert HD images for videos not data collection
        hd_renderer = hd_renderer_class(**hd_renderer_kwargs)
        img_env_for_video = InsertImagesEnv(state_env, {
            image_observation_key: renderer,
            'hd_image': hd_renderer
        })
        video_encoded_env = DictEncoderWrappedEnv(
            img_env_for_video,
            vae,
            encoder_input_key=image_observation_key,
            encoder_output_remapping={
                latent_obs_mean_key: 'p_z_mean',
                latent_obs_covariance_key: 'p_z_covariance',
            },
        )
        video_env = context_env_class(
            video_encoded_env,
            context_distribution=goal_distribution_params_distribution,
            reward_fn=reward_fn,
            unbatched_reward_fn=unbatched_reward_fn,
            observation_keys=observation_keys,
            contextual_diagnostics_fns=[
                set_diagnostics,
            ],
            update_env_info_fn=delete_info,
        )

        renderers = OrderedDict()
        renderers['reward'] = DynamicNumberEnvRenderer(
            dynamic_number_fn=ContextualRewardFnToRewardFn(reward_fn),
            modify_ax_fn=create_modify_fn(
                title='reward',
            ),
            modify_fig_fn=add_left_margin,
            **plot_renderer_kwargs
        )
        env_for_videos = InsertDebugImagesEnv(
            video_env,
            renderers=renderers,
        )
    return env, goal_distribution_params_distribution, reward_fn, env_for_videos


class DisCoVideoSaveFunction:
    def __init__(
        self,
        model: AutoEncoder,
        sets,
        data_collector,
        tag,
        save_video_period,
        reconstruction_key=None,
        decode_set_image_key=None,
        set_visualization_key=None,
        example_image_key=None,
        num_example_images_to_show=0,
        set_index_key=None,
        latent_obs_mean_key=None,
        video_path_collector_kwargs=None,
        extra_keys_to_save=None,
        **kwargs
    ):
        if extra_keys_to_save is None:
            extra_keys_to_save = []
        if video_path_collector_kwargs is None:
            video_path_collector_kwargs = {}
        self.model = model
        self.sets = sets
        self.data_collector = data_collector
        self.tag = tag
        self.decode_set_image_key = decode_set_image_key
        self.set_visualization_key = set_visualization_key
        self.reconstruction_key = reconstruction_key
        self.example_image_key = example_image_key
        self.num_example_images_to_show = num_example_images_to_show
        self.set_index_key = set_index_key
        self.latent_obs_mean_key = latent_obs_mean_key
        self.dump_video_kwargs = kwargs
        self.save_video_period = save_video_period
        self.video_path_collector_kwargs = video_path_collector_kwargs
        self.keys = ['image_observation'] + extra_keys_to_save
        if reconstruction_key:
            self.keys.append(reconstruction_key)
        if set_visualization_key:
            self.keys.append(set_visualization_key)
        if decode_set_image_key:
            self.keys.append(decode_set_image_key)
        if set_visualization_key:
            for ex_i in range(self.num_example_images_to_show):
                self.keys.append('example%d' % ex_i)
        self.logdir = logger.get_snapshot_dir()

    def __call__(self, algo, epoch):
        paths = self.data_collector.collect_new_paths(**self.video_path_collector_kwargs)
        if epoch % self.save_video_period == 0 or epoch == algo.num_epochs:
            filename = "video_{epoch}_{tag}.mp4".format(
                epoch=epoch, tag=self.tag
            )
            filepath = osp.join(self.logdir, filename)
            self.save_video_of_paths(paths, filepath)

    def save_video_of_paths(self, paths, filepath):
        if self.reconstruction_key:
            for i in range(len(paths)):
                self.add_reconstruction_to_path(paths[i])
        if self.set_visualization_key:
            for i in range(len(paths)):
                self.add_set_visualization_to_path(paths[i])
        if self.decode_set_image_key:
            for i in range(len(paths)):
                self.add_decoded_goal_to_path(paths[i])
        video.dump_paths(
            None, filepath, paths, self.keys, **self.dump_video_kwargs,
        )

    def add_set_visualization_to_path(self, path):
        set_idx = path["full_observations"][0][self.set_index_key]
        set = self.sets[set_idx]
        set_visualization = set.example_dict[self.example_image_key].mean(
            axis=0
        )
        examples = set.example_dict[self.example_image_key][
            :self.num_example_images_to_show
        ]
        for i_in_path, d in enumerate(path["full_observations"]):
            d[self.set_visualization_key] = set_visualization
            for ex_i, ex in enumerate(examples):
                d['example%d' % ex_i] = ex

    def add_decoded_goal_to_path(self, path):
        set_idx = path["full_observations"][0][self.set_index_key]
        set = self.sets[set_idx]
        sampled_data = set.example_dict[self.example_image_key]
        posteriors = self.model.encoder(ptu.from_numpy(sampled_data))
        learned_prior = set_vae_trainer.compute_prior(posteriors)
        decoded = self.model.decoder(learned_prior.mean)
        decoded_img = ptu.get_numpy(decoded.mean)[0]
        for i_in_path, d in enumerate(path["full_observations"]):
            d[self.decode_set_image_key] = np.clip(decoded_img, 0, 1)

    def add_reconstruction_to_path(self, path):
        for i_in_path, d in enumerate(path["full_observations"]):
            latent = d[self.latent_obs_mean_key]
            decoded_img = self.model.decode_one_np(latent)
            d[self.reconstruction_key] = np.clip(decoded_img, 0, 1)
