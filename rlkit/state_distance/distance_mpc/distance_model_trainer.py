from collections import OrderedDict
import numpy as np

from torch import optim as optim

from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.policies.simple import RandomPolicy
from rlkit.samplers.util import rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer


class DistanceModelTrainer(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            model,
            mpc_controller,
            obs_normalizer: TorchFixedNormalizer=None,
            action_normalizer: TorchFixedNormalizer=None,
            delta_normalizer: TorchFixedNormalizer=None,
            num_paths_for_normalization=0,
            learning_rate=1e-3,
            exploration_policy=None,
            vectorized=False,
            **kwargs
    ):
        if exploration_policy is None:
            exploration_policy = mpc_controller
        super().__init__(
            env,
            exploration_policy=exploration_policy,
            eval_policy=mpc_controller,
            **kwargs
        )
        self.model = model
        self.mpc_controller = mpc_controller
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.delta_normalizer = delta_normalizer
        self.vectorized = vectorized
        self.num_paths_for_normalization = num_paths_for_normalization

    def _do_training(self):
        tmp_batch = self.get_batch()
        random_state = tmp_batch['observations']

        losses = []
        batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        ob_deltas_pred = self.model(obs, actions)

        next_obs_pred = obs + ob_deltas_pred
        if self.vectorized:
            distance_to_random_state_pred = (
                    (next_obs_pred - random_state)**2
            )
            distance_to_random_state = (
                    (next_obs - random_state)**2
            )

            squared_errors = (
                 distance_to_random_state_pred - distance_to_random_state
            )**2
            loss = squared_errors.mean()
        else:
            distance_to_random_state_pred = (
                    (next_obs_pred - random_state)**2
            ).sum(1, keepdim=True)
            distance_to_random_state = (
                    (next_obs - random_state)**2
            ).sum(1, keepdim=True)

            squared_errors = (
                distance_to_random_state_pred - distance_to_random_state
            )**2
            loss = squared_errors.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses.append(ptu.get_numpy(loss))

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics.update(create_stats_ordered_dict(
                'Model Loss',
                losses,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Distance To Random State',
                ptu.get_numpy(distance_to_random_state),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Distance To Random State Predicted',
                ptu.get_numpy(distance_to_random_state_pred),
            ))

    def pretrain(self):
        if (
            self.num_paths_for_normalization == 0
            or (
                self.obs_normalizer is None
                and self.action_normalizer is None
            )
        ):
            return

        pretrain_paths = []
        random_policy = RandomPolicy(self.env.action_space)
        while len(pretrain_paths) < self.num_paths_for_normalization:
            path = rollout(self.env, random_policy, self.max_path_length)
            pretrain_paths.append(path)
        ob_mean, ob_std, delta_mean, delta_std, ac_mean, ac_std = (
            compute_normalization(pretrain_paths)
        )
        if self.obs_normalizer is not None:
            self.obs_normalizer.set_mean(ob_mean)
            self.obs_normalizer.set_std(ob_std)
        if self.action_normalizer is not None:
            self.action_normalizer.set_mean(ac_mean)
            self.action_normalizer.set_std(ac_std)

    def _can_evaluate(self):
        return self.eval_statistics is not None

    @property
    def networks(self):
        return [
            self.model
        ]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['model'] = self.model
        snapshot['mpc_controller'] = self.mpc_controller
        return snapshot

    def offline_evaluate(self, epoch):
        return self.evaluate(epoch)


def compute_normalization(paths):
    obs = np.vstack([path["observations"] for path in paths])
    next_obs = np.vstack([path["next_observations"] for path in paths])
    deltas = next_obs - obs
    ob_mean = np.mean(obs, axis=0)
    ob_std = np.std(obs, axis=0)
    delta_mean = np.mean(deltas, axis=0)
    delta_std = np.std(deltas, axis=0)
    actions = np.vstack([path["actions"] for path in paths])
    ac_mean = np.mean(actions, axis=0)
    ac_std = np.std(actions, axis=0)
    return ob_mean, ob_std, delta_mean, delta_std, ac_mean, ac_std
