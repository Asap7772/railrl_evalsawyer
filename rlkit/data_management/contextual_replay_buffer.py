import abc
from typing import Any, Dict

import numpy as np

from rlkit.core.distribution import DictDistribution
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.envs.contextual import ContextualRewardFn
from rlkit import pythonplusplus as ppp


class SampleContextFromObsDictFn(object, metaclass=abc.ABCMeta):
    """Interface definer, but you can also just pass in a function.

    This function maps an observation to some context that ``was achieved''.
    """

    @abc.abstractmethod
    def __call__(self, obs: dict) -> Any:
        pass


class RemapKeyFn(SampleContextFromObsDictFn):
    """A simple map that forwards observations to become the context."""

    def __init__(self, context_to_input_key: Dict[str, str]):
        self._context_to_input_key = context_to_input_key

    def __call__(self, obs: dict) -> Any:
        new_obs = {k: obs[v] for k, v in self._context_to_input_key.items()}
        return new_obs


class ContextualRelabelingReplayBuffer(ObsDictReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            context_keys,
            observation_keys,  # TODO: rename as observation_keys_to_save
            sample_context_from_obs_dict_fn: SampleContextFromObsDictFn,
            reward_fn: ContextualRewardFn,
            context_distribution: DictDistribution,
            fraction_future_context,
            fraction_distribution_context,
            fraction_next_context=0.,
            fraction_replay_buffer_context=0.0,
            post_process_batch_fn=None,
            observation_key='observation',
            save_data_in_snapshot=False,
            internal_keys=None,
            **kwargs
    ):
        ob_keys_to_save = observation_keys + context_keys
        super().__init__(
            max_size,
            env,
            ob_keys_to_save=ob_keys_to_save,
            internal_keys=internal_keys,
            observation_key=observation_key,
            save_data_in_snapshot=save_data_in_snapshot,
            **kwargs
        )
        if (
                fraction_distribution_context < 0
                or fraction_future_context < 0
                or fraction_next_context < 0
                or fraction_replay_buffer_context < 0
                or (fraction_future_context
                    + fraction_next_context
                    + fraction_distribution_context
                    + fraction_replay_buffer_context) > 1
        ):
            raise ValueError("Invalid fractions: {} and {}".format(
                fraction_future_context,
                fraction_distribution_context,
            ))
        self._observation_keys = observation_keys
        self._context_keys = context_keys
        self._context_distribution = context_distribution
        for k in context_keys:
            distribution_keys = set(self._context_distribution.spaces.keys())
            if k not in distribution_keys:
                raise TypeError(
                    "All context keys must be in context distribution."
                )
        self._sample_context_from_obs_dict_fn = sample_context_from_obs_dict_fn
        self._reward_fn = reward_fn
        self._fraction_next_context = fraction_next_context
        self._fraction_future_context = fraction_future_context
        self._fraction_distribution_context = (
            fraction_distribution_context
        )
        self._fraction_replay_buffer_context = fraction_replay_buffer_context

        def composed_post_process_batch_fn(
                batch, replay_buffer, obs_dict, next_obs_dict, new_contexts
        ):
            new_batch = batch
            if post_process_batch_fn:
                new_batch = post_process_batch_fn(
                    new_batch,
                    replay_buffer,
                    obs_dict, next_obs_dict, new_contexts
                )
            if self._reward_fn:
                new_batch['rewards'] = self._reward_fn(
                    obs_dict,
                    new_batch['actions'],
                    next_obs_dict,
                    new_contexts,
                )
            if len(new_batch['rewards'].shape) == 1:
                new_batch['rewards'] = new_batch['rewards'].reshape(-1, 1)
            return new_batch

        self._post_process_batch_fn = composed_post_process_batch_fn

    def random_batch(self, batch_size):
        num_future_contexts = int(batch_size * self._fraction_future_context)
        num_next_contexts = int(batch_size * self._fraction_next_context)
        num_replay_buffer_contexts = int(
            batch_size * self._fraction_replay_buffer_context
        )
        num_distrib_contexts = int(
            batch_size * self._fraction_distribution_context)
        num_rollout_contexts = (
                batch_size - num_future_contexts - num_distrib_contexts
                - num_next_contexts - num_replay_buffer_contexts
        )
        indices = self._sample_indices(batch_size)
        obs_dict = self._batch_obs_dict(indices)
        next_obs_dict = self._batch_next_obs_dict(indices)
        contexts = [{
            k: next_obs_dict[k][:num_rollout_contexts]
            for k in self._context_keys
        }]

        if num_distrib_contexts > 0:
            curr_obs = {
                k: next_obs_dict[k][num_rollout_contexts:num_rollout_contexts + num_distrib_contexts]
                for k in self._observation_keys
            }
            sampled_contexts = self._context_distribution.sample(
                num_distrib_contexts, context=curr_obs)
            sampled_contexts = {
                k: sampled_contexts[k] for k in self._context_keys}
            contexts.append(sampled_contexts)

        if num_replay_buffer_contexts > 0:
            replay_buffer_contexts = self._get_replay_buffer_contexts(
                num_replay_buffer_contexts,
            )
            replay_buffer_contexts = {
                k: replay_buffer_contexts[k] for k in self._context_keys}
            contexts.append(replay_buffer_contexts)

        if num_next_contexts > 0:
            start_idx = -(num_future_contexts + num_next_contexts)
            start_state_indices = indices[start_idx:-num_future_contexts]
            next_contexts = self._get_next_contexts(start_state_indices)
            contexts.append(next_contexts)

        if num_future_contexts > 0:
            start_state_indices = indices[-num_future_contexts:]
            future_contexts = self._get_future_contexts(start_state_indices)
            future_contexts = {
                k: future_contexts[k] for k in self._context_keys}
            contexts.append(future_contexts)

        actions = self._actions[indices]

        def concat(*x):
            return np.concatenate(x, axis=0)

        new_contexts = ppp.treemap(concat, *tuple(contexts),
                                   atomic_type=np.ndarray)

        batch = {
            'observations': obs_dict[self.observation_key],
            'actions': actions,
            'rewards': self._rewards[indices],
            'terminals': self._terminals[indices],
            'next_observations': next_obs_dict[self.observation_key],
            'indices': np.array(indices).reshape(-1, 1),
            **new_contexts
            # 'contexts': new_contexts,
        }
        new_batch = self._post_process_batch_fn(
            batch,
            self,
            obs_dict, next_obs_dict, new_contexts
        )
        return new_batch

    def _get_replay_buffer_contexts(self, batch_size):
        indices = self._sample_indices(batch_size)
        replay_buffer_obs_dict = self._batch_next_obs_dict(indices)
        return self._sample_context_from_obs_dict_fn(replay_buffer_obs_dict)

    def _get_future_contexts(self, start_state_indices):
        future_obs_idxs = self._get_future_obs_indices(start_state_indices)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
        return self._sample_context_from_obs_dict_fn(future_obs_dict)

    def _get_next_contexts(self, start_state_indices):
        next_obs_dict = self._batch_next_obs_dict(start_state_indices)
        return self._sample_context_from_obs_dict_fn(next_obs_dict)
