import numpy as np

import torch
from torch import distributions
from torch.distributions import kl_divergence

from rlkit.envs.contextual import ContextualRewardFn
from rlkit.envs.contextual.set_distributions import SetReward
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.distributions import MultivariateDiagonalNormal


class NormalLikelihoodRewardFn(ContextualRewardFn):
    def __init__(
            self,
            observation_key,
            mean_key,
            covariance_key,
            batched=True,
            drop_log_det_term=False,
            use_proper_scale_diag=True,
            sqrt_reward=False,
    ):
        self.observation_key = observation_key
        self.mean_key = mean_key
        self.covariance_key = covariance_key
        self.batched = batched
        self.drop_log_det_term = drop_log_det_term
        self.use_proper_scale_diag = use_proper_scale_diag
        self.sqrt_reward = sqrt_reward

    def __call__(self, states, actions, next_states, contexts):
        x = next_states[self.observation_key]
        mean = contexts[self.mean_key]
        covariance = contexts[self.covariance_key]
        if self.drop_log_det_term:
            reward = -((x - mean) ** 2) / (2 * covariance)
            reward = reward.sum(axis=-1)
            if self.sqrt_reward:
                reward = -np.sqrt(-reward)
        else:
            if self.use_proper_scale_diag:
                scale_diag = covariance ** 0.5
            else:
                scale_diag = covariance
            distribution = MultivariateDiagonalNormal(
                loc=ptu.from_numpy(mean), scale_diag=ptu.from_numpy(scale_diag)
            )
            reward = ptu.get_numpy(distribution.log_prob(ptu.from_numpy(x)))
        if not self.batched:
            reward = reward[0]
        return reward


class LatentRewardFn(ContextualRewardFn):
    NAME_TO_REWARD = {
        '-||mu_q - mu_p||': lambda q_z, prior, **_: - torch.norm(prior.mean - q_z.mean, dim=1),
        'log p(mu_q)': lambda q_z, prior, **_: prior.log_prob(q_z.mean),
        '-KL(q || p)': lambda q_z, prior, **_: -kl_divergence(q_z, prior),
        'e^{-KL(q || p)/T}': lambda q_z, prior, t: torch.exp(-kl_divergence(q_z, prior)/t),
        '-KL(p || q)': lambda q_z, prior, **_: -kl_divergence(prior, q_z),
        '-H(q, p)': lambda q_z, prior, **_: -kl_divergence(q_z, prior) - q_z.entropy(),
        '-H(p, q)': lambda q_z, prior, **_: -kl_divergence(prior, q_z) - prior.entropy(),
    }

    def __init__(
            self,
            observation_mean_key,
            observation_covariance_key,
            goal_mean_key,
            goal_covariance_key,
            batched=True,
            version='-KL(q || p)',
            extra_reward_kwargs=None,
    ):
        if version not in self.NAME_TO_REWARD:
            raise NotImplementedError(version)
        if extra_reward_kwargs is None:
            extra_reward_kwargs = {}
        self.observation_mean_key = observation_mean_key
        self.observation_covariance_key = observation_covariance_key
        self.goal_mean_key = goal_mean_key
        self.goal_covariance_key = goal_covariance_key
        self.batched = batched
        self.version = version
        self._extra_reward_kwargs = extra_reward_kwargs

    def __call__(self, states, actions, next_states, contexts):
        obs_mean = next_states[self.observation_mean_key]
        obs_std = next_states[self.observation_covariance_key] ** 0.5
        obs_posterior = MultivariateDiagonalNormal(
            loc=ptu.from_numpy(obs_mean),
            scale_diag=ptu.from_numpy(obs_std)
        )

        mean = contexts[self.goal_mean_key]
        covariance = contexts[self.goal_covariance_key]
        std = covariance ** 0.5
        set_prior = MultivariateDiagonalNormal(
            loc=ptu.from_numpy(mean), scale_diag=ptu.from_numpy(std)
        )
        reward = self.NAME_TO_REWARD[self.version](
                obs_posterior, set_prior,
                **self._extra_reward_kwargs
        )

        if not self.batched:
            reward = reward[0]
        return ptu.get_numpy(reward)


def create_ground_truth_set_rewards_fns(
        sets,
        set_index_key,
        state_observation_key,
        **kwargs
):
    reward_fn = SetReward(
        sets=sets,
        set_index_key=set_index_key,
        observation_key=state_observation_key,
        **kwargs
    )
    unbatched_reward_fn = SetReward(
        sets=sets,
        set_index_key=set_index_key,
        observation_key=state_observation_key,
        batched=False,
        **kwargs
    )
    return reward_fn, unbatched_reward_fn


def create_normal_likelihood_reward_fns(
        latent_observation_key,
        mean_key,
        covariance_key,
        reward_fn_kwargs,
):
    assert mean_key != covariance_key, "probably a typo"
    reward_fn = NormalLikelihoodRewardFn(
        observation_key=latent_observation_key,
        mean_key=mean_key,
        covariance_key=covariance_key,
        **reward_fn_kwargs
    )
    unbatched_reward_fn = NormalLikelihoodRewardFn(
        observation_key=latent_observation_key,
        mean_key=mean_key,
        covariance_key=covariance_key,
        batched=False,
        **reward_fn_kwargs
    )
    return reward_fn, unbatched_reward_fn


def create_latent_reward_fn(
        observation_mean_key,
        observation_covariance_key,
        goal_mean_key,
        goal_covariance_key,
        reward_fn_kwargs,
):
    assert goal_mean_key != goal_covariance_key, "probably a typo"
    reward_fn = LatentRewardFn(
        observation_mean_key=observation_mean_key,
        observation_covariance_key=observation_covariance_key,
        goal_mean_key=goal_mean_key,
        goal_covariance_key=goal_covariance_key,
        **reward_fn_kwargs
    )
    unbatched_reward_fn = LatentRewardFn(
        observation_mean_key=observation_mean_key,
        observation_covariance_key=observation_covariance_key,
        goal_mean_key=goal_mean_key,
        goal_covariance_key=goal_covariance_key,
        batched=False,
        **reward_fn_kwargs
    )
    return reward_fn, unbatched_reward_fn
