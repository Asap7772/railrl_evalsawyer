import os.path as osp
from collections import OrderedDict, namedtuple
from itertools import chain
from typing import Tuple, Optional, List, NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from gym import spaces
from matplotlib.patches import Ellipse
from torch import nn
from torch.distributions.kl import kl_divergence

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.loss import LossFunction
from rlkit.core.timer import timer
from rlkit.envs.encoder_wrappers import AutoEncoder, DictEncoder
from rlkit.misc import ml_util
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch.distributions import MultivariateDiagonalNormal, Distribution
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.sets import swae, mmd
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.vae.vae_torch_trainer import compute_vae_terms
from rlkit.visualization.image import combine_images_into_grid

ELLIPSE_COLORS = [
    np.array([5, 60, 94]) / 255.,
    np.array([94, 60, 5]) / 255.,
    # 'gold',
    # 'palegreen',
]

MEAN_COLORS = [
    'orange',
    'aqua',
]
PRIOR_COLORS = [
    # np.array([163, 22, 33]) / 255.,
    'r',
    'g',
]


LossStatistics = OrderedDict


# class SVAELosses(NamedTuple):
#     set_vae_loss: torch.Tensor

SVAELosses = namedtuple(
    'SVAELosses',
    'set_vae_loss',
)


SetVAETerms = namedtuple(
    'SetVAETerms',
    'likelihood kl_z kl_c q_c q_z p_x_given_c_z p_z_given_c c z',
)


class SetVAE(nn.Module, AutoEncoder, DictEncoder):
    """
    Hierarhical model. Using plate notation

        c
        |
        v

    |````````|
    | z -> x |
    |________|

    and what we observe is sets x that all correspond to the same `c`.

    Optionally, c only goes to z and not x.
    """
    def __init__(
            self,
            encoder_z: DistributionGenerator,
            encoder_c: DistributionGenerator,
            decoder: DistributionGenerator,
            prior_z_given_c: DistributionGenerator,
            prior_c: DistributionGenerator,
            z_dim: int,
            c_dim: int,
            x_depends_on_c=False,
    ):
        super().__init__()
        self.encoder_z = encoder_z
        self.encoder_c = encoder_c
        self.decoder = decoder
        self.prior_z_given_c = prior_z_given_c
        self.prior_c = prior_c
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_depends_on_c = x_depends_on_c

    def p_x_given_c_z(self, c, z):
        if self.x_depends_on_c:
            return self.decoder(c, z)
        else:
            return self.decoder(z)

    def approx_p_z_given_set(self, x):
        """Treat x as entire batch."""
        prior_c = self.encoder_c(x)
        c = prior_c.mean
        return self.prior_z_given_c(c)

    def q_zs_given_independent_xs(self, x):
        """Treat each x as a separate input"""
        prior_c = self.encoder_c(x, batch=True)
        c = prior_c.mean
        return self.encoder_z(c, x)

    def reconstruct(
            self,
            x,
            use_z_mean=True,
            use_c_mean=True,
            use_generative_model_mean=True,
    ):
        q_c = self.encoder_c(x)
        c = q_c.mean if use_c_mean else q_c.sample()
        batch_size = x.shape[0]
        c_batched = c.repeat(batch_size, 1)

        q_z = self.encoder_z(c_batched, x)
        z = q_z.mean if use_z_mean else q_z.sample()

        p_x = self.p_x_given_c_z(c, z)
        x_hat = p_x.mean if use_generative_model_mean else p_x.sample()
        return x_hat

    def sample(self, batch_size, use_generative_model_mean=True):
        p_c = self.prior_c()
        c = p_c.sample(torch.Size([batch_size])).squeeze(1)
        p_z = self.prior_z_given_c(c)
        z = p_z.sample()
        p_x = self.p_x_given_c_z(c, z)
        x = p_x.mean if use_generative_model_mean else p_x.sample()
        return x

    def set_sample(self, batch_size, data,
                   use_q_c_mean=True,
                   use_generative_model_mean=True):
        q_c = self.encoder_c(data)
        c = q_c.mean if use_q_c_mean else q_c.sample()
        c_batched = c.repeat(batch_size, 1)
        p_z = self.prior_z_given_c(c_batched)
        z = p_z.sample()
        p_x = self.p_x_given_c_z(c_batched, z)
        x = p_x.mean if use_generative_model_mean else p_x.sample()
        return x

    def latent_prior_and_posterior(self, x):
        terms = self.compute_set_vae_terms(x)
        return terms.p_z_given_c, terms.q_z

    @property
    def output_dict_space(self):
        return spaces.Dict({
            'p_z_mean': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.z_dim,),
            ),
            'p_z_covariance': spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.z_dim,),
            ),
            'p_c_mean': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.c_dim,),
            ),
            'p_c_covariance': spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.c_dim,),
            ),
        })

    def compute_set_vae_terms(self, x) -> SetVAETerms:
        q_c = self.encoder_c(x)
        p_c = self.prior_c()
        kl_c = kl_divergence(q_c, p_c)

        c = q_c.rsample()
        batch_size = x.shape[0]
        c_batched = c.repeat(batch_size, 1)
        q_z = self.encoder_z(c_batched, x)

        p_z_given_c = self.prior_z_given_c(c)
        kl_z = kl_divergence(q_z, p_z_given_c)

        z = q_z.rsample()
        p_x_given_c_z = self.p_x_given_c_z(c_batched, z)
        log_prob = p_x_given_c_z.log_prob(x)

        return SetVAETerms(
            likelihood=log_prob.mean(),
            kl_z=kl_z.mean(),
            kl_c=kl_c.mean(),
            q_c=q_c,
            q_z=q_z,
            p_x_given_c_z=p_x_given_c_z,
            p_z_given_c=p_z_given_c,
            c=c,
            z=z,
        )

    """Implement `AutoEncoder` interface"""
    @property
    def representation_size(self):
        return self.z_dim

    def encode_np(self, observations):
        x = ptu.from_numpy(observations)
        q_z = self.q_zs_given_independent_xs(x)
        if self.x_depends_on_c:
            raise NotImplementedError()
            # return np.concatenate([
            #     ptu.get_numpy(q_z.mean),
            #     ptu.get_numpy(c)[0],
            # ], axis=0)
        else:
            return ptu.get_numpy(q_z.mean)

    def encode_one_np(self, observation):
        x = ptu.from_numpy(observation[None])
        p_c = self.encoder_c(x)
        c = p_c.mean
        q_z = self.encoder_z(c, x)
        if self.x_depends_on_c:
            return np.concatenate([
                ptu.get_numpy(q_z.mean)[0],
                ptu.get_numpy(c)[0],
            ], axis=0)
        else:
            return ptu.get_numpy(q_z.mean)[0]

    def encode_to_dict_np(self, observation):
        x = ptu.from_numpy(observation)
        p_c = self.encoder_c(x)
        c = p_c.mean
        q_z = self.encoder_z(c, x)
        pytorch_dict = {
            'p_z_mean': q_z.mean,
            'p_z_covariance': q_z.variance,
            'p_c_mean': p_c.mean,
            'p_c_covariance': p_c.variance,
        }
        return {k: ptu.get_numpy(v) for k, v in pytorch_dict.items()}

    def decode_one_np(self, encoded_rep):
        if self.x_depends_on_c:
            encoded_rep_pt = ptu.from_numpy(encoded_rep[None])
            c = encoded_rep_pt[:, :self.c_dim]
            z = encoded_rep_pt[:, self.c_dim:]
        else:
            c = None
            z = ptu.from_numpy(encoded_rep[None])
        p_x = self.p_x_given_c_z(c, z)
        return ptu.get_numpy(p_x.mean)[0]


def compute_prior(q_z: Distribution):
    if not isinstance(q_z, MultivariateDiagonalNormal):
        raise NotImplementedError()
    second_moment = (q_z.variance + q_z.mean**2).mean(dim=0, keepdim=True)
    first_moment = q_z.mean.mean(dim=0, keepdim=True)
    variance = second_moment - first_moment**2
    stddev = torch.sqrt(variance)
    return MultivariateDiagonalNormal(loc=first_moment, scale_diag=stddev)


class PriorModel(DistributionGenerator):
    def __init__(self, size):
        super().__init__()
        self.mean = nn.Parameter(ptu.zeros(1, size))
        self.log_std = nn.Parameter(ptu.zeros(1, size))

    def initialize_values(self, q_z):
        second_moment = (q_z.variance + q_z.mean**2).mean(dim=0, keepdim=True)
        first_moment = q_z.mean.mean(dim=0, keepdim=True)
        variance = second_moment - first_moment**2
        stddev = torch.sqrt(variance)
        self.mean.data = first_moment
        self.log_std.data = torch.log(stddev)

    def forward(self, *ignored):
        if len(ignored) and len(ignored[0].shape) > 1 and ignored[0].shape[0] > 1:
            batch_size = ignored[0].shape[0]
            distribution = MultivariateDiagonalNormal(
                loc=self.mean.repeat(batch_size, 1),
                scale_diag=self.log_std.exp().repeat(batch_size, 1),
            )
        else:
            distribution = MultivariateDiagonalNormal(
                loc=self.mean,
                scale_diag=self.log_std.exp(),
            )
        return distribution


class FixedPriorModel(DistributionGenerator):
    def __init__(self, size):
        super().__init__()
        self.mean = ptu.zeros(1, size)
        self.log_std = ptu.zeros(1, size)

    def forward(self, *ignored):
        if len(ignored) and len(ignored[0].shape) > 1 and ignored[0].shape[0] > 1:
            batch_size = ignored[0].shape[0]
            distribution = MultivariateDiagonalNormal(
                loc=self.mean.repeat(batch_size, 1),
                scale_diag=self.log_std.exp().repeat(batch_size, 1),
            )
        else:
            distribution = MultivariateDiagonalNormal(
                loc=self.mean,
                scale_diag=self.log_std.exp(),
            )
        return distribution


class SetVAETrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            vae: SetVAE,
            vae_lr=1e-3,
            beta_z=1,
            beta_c=1,
            beta_z_scale_schedule_kwargs=None,
            beta_c_scale_schedule_kwargs=None,
            set_loss_weight=1,
            set_loss_weight_schedule_kwargs=None,
            loss_scale=1.0,
            vae_visualization_config=None,
            optimizer_class=optim.Adam,
            set_key='set',
            set_index_key='set_index',
            data_key='raw_next_observations',
            train_sets: Optional[List[torch.Tensor]] = None,
            eval_sets: Optional[List[torch.Tensor]] = None,
            debug_bad_recons=False,
            extra_log_fn=None,
            set_loss_version='kl',
            discriminators: Optional[List[nn.Module]] = None,
            prior_models: Optional[List[PriorModel]] = None,
            num_discriminator_samples=1,
            set_id_loss_weight=0.,
            train_prior_models=None,
            swae_kwargs=None,
            mmd_kwargs=None,
            init_prior_models_first_loop=True,
            debug_batch_size=32,
            estimate_set_nll_kwargs=None,
    ):
        super().__init__()
        if estimate_set_nll_kwargs is None:
            estimate_set_nll_kwargs = {}
        if beta_z_scale_schedule_kwargs is None:
            beta_z_scale_schedule_kwargs = {}
        if beta_c_scale_schedule_kwargs is None:
            beta_c_scale_schedule_kwargs = {}
        if swae_kwargs is None:
            swae_kwargs = {}
        if mmd_kwargs is None:
            mmd_kwargs = {}
        if set_loss_weight_schedule_kwargs is None:
            set_loss_weight_schedule_kwargs = {}
        if set_loss_version not in {
            'kl',
            'reverse_kl',
            'discriminator',
            'swae',
            'mmd',
        }:
            raise NotImplementedError(set_loss_version)
        if train_prior_models is None:
            train_prior_models = set_loss_version in {'discriminator', 'swae', 'mmd'}
        self.vae = vae
        self._base_beta_z = beta_z
        self._beta_z_scale_schedule = ml_util.create_schedule(
            **beta_z_scale_schedule_kwargs
        ) or ml_util.ConstantSchedule(1)

        self._base_beta_c = beta_c
        self._beta_c_scale_schedule = ml_util.create_schedule(
            **beta_c_scale_schedule_kwargs
        ) or ml_util.ConstantSchedule(1)

        self._set_loss_weight = set_loss_weight
        self.set_loss_weight_schedule = ml_util.create_schedule(
            **set_loss_weight_schedule_kwargs
        )
        self.set_loss_version = set_loss_version
        if train_prior_models:
            self.vae_optimizer = optimizer_class(
                chain(self.vae.parameters(), *(p.parameters() for p in prior_models)),
                lr=vae_lr,
            )
        else:
            self.vae_optimizer = optimizer_class(
                self.vae.parameters(),
                lr=vae_lr,
            )
        self.train_prior_models = train_prior_models
        self._need_to_update_eval_statistics = True
        self.loss_scale = loss_scale
        self.eval_statistics = OrderedDict()
        self.set_key = set_key
        self.set_index_key = set_index_key
        self.data_key = data_key  # TODO: get rid of this since it's unused
        self.train_sets = train_sets
        self.eval_sets = eval_sets
        self.debug_bad_recons = debug_bad_recons
        self.debug_batch_size = debug_batch_size
        self.extra_log_fn = extra_log_fn
        self.estimate_set_nll_kwargs = estimate_set_nll_kwargs

        self.vae_visualization_config = vae_visualization_config
        if not self.vae_visualization_config:
            self.vae_visualization_config = {}

        self.discriminators = discriminators
        self.num_discriminator_samples = num_discriminator_samples
        self.z_dim = vae.z_dim
        self.discriminator_loss_fn = nn.BCEWithLogitsLoss()
        self.example_batch = {}
        self._iteration = 0
        self._num_train_batches = 0
        self.swae_kwargs = swae_kwargs
        self.mmd_kwargs = mmd_kwargs
        self.init_prior_models_first_loop = init_prior_models_first_loop

        self.set_id_loss_weight = set_id_loss_weight
        self.set_identifier_model = nn.Linear(self.z_dim, len(self.train_sets))
        self.set_identifier_optimizer = optimizer_class(
            self.set_identifier_model.parameters(),
            lr=vae_lr,
        )
        self.set_identifier_loss = nn.CrossEntropyLoss()

    @property
    def _beta_z(self):
        return self._base_beta_z * self._beta_z_scale_schedule.get_value(
            self._iteration)

    @property
    def _beta_c(self):
        return self._base_beta_c * self._beta_c_scale_schedule.get_value(
            self._iteration)

    @property
    def set_loss_weight(self):
        if self.set_loss_weight_schedule is None:
            return self._set_loss_weight
        else:
            return self.set_loss_weight_schedule.get_value(
                self._iteration
            )

    def train_from_torch(self, batch):
        timer.start_timer('vae training', unique=False)
        self.vae.train()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )

        self.set_identifier_optimizer.zero_grad()

        self.vae_optimizer.zero_grad()
        losses.set_vae_loss.backward()
        self.vae_optimizer.step()
        self.set_identifier_optimizer.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False
            self.example_batch = batch
            self.eval_statistics['num_train_batches'] = self._num_train_batches
        self._num_train_batches += 1
        timer.stop_timer('vae training')

    def compute_loss(
            self,
            batch,
            skip_statistics=False
    ) -> Tuple[SVAELosses, LossStatistics]:
        vae_terms = self.vae.compute_set_vae_terms(batch[self.set_key])
        total_vae_loss = (
                - vae_terms.likelihood
                + self._beta_z * vae_terms.kl_z
                + self._beta_c * vae_terms.kl_c
        )
        losses = SVAELosses(set_vae_loss=total_vae_loss)

        eval_statistics = OrderedDict()
        if not skip_statistics:
            if self.extra_log_fn:
                eval_statistics.update(self.extra_log_fn(self))
            eval_statistics['log_likelihood'] = np.mean(ptu.get_numpy(
                vae_terms.likelihood
            ))
            eval_statistics['kl_z'] = np.mean(ptu.get_numpy(
                vae_terms.kl_z
            ))
            eval_statistics['kl_c'] = np.mean(ptu.get_numpy(
                vae_terms.kl_c
            ))
            eval_statistics['loss'] = np.mean(ptu.get_numpy(
                total_vae_loss
            ))
            eval_statistics['beta_z'] = self._beta_z
            eval_statistics['beta_c'] = self._beta_c
            eval_statistics['set_loss_weight'] = self.set_loss_weight
            for k, v in vae_terms.p_x_given_c_z.get_diagnostics().items():
                eval_statistics['p_x_given_c_z/{}'.format(k)] = v
            for k, v in vae_terms.q_z.get_diagnostics().items():
                eval_statistics['q_z_given_c_x/{}'.format(k)] = v
            for k, v in vae_terms.q_c.get_diagnostics().items():
                eval_statistics['q_c_given_x/{}'.format(k)] = v
            for name, set_list in [
                ('train', self.train_sets),
                ('eval', self.eval_sets),
            ]:
                for set_i, full_set in enumerate(set_list):
                    indices = np.random.randint(0, len(full_set), size=self.debug_batch_size)
                    set = full_set[indices]
                    vae_terms = self.vae.compute_set_vae_terms(set)
                    likelihood = vae_terms.likelihood
                    log_likelihood = np.mean(ptu.get_numpy(likelihood))
                    eval_statistics['{}/set{}/log_likelihood'.format(name, set_i)] = (
                        log_likelihood
                    )
                    nll_stats = self._compute_set_nll_stats(full_set, **self.estimate_set_nll_kwargs)
                    for k, v in nll_stats.items():
                        eval_statistics.update(
                            create_stats_ordered_dict(
                                '{}/set{}/{}'.format(name, set_i, k),
                                ptu.get_numpy(v),
                            )
                        )
                    if self.debug_bad_recons:
                        if log_likelihood < 0 and self._iteration > 40:
                            self.save_reconstruction(
                                'debug_bad_recons_iter{}'.format(self._iteration),
                                batch[self.data_key],
                            )
                        elif self._iteration % 50 == 0:
                            self.save_reconstruction(
                                'debug_good_recons_iter{}'.format(self._iteration),
                                batch[self.data_key],
                            )

                    eval_statistics['{}/set{}/kl_z'.format(name, set_i)] = np.mean(
                        ptu.get_numpy(vae_terms.kl_z))
                    eval_statistics['{}/set{}/kl_c'.format(name, set_i)] = np.mean(
                        ptu.get_numpy(vae_terms.kl_c))
                    set_prior = self.vae.encoder_c(set)
                    eval_statistics.update(
                        create_stats_ordered_dict(
                            '{}/set{}/learned_prior/mean'.format(name, set_i),
                            ptu.get_numpy(set_prior.mean)
                        )
                    )
                    eval_statistics.update(
                        create_stats_ordered_dict(
                            '{}/set{}/learned_prior/stddev'.format(name, set_i),
                            ptu.get_numpy(set_prior.stddev)
                        )
                    )
                    for k, v in vae_terms.p_x_given_c_z.get_diagnostics().items():
                        eval_statistics['{}/set{}/p_x_given_c_z/{}'.format(
                            name, set_i, k)] = v
                    for k, v in vae_terms.q_z.get_diagnostics().items():
                        eval_statistics['{}/set{}/q_z_given_x/{}'.format(
                            name, set_i, k)] = v

        return losses, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.dump_debug_images(epoch, **self.vae_visualization_config)
        self._iteration = epoch  # TODO: rename to iteration?

    def _compute_set_nll_stats(
            self,
            set,
            num_z_samples_list=None,
            infer_c_batch_size=None,
            eval_batch_size=None,
    ):
        if num_z_samples_list is None:
            num_z_samples_list = [1]
        if infer_c_batch_size and eval_batch_size:
            if infer_c_batch_size + eval_batch_size < len(set):
                idxs = np.random.choice(
                    np.arange(len(set)),
                    size=infer_c_batch_size + eval_batch_size,
                    replace=False,
                )
                infer_idxs = idxs[:infer_c_batch_size]
                eval_idxs = idxs[infer_c_batch_size:]
            else:
                infer_idxs = np.random.randint(0, len(set), size=infer_c_batch_size)
                eval_idxs = np.random.randint(0, len(set), size=eval_batch_size)
        elif infer_c_batch_size:
            infer_idxs = np.random.randint(0, len(set), size=infer_c_batch_size)
            eval_idxs = np.arange(len(set))
        elif eval_batch_size:
            infer_idxs = np.arange(len(set))
            eval_idxs = np.random.randint(0, len(set), size=eval_batch_size)
        else:
            infer_idxs = np.arange(len(set))
            eval_idxs = np.arange(len(set))

        infer_c_set = set[infer_idxs]
        eval_set = set[eval_idxs]

        nlls, nlls_biased, importance_weights = [], [], []
        stats = OrderedDict()
        max_num_z_samples = max(num_z_samples_list)
        for _ in range(max_num_z_samples):
            p_c = self.vae.encoder_c(infer_c_set)
            c = p_c.mean
            c_batched = c.repeat(1, 1, 1)
            p_z_given_c = self.vae.prior_z_given_c(c)

            q_z_given_c_x = self.vae.encoder_z(
                c.repeat(eval_set.shape[0], 1), eval_set)
            z_iwae = q_z_given_c_x.sample([1])
            p_x_iwae = self.vae.p_x_given_c_z(c_batched, z_iwae)
            importance_weight = (
                p_z_given_c.log_prob(z_iwae) - q_z_given_c_x.log_prob(z_iwae)
            ).exp()
            nll_biased = - p_x_iwae.log_prob(eval_set)

            z = p_z_given_c.sample([1])
            p_x = self.vae.p_x_given_c_z(c_batched, z)
            nll = - p_x.log_prob(eval_set)

            nlls.append(nll.detach())
            importance_weights.append(importance_weight.detach())
            nlls_biased.append(nll_biased.detach())

            num_z_samples = len(nlls)
            if num_z_samples in num_z_samples_list:
                nll_so_far = torch.cat(nlls, dim=0)
                nll_biased_so_far = torch.cat(nlls_biased, dim=0)
                iw_so_far = torch.cat(importance_weights, dim=0)
                nll_is_so_far = (
                    (nll_biased_so_far * iw_so_far).sum(dim=0)
                    / iw_so_far.sum(dim=0)
                )
                # log it!
                if num_z_samples == 1:
                    stats.update(OrderedDict([
                        ('nll/z{}/mc'.format(num_z_samples), nll_so_far),
                    ]))
                else:
                    stats.update(OrderedDict([
                        ('nll/z{}/matching/mc'.format(num_z_samples), nll_so_far.min(dim=0)[0]),
                        ('nll/z{}/mc'.format(num_z_samples), nll_so_far),
                        ('nll/z{}/is'.format(num_z_samples), nll_is_so_far),
                        ('nll/is_weights'.format(num_z_samples), importance_weight),
                    ]))
        return stats

    def save_reconstruction(
            self,
            name,
            batch,
            num_recons=None,
            image_format='CHW',
            unnormalize_images=True,
    ):
        logdir = logger.get_snapshot_dir()
        example_batch = ptu.get_numpy(batch)
        recon_examples_np = ptu.get_numpy(self.vae.reconstruct(batch))
        recon_examples_np = np.clip(recon_examples_np, 0, 1)

        if num_recons is None:
            top_row_example = example_batch
            bottom_row_recon = recon_examples_np
        else:
            top_row_example = example_batch[:num_recons]
            bottom_row_recon = recon_examples_np[:num_recons]

        # interleave images
        all_imgs = list(chain.from_iterable(
            zip(top_row_example, bottom_row_recon))
        )
        save_imgs(
            # imgs=list(top_row_example) + list(bottom_row_recon),
            imgs=all_imgs,
            file_path=osp.join(logdir, '{}.png'.format(name)),
            unnormalize=unnormalize_images,
            max_num_cols=min(len(top_row_example), 16),
            image_format=image_format,
        )

    # TODO: move this to launcher rather than in trainer
    def dump_debug_images(
            self,
            epoch,
            dump_images=True,
            num_recons=10,
            num_samples=25,
            debug_period=10,
            unnormalize_images=True,
            image_format='CHW',
            save_reconstructions=True,
            save_vae_samples=True,
            visualize_latent_space=False,
            visualize_c_latent_space=False,
    ):
        """

        :param epoch:
        :param dump_images: Set to False to not dump any images.
        :param num_recons:
        :param num_samples:
        :param debug_period: How often do you dump debug images?
        :param unnormalize_images: Should your unnormalize images before
            dumping them? Set to True if images are floats in [0, 1].
        :return:
        """
        self.vae.eval()
        if (not dump_images
                or debug_period <= 0
                or epoch % debug_period != 0
        ):
            return
        logdir = logger.get_snapshot_dir()

        def save_reconstruction(name, batch):
            top_row_example = ptu.get_numpy(batch)
            bottom_row_recon = ptu.get_numpy(self.vae.reconstruct(batch))

            save_imgs(
                imgs=list(top_row_example) + list(bottom_row_recon),
                file_path=osp.join(logdir, '{}_{}.png'.format(name, epoch)),
                unnormalize=unnormalize_images,
                max_num_cols=len(top_row_example),
                image_format=image_format,
            )

        # batch = self.example_batch[self.data_key]
        # if save_reconstructions:
        #     save_reconstruction('recon', batch[:num_recons])

        if save_vae_samples:
            raw_samples = ptu.get_numpy(self.vae.sample(num_samples))
            save_imgs(
                imgs=raw_samples,
                file_path=osp.join(logdir, 'vae_samples_{}.png'.format(epoch)),
                unnormalize=unnormalize_images,
                image_format=image_format,
            )

        for name, list_of_sets in [
            ('train', self.train_sets),
            ('eval', self.eval_sets),
        ]:
            for set_i, full_set in enumerate(list_of_sets):
                indices = np.random.randint(
                    0, len(full_set), size=num_recons//2)
                set = torch.cat(
                    (
                        full_set[:num_recons // 2],
                        full_set[indices],
                    ),
                    dim=0,
                )
                if save_reconstructions:
                    save_reconstruction('recon_{}_set{}'.format(name, set_i), set)

                if save_vae_samples:
                    set_samples = self.vae.set_sample(num_samples, set)
                    save_imgs(
                        imgs=ptu.get_numpy(set_samples),
                        file_path=osp.join(
                            logdir,
                            'vae_samples_{name}_set{set_i}_{epoch}.png'.format(
                                epoch=epoch, name=name, set_i=set_i,
                            ),
                        ),
                        unnormalize=unnormalize_images,
                        image_format=image_format,
                    )

            if visualize_latent_space:
                self.visualize_latent_space(list_of_sets, epoch, name)
            if visualize_c_latent_space:
                self.visualize_c_latent_space(list_of_sets, epoch, name)

    def visualize_c_latent_space(self, list_of_sets, epoch, name):
        ncols = self.vae.c_dim // 2
        fig, axes = plt.subplots(
            nrows=1, ncols=ncols,
            figsize=(2 * ncols, 2),
            squeeze=False,
        )
        axes = axes[0]
        p_c = self.vae.prior_c()
        q_cs = [self.vae.encoder_c(s) for s in list_of_sets]

        prior_mean = ptu.get_numpy(p_c.mean)[0]
        prior_std = ptu.get_numpy(p_c.stddev)[0]
        posterior_means = ptu.get_numpy(
            torch.cat([q_c.mean for q_c in q_cs], dim=0)
        )
        posterior_stds = ptu.get_numpy(
            torch.cat([q_c.stddev for q_c in q_cs], dim=0)
        )
        for ax_i, ax in enumerate(axes):
            mean_slc = posterior_means[:, 2 * ax_i:2 * ax_i + 2]
            std_slc = posterior_stds[:, 2 * ax_i:2 * ax_i + 2]
            prior_mean_slc = prior_mean[2 * ax_i:2 * ax_i + 2]
            prior_std_slc = prior_std[2 * ax_i:2 * ax_i + 2]
            for mean, std in zip(mean_slc, std_slc):
                e = Ellipse(
                    mean,
                    width=2 * std[0],
                    height=2 * std[1],
                    alpha=0.4,
                    color=ELLIPSE_COLORS[0],
                )
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
            ax.scatter(mean_slc[:, 0], mean_slc[:, 1], s=15,
                       color=MEAN_COLORS[0])

            prior_e = Ellipse(
                prior_mean_slc,
                width=2 * prior_std_slc[0],
                height=2 * prior_std_slc[1],
                edgecolor=PRIOR_COLORS[0],
                lw=4,
                facecolor='none',
                alpha=0.8
            )
            ax.add_artist(prior_e)
            prior_e.set_clip_box(ax.bbox)

            max_x = max(
                (mean_slc[:, 0] + std_slc[:, 0]).max(),
                (prior_mean_slc[0] + prior_std_slc[0]).max()
            )
            max_y = max(
                (mean_slc[:, 1] + std_slc[:, 1]).max(),
                (prior_mean_slc[1] + prior_std_slc[1]).max()
            )
            min_x = min(
                (mean_slc[:, 0] - std_slc[:, 0]).min(),
                (prior_mean_slc[0] - prior_std_slc[0]).min()
            )
            min_y = min(
                (mean_slc[:, 1] - std_slc[:, 1]).min(),
                (prior_mean_slc[1] - prior_std_slc[1]).min()
            )
            ax.set_xlim((min_x, max_x))
            ax.set_ylim((min_y, max_y))

        logdir = logger.get_snapshot_dir()
        save_path = osp.join(
            logdir,
            'c_latent_visualization_{}_{}.png'.format(
                name, epoch,
            )
        )
        fig.savefig(save_path)
        plt.close(fig)

    def visualize_latent_space(self, list_of_sets, epoch, name):
        ncols = self.vae.z_dim // 2
        nrows = len(list_of_sets)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(2 * ncols, 2 * nrows),
            squeeze=False,
        )
        for set_i, (set, set_axes) in enumerate(zip(list_of_sets, axes)):
        # fig, set_axes = plt.subplots(
        #     nrows=1, ncols=ncols,
        #     figsize=(2 * ncols, 2),
        #     squeeze=False,
        # )
        # set_axes = set_axes[0]
        # for set_i, set in enumerate(list_of_sets):
            if len(set) > 1024:
                set = set[np.random.randint(0, len(set), size=1024)]
            p_z, q_z = self.vae.latent_prior_and_posterior(set)
            # if self.train_prior_models:
            #     set_prior = self.prior_models[set_i]()
            # else:
            #     set_prior = compute_prior(q_z)
            prior_mean = ptu.get_numpy(p_z.mean)
            prior_std = ptu.get_numpy(p_z.stddev)
            if len(prior_mean.shape) == 2:
                prior_mean = prior_mean[0]
                prior_std = prior_std[0]

            fitted_prior = compute_prior(q_z)
            fitted_prior_mean = ptu.get_numpy(fitted_prior.mean)
            fitted_prior_std = ptu.get_numpy(fitted_prior.stddev)
            if len(fitted_prior_mean.shape) == 2:
                fitted_prior_mean = fitted_prior_mean[0]
                fitted_prior_std = fitted_prior_std[0]

            posterior_means = ptu.get_numpy(q_z.mean)
            posterior_stds = ptu.get_numpy(q_z.stddev)
            for ax_i, ax in enumerate(set_axes):
                mean_slc = posterior_means[:, 2 * ax_i:2 * ax_i + 2]
                std_slc = posterior_stds[:, 2 * ax_i:2 * ax_i + 2]
                for mean, std in zip(mean_slc, std_slc):
                    e = Ellipse(
                        mean,
                        width=2 * std[0],
                        height=2 * std[1],
                        alpha=0.1,
                        color=ELLIPSE_COLORS[0],
                    )
                    ax.add_artist(e)
                    e.set_clip_box(ax.bbox)
                ax.scatter(mean_slc[:, 0], mean_slc[:, 1], s=15,
                           color=MEAN_COLORS[0])

                prior_mean_slc = prior_mean[2 * ax_i:2 * ax_i + 2]
                prior_std_slc = prior_std[2 * ax_i:2 * ax_i + 2]
                prior_e = Ellipse(
                    prior_mean_slc,
                    width=2 * prior_std_slc[0],
                    height=2 * prior_std_slc[1],
                    edgecolor=PRIOR_COLORS[0],
                    lw=4,
                    facecolor='none',
                    alpha=0.8
                )

                ax.add_artist(prior_e)
                prior_e.set_clip_box(ax.bbox)

                fitted_prior_mean_slc = fitted_prior_mean[2 * ax_i:2 * ax_i + 2]
                fitted_prior_std_slc = fitted_prior_std[2 * ax_i:2 * ax_i + 2]
                fitted_prior_e = Ellipse(
                    fitted_prior_mean_slc,
                    width=2 * fitted_prior_std_slc[0],
                    height=2 * fitted_prior_std_slc[1],
                    edgecolor=PRIOR_COLORS[1],
                    lw=4,
                    facecolor='none',
                    alpha=0.8
                )
                ax.add_artist(fitted_prior_e)
                fitted_prior_e.set_clip_box(ax.bbox)

                max_x = max(
                    (mean_slc[:, 0] + std_slc[:, 0]).max(),
                    (prior_mean_slc[0] + prior_std_slc[0]).max(),
                    (fitted_prior_mean_slc[0] + fitted_prior_std_slc[0]).max(),
                )
                max_y = max(
                    (mean_slc[:, 1] + std_slc[:, 1]).max(),
                    (prior_mean_slc[1] + prior_std_slc[1]).max(),
                    (fitted_prior_mean_slc[1] + fitted_prior_std_slc[1]).max(),
                )
                min_x = min(
                    (mean_slc[:, 0] - std_slc[:, 0]).min(),
                    (prior_mean_slc[0] - prior_std_slc[0]).min(),
                    (fitted_prior_mean_slc[0] - fitted_prior_std_slc[0]).min(),
                )
                min_y = min(
                    (mean_slc[:, 1] - std_slc[:, 1]).min(),
                    (prior_mean_slc[1] - prior_std_slc[1]).min(),
                    (fitted_prior_mean_slc[1] - fitted_prior_std_slc[1]).min(),
                )
                ax.set_xlim((min_x, max_x))
                ax.set_ylim((min_y, max_y))

        logdir = logger.get_snapshot_dir()
        save_path = osp.join(
            logdir,
            'latent_visualization_{}_{}.png'.format(
                name, epoch,
            )
        )
        fig.savefig(save_path)
        plt.close(fig)

    @property
    def networks(self):
        nets = [
            self.vae,
            self.set_identifier_model,
        ]
        if self.discriminators:
            nets += self.discriminators
        return nets

    @property
    def optimizers(self):
        return [
            self.vae_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            vae=self.vae,
            train_sets=self.train_sets,
            eval_sets=self.eval_sets,
        )


def save_imgs(imgs, file_path, **kwargs):
    imwidth = imgs[0].shape[1]
    imheight = imgs[0].shape[2]
    imgs = np.clip(imgs, 0, 1)
    combined_img = combine_images_into_grid(
        imgs=list(imgs),
        imwidth=imwidth,
        imheight=imheight,
        **kwargs
    )
    cv2_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, cv2_img)


class CustomDictLoader(object):
    r"""Method for iterating over dictionaries."""

    def __init__(
            self,
            data,
            sets,
            data_key='data',
            set_key='set',
            set_index_key='set_index',
            batch_size=32,
    ):
        self.data = data
        self.sets = sets
        self.data_key = data_key
        self.set_key = set_key
        self.set_index_key = set_index_key
        self.batch_size = batch_size
        # TODO: maybe just reuse the built in pytorch data loaders?
        self.set_i_to_shuffled_indices = [
            np.random.shuffle(np.arange(len(s)))
            for k, s in enumerate(self.sets)
        ]

    def __iter__(self):
        for i, datum in enumerate(self.data):
            set_i = i % len(self.sets)
            full_set = self.sets[set_i]
            if self.batch_size:
                # all_indices = self.set_i_to_shuffled_indices[set_i]
                # end_i = min((i+1) * self.batch_size, len(all_indices))
                # indices = all_indices[i*self.batch_size:(i+1) * self.batch_size]
                indices = np.random.randint(0, len(full_set), size=self.batch_size)
                set = full_set[indices]
            else:
                set = full_set
            yield {
                # self.data_key: datum,
                self.set_key: set,
                self.set_index_key: set_i,
            }
        self.set_i_to_shuffled_indices = [
            np.random.shuffle(np.arange(len(s)))
            for k, s in enumerate(self.sets)
        ]

    def __len__(self):
        return len(self.data)