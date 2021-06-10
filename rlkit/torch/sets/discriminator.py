from collections.__init__ import OrderedDict
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch import autograd

from rlkit.core.loss import LossFunction
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.sets.set_vae_trainer import PriorModel
from rlkit.torch.sets.batch_algorithm import DictLoader
from rlkit.torch.supervised_learning.supervised_algorithm import LossStatistics
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.vae.vae_torch_trainer import VAE


class DiscriminatorDataset(DictLoader):
    """Generate data for the discriminator."""
    def __init__(
            self,
            vae: VAE,
            examples: torch.FloatTensor,
            prior_model: PriorModel,
            num_samples_per_class: int,
            batch_size=128,
            x_key='x',
            y_key='y',
    ):
        self.vae = vae
        self.prior_model = prior_model
        self.num_samples_per_class = num_samples_per_class
        self.z_dim = vae.representation_size
        self.batch_size = batch_size
        self.examples_iterator = data.DataLoader(
            examples, batch_size=batch_size
        )

        self.x_key = x_key
        self.y_key = y_key

    def __iter__(self):
        for examples in self.examples_iterator:
            prior = self.prior_model()
            q_z = self.vae.encoder(examples)
            posterior_samples = q_z.sample((self.num_samples_per_class,))
            posterior_samples = posterior_samples.view(-1, self.z_dim)
            num_latent_samples = posterior_samples.shape[0]
            prior_samples = prior.rsample((num_latent_samples,))
            prior_samples = prior_samples.view(-1, self.z_dim)
            y = torch.cat(
                (
                    ptu.zeros((num_latent_samples, 1)),
                    ptu.ones((num_latent_samples, 1)),
                ),
                dim=0,
            )
            x = torch.cat((posterior_samples, prior_samples), dim=0)
            # if x.max() > 10:
            #     import ipdb; ipdb.set_trace()
            # if x.min() < -10:
            #     import ipdb; ipdb.set_trace()
            yield {
                self.x_key: x,
                self.y_key: y,
            }

    def __len__(self):
        return len(self.examples_iterator)


_str_to_optimizer = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
}


class DiscriminatorTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            model: nn.Module,
            prior: PriorModel,
            lr=1e-3,
            prior_lr=1e-3,
            optimizer_class=optim.Adam,
            prior_optimizer_class=optim.Adam,
            x_key='x',
            y_key='y',
            gradient_penalty_weight=0.,
            target_gradient_norm=1.,
            weight_clip=0.,
            discriminator_steps_per_prior_steps=1,
            name='discriminator',
    ):
        super().__init__(track_num_train_steps=False)
        if isinstance(optimizer_class, str):
            optimizer_class = ptu.optimizer_from_string(optimizer_class)
        if isinstance(prior_optimizer_class, str):
            prior_optimizer_class = ptu.optimizer_from_string(prior_optimizer_class)
        self.model = model
        self.discriminator_optimizer = optimizer_class(
            self.model.parameters(), lr=lr)
        self.prior_optimizer = prior_optimizer_class(
            prior.parameters(), lr=prior_lr)

        self.example_batch = {}
        self._iteration = 0
        self._num_train_batches = 0
        self._need_to_update_eval_statistics = True
        self.x_key = x_key
        self.y_key = y_key
        self.gradient_penalty_weight = gradient_penalty_weight
        self.target_gradient_norm = target_gradient_norm
        self.weight_clip = weight_clip
        self.discriminator_steps_per_prior_steps = (
            discriminator_steps_per_prior_steps
        )
        self.sometimes_skip_prior_update = (
            self.discriminator_steps_per_prior_steps >= 1
        )
        self.prior_steps_per_discriminator_steps = int(
            1./self.discriminator_steps_per_prior_steps
        )

        self.discriminator_loss_fn = nn.BCEWithLogitsLoss()
        self.eval_statistics = OrderedDict()
        self.name = name

    def train_from_torch(self, batch):
        discriminator_loss, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )

        prior_loss = - discriminator_loss
        if self.sometimes_skip_prior_update:
            if (
                    self._num_train_batches
                    % self.discriminator_steps_per_prior_steps == 0
            ):
                self.prior_optimizer.zero_grad()
                prior_loss.backward(retain_graph=True)
                self.prior_optimizer.step()

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
        else:
            train_discrim = (
                    self._num_train_batches
                    % self.prior_steps_per_discriminator_steps == 0
            )
            self.prior_optimizer.zero_grad()
            prior_loss.backward(retain_graph=train_discrim)
            self.prior_optimizer.step()
            if train_discrim:
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

        if self.weight_clip > 0:
            for p in self.model.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False
            self.example_batch = batch
            self.eval_statistics['num_train_batches'] = self._num_train_batches
        self._num_train_batches += 1

    def compute_loss(
            self,
            batch,
            skip_statistics=False,
            **kwargs
    ) -> Tuple[torch.FloatTensor, LossStatistics]:
        y = batch[self.y_key]
        x = batch[self.x_key]
        # if not skip_statistics and self._iteration >= 4:
        #     if x.max() > 10:
        #         import ipdb; ipdb.set_trace()
        #     if x.min() < -10:
        #         import ipdb; ipdb.set_trace()
        #     epoch = self._iteration + 1
        #     import matplotlib.pyplot as plt
        #
        #     fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(2 * 8, 2))
        #     xnp = x.cpu().detach().numpy()
        #     pos = xnp[:128]
        #     neg = xnp[128:]
        #     for i, ax in enumerate(axes):
        #         pos_x = pos[:, 2*i]
        #         pos_y = pos[:, 2*i + 1]
        #         neg_x = neg[:, 2*i]
        #         neg_y = neg[:, 2*i + 1]
        #         ax.scatter(pos_x, pos_y, color='r')
        #         ax.scatter(neg_x, neg_y, color='b')
        #     from rlkit.core import logger
        #     logdir = logger.get_snapshot_dir()
        #     plt.savefig(logdir + '/samples_{}_{}.png'.format(epoch, self.name))
        logits = self.model(x)
        loss = self.discriminator_loss_fn(logits, y)

        if self.gradient_penalty_weight != 0.:
            gradient_penalty, grad_norms = calc_gradient_penalty(
                inputs=x, outputs=logits,
                target_grad_norm=self.target_gradient_norm,
            )
            loss = loss + self.gradient_penalty_weight * gradient_penalty

        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['loss'] = np.mean(ptu.get_numpy(
                loss
            ))
            y_hat = logits > 0
            accuracy = (y_hat.to(y.dtype) == y).to(torch.float).mean().item()
            positive_prob = y_hat.to(torch.float).mean().item()
            eval_statistics['accuracy'] = accuracy
            eval_statistics['positive_prob'] = positive_prob
            eval_statistics.update(
                create_stats_ordered_dict(
                    'logits',
                    ptu.get_numpy(logits)
                )
            )
            if self.gradient_penalty_weight != 0.:
                eval_statistics['gradient_penalty'] = gradient_penalty.item()
                eval_statistics.update(
                    create_stats_ordered_dict(
                        'gradient_norm',
                        ptu.get_numpy(grad_norms)
                    )
                )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self._iteration = epoch

    @property
    def networks(self):
        return [self.model]

    @property
    def optimizers(self):
        return [self.model]

    def get_snapshot(self):
        return dict(
            model=self.model
        )


def calc_gradient_penalty(inputs, outputs, target_grad_norm=1):
    """Based on https://github.com/caogang/wgan-gp/blob/master/gan_toy.py"""
    gradients = autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=ptu.ones(outputs.size()),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_norms = gradients.norm(2, dim=1)

    delta = torch.clamp_min(gradient_norms - target_grad_norm, target_grad_norm)
    grad_penalty = (delta**2).mean()

    return grad_penalty, gradient_norms
