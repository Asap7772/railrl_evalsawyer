from collections import OrderedDict
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sets import set_vae_trainer as svt
from rlkit.torch.sets import models
from rlkit.torch.sets.discriminator import (
    DiscriminatorDataset,
    DiscriminatorTrainer,
)
from rlkit.torch.sets.set_vae_trainer import PriorModel, CustomDictLoader
from rlkit.torch.sets.batch_algorithm import (
    BatchTorchAlgorithm,
)
from rlkit.torch.sets.parallel_algorithms import ParallelAlgorithms
from torch.utils import data
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.vae_torch_trainer import VAE


def create_circle_dataset(num_examples, radius=3, scale=0.5, origin=(0, 0)):
    angle = np.random.uniform(size=(num_examples, 1)) * 2 * np.pi
    r = scale * np.random.randn(num_examples, 1) + radius
    y = r * np.sin(angle) + origin[1]
    x = r * np.cos(angle) + origin[0]
    return np.concatenate([x, y], axis=1)


def create_box_dataset(num_examples, xlim, ylim):
    x = np.random.uniform(xlim[0], xlim[1], size=(num_examples, 1))
    y = np.random.uniform(ylim[0], ylim[1], size=(num_examples, 1))
    return np.concatenate([x, y], axis=1)


def create_datasets(create_set_kwargs_list=None):
    if create_set_kwargs_list is None:
        create_set_kwargs_list = [
            dict(num_examples=128, version='circle'),
            dict(num_examples=128, version='box', xlim=(0, 2), ylim=(0, 2)),
            dict(num_examples=128, version='box', xlim=(-2, 0), ylim=(-2, 0)),
            dict(num_examples=128, version='box', xlim=(0, 2), ylim=(-2, 0)),
            dict(num_examples=128, version='box', xlim=(-2, 2), ylim=(0, 2)),
        ]
    return np.array([
        create_set(**kwargs) for kwargs in create_set_kwargs_list
    ])


def create_set(version, **kwargs):
    if version == 'circle':
        return create_circle_dataset(**kwargs)
    elif version == 'box':
        return create_box_dataset(**kwargs)
    else:
        raise NotImplementedError()


def setup_discriminator(
        vae: VAE,
        examples,
        prior,
        discriminator_kwargs=None,
        dataset_kwargs=None,
        trainer_kwargs=None,
        algo_kwargs=None,
        name='',
):
    if discriminator_kwargs is None:
        discriminator_kwargs = {}
    if dataset_kwargs is None:
        dataset_kwargs = {}
    if trainer_kwargs is None:
        trainer_kwargs = {}
    if algo_kwargs is None:
        algo_kwargs = {}
    discriminator = ConcatMlp(
        input_size=vae.representation_size,
        output_size=1,
        **discriminator_kwargs
    )
    discriminator_data_loader = DiscriminatorDataset(
        vae, examples, prior, **dataset_kwargs)
    discriminator_trainer = DiscriminatorTrainer(
        discriminator,
        prior,
        name=name,
        **trainer_kwargs,
    )
    discriminator_algo = BatchTorchAlgorithm(
        discriminator_trainer,
        discriminator_data_loader,
        **algo_kwargs
    )
    return discriminator_algo, discriminator, prior


def train_2d_set_vae(
        create_set_vae_kwargs,
        vae_trainer_kwargs,
        vae_algo_kwargs,
        debug_kwargs,
        num_iters,
        x_depends_on_c=False,
        vae_data_loader_kwargs=None,
        create_train_dataset_kwargs=None,
        create_eval_dataset_kwargs=None,
        setup_discriminator_kwargs=None,
        set_dict_loader_kwargs=None,
):
    if set_dict_loader_kwargs is None:
        set_dict_loader_kwargs = {}
    if vae_data_loader_kwargs is None:
        vae_data_loader_kwargs = {}
    if setup_discriminator_kwargs is None:
        setup_discriminator_kwargs = {}
    if create_eval_dataset_kwargs is None:
        create_eval_dataset_kwargs = create_train_dataset_kwargs
    data_dim = 2
    eval_sets = create_datasets(**create_eval_dataset_kwargs)
    train_sets = create_datasets(**create_train_dataset_kwargs)
    for set_ in train_sets:
        plt.scatter(*set_.T)
    all_obs = np.vstack(train_sets)

    # vae = models.create_vector_vae(
        # data_dim=data_dim,
        # **create_vae_kwargs,
    # )
    vae = models.create_vector_set_vae(
        data_dim=data_dim,
        x_depends_on_c=x_depends_on_c,
        **create_set_vae_kwargs,
    )
    data_key = 'data'
    set_key = 'set'
    set_index_key = 'set_index'

    train_sets_pt = [ptu.from_numpy(s) for s in train_sets]
    eval_sets_pt = [ptu.from_numpy(s) for s in eval_sets]
    all_obs_pt = ptu.from_numpy(all_obs)
    all_obs_iterator_pt = data.DataLoader(all_obs_pt, **vae_data_loader_kwargs)
    dict_loader = CustomDictLoader(
        data=all_obs_iterator_pt,
        sets=train_sets_pt,
        data_key=data_key,
        set_key=set_key,
        set_index_key=set_index_key,
        **set_dict_loader_kwargs
    )

    algos = OrderedDict()
    discriminator_algos = []
    discriminators = []
    if setup_discriminator_kwargs:
        prior_models = [PriorModel(vae.representation_size) for _ in train_sets_pt]
        for i, examples in enumerate(train_sets_pt):
            discriminator_algo, discriminator, prior_m = setup_discriminator(
                vae,
                examples,
                prior_models[i],
                name='discriminator{}'.format(i),
                **setup_discriminator_kwargs
            )
            discriminator_algos.append(discriminator_algo)
            discriminators.append(discriminator)
    else:
        prior_models = None
    vae_trainer = svt.SetVAETrainer(
        vae=vae,
        set_key=set_key,
        data_key=data_key,
        train_sets=train_sets_pt,
        eval_sets=eval_sets_pt,
        prior_models=prior_models,
        discriminators=discriminators,
        **vae_trainer_kwargs)
    vae_algorithm = BatchTorchAlgorithm(
        vae_trainer,
        dict_loader,
        **vae_algo_kwargs,
    )
    algos['vae'] = vae_algorithm
    for i, algo in enumerate(discriminator_algos):
        algos['discriminator_{}'.format(i)] = algo
    algorithm = ParallelAlgorithms(algos, num_iters)
    algorithm.to(ptu.device)

    set_up_debugging(vae_algorithm, prior_models, discriminator_algos, **debug_kwargs)

    algorithm.run()


def set_up_debugging(
        vae_algorithm,
        prior_models,
        discriminator_algos,
        debug_period=10,
        num_samples=25,
        dump_posterior_and_prior_samples=False,
):
    from rlkit.core import logger
    logdir = logger.get_snapshot_dir()
    set_loss_version = vae_algorithm.trainer.set_loss_version

    # visualize the train/eval set once
    plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    xmin = xmax = ymin = ymax = 0
    for name, list_of_sets in [
        ('train', vae_algorithm.trainer.train_sets),
        ('eval', vae_algorithm.trainer.eval_sets),
    ]:
        plt.figure()
        for i, set in enumerate(list_of_sets):
            set_examples = ptu.get_numpy(set)
            plt.scatter(*set_examples.T, color=plt_colors[i])

        xmin, xmax, ymin, ymax = plt.axis()
        plt.savefig(osp.join(logdir, '{}_set_visualization.png'.format(name)))
        plt.close()

    def dump_debug_images(
            algo,
            epoch,
            tag='',
    ):
        trainer = algo.trainer
        trainer.vae.train()
        if debug_period <= 0 or epoch % debug_period != 0:
            return

        def draw_reconstruction(batch, color=None):
            x_np = ptu.get_numpy(batch)
            x_hat_np = ptu.get_numpy(trainer.vae.reconstruct(batch))
            delta = x_hat_np - x_np
            plt.quiver(
                x_np[:, 0],
                x_np[:, 1],
                delta[:, 0],
                delta[:, 1],
                scale=1.,
                scale_units='xy',
                linewidth=0.5,
                alpha=0.5,
                color=color,
            )

        # batch = trainer.example_batch[trainer.data_key]
        # plt.figure()
        # draw_reconstruction(batch)
        # plt.savefig(osp.join(logdir, '{}_recon.png'.format(epoch)))
        #
        raw_samples = ptu.get_numpy(trainer.vae.sample(num_samples))
        plt.figure()
        plt.scatter(*raw_samples.T)
        plt.title('samples, epoch {}'.format(epoch))
        plt.savefig(osp.join(logdir, 'vae_samples_{epoch}.png'.format(
            epoch=epoch)))
        plt.close()

        for prefix, list_of_sets in [
            ('eval', trainer.eval_sets),
        ]:
            name = prefix + tag
            plt.figure()
            for i, set in enumerate(list_of_sets):
                draw_reconstruction(set, color=plt_colors[i])
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            plt.title('{}, epoch {}'.format(name, epoch))
            plt.savefig(
                osp.join(logdir, 'set_recons_{name}_{epoch}.png'.format(
                    epoch=epoch, name=name)))
            plt.close()
        for prefix, list_of_sets in [
            ('train', trainer.train_sets),
            ('eval', trainer.eval_sets),
        ]:
            name = prefix + tag
            for fix_xy_lims in [True, False]:
                plt.figure()
                for set_i, set in enumerate(list_of_sets):
                    set_samples = ptu.get_numpy(
                        trainer.vae.set_sample(num_samples, set))
                    plt.scatter(*set_samples.T, color=plt_colors[set_i])
                if fix_xy_lims:
                    plt.xlim((xmin, xmax))
                    plt.ylim((ymin, ymax))
                    file_name = 'set_vae_samples_fixed_axes_{name}_{epoch}.png'.format(
                        epoch=epoch, name=name,
                    )
                else:
                    file_name = 'set_vae_samples_{name}_{epoch}.png'.format(
                        epoch=epoch, name=name,
                    )
                plt.title('{}, epoch {}'.format(name, epoch))
                plt.savefig(osp.join(logdir, file_name))
                plt.close()

            plt.figure()
            for i, set in enumerate(list_of_sets):
                draw_reconstruction(set, color=plt_colors[i])
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            plt.title('{}, epoch {}'.format(name, epoch))
            plt.savefig(
                osp.join(logdir, 'set_recons_{name}_{epoch}.png'.format(
                    epoch=epoch, name=name)))
            plt.close()

    def dump_samples(
            algo,
            epoch,
    ):
        if debug_period <= 0 or epoch % debug_period != 0:
            return
        # visualize the train/eval set once
        data_loaders = [algo.data_loader for algo in discriminator_algos]

        def get_last_batch(dl):
            batch = None
            for batch in dl:
                pass
            return batch

        batches = [get_last_batch(dl) for dl in data_loaders]
        nrows = len(batches)
        ncols = algo.trainer.vae.representation_size // 2
        fig, list_of_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))
        for batch, axes in zip(batches, list_of_axes):
            # y = batch['y']
            x = batch['x']
            xnp = x.cpu().detach().numpy()
            posterior_samples = xnp[:128]
            prior_samples = xnp[128:]
            for i, ax in enumerate(axes):
                post_x = posterior_samples[:, 2*i]
                post_y = posterior_samples[:, 2*i + 1]
                prior_x = prior_samples[:, 2*i]
                prior_y = prior_samples[:, 2*i + 1]
                ax.scatter(post_x, post_y, color='r')
                ax.scatter(prior_x, prior_y, color='b')

        plt.title('{}, epoch {}'.format(name, epoch))
        plt.savefig(logdir + '/discriminator_samples_{epoch}.png'.format(
            epoch=epoch,
        ))
        plt.close()

    vae_algorithm.post_epoch_funcs.append(dump_debug_images)
    if dump_posterior_and_prior_samples:
        vae_algorithm.post_epoch_funcs.append(dump_samples)
    # if discriminator_algos:
    #     vae_algorithm.pre_train_funcs.append(
    #         functools.partial(dump_debug_images, tag='-pre-vae')
    #     )
