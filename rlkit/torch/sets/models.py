import json

import numpy as np
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.launchers.experiments.disentanglement import (
    contextual_encoder_distance_launcher as cedl,
)
from rlkit.misc import asset_loader
from rlkit.torch.core import PyTorchModule
from rlkit.torch.distributions import MultivariateDiagonalNormal
from rlkit.torch.networks import (
    AverageSetConcatMultiHeadedMlp,
    BasicCNN,
    Flatten,
    Mlp,
    ConcatMultiHeadedMlp,
    Reshape,
)
from rlkit.torch.networks import basic
from rlkit.torch.networks.dcnn import BasicDCNN
from rlkit.torch.networks.mlp import MultiHeadedMlp, ConcatMlp
from rlkit.torch.networks.stochastic.distribution_generator import (
    BernoulliGenerator,
    Gaussian,
    IndependentGenerator,
)
from rlkit.torch.sets.fancy_vae_architecture import (
    get_fancy_vae,
    get_fancy_set_vae_networks,
)
from rlkit.torch.sets.set_vae_trainer import PriorModel, SetVAE, FixedPriorModel
from rlkit.torch.vae.vae_torch_trainer import VAE


class DummyNetwork(PyTorchModule):
    def __init__(self, *output_shapes):
        super().__init__()
        self._output_shapes = output_shapes
        # if len(output_shapes) == 1:
        #     self.output = ptu.zeros(output_shapes[0])
        # else:
        #     self.output = tuple(
        #         ptu.zeros(shape) for shape in output_shapes
        #     )

    def forward(self, input):
        if len(self._output_shapes) == 1:
            return ptu.zeros((input.shape[0], *self._output_shapes[0]))
        else:
            return tuple(
                ptu.zeros((input.shape[0], *shape))
                for shape in self._output_shapes
            )
        # return self.output


def create_dummy_image_vae(
        img_chw,
        latent_dim=1,
        *args,
        **kwargs
) -> VAE:
    encoder_network = DummyNetwork((latent_dim,), (latent_dim,))
    decoder_network = DummyNetwork(img_chw)
    encoder = Gaussian(encoder_network)
    decoder = Gaussian(decoder_network, std=1, reinterpreted_batch_ndims=3)
    prior = MultivariateDiagonalNormal(
        loc=ptu.zeros(1, latent_dim), scale_diag=ptu.ones(1, latent_dim),
    )
    return VAE(encoder, decoder, prior)


def load_pretrained_vae(
        filepath,
        file_type='torch',
        pretrained_key='trainer/vae',
):
    data = asset_loader.load_local_or_remote_file(
        filepath, file_type=file_type, map_location=ptu.device)
    new_file_path = '/'.join(filepath.split('/')[:-1] + ['variant.json'])
    pretrained_variant_path = asset_loader.local_path_from_s3_or_local_path(
        new_file_path
    )
    pretrained_variant = json.load(open(pretrained_variant_path, 'r'))
    my_variant_path = logger.get_snapshot_dir() + '/variant.json'
    my_variant = json.load(open(my_variant_path, 'r'))
    my_variant['pretrained_vae_variant'] = pretrained_variant
    logger.log_variant(my_variant_path, my_variant)

    if 'trainer/policy' in data:  # RL algorithm snapshot
        env = data['evaluation/env']
        vae = env.model
    else:  # just a VAE training snapshot
        vae = data[pretrained_key]

    return vae


def create_image_vae(
        img_chw=None,
        latent_dim=None,
        encoder_cnn_kwargs=None,
        encoder_mlp_kwargs=None,
        decoder_mlp_kwargs=None,
        decoder_dcnn_kwargs=None,
        use_mlp_decoder=False,
        decoder_distribution="bernoulli",
        use_fancy_architecture=False,
) -> VAE:
    img_num_channels, img_height, img_width = img_chw
    if use_fancy_architecture:
        decoder_network, encoder_network = get_fancy_vae(img_chw, latent_dim)
    else:
        encoder_network = create_image_encoder(
            img_chw, latent_dim, encoder_cnn_kwargs, encoder_mlp_kwargs,
        )
        if decoder_mlp_kwargs is None:
            decoder_mlp_kwargs = cedl.invert_encoder_mlp_params(
                encoder_mlp_kwargs
            )
        if use_mlp_decoder:
            decoder_network = create_mlp_image_decoder(
                img_chw,
                latent_dim,
                decoder_mlp_kwargs,
                two_headed=decoder_distribution == 'gaussian_learned_variance',
            )
        else:
            if decoder_distribution == "gaussian_learned_variance":
                raise NotImplementedError()
            pre_dcnn_chw = encoder_network._modules["0"].output_shape
            if decoder_dcnn_kwargs is None:
                decoder_dcnn_kwargs = cedl.invert_encoder_params(
                    encoder_cnn_kwargs, img_num_channels,
                )
            decoder_network = create_image_decoder(
                pre_dcnn_chw,
                latent_dim,
                decoder_dcnn_kwargs,
                decoder_mlp_kwargs,
            )
    encoder = Gaussian(encoder_network)
    encoder.input_size = encoder_network.input_size
    if decoder_distribution in {
        "gaussian_learned_global_scalar_variance",
        "gaussian_learned_global_image_variance",
        "gaussian_learned_variance",
    }:
        if decoder_distribution == "gaussian_learned_global_image_variance":
            log_std = basic.LearnedPositiveConstant(
                ptu.zeros((img_num_channels, img_height, img_width))
            )
            decoder_network = basic.ApplyMany(decoder_network, log_std)
        elif decoder_distribution == "gaussian_learned_global_scalar_variance":
            log_std = basic.LearnedPositiveConstant(ptu.zeros(1))
            decoder_network = basic.ApplyMany(decoder_network, log_std)
        decoder = Gaussian(decoder_network, reinterpreted_batch_ndims=3)
    elif decoder_distribution == "gaussian_fixed_unit_variance":
        decoder = Gaussian(decoder_network, std=1, reinterpreted_batch_ndims=3)
    elif decoder_distribution == "bernoulli":
        decoder = IndependentGenerator(
            BernoulliGenerator(decoder_network), reinterpreted_batch_ndims=3
        )
    else:
        raise NotImplementedError(decoder_distribution)
    prior = MultivariateDiagonalNormal(
        loc=ptu.zeros(1, latent_dim), scale_diag=ptu.ones(1, latent_dim),
    )
    return VAE(encoder, decoder, prior)


def create_image_set_vae(
        img_chw,
        c_dim,
        z_dim,
        x_depends_on_c=False,
        p_z_given_c_kwargs=None,
        decoder_distribution="bernoulli",
        use_fancy_architecture=False,
        p_z_given_c_version='network',
        encoder_c_kwargs=None,
        encoder_z_kwargs=None,
) -> VAE:
    if encoder_z_kwargs is None:
        encoder_z_kwargs = {}
    if encoder_c_kwargs is None:
        encoder_c_kwargs = {}
    if p_z_given_c_version not in {
        'network',
        'learned_ignore_c',
        'fixed'
    }:
        raise NotImplementedError(p_z_given_c_version)
    if p_z_given_c_kwargs is None:
        p_z_given_c_kwargs = {}
    img_num_channels, img_height, img_width = img_chw
    if use_fancy_architecture:
        decoder_net, encoder_c_net, encoder_z_net = get_fancy_set_vae_networks(
            img_chw, c_dim, z_dim, x_depends_on_c,
            z_ignores_c_debug=(p_z_given_c_kwargs != 'network'),
        )
    else:
        raise NotImplementedError()
    encoder_c = Gaussian(encoder_c_net, **encoder_c_kwargs)

    encoder_z = Gaussian(encoder_z_net, **encoder_z_kwargs)
    if decoder_distribution in {
        "gaussian_learned_global_scalar_variance",
        "gaussian_learned_global_image_variance",
        "gaussian_learned_variance",
    }:
        if decoder_distribution == "gaussian_learned_global_image_variance":
            log_std = basic.LearnedPositiveConstant(
                ptu.zeros((img_num_channels, img_height, img_width))
            )
            decoder_net = basic.ApplyMany(decoder_net, log_std)
        elif decoder_distribution == "gaussian_learned_global_scalar_variance":
            log_std = basic.LearnedPositiveConstant(ptu.zeros(1))
            decoder_net = basic.ApplyMany(decoder_net, log_std)
        decoder = Gaussian(decoder_net, reinterpreted_batch_ndims=3)
    elif decoder_distribution == "gaussian_fixed_unit_variance":
        decoder = Gaussian(decoder_net, std=1, reinterpreted_batch_ndims=3)
    elif decoder_distribution == "bernoulli":
        decoder = IndependentGenerator(
            BernoulliGenerator(decoder_net), reinterpreted_batch_ndims=3
        )
    else:
        raise NotImplementedError(decoder_distribution)

    if p_z_given_c_version == 'fixed':
        prior_z_given_c = FixedPriorModel(z_dim)
    elif p_z_given_c_version == 'learned_ignore_c':
        prior_z_given_c = PriorModel(z_dim)
    elif p_z_given_c_version == 'network':
        prior_z_net = create_vector_encoder(c_dim, z_dim, p_z_given_c_kwargs)
        prior_z_given_c = Gaussian(prior_z_net)
    else:
        raise NotImplementedError(p_z_given_c_version)
    prior_c = PriorModel(c_dim)
    return SetVAE(
        encoder_z,
        encoder_c,
        decoder,
        prior_z_given_c,
        prior_c,
        z_dim,
        c_dim,
        x_depends_on_c=x_depends_on_c,
    )


def create_image_encoder(
    img_chw, latent_dim, encoder_cnn_kwargs, encoder_kwargs,
):
    img_num_channels, img_height, img_width = img_chw
    cnn = BasicCNN(
        input_width=img_width,
        input_height=img_height,
        input_channels=img_num_channels,
        **encoder_cnn_kwargs
    )
    cnn_output_size = np.prod(cnn.output_shape)
    mlp = MultiHeadedMlp(
        input_size=cnn_output_size,
        output_sizes=[latent_dim, latent_dim],
        **encoder_kwargs
    )
    enc = nn.Sequential(cnn, Flatten(), mlp)
    enc.input_size = img_width * img_height * img_num_channels
    enc.output_size = latent_dim
    return enc


def create_image_decoder(
    pre_dcnn_chw,
    latent_dim,
    decoder_dcnn_kwargs,
    decoder_kwargs,
):
    dcnn_in_channels, dcnn_in_height, dcnn_in_width = pre_dcnn_chw
    dcnn_input_size = dcnn_in_channels * dcnn_in_width * dcnn_in_height
    dcnn = BasicDCNN(
        input_width=dcnn_in_width,
        input_height=dcnn_in_height,
        input_channels=dcnn_in_channels,
        **decoder_dcnn_kwargs
    )
    mlp = Mlp(
        input_size=latent_dim, output_size=dcnn_input_size, **decoder_kwargs
    )
    dec = nn.Sequential(mlp, dcnn)
    dec.input_size = latent_dim
    return dec


def create_mlp_image_decoder(
    img_chw, latent_dim, decoder_kwargs, two_headed,
):
    img_num_channels, img_height, img_width = img_chw
    output_size = img_num_channels * img_height * img_width
    if two_headed:
        dec = nn.Sequential(
            MultiHeadedMlp(
                input_size=latent_dim,
                output_sizes=[output_size, output_size],
                **decoder_kwargs
            ),
            basic.Map(Reshape(img_num_channels, img_height, img_width)),
        )
    else:
        dec = nn.Sequential(
            Mlp(
                input_size=latent_dim, output_size=output_size, **decoder_kwargs
            ),
            Reshape(img_num_channels, img_height, img_width),
        )
    dec.input_size = latent_dim
    dec.output_size = img_num_channels * img_height * img_width
    return dec


def create_vector_vae(
        data_dim, latent_dim, encoder_kwargs,
        decoder_distribution="gaussian_fixed_unit_variance",
        decoder_kwargs=None,
):
    encoder_net = create_vector_encoder(data_dim, latent_dim, encoder_kwargs)
    decoder_kwargs = decoder_kwargs or cedl.invert_encoder_mlp_params(encoder_kwargs)
    decoder_net = create_vector_decoder(data_dim, latent_dim, decoder_kwargs)
    prior = MultivariateDiagonalNormal(
        loc=ptu.zeros(1, latent_dim), scale_diag=ptu.ones(1, latent_dim),
    )
    encoder = Gaussian(encoder_net)
    encoder.input_size = encoder_net.input_size
    if decoder_distribution == "gaussian_fixed_unit_variance":
        decoder = Gaussian(decoder_net, std=1, reinterpreted_batch_ndims=1)
    elif decoder_distribution == "gaussian_learned_global_scalar_variance":
        log_std = basic.LearnedPositiveConstant(ptu.zeros(1))
        decoder_net = basic.ApplyMany(decoder_net, log_std)
        decoder = Gaussian(decoder_net, reinterpreted_batch_ndims=1)
    else:
        raise NotImplementedError(decoder_distribution)
    return VAE(encoder, decoder, prior)


def create_distribution_generator(mean_net, distribution, data_dim):
    if distribution == 'gaussian_fixed_unit_variance':
        distrib_gen = Gaussian(
                mean_net, std=1, reinterpreted_batch_ndims=data_dim)
    elif distribution == "gaussian_learned_global_scalar_variance":
        log_std = basic.LearnedPositiveConstant(ptu.zeros(1))
        mean_net = basic.ApplyMany(mean_net, log_std)
        distrib_gen = Gaussian(mean_net, reinterpreted_batch_ndims=data_dim)
    else:
        raise NotImplementedError(distribution)
    return distrib_gen


def create_vector_set_vae(
        data_dim,
        z_dim,
        c_dim,
        x_depends_on_c,
        encoder_kwargs,
        decoder_distribution="gaussian_fixed_unit_variance",
        decoder_kwargs=None,
        set_encoder_kwargs=None,
        p_z_given_c_kwargs=None,
):
    if set_encoder_kwargs is None:
        set_encoder_kwargs = encoder_kwargs
    if p_z_given_c_kwargs is None:
        p_z_given_c_kwargs = encoder_kwargs
    encoder_net_c = create_vector_set_encoder(data_dim, c_dim, set_encoder_kwargs)
    encoder_c = Gaussian(encoder_net_c)
    encoder_c.input_size = encoder_net_c.input_size

    encoder_net_z = create_vector_encoder(data_dim + c_dim, z_dim, encoder_kwargs)
    encoder_z = Gaussian(encoder_net_z)
    encoder_z.input_size = encoder_net_z.input_size

    decoder_kwargs = decoder_kwargs or cedl.invert_encoder_mlp_params(encoder_kwargs)
    if x_depends_on_c:
        decoder_net = create_vector_decoder(data_dim, c_dim + z_dim, decoder_kwargs)
    else:
        decoder_net = create_vector_decoder(data_dim, z_dim, decoder_kwargs)
    decoder = create_distribution_generator(decoder_net, decoder_distribution, 1)

    prior_z_net = create_vector_encoder(c_dim, z_dim, p_z_given_c_kwargs)
    prior_z_given_c = Gaussian(prior_z_net)

    prior_c = PriorModel(c_dim)

    return SetVAE(
        encoder_z,
        encoder_c,
        decoder,
        prior_z_given_c,
        prior_c,
        z_dim,
        c_dim,
        x_depends_on_c=x_depends_on_c,
    )


def create_vector_encoder(data_dim, latent_dim, encoder_kwargs):
    enc = ConcatMultiHeadedMlp(
        input_size=data_dim,
        output_sizes=[latent_dim, latent_dim],
        **encoder_kwargs
    )
    enc.input_size = data_dim
    enc.output_size = latent_dim
    return enc


def create_vector_set_encoder(data_dim, latent_dim, encoder_kwargs):
    enc = AverageSetConcatMultiHeadedMlp(
        input_size=data_dim,
        output_sizes=[latent_dim, latent_dim],
        **encoder_kwargs
    )
    enc.input_size = data_dim
    enc.output_size = latent_dim
    return enc


def create_vector_decoder(data_dim, latent_dim, decoder_kwargs):
    dec = ConcatMlp(input_size=latent_dim, output_size=data_dim, **decoder_kwargs)
    return dec
