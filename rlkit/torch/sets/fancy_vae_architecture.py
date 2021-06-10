from __future__ import print_function
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import CNN, DCNN, AverageSetConcatMultiHeadedMlp
from rlkit.torch.networks import Flatten, Reshape
from rlkit.torch.networks.basic import (
    Concat, ZipApply,
    MultiInputSequential,
    DropFeatures,
    ReshapeWrapper,
)
from rlkit.torch.networks.mlp import MultiHeadedMlp, Mlp
from rlkit.torch.vae.vae_base import (
    compute_bernoulli_log_prob,
    GaussianLatentVAE,
)
from rlkit.torch.vae.vq_vae import Encoder, Decoder


def _get_fancy_autoencoder_cnns(
            embedding_dim=3,
            input_channels=3,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
):
    encoder_cnn = Encoder(
        input_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    )
    pre_rep_conv = nn.Conv2d(
        in_channels=num_hiddens,
        out_channels=embedding_dim,
        kernel_size=1,
        stride=1,
    )
    encoder_network = nn.Sequential(encoder_cnn, pre_rep_conv)
    decoder_network = Decoder(
        embedding_dim,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    )
    return encoder_network, decoder_network


class CustomSequential(nn.Sequential):
    def __init__(self, preprocess, reshape, *tail):
        super().__init__(*tail)
        self.preprocess = preprocess
        self.reshape = reshape

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        x = self.preprocess(x)
        original_shape = x.shape
        x = self.reshape(x)
        out = super().forward(x)
        return out.view(*tuple(*original_shape[:2], *out.shape[1:]))


def get_fancy_vae(img_chw, latent_dim):
    img_num_channels, img_height, img_width = img_chw
    encoder_cnn, decoder_cnn = _get_fancy_autoencoder_cnns()
    # stub_vae = SashaVAE(latent_dim)
    # encoder_cnn = nn.Sequential(stub_vae._encoder, stub_vae._pre_rep_conv, )
    encoder_cnn.eval()
    test_mat = torch.zeros(1, img_num_channels, img_width, img_height, )
    encoder_cnn_output_shape = encoder_cnn(test_mat).shape[1:]
    encoder_cnn.train()
    encoder_cnn_output_size = np.prod(encoder_cnn_output_shape)
    encoder_mlp = MultiHeadedMlp(
        input_size=encoder_cnn_output_size,
        output_sizes=[latent_dim, latent_dim],
        hidden_sizes=[],
    )
    encoder_network = nn.Sequential(encoder_cnn, Flatten(), encoder_mlp)
    encoder_network.input_size = img_width * img_height * img_num_channels

    # decoder_cnn = stub_vae._decoder
    decoder_mlp = nn.Linear(latent_dim, encoder_cnn_output_size)
    # decoder_network = nn.Sequential(
    decoder_network = CustomSequential(
        decoder_mlp, Reshape(*encoder_cnn_output_shape), decoder_cnn,
    )
    decoder_network.input_size = encoder_cnn_output_size
    return decoder_network, encoder_network


class SetOrSingletonEncoder(nn.Module):
    def __init__(self, preprocessor, set_module):
        super().__init__()
        self.preprocessor = preprocessor
        self.set_module = set_module

    def forward(self, *args, batch=False):
        output = self.preprocessor(*args)
        if isinstance(output, tuple):
            return self.set_module(*output, batch=batch)
        else:
            return self.set_module(output, batch=batch)


def get_fancy_set_vae_networks(
        img_chw, c_dim, z_dim, x_depends_on_c,
        z_ignores_c_debug=False,
):
    img_num_channels, img_height, img_width = img_chw
    encoder_cnn, decoder_cnn = _get_fancy_autoencoder_cnns()
    encoder_cnn.eval()
    test_mat = torch.zeros(1, img_num_channels, img_width, img_height, )
    encoder_cnn_output_shape = encoder_cnn(test_mat).shape[1:]
    encoder_cnn.train()
    encoder_cnn_output_size = int(np.prod(encoder_cnn_output_shape))
    encoder_cnn_flat = nn.Sequential(encoder_cnn, Flatten())
    encoder_c_mlp = AverageSetConcatMultiHeadedMlp(
        input_size=encoder_cnn_output_size,
        output_sizes=[c_dim, c_dim],
        hidden_sizes=[],
    )
    encoder_c_network = SetOrSingletonEncoder(encoder_cnn_flat, encoder_c_mlp)
    encoder_c_network.input_size = img_width * img_height * img_num_channels

    if z_ignores_c_debug:
        encoder_z_mlp = MultiHeadedMlp(
            input_size=encoder_cnn_output_size,
            output_sizes=[z_dim, z_dim],
            hidden_sizes=[32, 32],
        )
        encoder_z_network = MultiInputSequential(
            ZipApply(DropFeatures(), encoder_cnn_flat),
            Concat(),
            Flatten(),
            encoder_z_mlp
        )
    else:
        encoder_z_mlp = MultiHeadedMlp(
            input_size=c_dim + encoder_cnn_output_size,
            output_sizes=[z_dim, z_dim],
            hidden_sizes=[32, 32],
        )
        c_processor_for_z = nn.Identity()
        encoder_z_network = MultiInputSequential(
            ZipApply(c_processor_for_z, encoder_cnn_flat),
            Concat(),
            Flatten(),
            encoder_z_mlp
        )

    if x_depends_on_c:
        decoder_mlp = nn.Sequential(
            Concat(),
            nn.Linear(c_dim + z_dim, encoder_cnn_output_size),
        )
    else:
        decoder_mlp = nn.Linear(z_dim, encoder_cnn_output_size)
    # decoder_network = CustomSequential(
    decoder_network = nn.Sequential(
        # decoder_mlp, Reshape(*encoder_cnn_output_shape), decoder_cnn,
        decoder_mlp, ReshapeWrapper(encoder_cnn_output_shape, decoder_cnn),
    )
    decoder_network.input_size = encoder_cnn_output_size
    return decoder_network, encoder_c_network, encoder_z_network


class FancyVAE(GaussianLatentVAE):
    # Mimic original RIG code
    def __init__(
            self,
            representation_size,
            architecture,

            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
    ):
        """

        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.architecture = architecture

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size=deconv_args['deconv_input_width']* \
                         deconv_args['deconv_input_height']* \
                         deconv_args['deconv_input_channels']
        self.decoder, self.encoder = get_fancy_vae(
            (input_channels, imsize, imsize),
            representation_size,
        )

        self.epoch = 0
        self.decoder_distribution = decoder_distribution

    def encode(self, input):
        input = input.view(input.shape[0],
                           self.input_channels,
                           self.imsize,
                           self.imsize)
        h = self.encoder(input)
        mu, logvar = h
        # h = h.view(h.shape[0], -1)
        # mu = self.fc1(h)
        # if self.log_min_variance is None:
        #     logvar = self.fc2(h)
        # else:
        #     logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return (mu, logvar)

    def decode(self, latents):
        decoded = self.decoder(latents).view(-1, self.imsize*self.imsize*self.input_channels)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1), torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            log_prob = compute_bernoulli_log_prob(inputs, obs_distribution_params[0]) * self.imlength
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0], reduction='elementwise_mean')
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))
