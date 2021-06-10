import torch
import torch.nn as nn
from torchvision import datasets, transforms
from rlkit.torch import pytorch_util as ptu
from os import path as osp
import numpy as np
from torchvision.utils import save_image
import time
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
from PIL import Image
from rlkit.misc.asset_loader import load_local_or_remote_file
import os
from tqdm import tqdm
import pickle
import sys

from rlkit.torch.vae.initial_state_pixelcnn import GatedPixelCNN
import rlkit.torch.vae.pixelcnn_utils
from rlkit.torch.vae.vq_vae import VQ_VAE

from rlkit.core import logger
from rlkit.torch.vae.pixelcnn_trainer import PixelCNNTrainer
from rlkit.data_management.dataset  import InfiniteBatchLoader

"""
data loaders
"""

def train_pixelcnn(
    vqvae=None,
    vqvae_path=None,
    num_epochs=100,
    batch_size=32,
    n_layers=15,
    dataset_path=None,
    save=True,
    save_period=10,
    cached_dataset_path=False,
    trainer_kwargs=None,
    model_kwargs=None,
    object_list=None,
    debug=False,
):
    trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs
    model_kwargs = {} if model_kwargs is None else model_kwargs

    # Load VQVAE + Define Args
    if vqvae is None:
        vqvae = load_local_or_remote_file(vqvae_path)
        vqvae.to(ptu.device)
        vqvae.eval()

    root_len = vqvae.root_len
    num_embeddings = vqvae.num_embeddings
    embedding_dim = vqvae.embedding_dim
    cond_size = vqvae.num_embeddings
    imsize = vqvae.imsize
    discrete_size = root_len * root_len
    representation_size = embedding_dim * discrete_size
    input_channels = vqvae.input_channels
    imlength = imsize * imsize * input_channels

    log_dir = logger.get_snapshot_dir()

    # Define data loading info
    new_path = osp.join(log_dir, 'pixelcnn_data.npy')

    def prep_sample_data(cached_path):
        data = load_local_or_remote_file(cached_path).item()
        train_data = data['train']#.reshape(-1, discrete_size)
        test_data = data['test']#.reshape(-1, discrete_size)
        return train_data, test_data

    def process_object_list(object_list, dataset):
        keep_ind = 0
        objects_np = np.array(dataset['object'])
        for o in object_list:
            keep_ind += (objects_np == o)

        keep_ind = np.where(keep_ind == 1)
        dataset['env'] = dataset['env'][keep_ind]
        dataset['observations'] = dataset['observations'][keep_ind]

    def encode_dataset(path, object_list):
        data = load_local_or_remote_file(path)
        data = data.item()

        if object_list is not None:
            process_object_list(object_list, data)

        all_data = []

        n = data["observations"].shape[0]

        for i in tqdm(range(n)):
            obs = ptu.from_numpy(data["observations"][i] / 255.0)
            latent = vqvae.encode(obs, cont=False)
            all_data.append(latent)

        encodings = ptu.get_numpy(torch.stack(all_data, dim=0))
        return encodings

    if cached_dataset_path:
        train_data, test_data = prep_sample_data(cached_dataset_path)
    else:
        train_data = encode_dataset(dataset_path['train'], object_list)
        test_data = encode_dataset(dataset_path['test'], None)
    dataset = {'train': train_data, 'test': test_data}
    np.save(new_path, dataset)

    _, _, train_loader, test_loader, _ = \
        rlkit.torch.vae.pixelcnn_utils.load_data_and_data_loaders(new_path, 'COND_LATENT_BLOCK', batch_size)

    #train_dataset = InfiniteBatchLoader(train_loader)
    #test_dataset = InfiniteBatchLoader(test_loader)

    print("Finished loading data")

    model = GatedPixelCNN(
        num_embeddings,
        root_len**2,
        n_classes=representation_size,
        **model_kwargs
    ).to(ptu.device)
    trainer = PixelCNNTrainer(
        model,
        vqvae,
        batch_size=batch_size,
        **trainer_kwargs,
    )

    print("Starting training")

    BEST_LOSS = 999
    for epoch in range(num_epochs):
        should_save = (epoch % save_period == 0) and (epoch > 0)
        trainer.train_epoch(epoch, train_loader)
        trainer.test_epoch(epoch, test_loader)
        
        trainer.dump_samples(epoch, test_data, test=True)
        trainer.dump_samples(epoch, train_data, test=False)

        if should_save:
            logger.save_itr_params(epoch, model)

        stats = trainer.get_diagnostics()

        cur_loss = stats["test/loss"]
        if cur_loss < BEST_LOSS:
            BEST_LOSS = cur_loss
            vqvae.set_pixel_cnn(model)
            logger.save_extra_data(vqvae, 'best_vqvae', mode='torch')
        else:
            return vqvae

        for k, v in stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        trainer.end_epoch(epoch)

    return vqvae
