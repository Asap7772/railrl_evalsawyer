import torch
import torch.nn as nn
from torchvision import datasets, transforms
from rlkit.torch import pytorch_util as ptu
from os import path as osp
from sklearn import neighbors
import numpy as np
from torchvision.utils import save_image
import time
import os 
import pickle
import sys
"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
current_dir = sys.path.append(os.getcwd())
pixelcnn_dir = sys.path.append(os.getcwd()+ '/pixelcnn')

from models import GatedPixelCNN
import utils

"""
Hyperparameters
"""
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument("--filepath", type=str)
parser.add_argument("--vaepath", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true", default=True)

parser.add_argument("--dataset",  type=str, default='LATENT_BLOCK',
    help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--img_dim", type=int, default=12)
parser.add_argument("--input_dim", type=int, default=1,
    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=512,
    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    # vae = torch.load(local_path, map_location='cpu')
    print("loaded", local_path)
    vae.to("cpu")
    return vae

"""
data loaders
"""
all_data = np.load(args.filepath, allow_pickle=True)
vqvae = load_vae(args.vaepath)


def ind_to_cont(e_indices):
    input_shape = e_indices.shape + (vqvae.representation_size,)
    e_indices = e_indices.reshape(-1).unsqueeze(1)#, input_shape[1]*input_shape[2])
    
    min_encodings = torch.zeros(e_indices.shape[0], vqvae.num_embeddings, device=e_indices.device)
    min_encodings.scatter_(1, e_indices, 1)

    e_weights = vqvae._embedding.weight
    quantized = torch.matmul(
        min_encodings, e_weights).view(input_shape)
    
    z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    return z_q

#all_data_cont = ind_to_cont(torch.LongTensor(all_data.reshape(-1, 12, 12)))
#tree = neighbors.KDTree(all_data, metric="chebyshev")

def get_closest_stats(latents):
    #latents = ind_to_cont(latents)
    latents = ptu.get_numpy(latents).reshape(-1, 144)
    all_dists = []
    all_index = []
    for i in range(latents.shape[0]):
        smallest_dist = float('inf')
        index = 0
        for j in range(all_data.shape[0]):
            dist = np.count_nonzero(latents[i]!= all_data[j])
            if dist < smallest_dist:
                smallest_dist = dist
                index = j


        #dist, index = tree.query(latents[i].cpu())
        all_dists.append(smallest_dist)
        all_index.append(index)
    all_dists = np.array(all_dists)
    all_index = np.array(all_index)
    print("Mean:", np.mean(all_dists))
    print("Std:", np.std(all_dists))
    print("Min:", np.min(all_dists))
    print("Max:", np.max(all_dists))
    return torch.LongTensor(all_data[all_index].reshape(-1, 12, 12))


if args.dataset == 'LATENT_BLOCK':
    _, _, train_loader, test_loader, _ = utils.load_data_and_data_loaders(args.filepath, 'LATENT_BLOCK', args.batch_size)
else:
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=True, download=True,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_Size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=False,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

model = GatedPixelCNN(args.n_embeddings, args.img_dim**2, args.n_layers).to(device)
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


"""
train, test, and log
"""

def train():
    train_loss = []
    for batch_idx, x in enumerate(train_loader):
        start_time = time.time()
        
        #x = (x[:, 0]).cuda()
        x = x.cuda()

        # Train PixelCNN with images
        logits = model(x)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, args.n_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % args.log_interval == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                args.log_interval * batch_idx / len(train_loader),
                np.asarray(train_loss)[-args.log_interval:].mean(0),
                time.time() - start_time
            ))


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, x in enumerate(test_loader):
        #x = (x[:, 0]).cuda()
            x = x.cuda()
            logits = model(x)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )
            
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples(epoch, batch_size=64):
    e_indices = model.generate(shape=(args.img_dim, args.img_dim), batch_size=batch_size).reshape(-1, args.img_dim**2)
    closest_index = get_closest_stats(e_indices)
    samples = vqvae.decode(e_indices.cpu())
    closest = vqvae.decode(closest_index.cpu())

    save_dir1 = "/home/ashvin/Desktop/vqvae_samples/sample{0}.png".format(epoch)
    save_dir2 = "/home/ashvin/Desktop/vqvae_samples/closest{0}.png".format(epoch)

    save_image(
        samples.data.view(batch_size, 3, 48, 48).transpose(2, 3),
        save_dir1
    )


    save_image(
        closest.data.view(batch_size, 3, 48, 48).transpose(2, 3),
        save_dir2
    )


BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, args.epochs):
    vqvae.set_pixel_cnn(model)
    print("\nEpoch {}:".format(epoch))
    train()
    cur_loss = test()

    if args.save or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        try:
            torch.save(model.state_dict(), '/home/ashvin/Desktop/sim-pusher-pixelcnn/pixelcnn.pt')
            torch.save(vqvae.state_dict(), '/home/ashvin/Desktop/sim-pusher-pixelcnn/vqvae.pt')
            #torch.save(model.state_dict(), 'results/{}_pixelcnn.pt'.format(args.dataset))
        except:
            torch.save(model.state_dict(), 'results/{}_pixelcnn.pt'.format(args.dataset))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    if args.gen_samples:
        generate_samples(epoch)