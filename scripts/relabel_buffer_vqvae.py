from rlkit.misc.asset_loader import load_local_or_remote_file
import numpy as np
import torch
from skimage.transform import rescale, resize, downscale_local_mean

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--buffer_path', type=str, default='/nfs/kun1/users/asap7772/real_data_drawer/val_data/fixed_pot_demos.npy')
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--visualize', action='store_true', default=False)
args = parser.parse_args()

vqvae_path = '/nfs/kun1/users/asap7772/best_vqvae.pt'
vqvae = load_local_or_remote_file(vqvae_path)

path = args.buffer_path
data = np.load(path, allow_pickle=True)

def resize_small(img):
    if img.shape[0] == 6912 or img.flatten().shape[0] == 921600:
        return img
    img = img.reshape(3,64,64)
    img = img.transpose(1,2,0)
    img = resize(img, (48, 48), anti_aliasing=True)
    img = img.transpose(2,1,0).flatten()
    return img

import matplotlib.pyplot as plt
from torchvision import transforms
def plot_img(obs_img, save='a.png'):
    plt.figure()
    if type(obs_img) == torch.Tensor:
        im_new = transforms.ToPILImage()(obs_img.cpu())
    else:
        im_new = obs_img
    plt.imshow(im_new)
    plt.savefig('/home/ashvin/ros_ws/src/railrl-private_anikait/image_vqvae/' + save)
    plt.close()

for i in range(len(data)):
    print('done', i)
    obs = data[i]['observations']
    next_obs = data[i]['next_observations']

    data[i]['latents'] = []
    data[i]['next_latents'] = []

    for j in range(len(obs)):
        img = torch.from_numpy(resize_small(obs[j]['image_observation']).reshape(3,48,48)).float().cuda()
        next_img = torch.from_numpy(resize_small(obs[j]['image_observation']).reshape(3,48,48)).float().cuda()
        
        if args.visualize:
            plot_img(img, 'a.png')
            plot_img(next_img, 'b.png')

        data[i]['latents'].append(vqvae.encode(img).detach().cpu().numpy())
        data[i]['next_latents'].append(vqvae.encode(next_img).detach().cpu().numpy())

        if args.visualize:
            plot_img(vqvae.decode(vqvae.encode(img)).squeeze(), 'reconstructed_a.png')
            plot_img(vqvae.decode(vqvae.encode(next_img)).squeeze(), 'reconstructed_b.png')
        
if args.save_path is None:
    new_path = args.buffer_path.split('.npy')[0] + '_latent.npy'
else:
    new_path = args.save_path
np.save(new_path, data)
print('saved', new_path)