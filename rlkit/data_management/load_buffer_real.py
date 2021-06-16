import numpy as np
from numpy.core.fromnumeric import trace
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
import os
from rlkit.envs.dummy_env import DummyEnv
import pickle

# TODO Clean up this file
MAX_SIZE = int(1E5)
def get_buffer(observation_key='image', buffer_size=MAX_SIZE, image_shape=(64,64,3), imgstate=False, color_jitter=False):
    expl_env = DummyEnv(image_shape=image_shape)
    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
        internal_keys=None,
        color_jitter=color_jitter,
        jit_percent=0.1, #TODO Change
    )
    return replay_buffer

import torch

#Stephen commented on 6-10 because it was preventing headless rendering in bottleneck setup.
# import matplotlib.pyplot as plt
# def plot_img(obs_img):
#     plt.figure()
#     if type(obs_img) == torch.Tensor:
#         im_new = transforms.ToPILImage()(obs_img)
#     else:
#         im_new = obs_img
#     plt.imshow(im_new)
#     plt.savefig('/nfs/kun1/users/asap7772/cog/a.png')
#     plt.show()

from torchvision import transforms
from skimage.transform import rescale, resize, downscale_local_mean

def resize_small(img):
    if img.shape[0] == 6912 or img.flatten().shape[0] == 921600:
        return img
    img = img.reshape(3,64,64)
    img = img.transpose(1,2,0)
    img = resize(img, (48, 48), anti_aliasing=True)
    img = img.transpose(2,0,1).flatten()
    return img

def load_path(path, rew_path, replay_buffer, small_img=False, bc=False, imgstate=False):
    data = np.load(path,allow_pickle=True)
    if rew_path is not None:
        rew = pickle.load(open(rew_path, 'rb'))
        assert len(data) == len(rew)
        for i in range(len(data)):
            data[i]['rewards'] = np.array(rew[i]).tolist()

    if bc:
        data = [data[i] for i in range(len(data)) if sum(data[i]['rewards']) > 0]
        print('kept', len(data), 'traj')

    if small_img:
        for i in range(len(data)):
            for j in range(len(data[i]['observations'])):
                data[i]['observations'][j]['image_observation'] = resize_small(data[i]['observations'][j]['image_observation'])
                data[i]['next_observations'][j]['image_observation'] = resize_small(data[i]['next_observations'][j]['image_observation'])
    add_data_to_buffer(data, replay_buffer, small_img=small_img, imgstate=imgstate)

def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions

def add_data_to_buffer(data, replay_buffer, scale_rew=False, scale=200, shift=1, drop_last=True, small_img=False, imgstate=False):
    for j in range(len(data)):
        # import ipdb; ipdb.set_trace()
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        if 'next_actions' not in data[j]:
            data[j]['next_actions'] = np.concatenate((data[j]['actions'][1:], data[j]['actions'][-1:]))
        rew = np.array(data[j]['rewards']) * (0.99**np.arange(len(data[j]['rewards'])))
        mcrewards = np.cumsum(rew[::-1])[::-1].tolist() 

        if 'latents' in data[j]:
            path = dict(
                rewards=[np.asarray([r]) for r in data[j]['rewards']],
                mcrewards=[np.asarray([r]) for r in mcrewards],
                actions=data[j]['actions'],
                next_actions =data[j]['next_actions'],
                terminals=[np.asarray([t]) for t in data[j]['terminals']],
                observations=process_images(data[j]['observations'], small_img=small_img, imgstate=imgstate),
                next_observations=process_images(data[j]['next_observations'], small_img=small_img, imgstate=imgstate),
                latents = data[j]['latents'],
                next_latents = data[j]['next_latents'],
            )
        else:
            path = dict(
                rewards=[np.asarray([r]) for r in data[j]['rewards']],
                mcrewards=[np.asarray([r]) for r in mcrewards],
                actions=data[j]['actions'],
                next_actions =data[j]['next_actions'],
                terminals=[np.asarray([t]) for t in data[j]['terminals']],
                observations=process_images(data[j]['observations'], small_img=small_img, imgstate=imgstate),
                next_observations=process_images(data[j]['next_observations'], small_img=small_img, imgstate=imgstate),
            )


        if drop_last:
            for x in path:
                #drop last transition since image size is larger
                path[x] = path[x][:-1]

        if scale_rew:
            path['rewards'] = [np.asarray([r*scale + shift]) for r in data[j]['rewards']]
        replay_buffer.add_path(path)

def process_images(observations, small_img=False, imgstate=False):
    output = []
    for i in range(len(observations)):
        try:
            if small_img:
                image = observations[i]['image_observation'].reshape(3,48,48)
            else:
                image = observations[i]['image_observation'].reshape(3,64,64)
            state = observations[i]['state_observation']
        except:
            if small_img:
                image = observations[i-1]['image_observation'].reshape(3,48,48)
            else:
                image = observations[i-1]['image_observation'].reshape(3,64,64)
            state = observations[i-1]['state_observation']
        if len(image.shape) == 3:
            image = image.flatten()
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(state=state, image=image) if imgstate else dict(image=image))     
    return output

if __name__ == "__main__":
    args = lambda:0 #RANDOM Object
    paths = []
    data_path = '/nfs/kun1/users/ashvin/data/val_data'
    paths.append( (os.path.join(data_path,'fixed_pot_demos.npy'), os.path.join(data_path,'fixed_pot_demos_putlidon_rew.pkl')))
    
    replay_buffer = get_buffer()
    for path, rew_path in paths:
        load_path(path, rew_path, replay_buffer)
    import ipdb; ipdb.set_trace()
    
