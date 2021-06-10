from matplotlib.pyplot import hist
from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.misc import eval_util
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
import argparse
import pickle
import uuid
from rlkit.core import logger
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN
import torch
from sawyer_control.envs.sawyer_grip import SawyerGripEnv

filename = str(uuid.uuid4())
import numpy as np
from rlkit.torch.sac.policies_v2 import TanhGaussianPolicy, MakeDeterministic

def simulate_policy(args):
    action_dim = args.action_dim
    
    cnn_params=dict(
        kernel_sizes=[3, 3, 3],
        n_channels=[16, 16, 16],
        strides=[1, 1, 1],
        hidden_sizes=[1024, 512, 256],
        paddings=[1, 1, 1],
        pool_type='max2d',
        pool_sizes=[2, 2, 1],  # the one at the end means no pool
        pool_strides=[2, 2, 1],
        pool_paddings=[0, 0, 0],
        image_augmentation=True,
        image_augmentation_padding=4)
    
    cnn_params.update(
        input_width=64,
        input_height=64,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )

    if args.bottleneck:
        qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=16,deterministic=False, width=64, height=64)
    else:
        qf1 = ConcatCNN(**cnn_params)

    parameters = torch.load(args.policy_path)
    qf1.load_state_dict(parameters['qf1_state_dict'])
    env = SawyerGripEnv(action_mode='position',
            config_name='ashvin_config',
            reset_free=False,
            position_action_scale=0.05,
            max_speed=0.4,
            step_sleep_time=0.2,
            crop_version_str="crop_val_torch")

    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Q Function loaded")

    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.gpu:
        ptu.set_gpu_mode(True)
    if hasattr(qf1, "to"):
        qf1.to(ptu.device)
    if hasattr(env, "vae"):
        env.vae.to(ptu.device)

    # if args.deterministic:
    #     policy = MakeDeterministic(policy)

    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(qf1, PyTorchModule):
        qf1.train(False)
    
    from skimage import data, color
    from skimage.transform import rescale, resize, downscale_local_mean
    import torchvision.transforms.functional as F
    from PIL import Image
    import matplotlib.pyplot as plt

    def plot_img(obs_img):
        plt.figure()
        if type(obs_img) == torch.Tensor:
            from torchvision import transforms
            im_new = transforms.ToPILImage()(obs_img)
        else:
            im_new = obs_img
        plt.imshow(im_new)
        plt.show()

    def crop(img, img_dim = (64,64)):
        from matplotlib import cm
        img = img.astype(float)
        img /= 255.
        img = img[:, 50:530, :] 
        img = Image.fromarray(np.uint8(img*255))
        
        img = F.resize(img, img_dim, Image.ANTIALIAS)
        img = np.array(img)

        img = img*1.0/255
        img = img.transpose([2,0,1]) #.flatten()
        return torch.from_numpy(img).float()

    paths = []
    for i in range(args.N):
        print('traj', i)
        next_observations = []
        observations = []
        cropped_images = []
        actions = []
        rewards = []
        dones = []
        infos = []
        
        observation = env.reset()
        if args.debug:
            while(True):
                import ipdb; ipdb.set_trace()
                env.step(np.random.rand(4))
        else:
            for j in range(args.H):
                print('trans', j)
                obs_img = crop(np.flip(observation['hires_image_observation'], axis=-1))
                if args.save_img:
                    plot_img(obs_img)
                
                obs_img = obs_img.flatten()
                obs_img = obs_img.repeat(args.num_random, 1).to(ptu.device)
                rand_actions = torch.rand(args.num_random, action_dim).to(ptu.device)
                pred_qs = qf1(obs_img, rand_actions)

                action = rand_actions[torch.argmax(pred_qs, dim=0)[0]]
                action = action.cpu().numpy()
                print('action', action)
                old_obs = observation
                observation, reward, done, info = env.step(action)

                observations.append(old_obs)
                next_observations.append(observation)
                cropped_images.append(obs_img)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

        paths.append(dict(observations=observations,next_observations=next_observations,cropped_images=cropped_images,actions=actions, rewards = rewards, dones=dones, infos=infos))
        np.save(args.out_path, paths)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10,
                        help='Number of Trajectories')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    # parser.add_argument()
    parser.add_argument('--policy_path', type=str, default='evaluation')
    parser.add_argument('--out_path', type=str, default='evaluation')
    parser.add_argument('--env_type', type=str, default='evaluation')
    parser.add_argument('--bottleneck', action='store_true')
    parser.add_argument('--num_random', type=int, default=128)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--log_diagnostics', action='store_true')
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--save_img', action='store_true')
    args = parser.parse_args()
    simulate_policy(args)
