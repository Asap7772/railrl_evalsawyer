from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.misc import eval_util
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
import argparse
import pickle
import uuid
from rlkit.core import logger
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN,VQVAEEncoderCNN
import torch
from sawyer_control.envs.sawyer_grip import SawyerGripEnv
import matplotlib.pyplot as plt
filename = str(uuid.uuid4())
import numpy as np
from rlkit.torch.sac.policies_v2 import TanhGaussianPolicy, GaussianPolicy, MakeDeterministic

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


    if args.deeper_net:
        print('deeper conv net')
        cnn_params.update(
            kernel_sizes=[3, 3, 3, 3, 3],
            n_channels=[32, 32, 32, 32, 32],
            strides=[1, 1, 1, 1, 1],
            paddings=[1, 1, 1, 1, 1],
            pool_sizes=[2, 2, 1, 1, 1],
            pool_strides=[2, 2, 1, 1, 1],
            pool_paddings=[0, 0, 0, 0, 0]
        )
    
    cnn_params.update(
        input_width=64,
        input_height=64,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )
    
    cnn_params.update(
        output_size=256,
        added_fc_input_size=args.statedim if args.imgstate else 0,
        hidden_sizes=[1024, 512],
    ) 

    print(cnn_params)

    if args.vqvae_enc:
        policy_obs_processor = VQVAEEncoderCNN(**cnn_params)
    else:
        policy_obs_processor = CNN(**cnn_params)

    policy_class = GaussianPolicy if args.gaussian_policy else TanhGaussianPolicy
    policy = policy_class(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    parameters = torch.load(args.policy_path)
    policy.load_state_dict(parameters['policy_state_dict'])
        
    env = SawyerGripEnv(action_mode='position',
            config_name='ashvin_config',
            reset_free=False,
            position_action_scale=0.05,
            max_speed=0.4,
            step_sleep_time=0.2,
            crop_version_str="crop_val_torch")

    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")

    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.gpu:
        ptu.set_gpu_mode(True)
    if hasattr(policy, "to"):
        policy.to(ptu.device)
    if hasattr(env, "vae"):
        env.vae.to(ptu.device)

    if args.deterministic:
        policy = MakeDeterministic(policy)

    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    
    from skimage import data, color
    from skimage.transform import rescale, resize, downscale_local_mean
    import torchvision.transforms.functional as F
    from PIL import Image

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
        for j in range(args.H):
            print('trans', j)
            obs_img = crop(np.flip(observation['hires_image_observation'], axis=-1), img_dim=(48,48) if args.smdim else (64,64))
            obs_img = torch.from_numpy(obs_img.numpy().swapaxes(-2,-1))
            if args.save_img:
                plot_img(torch.from_numpy(obs_img.numpy().swapaxes(-2,-1)))
            
            if args.debug:
                action = np.random.rand(4)
            else:
                action = policy.forward(obs_img.flatten()[None],extra_fc_input=torch.from_numpy(observation['state_observation'])[None].float() if args.imgstate else None)[0].squeeze().detach().cpu().numpy()
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
        print('saved', args.out_path)
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
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--gaussian_policy', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--log_diagnostics', action='store_true')
    parser.add_argument('--smdim', action='store_true')
    parser.add_argument('--vqvae_enc', action='store_true')
    parser.add_argument('--deeper_net', action='store_true')
    parser.add_argument('--imgstate', action='store_true')
    parser.add_argument('--pickle', action='store_true')
    parser.add_argument('--statedim', type=int, default=3)
    parser.add_argument('--action_dim', type=int, default=4)
    args = parser.parse_args()
    simulate_policy(args)
