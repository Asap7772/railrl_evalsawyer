import argparse
import pickle

import numpy as np

from rlkit.core import logger
from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.envs.wrappers import ImageMujocoEnv
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode


def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    data = pickle.load(open(args.file, "rb")) # joblib.load(args.file)
    policy = data['policy']

    env = data['env']
    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.multitaskpause:
        env.pause_on_goal = True
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    paths = []

    from rlkit.misc.inspect_q_util import debug_q
    debug_q(data["ensemble_qs"], policy)

    while True:
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key='observation',
            desired_goal_key='desired_goal',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
