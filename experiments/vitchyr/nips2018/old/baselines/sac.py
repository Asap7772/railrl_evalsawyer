import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu

from rlkit.envs.mujoco.sawyer_gripper_env import SawyerPushXYEnv
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = MultitaskToFlatEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = ConcatMlp(
        input_size=obs_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=300,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
        ),
        env_class=SawyerPushXYEnv,
        env_kwargs=dict(
            frame_skip=50,
            only_reward_block_to_goal=False,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='SAC',
        version='normal',
        normalize=True,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer-sim-push-easy-ish-check-fixed-env'

    search_space = {
        # 'env_kwargs.randomize_goals': [True, False],
        # 'env_kwargs.only_reward_block_to_goal': [False, True],
        'algo_kwargs.max_path_length': [50, 100],
        'algo_kwargs.reward_scale': [10, 100, 1000],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
