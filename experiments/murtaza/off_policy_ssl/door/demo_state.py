from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.murtaza.rfeatures_rl import state_td3bc_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='SawyerDoorHookResetFreeEnv-v1',
        algo_kwargs=dict(
            batch_size=1024,
            num_epochs=170,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=500,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=100,
        ),
        td3_trainer_kwargs=dict(
            discount=0.99,
        ),
        td3_bc_trainer_kwargs=dict(
            discount=0.99,
            demo_path="demos/door_demos_action_noise_1000.npy",
            demo_off_policy_path=None,
            bc_num_pretrain_steps=10000,
            q_num_pretrain_steps=10000,
            rl_weight=1.0,
            bc_weight=0,
            reward_scale=1.0,
            target_update_period=2,
            policy_update_period=2,
            add_demos_to_replay_buffer=True,
            obs_key='state_observation',
            env_info_key='angle_difference',
            max_path_length=100,

        ),
        replay_buffer_kwargs=dict(
            max_size=int(1e6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        exploration_noise=.8,
        load_demos=True,
        pretrain_rl=False,
        pretrain_policy=False,
        es='ou',
        td3_bc=True,
        save_video=True,
        image_env_kwargs=dict(
            imsize=48,
            init_camera=sawyer_door_env_camera_v0,
            transpose=True,
            normalize=True,
        ),
        presampled_goals="goals/door_goals.npy",
    )

    search_space = {
        # 'td3_bc_trainer_kwargs.use_awr':[False, True],
        # 'td3_bc_trainer_kwargs.demo_beta':[1, 10],
        'td3_bc_trainer_kwargs.bc_weight':[10, 1, .1, 0],
        # 'td3_bc_trainer_kwargs.rl_weight':[0],
        # 'algo_kwargs.num_epochs':[100],
        # 'algo_kwargs.num_eval_steps_per_epoch':[100],
        # 'algo_kwargs.num_expl_steps_per_train_loop':[100],
        # 'algo_kwargs.min_num_steps_before_training':[0],
        # 'td3_bc_trainer_kwargs.add_demos_to_replay_buffer':[True, False],
        # 'td3_bc_trainer_kwargs.num_trains_per_train_loop':[1000, 2000, 4000, 10000, 16000],
        'exploration_noise':[0.1, .3, .5],
        # 'pretrain_rl':[True],
        # 'pretrain_policy':[False],
        'pretrain_rl': [False],
        'pretrain_policy': [False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'test1'

    n_seeds = 2
    mode = 'gcp'
    exp_name = 'door_state_td3_bc_noisy_demo_v2'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # if variant['td3_bc_trainer_kwargs']['bc_weight'] == 0 and variant['td3_bc_trainer_kwargs']['demo_beta'] != 1:
        #     continue
        for _ in range(n_seeds):
            run_experiment(
                state_td3bc_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                num_exps_per_instance=2,
                skip_wait=False,
                gcp_kwargs=dict(
                    preemptible=False,
                )
            )