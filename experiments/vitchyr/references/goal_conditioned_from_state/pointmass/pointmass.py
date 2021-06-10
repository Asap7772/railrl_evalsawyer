import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rl_exp_launcher_util import twin_sac_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='Point2DLargeEnv-offscreen-v0',
        # env_id='Point2DEnv-Train-Everything-Eval-Everything-v1',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        twin_sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=100,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=100,
        ),
        replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
            max_size=int(1e6),
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        save_video=False,
        save_video_period=10,
        do_state_exp=True,
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'upstream/master-vitchyr'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                twin_sac_experiment,
                unpack_variant=False,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.5*24*60),
            )
