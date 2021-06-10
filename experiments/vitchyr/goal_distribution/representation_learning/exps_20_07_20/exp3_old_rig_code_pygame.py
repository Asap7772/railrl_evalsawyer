import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.contextual.rig.rig_launcher import (
    rig_experiment
)

if __name__ == "__main__":
    variant = dict(
        env_class=PickAndPlaceEnv,
        env_kwargs=dict(
            # Environment dynamics
            action_scale=1.0,
            boundary_dist=4,
            ball_radius=0.75,
            object_radius=0.50,
            cursor_visual_radius=1.,
            object_visual_radius=1.,
            min_grab_distance=0.5,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense",
            success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            object_reward_only=False,

            init_position_strategy='random',
            num_objects=2,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=101,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.3,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            pad_color=0,
            subpad_length=1,
            pad_length=1,
            num_columns_per_rollout=2,
        ),
        train_vae_kwargs=dict(
            vae_path=None,
            representation_size=4,
            beta=10.0,
            # beta_schedule_kwargs=dict(
            #     x_values=(0, 500),
            #     y_values=(1  / 128.0, 50  / 128.0),
            # ),
            num_epochs=101,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=False,
                save_trajectories=False,
                enviorment_dataset=False,
                use_cached=False,
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
                # weight_decay=0.01,
            ),
            save_period=5,
        ),
        renderer_kwargs=dict(
            # create_image_format='HWC',
            output_image_format='CWH',
            # output_image_format='CHW',
            flatten_image=True,
            # normalize_image=False,
        ),
        evaluation_goal_sampling_mode="reset_of_env",
        exploration_goal_sampling_mode="vae_prior",
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 2
    mode = 'sss'
    exp_prefix = 'exp3-old-rig-code-pygame-take2'

    search_space = {
        'train_vae_kwargs.representation_size': [
            64, 32, 16
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['exp_id'] = exp_id
        for _ in range(n_seeds):
            run_experiment(
                rig_experiment,
                exp_name=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )
