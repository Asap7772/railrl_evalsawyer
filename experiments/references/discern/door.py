import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.door.generate_uniform_dataset import generate_uniform_dataset_door
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.torch.vae.dataset.generate_goal_dataset import generate_goal_dataset_using_policy

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        env_id='SawyerDoorHookResetFreeEnv-v0',
        init_camera=sawyer_door_env_camera_v0,
        grill_variant=dict(
            use_discern_sampling=True,
            save_video=True,
            online_vae_beta=5,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=1010,
                    num_steps_per_epoch=500,
                    num_steps_per_eval=500,
                    min_num_steps_before_training=10000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    # collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=1,
                    ),
                    reward_scale=1,
                ),
                her_kwargs=dict(
                ),
                twin_sac_kwargs=dict(
                    train_policy_with_reparameterization=True,
                    soft_target_tau=1e-3,  # 1e-2
                    policy_update_period=1,
                    target_update_period=1,  # 1
                    use_automatic_entropy_tuning=True,
                ),
                online_vae_kwargs=dict(
                    vae_training_schedule=vae_schedules.custom_schedule,
                    oracle_data=False,
                    vae_save_period=50,
                    parallel_vae_train=False,
                ),
                diverse_kwargs=dict(
                    p_replace=.05,
                    p_add_non_diverse=.05,
                    goal_buffer_size=1024,
                ),

            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.0,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='None',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=0,
            ),
            normalize=False,
            render=False,
            exploration_noise=0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            generate_goal_dataset_fctn=generate_goal_dataset_using_policy,
            goal_generation_kwargs=dict(
                num_goals=1000,
                use_cached_dataset=True,
                policy_file='11-09-her-twin-sac-door/11-09-her-twin-sac-door_2018_11_10_02_17_10_id000--s16215/params.pkl',
                path_length=100,
                show=False,
                tag='_twin_sac'
            ),
            presampled_goals_path='door_goals_bright_sawyer.npy',
            presample_goals=True,
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='ONLINE-VAE-SAC-BERNOULLI-HER-TD3',
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=1.0,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='gaussian',
            generate_vae_dataset_kwargs=dict(
                N=2,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                n_random_steps=1,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                decoder_distribution='gaussian_identity_variance',
                input_channels=3,
                architecture=imsize48_default_architecture,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                lr=1e-3,
            ),
            save_period=1,
        ),
    )

    search_space = {
        'grill_variant.algo_kwargs.diverse_kwargs.p_replace': [.01],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 3
    mode = 'gcp'
    exp_prefix = 'steven-door-discern-new-visuals-comp-2'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                snapshot_gap=50,
                snapshot_mode='gap_and_last',
                use_gpu=True,
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-west1-b',
                    gpu_kwargs=dict(
                        # gpu_model='nvidia-tesla-p4',
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                )
          )
