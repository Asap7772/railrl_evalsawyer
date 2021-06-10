import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import (
    init_sawyer_camera_v4,
    sawyer_init_camera_zoomed_in,
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPS-v0',
        grill_variant=dict(
            save_video=True,
            online_vae_beta=1,
            save_video_period=100,
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
                    num_epochs=300,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=1024,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    collection_mode='online',
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
                    vae_training_schedule=vae_schedules.every_other,
                    oracle_data=True,
                    vae_save_period=50,
                    parallel_vae_train=False,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(100000),
                fraction_goals_rollout_goals=0.0,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='image_bernoulli_inv_prob',
                priority_function_kwargs=dict(
                    sampling_method='correct',
                    num_latents_to_sample=10,
                ),
                power=1/8,
            ),
            normalize=False,
            render=False,
            exploration_noise=0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                # type='latent_distance',
                type='mahalanobis_distance',
            ),
            observation_key='posterior_params_observation',
            # desired_goal_key='posterior_params_desired_goal',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='ONLINE-VAE-SAC-BERNOULLI',
            generate_uniform_dataset_kwargs=dict(
                init_camera=sawyer_init_camera_zoomed_in,
                env_id='SawyerPushNIPS-v0',
                num_imgs=10000,
                use_cached_dataset=False,
                show=False,
                save_file_prefix='push-nips',
            ),
            generate_uniform_dataset_fn=generate_uniform_dataset_reacher,
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=1.0,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                N=10000,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=True,
                n_random_steps=100,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
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
        'grill_variant.replay_buffer_kwargs.vae_priority_type': [
            'None',
        ],
        'grill_variant.reward_params.type': [
            'mahalanobis_distance',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 2
    # mode = 'ec2'
    exp_prefix = 'generate-pusher-nips-dataset'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )
