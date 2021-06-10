import rlkit.misc.hyperparameter as hyp
from rlkit.demos.source.dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.awac_rig import awac_rig_experiment
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
# from roboverse.envs.sawyer_rig_multiobj_tray_v0 import SawyerRigMultiobjTrayV0
# from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0
from sawyer_control.envs.sawyer_grip import SawyerGripEnv
# from sawyer_control.envs.sawyer_grip_stub import SawyerGripEnv
from rlkit.envs.encoder_wrappers import ConditionalEncoderWrappedEnv
from rlkit.torch.networks import Clamp
from rlkit.torch.vae.vq_vae import CCVAE
from rlkit.torch.vae.vq_vae_trainer import CCVAETrainer
from rlkit.torch.grill.common import train_vae

demo_paths = [ # 
    dict( # all data together
        path='/home/ashvin/data/s3doodad/demos/icra2021/dataset_v2/offpolicy_ccrig_v2_1.npy',
        obs_dict=True,
        is_demo=True,
    ),
    dict( # all data together
        path='/home/ashvin/data/s3doodad/demos/icra2021/dataset_v2/offpolicy_ccrig_v2_2.npy',
        obs_dict=True,
        is_demo=True,
    ),
    dict( # all data together
        path='/home/ashvin/data/s3doodad/demos/icra2021/dataset_v2/offpolicy_ccrig_v2_3.npy',
        obs_dict=True,
        is_demo=True,
    ),
]

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        env_class=SawyerGripEnv,
        env_kwargs=dict(
            action_mode='position',
            config_name='ashvin_config',
            reset_free=False,
            position_action_scale=0.05,
            max_speed=0.4,
            step_sleep_time=0.2,
        ),
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=False,
            alpha=0,
            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0, #25000
            policy_weight_decay=1e-4,
            q_weight_decay=0,
            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            compute_bc=True,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,
            reward_transform_kwargs=None,
            terminal_transform_kwargs=None,
        ),
        max_path_length=50, #65
        algo_kwargs=dict(
            batch_size=1024, #1024
            num_epochs=1001, #1001
            num_eval_steps_per_epoch=500, #1000
            num_expl_steps_per_train_loop=500, #1000
            num_trains_per_train_loop=500, #1000
            min_num_steps_before_training=500, #4000
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
            max_size=int(2E5),
        ),
        demo_replay_buffer_kwargs=dict(
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
        ),
        reward_kwargs=dict(
            reward_type='dense',
            epsilon=1.0,
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=1,
            pad_color=0,
        ),
        reset_keys_map=dict(
            image_observation="initial_latent_state"
        ),
        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=dict(
            recompute_reward=True,
            condition_encoding=True,
        ),
        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),
        add_env_demos=False,
        add_env_offpolicy_data=False,
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        evaluation_goal_sampling_mode="presampled_images",
        exploration_goal_sampling_mode="conditional_vae_prior",
        train_vae_kwargs=dict(
            beta=1,
            imsize=48,
            embedding_dim=1,
            beta_schedule_kwargs=dict(
                x_values=(0, 1501),
                y_values=(0, 50)
            ),
            num_epochs=1501,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                N=1000,
                n_random_steps=2,
                test_p=.9,
                dataset_path={'train': None,
                              'test': None,
                              },
                augment_data=False,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
                tag="ccrig_tuning_orig_network",
            ),
            vae_trainer_class=CCVAETrainer,
            vae_class=CCVAE,
            vae_kwargs=dict(
                input_channels=3,
                imsize=48,
            ),
            algo_kwargs=dict(
                key_to_reconstruct='x_t',
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                weight_decay=0.0,
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),
            save_period=50,
        ),
        ccvae_or_cbigan_exp=True,
        train_model_func=train_vae,
        encoder_wrapper=ConditionalEncoderWrappedEnv,
        # presampled_goal_kwargs=dict(
        #     eval_goals='', #HERE
        #     expl_goals='',
        # ),
        presampled_goal_kwargs=dict(
            # eval_goals='s3doodad/demos/icra2021/dataset_v2/goals_drawervertical_close_final.npy', # HERE
            eval_goals='s3doodad/demos/icra2021/dataset_v2/goals_drawervertical_open_final.npy', # HERE
            expl_goals='',
        ),
        pickle_paths=True,
        launcher_config=dict(
            unpack_variant=True,
            region='us-east-2', #HERE
        ),
        pretrained_vae_path="/home/ashvin/real_world_ccvae.pt",
        pretrained_algo_path="/home/ashvin/data/ashvin/icra2021/widowx/sawyer-ccrig-drawervertical-close1/run0/id0/itr_0.pt",
    )
    search_space = {
        "seed": range(3),
        'path_loader_kwargs.demo_paths': [demo_paths],
        'trainer_kwargs.beta': [0.3],
        'num_pybullet_objects':[None],
        'policy_kwargs.min_log_std': [-6],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.clip_score': [2, ],
        'trainer_kwargs.awr_min_q': [True, ],
        'trainer_kwargs.reward_transform_kwargs': [None, ],
        'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0),],
        'qf_kwargs.output_activation': [Clamp(max=0)],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)
    run_variants(awac_rig_experiment, variants, run_id=1)