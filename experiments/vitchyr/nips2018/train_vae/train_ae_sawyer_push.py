
import rlkit.misc.hyperparameter as hyp
from rlkit.images.camera import (
    sawyer_init_camera_zoomed_in,
    sawyer_init_camera,
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE, AutoEncoder
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.sawyer2d_push_new_easy_data import generate_vae_dataset


def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['get_data_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = AutoEncoder(representation_size, input_channels=3)
    if ptu.gpu_enabled():
        m.to(ptu.device)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(epoch, save_reconstruction=should_save_imgs,
                     save_scatterplot=should_save_imgs)
        if should_save_imgs:
            t.dump_samples(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'
    use_gpu = True

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'vae-sawyer-new-push-easy-zoomed-in-1000'
    exp_prefix = 'ae-sawyer-push-lre-2'

    variant = dict(
        beta=0,
        num_epochs=100,
        get_data_kwargs=dict(
            # N=1000,
            # init_camera=sawyer_init_camera_zoomed_in,
            dataset_path='05-22-sawyer_push_dataset'
                         # '/sawyer_push_new_easy1000_sawyer_init_camera.npy',
                         '/sawyer_push_new_easy1000_sawyer_init_camera_zoomed_in.npy',
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-2,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 30, 100],
            # y_values=[0, 0, 0.1, 0.5],
            y_values=[0, 5, 5],
        ),
        save_period=5,
    )

    search_space = {
        'representation_size': [16],
        # 'beta_schedule_kwargs.y_values': [
        #     [0, 0, 0.1, 0.5],
        #     [0, 0, 0.1, 0.1],
        #     [0, 0, 5, 5],
        # ],
        # 'algo_kwargs.lr': [1e-3, 1e-2],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=use_gpu,
                trial_dir_suffix='r'+str(variant.get('representation_size', 0)),
            )
