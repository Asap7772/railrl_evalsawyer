import typing
from collections import OrderedDict

from torch.utils import data

from rlkit.core import logger
from rlkit.core.logging import append_log
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.core.timer import Timer
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DictLoader(object):
    r"""Method for iterating over dictionaries."""

    def __init__(self, key_to_batch_sampler: typing.Dict[typing.Any, data.DataLoader]):
        if len(key_to_batch_sampler) == 0:
            raise ValueError("need at least one sampler")
        self.keys, self.samplers = zip(*key_to_batch_sampler.items())

    def __iter__(self):
        # values = []
        for values in zip(*self.samplers):
            yield dict(zip(self.keys, values))

    def __len__(self):
        return len(self.samplers[0])


class BatchTorchAlgorithm(object):
    def __init__(
            self,
            trainer: TorchTrainer,
            data_loader: DictLoader,
            num_iters: int,
            num_epochs_per_iter=1,
            progress_csv_file_name='progress.csv',
            start_epoch=0,
            snapshot_prefix='',
    ):
        self.trainer = trainer
        self.data_loader = data_loader
        self.epoch = start_epoch
        self.num_iters = num_iters
        self.num_epochs_per_iter = num_epochs_per_iter
        self.progress_csv_file_name = progress_csv_file_name

        self.pre_train_funcs = []
        self.post_train_funcs = []
        self.post_epoch_funcs = []
        self.timer = Timer(return_global_times=True)
        self._snapshot_prefix = snapshot_prefix

    def run(self):
        if self.progress_csv_file_name != 'progress.csv':
            logger.remove_tabular_output(
                'progress.csv', relative_to_snapshot_dir=True
            )
            logger.add_tabular_output(
                self.progress_csv_file_name, relative_to_snapshot_dir=True,
            )
        for epoch in range(self.num_iters):
            self.begin_epoch()
            self.timer.start_timer('saving')
            logger.save_itr_params(
                self.epoch,
                self.get_snapshot(),
                snapshot_prefix=self._snapshot_prefix,
            )
            self.timer.stop_timer('saving')
            log_dict = self.train()
            logger.record_dict(log_dict)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
            self.end_epoch(epoch)
        logger.save_itr_params(
            self.epoch,
            self.get_snapshot(),
            snapshot_prefix=self._snapshot_prefix,
        )
        if self.progress_csv_file_name != 'progress.csv':
            logger.remove_tabular_output(
                self.progress_csv_file_name, relative_to_snapshot_dir=True,
            )
            logger.add_tabular_output(
                'progress.csv', relative_to_snapshot_dir=True,
            )

    def train(self):
        self.timer.start_timer('training', unique=False)
        for _ in range(self.num_epochs_per_iter):
            for batch in self.data_loader:
                self.trainer.train_from_torch(batch)
        self.timer.stop_timer('training')
        log_stats = self.get_diagnostics()
        return log_stats

    def begin_epoch(self):
        self.timer.reset()

    def end_epoch(self, epoch):
        for post_train_func in self.post_train_funcs:
            post_train_func(self, epoch)

        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)
        self.epoch = epoch + 1

    def get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        return snapshot

    def get_diagnostics(self):
        self.timer.start_timer('logging', unique=False)
        algo_log = OrderedDict()
        append_log(algo_log, self.trainer.get_diagnostics(), prefix='trainer/')
        append_log(algo_log, _get_epoch_timings(self.timer))
        algo_log['epoch'] = self.epoch
        self.timer.stop_timer('logging')
        return algo_log

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


