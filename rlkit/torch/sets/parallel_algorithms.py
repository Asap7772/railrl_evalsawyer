from collections.__init__ import OrderedDict

from rlkit.core import logger
from rlkit.core.logging import append_log
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.core.timer import Timer


class ParallelAlgorithms(object):
    def __init__(
            self,
            algorithms: OrderedDict,
            num_iters: int,
    ):
        self.algorithms = algorithms
        self.num_iters = num_iters
        self._start_epoch = 0
        self.epoch = self._start_epoch

        self.post_train_funcs = []
        self.post_epoch_funcs = []
        self.timer = Timer()

    def run(self):
        self.timer.return_global_times = True
        for epoch in range(self.num_iters):
            self.begin_epoch()
            self.timer.start_timer('saving')
            logger.save_itr_params(self.epoch, self._get_snapshot())
            self.timer.stop_timer('saving')
            log_dict = self.train(epoch)
            logger.record_dict(log_dict)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
            self.end_epoch(epoch)
        logger.save_itr_params(self.epoch, self._get_snapshot())

    def train(self, epoch):
        self.timer.start_timer('training', unique=False)
        for algorithm in self.algorithms.values():
            for fn in algorithm.pre_train_funcs:
                fn(algorithm, epoch)
            algorithm.train()
            for fn in algorithm.post_train_funcs:
                fn(algorithm, epoch)
        self.timer.stop_timer('training')
        log_stats = self.get_diagnostics()
        return log_stats

    def begin_epoch(self):
        self.timer.reset()

    def end_epoch(self, epoch):
        for post_train_func in self.post_train_funcs:
            post_train_func(self, epoch)

        for algorithm in self.algorithms.values():
            algorithm.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)
        self.epoch = epoch + 1

    def _get_snapshot(self):
        snapshot = {}
        for name, algo in self.algorithms.items():
            for k, v in algo.get_snapshot().items():
                snapshot['{}/{}'.format(name, k)] = v
        return snapshot

    def get_diagnostics(self):
        self.timer.start_timer('logging', unique=False)
        algo_log = OrderedDict()
        for name, algorithm in self.algorithms.items():
            append_log(
                algo_log,
                algorithm.get_diagnostics(),
                prefix=name,
                divider='/',
            )
        append_log(algo_log, _get_epoch_timings())
        algo_log['epoch'] = self.epoch
        self.timer.stop_timer('logging')
        return algo_log

    def to(self, device):
        for algo in self.algorithms.values():
            algo.to(device)