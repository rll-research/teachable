import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker
from queue import Empty
import numpy as np


class WorkerModel(Worker):
    def __init__(self):
        super().__init__()
        self.sum_model_itr = 0
        self.with_new_data = None
        self.remaining_model_idx = None
        self.valid_loss_rolling_average = None
        self.dynamics_model = None

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):
        self.dynamics_model = pickle.loads(dynamics_model_pickle)

    def prepare_start(self):
        samples_data_arr = pickle.loads(self.queue.get())
        self._synch(samples_data_arr, check_init=True)
        self.step()
        self.queue_next.put(pickle.dumps(self.dynamics_model))

    def process_queue(self):
        do_push = 0
        samples_data_arr = []
        while True:
            try:
                if not self.remaining_model_idx:
                    logger.log('Model at iteration {} is block waiting for data'.format(self.itr_counter))
                    # FIXME: check stop_cond
                    time_wait = time.time()
                    samples_data_arr_pickle = self.queue.get()
                    time_wait = time.time() - time_wait
                    logger.logkv('Model-TimeBlockWait', time_wait)
                    self.remaining_model_idx = list(range(self.dynamics_model.num_models))
                else:
                    if self.verbose:
                        logger.log('Model try get_nowait.........')
                    samples_data_arr_pickle = self.queue.get_nowait()
                if samples_data_arr_pickle == 'push':
                    # Only push once before executing another step
                    if do_push == 0:
                        do_push = 1
                        self.push()
                else:
                    samples_data_arr.extend(pickle.loads(samples_data_arr_pickle))
            except Empty:
                break

        do_synch = len(samples_data_arr)
        if do_synch:
            self._synch(samples_data_arr)

        do_step = 1

        if self.verbose:
            logger.log('Model finishes processing queue with {}, {}, {}......'.format(do_push, do_synch, do_step))

        return do_push, do_synch, do_step

    def step(self, obs=None, act=None, obs_next=None):
        time_model_fit = time.time()

        """ --------------- fit dynamics model --------------- """

        if self.verbose:
            logger.log('Model at iteration {} is training for one epoch...'.format(self.itr_counter))
        self.sum_model_itr += len(self.remaining_model_idx)
        self.remaining_model_idx, self.valid_loss_rolling_average = self.dynamics_model.fit_one_epoch(
            remaining_model_idx=self.remaining_model_idx,
            valid_loss_rolling_average_prev=self.valid_loss_rolling_average,
            with_new_data=self.with_new_data,
            verbose=self.verbose,
            log_tabular=True,
            prefix='Model-',
        )
        self.with_new_data = False
        time_model_fit = time.time() - time_model_fit

        logger.logkv('Model-TimeStep', time_model_fit)

    def _synch(self, samples_data_arr, check_init=False):
        time_synch = time.time()
        if self.verbose:
            logger.log('Model at {} is synchronizing...'.format(self.itr_counter))
        obs = np.concatenate([samples_data['observations'] for samples_data in samples_data_arr])
        act = np.concatenate([samples_data['actions'] for samples_data in samples_data_arr])
        obs_next = np.concatenate([samples_data['next_observations'] for samples_data in samples_data_arr])
        self.dynamics_model.update_buffer(
            obs=obs,
            act=act,
            obs_next=obs_next,
            check_init=check_init,
        )

        # Reset variables for early stopping condition
        logger.logkv('Model-AvgEpochs', self.sum_model_itr/self.dynamics_model.num_models)
        self.sum_model_itr = 0
        self.with_new_data = True
        self.remaining_model_idx = list(range(self.dynamics_model.num_models))
        self.valid_loss_rolling_average = None
        time_synch = time.time() - time_synch

        logger.logkv('Model-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        state_pickle = pickle.dumps(self.dynamics_model.get_shared_param_values())
        assert state_pickle is not None
        while self.queue_next.qsize() > 5:
            try:
                logger.log('Model is off loading data from queue_next...')
                _ = self.queue_next.get_nowait()
            except Empty:
                break
        self.queue_next.put(state_pickle)
        time_push = time.time() - time_push

        logger.logkv('Model-TimePush', time_push)


