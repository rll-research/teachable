import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


class WorkerData(Worker):
    def __init__(self, simulation_sleep):
        super().__init__()
        self.simulation_sleep = simulation_sleep
        self.env = None
        self.env_sampler = None
        self.dynamics_sample_processor = None
        self.samples_data_arr = []

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,  # UNUSED
            dynamics_model_pickle,
            feed_dict,
    ):

        from meta_mb.samplers.sampler import Sampler
        from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)

        self.env = env
        self.env_sampler = Sampler(env=env, policy=policy, **feed_dict['sampler'])
        self.dynamics_sample_processor = ModelSampleProcessor(
            **feed_dict['sample_processor']
        )

    def prepare_start(self):
        random_sinusoid = self.queue.get()
        self.step(random_sinusoid)
        self.push()

    def step(self, random_sinusoid=(False, False)):
        time_step = time.time()

        if self.itr_counter == 1 and self.env_sampler.policy.dynamics_model.normalization is None:
            if self.verbose:
                logger.log('Data starts first step...')
            self.env_sampler.policy.dynamics_model = pickle.loads(self.queue.get())
            if self.verbose:
                logger.log('Data first step done...')

        '''------------- Obtaining samples from the environment -----------'''

        if self.verbose:
            logger.log("Data is obtaining samples...")
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random_sinusoid[0],
            sinusoid=random_sinusoid[1],
            log_prefix='Data-EnvSampler-',
        )

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Data is processing samples...")
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='Data-EnvTrajs-',
        )

        self.samples_data_arr.append(samples_data)
        time_step = time.time() - time_step

        time_sleep = max(self.simulation_sleep - time_step, 0)
        time.sleep(time_sleep)

        logger.logkv('Data-TimeStep', time_step)
        logger.logkv('Data-TimeSleep', time_sleep)

    def _synch(self, dynamics_model_state_pickle):
        time_synch = time.time()
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.env_sampler.policy.dynamics_model.set_shared_params(dynamics_model_state)
        time_synch = time.time() - time_synch

        logger.logkv('Data-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        self.queue_next.put(pickle.dumps(self.samples_data_arr))
        self.samples_data_arr = []
        time_push = time.time() - time_push

        logger.logkv('Data-TimePush', time_push)

    def set_stop_cond(self):
        if self.itr_counter >= self.n_itr:
            self.stop_cond.set()

