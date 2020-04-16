import time, pickle
from queue import Empty
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


class WorkerPolicy(Worker):
    def __init__(self, algo_str, sampler_str='metrpo'):
        super().__init__()
        self.policy = None
        self.baseline = None
        self.model_sampler = None
        self.model_sample_processor = None
        self.algo = algo_str
        self.sampler_str = sampler_str

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):

        from meta_mb.samplers.metrpo_samplers.metrpo_sampler import METRPOSampler
        from meta_mb.samplers.bptt_samplers.bptt_sampler import BPTTSampler
        from meta_mb.samplers.base import SampleProcessor
        from meta_mb.algos.ppo import PPO
        from meta_mb.algos.trpo import TRPO

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)
        dynamics_model = pickle.loads(dynamics_model_pickle)

        self.policy = policy
        self.baseline = baseline
        if self.sampler_str == 'metrpo':
            self.model_sampler = METRPOSampler(env=env, policy=policy, dynamics_model=dynamics_model, **feed_dict['model_sampler'])
        elif self.sampler_str == 'bptt':
            self.model_sampler = BPTTSampler(env=env, policy=policy, dynamics_model=dynamics_model, **feed_dict['model_sampler'])
        else:
            raise NotImplementedError
        self.model_sample_processor = SampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
        if self.algo == 'meppo':
            self.algo = PPO(policy=policy, **feed_dict['algo'])
        elif self.algo == 'metrpo':
            self.algo = TRPO(policy=policy, **feed_dict['algo'])
        else:
            raise NotImplementedError('algo_str must be meppo or metrpo')

    def prepare_start(self):
        dynamics_model = pickle.loads(self.queue.get())
        self.model_sampler.dynamics_model = dynamics_model
        if hasattr(self.model_sampler, 'vec_env'):
            self.model_sampler.vec_env.dynamics_model = dynamics_model
        self.step()
        # self.queue_next.put(pickle.dumps(self.result))
        self.push()

    def step(self):
        time_step = time.time()

        """ -------------------- Sampling --------------------------"""

        if self.verbose:
            logger.log("Policy is obtaining samples ...")
        paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-')

        """ ----------------- Processing Samples ---------------------"""

        if self.verbose:
            logger.log("Policy is processing samples ...")
        samples_data = self.model_sample_processor.process_samples(
            paths,
            log='all',
            log_prefix='Policy-'
        )

        if type(paths) is list:
            self.log_diagnostics(paths, prefix='Policy-')
        else:
            self.log_diagnostics(sum(paths.values(), []), prefix='Policy-')

        """ ------------------ Policy Update ---------------------"""

        if self.verbose:
            logger.log("Policy optimization...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        self.algo.optimize_policy(samples_data, log=True, verbose=self.verbose, prefix='Policy-')

        self.policy = self.model_sampler.policy
        time_step = time.time() - time_step

        logger.logkv('Policy-TimeStep', time_step)

    def _synch(self, dynamics_model_state_pickle):
        time_synch = time.time()
        if self.verbose:
            logger.log('Policy is synchronizing...')
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.model_sampler.dynamics_model.set_shared_params(dynamics_model_state)
        if hasattr(self.model_sampler, 'vec_env'):
            self.model_sampler.vec_env.dynamics_model.set_shared_params(dynamics_model_state)
        time_synch = time.time() - time_synch

        logger.logkv('Policy-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        policy_state_pickle = pickle.dumps(self.policy.get_shared_param_values())
        assert policy_state_pickle is not None
        while self.queue_next.qsize() > 5:
            try:
                logger.log('Policy is off loading data from queue_next...')
                _ = self.queue_next.get_nowait()
            except Empty:
                # very rare chance to reach here
                break
        self.queue_next.put(policy_state_pickle)
        time_push = time.time() - time_push

        logger.logkv('Policy-TimePush', time_push)

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
