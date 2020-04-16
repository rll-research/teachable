import time, pickle
from queue import Empty
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


class WorkerPolicy(Worker):
    def __init__(self, num_inner_grad_steps, sampler_str='mbmpo'):
        super().__init__()
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = None
        self.baseline = None
        self.model_sampler = None
        self.model_sample_processor = None
        self.algo = None
        self.sampler_str = sampler_str

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
    ):

        from meta_mb.samplers.mbmpo_samplers.mbmpo_sampler import MBMPOSampler
        from meta_mb.samplers.bptt_samplers.meta_bptt_sampler import MetaBPTTSampler
        from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
        from meta_mb.meta_algos.trpo_maml import TRPOMAML

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)
        dynamics_model = pickle.loads(dynamics_model_pickle)

        self.policy = policy
        self.baseline = baseline
        if self.sampler_str == 'mbmpo':
            self.model_sampler = MBMPOSampler(env=env, policy=policy, dynamics_model=dynamics_model, **feed_dict['model_sampler'])
        elif self.sampler_str == 'bptt':
            self.model_sampler = MetaBPTTSampler(env=env, policy=policy, dynamics_model=dynamics_model,
                                                 **feed_dict['model_sampler'])
        else:
            raise NotImplementedError
        self.model_sample_processor = MAMLSampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
        self.algo = TRPOMAML(policy=policy, **feed_dict['algo'])

    def prepare_start(self):
        dynamics_model = pickle.loads(self.queue.get())
        self.model_sampler.dynamics_model = dynamics_model
        if hasattr(self.model_sampler, 'vec_env'):
            self.model_sampler.vec_env.dynamics_model = dynamics_model
        self.step()
        self.push()

    def step(self):
        time_step = time.time()

        ''' --------------- MAML steps --------------- '''

        self.policy.switch_to_pre_update()  # Switch to pre-update policy
        all_samples_data = []

        for step in range(self.num_inner_grad_steps+1):
            if self.verbose:
                logger.log("Policy Adaptation-Step %d **" % step)

            """ -------------------- Sampling --------------------------"""

            #time_sampling = time.time()
            paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-', buffer=None)
            #time_sampling = time.time() - time_sampling

            """ ----------------- Processing Samples ---------------------"""

            #time_sample_proc = time.time()
            samples_data = self.model_sample_processor.process_samples(
                paths,
                log='all',
                log_prefix='Policy-'
            )
            all_samples_data.append(samples_data)
            #time_sample_proc = time.time() - time_sample_proc

            self.log_diagnostics(sum(list(paths.values()), []), prefix='Policy-')

            """ ------------------- Inner Policy Update --------------------"""

            #time_algo_adapt = time.time()
            if step < self.num_inner_grad_steps:
                self.algo._adapt(samples_data)
            #time_algo_adapt = time.time() - time_algo_adapt

        """ ------------------ Outer Policy Update ---------------------"""

        if self.verbose:
            logger.log("Policy is optimizing...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        #time_algo_opt = time.time()
        self.algo.optimize_policy(all_samples_data, prefix='Policy-')
        #time_algo_opt = time.time() - time_algo_opt

        time_step = time.time() - time_step
        self.policy = self.model_sampler.policy

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
