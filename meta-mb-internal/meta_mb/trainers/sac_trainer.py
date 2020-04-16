import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger
from meta_mb.replay_buffers.simple_replay_buffer import SimpleReplayBuffer


class Trainer(object):
    """
    Performs steps for MAML
    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            task=None,
            sess=None,
            n_initial_exploration_steps=1e3,
            num_grad_steps=None,
            max_replay_buffer_size=1e5,
            ):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.task = task
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.replay_buffer = SimpleReplayBuffer(self.env, max_replay_buffer_size)
        if num_grad_steps is None:
            self.num_grad_steps = self.sampler.total_samples
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo
        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """

        with self.sess.as_default() as sess:
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))
            start_time = time.time()

            if self.start_itr == 0:
                self.algo._update_target(tau=1.0)
                if self.n_initial_exploration_steps > 0:
                    while self.replay_buffer._size < self.n_initial_exploration_steps:
                        paths = self.sampler.obtain_samples(log=True, log_prefix='train-', random=True)
                        samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                        sample_num = samples_data['observations'].shape[0]
                        for i in range(sample_num):
                            self.replay_buffer.add_sample(samples_data['observations'][i],
                                                          samples_data['actions'][i], samples_data['rewards'][i],
                                                          samples_data['dones'][i], samples_data['next_observations'][i],
                                                          )

            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""

                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.sampler.obtain_samples(log=True, log_prefix='train-')
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""
                # check how the samples are processed
                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                sample_num = samples_data['observations'].shape[0]
                for i in range(sample_num):
                    self.replay_buffer.add_sample(samples_data['observations'][i], samples_data['actions'][i],
                                                  samples_data['rewards'][i], samples_data['dones'][i],
                                                  samples_data['next_observations'][i])
                proc_samples_time = time.time() - time_proc_samples_start

                paths = self.sampler.obtain_samples(log=True, log_prefix='eval-', deterministic=True)
                _ = self.sample_processor.process_samples(paths, log='all', log_prefix='eval-')

                self.log_diagnostics(paths, prefix='train-')

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")

                time_optimization_step_start = time.time()
                self.algo.optimize_policy(self.replay_buffer, itr - self.start_itr, self.num_grad_steps)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time-Optimization', time.time() - time_optimization_step_start)
                logger.logkv('Time-SampleProc', np.sum(proc_samples_time))
                logger.logkv('Time-Sampling', sampling_time)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
