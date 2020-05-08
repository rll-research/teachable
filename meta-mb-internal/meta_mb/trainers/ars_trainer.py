import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger


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
            model_sampler,
            env_sampler,
            ars_sample_processor,
            dynamics_sample_processor,
            policy,
            n_itr,
            num_deltas,
            dynamics_model=None,
            start_itr=0,
            steps_per_iter=30,
            initial_random_samples=False,
            sess=None,
            dynamics_model_max_epochs=200,
            log_real_performance=True,
            delta_std=0.03,
            gpu_frac=0.5,
            sample_from_buffer=False,
            ):
        self.algo = algo
        self.env = env
        self.model_sampler = model_sampler
        self.ars_sample_processor = ars_sample_processor
        self.env_sampler = env_sampler
        self.dynamics_sample_processor = dynamics_sample_processor
        self.dynamics_model = dynamics_model
        self.baseline = ars_sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.num_deltas = num_deltas
        self.delta_std = delta_std
        self.sample_from_buffer = sample_from_buffer

        self.initial_random_samples = initial_random_samples
        self.steps_per_iter = steps_per_iter
        self.log_real_performance = log_real_performance

        self._prev_policy = None
        self._last_returns = -1e8

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
            sess = tf.Session(config=config)
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

            if type(self.steps_per_iter) is tuple:
                steps_per_iter = np.linspace(self.steps_per_iter[0], self.steps_per_iter[1], self.n_itr).astype(np.int)
            else:
                steps_per_iter = [self.steps_per_iter] * self.n_itr

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.env_sampler.obtain_samples(log=True, random=True, log_prefix='EnvSampler-')

                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    env_paths = self.env_sampler.obtain_samples(log=True, log_prefix='EnvSampler-')
                    self.policy.obs_filter.stats_increment()

                logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
                logger.log("Processing environment samples...")

                # first processing just for logging purposes
                time_env_samp_proc = time.time()
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                              log=True,
                                                                              log_prefix='EnvTrajs-')

                self.env.log_diagnostics(env_paths, prefix='EnvTrajs-')

                logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                ''' --------------- fit dynamics model --------------- '''

                time_fit_start = time.time()

                logger.log("Training dynamics model for %i epochs ..." % self.dynamics_model_max_epochs)
                if self.dynamics_model is not None:
                    self.dynamics_model.fit(samples_data['observations'],
                                            samples_data['actions'],
                                            samples_data['next_observations'],
                                            epochs=self.dynamics_model_max_epochs,
                                            verbose=False,
                                            log_tabular=True,
                                            compute_normalization=True)

                buffer = None if not self.sample_from_buffer else samples_data

                logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                # returns = np.mean(samples_data['returns'])
                # if returns < self._last_returns:
                #     self.policy.set_params(self._prev_policy)
                #     self._last_returns = returns
                # self._prev_policy = self.policy.get_params()

                ''' ------------ log real performance --------------- '''

                # if self.log_real_performance:
                #     logger.log("Evaluating the performance of the real policy")
                #     env_paths = self.env_sampler.obtain_samples(log=True, log_prefix='RealPolicy-')
                #     _ = self.model_sample_processor.process_samples(env_paths, log='all', log_prefix='PrePolicy-')

                ''' --------------- RS steps --------------- '''

                times_dyn_sampling = []
                times_dyn_sample_processing = []
                times_itr = []
                times_rs_steps = []
                list_sampling_time = []
                list_proc_samples_time = []
                for rs_itr in range(steps_per_iter[itr]):
                    time_itr_start = time.time()
                    logger.log("\n -------------- RS-Step %d --------------" % int(sum(steps_per_iter[:itr]) + rs_itr))
                    deltas = self.policy.get_deltas(self.num_deltas)
                    self.policy.set_deltas(deltas, delta_std=self.delta_std, symmetrical=True)

                    """ -------------------- Sampling --------------------------"""
                    logger.log("Obtaining samples...")
                    time_env_sampling_start = time.time()
                    samples_data = self.model_sampler.obtain_samples(log=True,
                                                                     log_prefix='Models-',
                                                                     buffer=buffer)
                    list_sampling_time.append(time.time() - time_env_sampling_start)

                    """ ---------------------- Processing --------------------- """
                    # TODO: Add preprocessing of the state to see what sort of update rule between the models we want
                    samples_data = self.ars_sample_processor.process_samples(samples_data,
                                                                                 log=True,
                                                                                 log_prefix='step%d-' % rs_itr)

                    if self.dynamics_model is None:
                        self.policy.stats_increment()

                    """ ------------------ Outer Policy Update ---------------------"""
                    logger.log("Optimizing policy...")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    time_rs_start = time.time()
                    self.algo.optimize_policy(samples_data['returns'], deltas)

                    times_dyn_sampling.append(list_sampling_time)
                    times_dyn_sample_processing.append(list_proc_samples_time)
                    times_rs_steps.append(time.time() - time_rs_start)
                    times_itr.append(time.time() - time_itr_start)


                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                if self.dynamics_model is None:
                    logger.logkv('n_timesteps', self.model_sampler.total_timesteps_sampled)
                else:
                    logger.logkv('n_timesteps', self.env_sampler.total_timesteps_sampled)

                logger.logkv('AvgTime-RS', np.mean(times_rs_steps))
                logger.logkv('AvgTime-SampleProc', np.mean(times_dyn_sample_processing))
                logger.logkv('AvgTime-Sampling', np.mean(times_dyn_sampling))
                logger.logkv('AvgTime-ModelItr', np.mean(times_itr))

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
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
