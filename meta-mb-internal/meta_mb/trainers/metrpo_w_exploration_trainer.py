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
            model_sample_processor,
            dynamics_sample_processor,
            policy,
            dynamics_model,
            n_itr,
            num_rollouts_per_iter,
            start_itr=0,
            grad_steps_per_rollout=5,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            log_real_performance=True,
            sample_from_buffer=True,
            ):
        self.algo = algo
        self.env = env
        self.model_sampler = model_sampler
        self.model_sample_processor = model_sample_processor
        self.env_sampler = env_sampler
        self.dynamics_sample_processor = dynamics_sample_processor
        self.dynamics_model = dynamics_model
        self.baseline = model_sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_rollouts_per_iter = num_rollouts_per_iter
        self.dynamics_model_max_epochs = dynamics_model_max_epochs

        self.initial_random_samples = initial_random_samples
        self.grad_steps_per_rollout = grad_steps_per_rollout
        self.log_real_performance = log_real_performance
        self.sample_from_buffer = sample_from_buffer

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
            # uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            # sess.run(tf.variables_initializer(uninit_vars))
            sess.run(tf.global_variables_initializer())

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    self.env_sampler.total_samples *= self.num_rollouts_per_iter
                    env_paths = self.env_sampler.obtain_samples(log=True,
                                                                random=self.initial_random_samples,
                                                                log_prefix='Data-EnvSampler-',
                                                                verbose=True)
                    self.env_sampler.total_samples /= self.num_rollouts_per_iter

                time_env_samp_proc = time.time()
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                              log=True,
                                                                              log_prefix='Data-EnvTrajs-')
                self.env.log_diagnostics(env_paths, prefix='Data-EnvTrajs-')
                logger.record_tabular('Data-TimeEnvSampleProc', time.time() - time_env_samp_proc)

                buffer = samples_data if self.sample_from_buffer else None

                ''' --------------- fit dynamics model --------------- '''
                logger.log("Training dynamics model for %i epochs ..." % self.dynamics_model_max_epochs)
                time_fit_start = time.time()
                self.dynamics_model.fit(samples_data['observations'],
                                        samples_data['actions'],
                                        samples_data['next_observations'],
                                        epochs=self.dynamics_model_max_epochs, verbose=False,
                                        log_tabular=True, prefix='Model-')

                logger.record_tabular('Model-TimeModelFit', time.time() - time_fit_start)

                env_paths = []
                for id_rollout in range(self.num_rollouts_per_iter):
                    times_dyn_sampling = []
                    times_dyn_sample_processing = []
                    times_optimization = []
                    times_step = []

                    grad_steps_per_rollout = self.grad_steps_per_rollout
                    for step in range(grad_steps_per_rollout):

                        # logger.log("\n ---------------- Grad-Step %d ----------------" % int(grad_steps_per_rollout*itr*self.num_rollouts_per_iter,
                        #                                                             + id_rollout * grad_steps_per_rollout + step))
                        step_start_time = time.time()

                        """ -------------------- Sampling --------------------------"""

                        logger.log("Obtaining samples from the model...")
                        time_env_sampling_start = time.time()
                        paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-', buffer=buffer)
                        sampling_time = time.time() - time_env_sampling_start

                        """ ----------------- Processing Samples ---------------------"""

                        logger.log("Processing samples from the model...")
                        time_proc_samples_start = time.time()
                        samples_data = self.model_sample_processor.process_samples(paths, log='all',
                                                                                   log_prefix='Policy-')
                        proc_samples_time = time.time() - time_proc_samples_start

                        if type(paths) is list:
                            self.log_diagnostics(paths, prefix='Policy-')
                        else:
                            self.log_diagnostics(sum(paths.values(), []), prefix='Policy-')

                        """ ------------------ Policy Update ---------------------"""

                        logger.log("Optimizing policy...")
                        time_optimization_step_start = time.time()
                        self.algo.optimize_policy(samples_data)
                        optimization_time = time.time() - time_optimization_step_start

                        times_dyn_sampling.append(sampling_time)
                        times_dyn_sample_processing.append(proc_samples_time)
                        times_optimization.append(optimization_time)
                        times_step.append(time.time() - step_start_time)

                    logger.log("Obtaining random samples from the environment...")
                    env_paths.extend(self.env_sampler.obtain_samples(log=True,
                                                                     log_prefix='Data-EnvSampler-',
                                                                     verbose=True))

                logger.record_tabular('Data-TimeEnvSampling', time.time() - time_env_sampling_start)
                logger.log("Processing environment samples...")

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Iteration', itr)
                logger.logkv('n_timesteps', self.env_sampler.total_timesteps_sampled)
                logger.logkv('Policy-TimeSampleProc', np.sum(times_dyn_sample_processing))
                logger.logkv('Policy-TimeSampling', np.sum(times_dyn_sampling))
                logger.logkv('Policy-TimeAlgoOpt', np.sum(times_optimization))
                logger.logkv('Policy-TimeStep', np.sum(times_step))

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

            logger.logkv('Trainer-TimeTotal', time.time() - start_time)

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # self.env.log_diagnostics(paths, prefix) # FIXME: Currently this doesn't work with the model
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
