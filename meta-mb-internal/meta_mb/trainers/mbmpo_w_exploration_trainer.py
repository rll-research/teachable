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
            num_inner_grad_steps=1,
            meta_steps_per_iter=30,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            log_real_performance=True,
            sample_from_buffer=False,
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
        self.num_inner_grad_steps = num_inner_grad_steps
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        assert policy.meta_batch_size % num_rollouts_per_iter == 0
        self.num_rollouts_per_iter = num_rollouts_per_iter

        self.initial_random_samples = initial_random_samples
        self.meta_steps_per_iter = meta_steps_per_iter
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

            if type(self.meta_steps_per_iter) is tuple:
                meta_steps_per_iter = np.linspace(self.meta_steps_per_iter[0]
                                                  , self.meta_steps_per_iter[1], self.n_itr).astype(np.int)
            else:
                meta_steps_per_iter = [self.meta_steps_per_iter] * self.n_itr
            start_time = time.time()

            for itr in range(self.start_itr, self.n_itr):
                self.policy.switch_to_pre_update()
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.env_sampler.obtain_samples(log=True,
                                                                random=self.initial_random_samples,
                                                                log_prefix='Data-EnvSampler-',
                                                                verbose=True,
                                                                )

                    if type(env_paths) is dict:
                        env_paths = list(env_paths.values()) if env_paths is dict else env_paths
                        idxs = np.random.choice(range(len(env_paths)),
                                                size=self.num_rollouts_per_iter,
                                                replace=False)
                        env_paths = sum([env_paths[idx] for idx in idxs], [])

                    elif type(env_paths) is list:
                        idxs = np.random.choice(range(len(env_paths)),
                                                size=self.num_rollouts_per_iter,
                                                replace=False)
                        env_paths = [env_paths[idx] for idx in idxs]


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

                ''' ------------ log real performance --------------- '''

                if self.log_real_performance:
                    logger.log("Evaluating the performance of the real policy")
                    self.policy.switch_to_pre_update()
                    log_env_paths = self.env_sampler.obtain_samples(log=True, log_prefix='PrePolicy-')
                    log_samples_data = self.model_sample_processor.process_samples(log_env_paths , log='all',
                                                                               log_prefix='PrePolicy-')
                    self.algo._adapt(log_samples_data )
                    log_env_paths = self.env_sampler.obtain_samples(log=True, log_prefix='PostPolicy-')
                    self.model_sample_processor.process_samples(log_env_paths , log='all', log_prefix='PostPolicy-')

                ''' --------------- MAML steps --------------- '''

                env_paths = []
                for id_rollout in range(self.num_rollouts_per_iter):
                    logger.log("Obtaining random samples from the environment...")
                    self.policy.switch_to_pre_update()
                    env_paths.append(self.env_sampler.obtain_samples(log=True,
                                                                     log_prefix='Data-EnvSampler-',
                                                                     verbose=True)[0])
                    times_dyn_sampling = []
                    times_dyn_sample_processing = []
                    times_meta_sampling = []
                    times_inner_step = []
                    times_total_inner_step = []
                    times_outer_step = []
                    times_maml_steps = []

                    meta_grad_steps = int(np.ceil(meta_steps_per_iter[itr] / self.num_rollouts_per_iter))
                    for meta_itr in range(meta_grad_steps):

                        logger.log("\n ---------------- Meta-Step %d ----------------" % int(sum(meta_steps_per_iter[:itr])
                                                                                             + id_rollout * meta_grad_steps + meta_itr))
                        self.policy.switch_to_pre_update()  # Switch to pre-update policy

                        all_samples_data, all_paths = [], []
                        list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
                        time_maml_steps_start = time.time()
                        start_total_inner_time = time.time()

                        for step in range(self.num_inner_grad_steps+1):
                            logger.log("\n ** Adaptation-Step %d **" % step)

                            """ -------------------- Sampling --------------------------"""

                            logger.log("Obtaining samples...")
                            time_env_sampling_start = time.time()
                            paths = self.model_sampler.obtain_samples(log=True,
                                                                      log_prefix='Step_%d-' % step,
                                                                      buffer=buffer)
                            list_sampling_time.append(time.time() - time_env_sampling_start)
                            all_paths.append(paths)

                            """ ----------------- Processing Samples ---------------------"""

                            logger.log("Processing samples...")
                            time_proc_samples_start = time.time()
                            samples_data = self.model_sample_processor.process_samples(paths, log='all', log_prefix='Step_%d-' % step)
                            all_samples_data.append(samples_data)
                            list_proc_samples_time.append(time.time() - time_proc_samples_start)

                            self.log_diagnostics(sum(list(paths.values()), []), prefix='Step_%d-' % step)

                            """ ------------------- Inner Policy Update --------------------"""

                            time_inner_step_start = time.time()
                            if step < self.num_inner_grad_steps:
                                logger.log("Computing inner policy updates...")
                                self.algo._adapt(samples_data)

                            list_inner_step_time.append(time.time() - time_inner_step_start)
                        total_inner_time = time.time() - start_total_inner_time

                        time_maml_opt_start = time.time()

                        """ ------------------ Outer Policy Update ---------------------"""

                        logger.log("Optimizing policy...")
                        # This needs to take all samples_data so that it can construct graph for meta-optimization.
                        time_outer_step_start = time.time()
                        self.algo.optimize_policy(all_samples_data)

                    times_inner_step.append(list_inner_step_time)
                    times_total_inner_step.append(total_inner_time)
                    times_outer_step.append(time.time() - time_outer_step_start)
                    times_meta_sampling.append(np.sum(list_sampling_time))
                    times_dyn_sampling.append(list_sampling_time)
                    times_dyn_sample_processing.append(list_proc_samples_time)
                    times_maml_steps.append(time.time() - time_maml_steps_start)


                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                if self.log_real_performance:
                    logger.logkv('n_timesteps', self.env_sampler.total_timesteps_sampled/(3 * self.policy.meta_batch_size))
                else:
                    logger.logkv('n_timesteps', self.env_sampler.total_timesteps_sampled/self.policy.meta_batch_size)
                logger.logkv('AvgTime-OuterStep', np.mean(times_outer_step))
                logger.logkv('AvgTime-InnerStep', np.mean(times_inner_step))
                logger.logkv('AvgTime-TotalInner', np.mean(times_total_inner_step))
                logger.logkv('AvgTime-InnerStep', np.mean(times_inner_step))
                logger.logkv('AvgTime-SampleProc', np.mean(times_dyn_sample_processing))
                logger.logkv('AvgTime-Sampling', np.mean(times_dyn_sampling))
                logger.logkv('AvgTime-MAMLSteps', np.mean(times_maml_steps))

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
        # self.env.log_diagnostics(paths, prefix) # FIXME: Currently this doesn't work with the model
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
