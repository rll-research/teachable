import tensorflow as tf
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
            sampler,
            sample_processor,
            policy,
            dynamics_model,
            value_function,
            n_itr,
            start_itr=0,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            vfun_max_epochs=200,

            ):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.vfun_max_epochs = vfun_max_epochs

        self.initial_random_samples = initial_random_samples

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
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                logger.log("Obtaining samples from the environment using the policy...")
                env_paths = self.sampler.obtain_samples(log=True, log_prefix='')

                logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
                logger.log("Processing environment samples...")

                # first processing just for logging purposes
                time_env_samp_proc = time.time()
                samples_data = self.sample_processor.process_samples(env_paths,
                                                                     log=True,
                                                                     log_prefix='EnvTrajs-')

                logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                ''' --------------- fit dynamics model --------------- '''

                time_fit_start = time.time()

                logger.log("Training dynamics model for %i epochs ..." % self.dynamics_model_max_epochs)
                self.dynamics_model.fit(samples_data['observations'],
                                        samples_data['actions'],
                                        samples_data['next_observations'],
                                        epochs=self.dynamics_model_max_epochs,
                                        verbose=False, log_tabular=True,
                                        early_stopping=True, compute_normalization=False)

                logger.log("Training the value function for %i epochs ..." % self.vfun_max_epochs)
                self.value_function.fit(samples_data['observations'],
                                        samples_data['returns'],
                                        epochs=self.vfun_max_epochs,
                                        verbose=False, log_tabular=True, compute_normalization=False)

                logger.log("Training the policy ...")
                self.algo.optimize_policy(samples_data)

                logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                self.log_diagnostics(env_paths, '')
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
        return dict(itr=itr, policy=self.policy, env=self.env,
                    dynamics_model=self.dynamics_model, value_function=self.value_function)

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
