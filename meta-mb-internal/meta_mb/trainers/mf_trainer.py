import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger
from meta_mb.samplers.utils import rollout

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
            use_rp_inner=False,
            use_rp_outer=False,
            reward_threshold=0.8,
            exp_name="",
            videos_every=5,
            curriculum_step=0,
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
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.use_rp_inner = use_rp_inner
        self.use_rp_outer = use_rp_outer
        self.reward_threshold = reward_threshold
        self.curriculum_step = curriculum_step
        self.exp_name = exp_name
        self.videos_every = videos_every

    def check_advance_curriculum(self, data):
        rewards = data['avg_reward']
        should_advance = rewards > self.reward_threshold
        if should_advance:
            self.curriculum_step += 1
        return should_advance

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
            advance_curriculum = False

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                self.sampler.update_tasks()
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""

                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.sampler.obtain_samples(log=True, log_prefix='train-', advance_curriculum=advance_curriculum)
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""

                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                advance_curriculum = self.check_advance_curriculum(samples_data)
                proc_samples_time = time.time() - time_proc_samples_start

                """ ------------------ Reward Predictor Splicing ---------------------"""
                r_discrete, logprobs = self.algo.reward_predictor.get_actions(samples_data['env_infos']['next_obs_rewardfree'])
                self.log_rew_pred(r_discrete[:,:,0], samples_data['rewards'], samples_data['env_infos'])
                # Splice into the inference process
                if self.use_rp_inner:
                    samples_data['observations'][:,:, -2] = r_discrete[:, :, 0]
                # Splice into the meta-learning process
                if self.use_rp_outer:
                    samples_data['rewards'] = logprobs[:, :, 1]   
                
                """ ------------------ End Reward Predictor Splicing ---------------------"""

                if type(paths) is list:
                    self.log_diagnostics(paths, prefix='train-')
                else:
                    self.log_diagnostics(sum(paths.values(), []), prefix='train-')

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_optimization_step_start = time.time()
                self.algo.optimize_policy(samples_data)
                # TODO: Make sure we optimize this for more steps
                self.algo.optimize_reward(samples_data)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time-Optimization', time.time() - time_optimization_step_start)
                logger.logkv('Time-SampleProc', np.sum(proc_samples_time))
                logger.logkv('Time-Sampling', sampling_time)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.logkv('Curriculum Step', self.curriculum_step)
                logger.logkv('Curriculum Percent', self.curriculum_step / len(self.env.levels_list))

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                step = self.curriculum_step
                if advance_curriculum:
                    step -= 1
                logger.save_itr_params(itr, step, params)
                logger.log("Saved")

                logger.dumpkvs()


                # Save videos of the progress periodically, or right before we advance levels
                if advance_curriculum:
                    # Save a video of the original level
                    self.save_videos(step, save_name='ending_video', num_rollouts=10)
                    # Save a video of the new level
                    self.save_videos(step + 1, save_name='beginning_video', num_rollouts=10)
                elif itr % self.videos_every == 0:
                    self.env.set_level_distribution(step)
                    self.save_videos(step, save_name='intermediate_video', num_rollouts=2)

                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def save_videos(self, step, save_name='sample_video', num_rollouts=2):
        paths = rollout(self.env, self.policy, max_path_length=200, reset_every=2, show_last=10, stochastic=True,
                        batch_size=100,
                        video_filename=self.exp_name + '/' + save_name + str(step) + '.mp4', num_rollouts=num_rollouts)
        print('Average Returns: ', np.mean([sum(path['rewards']) for path in paths]))
        print('Average Path Length: ', np.mean([path['env_infos'][-1]['episode_length'] for path in paths]))
        print('Average Success Rate: ', np.mean([path['env_infos'][-1]['success'] for path in paths]))

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        if self.algo.reward_predictor is not None:
            return dict(itr=itr,
                        policy=self.policy,
                        env=self.env,
                        baseline=self.baseline,
                        curriculum_step=self.curriculum_step)
        else:
            return dict(itr=itr,
                        policy=self.policy,
                        env=self.env,
                        baseline=self.baseline,
                        curriculum_step=self.curriculum_step,
                        reward_predictor=self.algo.reward_predictor)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)

    def _log_rew_pred(self, r_discrete, rewards, log_prefix):
        correct = rewards == r_discrete
        positives = rewards > 0
        pred_positives = r_discrete > 0
        negatives = 1 - positives
        incorrect = 1 - correct

        if positives.sum() > 0:
            false_negatives = (incorrect * positives).sum() / positives.sum()
            logger.logkv(log_prefix + 'FalseNegative', false_negatives)
            recall = (correct * positives).sum() / positives.sum()
            logger.logkv(log_prefix + 'Recall', recall)
        else:
            logger.logkv(log_prefix + 'FalseNegative', -1) # Indicates none available

        if negatives.sum() > 0:
            false_positives = (incorrect * negatives).sum() / negatives.sum()
            logger.logkv(log_prefix + 'FalsePositive', false_positives)
        else:
            logger.logkv(log_prefix + 'FalsePositive', -1)

        if pred_positives.sum():
            precision = (correct * positives).sum() / pred_positives.sum()
            logger.logkv(log_prefix + 'Precision', precision)
        else:
            logger.logkv(log_prefix + 'Precision', -1)

        if positives.sum() > 0 and pred_positives.sum() > 0:
            logger.logkv(log_prefix + 'F1', precision * recall / (precision + recall))
        else:
            logger.logkv(log_prefix + 'F1', -1)



    def log_rew_pred(self, r_discrete, rewards, env_infos):
        log_prefix = "RewPred"

        # Flatten, trim out any which are just there for padding
        # Elements where step=0 are just padding on the end.
        curr_elements = 1 - (env_infos['step'].flatten() == 0)
        r_discrete = np.stack([data for data, curr_bool in zip(r_discrete.flatten(), curr_elements) if curr_bool])
        rewards = np.stack([data for data, curr_bool in zip(rewards.flatten(), curr_elements) if curr_bool])
        step = np.stack([data for data, curr_bool in zip(env_infos['step'].flatten(), curr_elements) if curr_bool])


        self._log_rew_pred(r_discrete, rewards, log_prefix + "-")

        # Log split by index in meta-task
        unique_steps = np.unique(env_infos['step'])
        for i in unique_steps:
            if i == 0:
                continue  # Remove 0, which is just filler
            curr_elements = step == i
            r_discrete_i = np.stack([data for data, curr_bool in zip(r_discrete, curr_elements) if curr_bool])
            rewards_i = np.stack([data for data, curr_bool in zip(rewards, curr_elements) if curr_bool])
            self._log_rew_pred(r_discrete_i, rewards_i, log_prefix + str(i) + "-")

        # Log split by dropout
        no_goal = np.stack([dropout for dropout, curr_bool in zip(env_infos['dropout_goal'].flatten(), curr_elements)
                            if curr_bool])
        no_corrections = np.stack(
            [dropout for dropout, curr_bool in zip(env_infos['dropout_corrections'].flatten(), curr_elements)
             if curr_bool])
        yes_goal = np.stack(
            [not dropout for dropout, curr_bool in zip(env_infos['dropout_goal'].flatten(), curr_elements)
             if curr_bool])
        yes_corrections = np.stack(
            [not dropout for dropout, curr_bool in zip(env_infos['dropout_corrections'].flatten(), curr_elements)
             if curr_bool])
        masks = [no_goal, yes_goal, no_corrections, yes_corrections]
        names = ["no_goal", "yes_goal", "no_corrections", "yes_corrections"]
        for name, mask in zip(names, masks):
            # Skip logging if there aren't any in this category (e.g. if we aren't using dropout)
            if mask.sum() == 0:
                continue
            if name == 'yes_corrections':
                print("hi")
            log_prefix_i = log_prefix + name + "-"
            r_discrete_i = np.stack([data for data, curr_bool in zip(r_discrete, mask) if curr_bool.all()])
            rewards_i = np.stack([data for data, curr_bool in zip(rewards, mask) if curr_bool.all()])
            self._log_rew_pred(r_discrete_i, rewards_i, log_prefix_i)