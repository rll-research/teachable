import tensorflow as tf
import torch
import numpy as np
from meta_mb.logger import logger
from meta_mb.samplers.utils import rollout
import os.path as osp
import joblib
import time
import psutil
import os

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
        success_threshold=0.95,
        accuracy_threshold=0.9,
        exp_name="",
        videos_every=5,
        curriculum_step=0,
        config=None,
        log_and_save=True,
        increase_dropout_threshold=float('inf'),
        increase_dropout_increment=None,
        advance_without_teacher=False,
        teacher_info=[],
        sparse_rewards=True,
        distill_only=False,
        mode='collection',
        num_batches=None,
        data_path=None,
        il_trainer=None,
        source='agent',
        batch_size=100,
        eval_every=100,
        save_every=100,
        log_every=10,
        save_videos_every=1000,
        distill_with_teacher=False,
        supervised_model=None,
        reward_predictor=None):
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
        self.success_threshold = success_threshold
        self.accuracy_threshold = accuracy_threshold
        self.curriculum_step = curriculum_step
        self.exp_name = exp_name
        self.videos_every = videos_every
        self.config = config
        self.log_and_save = log_and_save
        self.increase_dropout_threshold = increase_dropout_threshold
        self.increase_dropout_increment = increase_dropout_increment
        self.advance_without_teacher = advance_without_teacher
        self.teacher_info = teacher_info
        self.sparse_rewards = sparse_rewards
        self.distill_only = distill_only
        self.mode = mode
        self.num_batches = num_batches
        self.data_path = data_path
        self.il_trainer = il_trainer
        self.source = source
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.save_every = save_every
        self.save_videos_every = save_videos_every
        self.log_every = log_every
        self.distill_with_teacher = distill_with_teacher
        self.supervised_model = supervised_model
        self.reward_predictor = reward_predictor
        if self.num_batches is not None:
            self.num_train_batches = (self.num_batches * 0.9)
            self.num_val_batches = self.num_batches - self.num_train_batches
            assert self.num_train_batches > 0

    def check_advance_curriculum(self, data):
        num_total_episodes = data['dones'].sum()
        num_successes = data['env_infos']['success'].sum()
        avg_success = num_successes / num_total_episodes
        # Episode length contains the timestep, starting at 1.  Padding values are 0.
        pad_steps = (data['env_infos']['episode_length'] == 0).sum()
        correct_actions = (data['actions'] == data['env_infos']['teacher_action']).sum() - pad_steps
        avg_accuracy = correct_actions / (np.prod(data['actions'].shape) - pad_steps)
        # We take the max since runs which end early will be 0-padded
        dropout_level = np.max(data['env_infos']['dropout_proportion'])
        should_advance_curriculum = (avg_success >= self.success_threshold) and (dropout_level == 1) and (avg_accuracy >= self.accuracy_threshold)
        should_increase_dropout = avg_success >= self.increase_dropout_threshold
        return should_advance_curriculum, should_increase_dropout

    def load_data(self, start_index, end_index):
        batch_index = np.random.randint(start_index, end_index)
        batch_path = osp.join(self.data_path, 'batch_%d.pkl' % batch_index)
        samples_data = joblib.load(batch_path)
        curr_batch_len = len(samples_data['observations'])
        if curr_batch_len < self.batch_size:
            print(f'Found batch of length {curr_batch_len}, which is smaller than desired batch size {self.batch_size}')
        elif curr_batch_len > self.batch_size:
            diff = curr_batch_len - self.batch_size
            start_index = np.random.choice(diff + 1)
            samples_data2 = {}
            for k, v in samples_data.items():
                if type(v) is np.ndarray:
                    samples_data2[k] = v[start_index: start_index + self.batch_size]
                elif type(v) is dict:
                    v2 = {}
                    for vk, vv in v.items():
                        v2[vk] = vv[start_index: start_index + self.batch_size]
                    samples_data2[k] = v2
                else:
                    samples_data2[k] = v
            samples_data = samples_data2
        return samples_data

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
            dropout_proportion = 1 if self.increase_dropout_increment is None else 0  # TODO: remember 2 reset
            start_time = time.time()

            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""

                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.sampler.obtain_samples(log=True, log_prefix='train/',
                                                    advance_curriculum=advance_curriculum,
                                                    dropout_proportion=dropout_proportion)
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""

                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train/')
                advance_curriculum, increase_dropout = self.check_advance_curriculum(samples_data)
                proc_samples_time = time.time() - time_proc_samples_start

                """ ------------------ Reward Predictor Splicing ---------------------"""
                # r_discrete, logprobs = self.reward_predictor.get_actions(samples_data['env_infos']['next_obs_rewardfree'])
                # if self.supervised_model is not None:
                #     self.log_supervised(samples_data)
                # if self.sparse_rewards:
                #     self.log_rew_pred(r_discrete[:,:,0], samples_data['rewards'], samples_data['env_infos'])
                # # Splice into the inference process
                # if self.use_rp_inner:
                #     samples_data['observations'][:,:, -2] = r_discrete[:, :, 0]
                # # Splice into the meta-learning process
                # if self.use_rp_outer:
                #     samples_data['rewards'] = r_discrete[:, :, 0]
                # if 'teacher_action' in samples_data['env_infos']:
                #     samples_data['env_infos']['teacher_action'] = samples_data['env_infos']['teacher_action'].astype(np.int32)
                
                """ ------------------ End Reward Predictor Splicing ---------------------"""

                if type(paths) is list:
                    self.log_diagnostics(paths, prefix='train-')
                else:
                    self.log_diagnostics(sum(paths.values(), []), prefix='train-')

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_optimization_step_start = time.time()
                if not self.distill_only:
                    self.algo.optimize_policy(samples_data)
                    # self.algo.optimize_reward(samples_data)
                if self.supervised_model is not None and advance_curriculum:
                    samples_data = self.load_data(0, self.num_train_batches)
                    distill_log = self.distill(samples_data, is_training=True)  # TODO: do this more!
                    for k, v in distill_log.items():
                        logger.logkv(f"Distill/{k}_Train", v)

                    if itr % self.eval_every == 0 or itr == self.n_itr - 1:
                        with torch.no_grad():
                            # Accuracy on the validation set
                            # samples_data = self.load_data(self.num_train_batches, self.num_batches)  # TODO: collect differently
                            # self.sampler.supervised_model.reset(dones=[True] * len(samples_data['observations']))
                            distill_log = self.distill(samples_data, is_training=False)
                            for k, v in distill_log.items():
                                logger.logkv(f"Distill/{k}_Validation", v)
                            self.sampler.supervised_model.reset(dones=[True] * len(samples_data['observations']))
                            logger.log("Running supervised model")
                            advance_curriculum_s, increase_dropout_s = self.run_supervised()
                            if self.advance_without_teacher:
                                advance_curriculum = advance_curriculum_s
                                increase_dropout = increase_dropout_s
                            logger.log('Evaluating supervised')
                            self.sampler.supervised_model.reset(dones=[True] * len(samples_data['observations']))

                            should_save_video = itr % self.save_videos_every == 0
                            self.sampler.supervised_model.reset(dones=[True])
                            paths, accuracy = self.save_videos(itr, self.il_trainer.acmodel, save_name='video',
                                                               num_rollouts=10,
                                                               use_teacher=self.distill_with_teacher,
                                                               save_video=should_save_video)
                            logger.logkv("Distill/RolloutAcc", accuracy)
                    else:
                        if self.advance_without_teacher:
                            advance_curriculum = False

                if advance_curriculum:
                    self.curriculum_step += 1

                if increase_dropout:
                    dropout_proportion += self.increase_dropout_increment
                    dropout_proportion = min(1, dropout_proportion)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time/Optimization', time.time() - time_optimization_step_start)
                logger.logkv('Time/SampleProc', np.sum(proc_samples_time))
                logger.logkv('Time/Sampling', sampling_time)

                logger.logkv('Time/Total', time.time() - start_time)
                logger.logkv('Time/Itr', time.time() - itr_start_time)

                logger.logkv('Curriculum Step', self.curriculum_step)
                logger.logkv('Curriculum Percent', self.curriculum_step / len(self.env.levels_list))

                process = psutil.Process(os.getpid())
                memory_use = process.memory_info().rss / float(2 ** 20)
                print("Memory Use MiB", memory_use)
                logger.logkv('Memory MiB', memory_use)

                logger.log(self.exp_name)

                params = self.get_itr_snapshot(itr)
                step = self.curriculum_step
                if advance_curriculum:
                    step -= 1

                if self.log_and_save:
                    logger.log("Saving snapshot...")
                    logger.save_itr_params(itr, step, params)
                    logger.log("Saved")
                    logger.dumpkvs()


        logger.log("Training finished")
        # self.sess.close()  # TODO: is this okay?


    def save_data(self, data, itr):
        file_name = osp.join(self.exp_name, 'batch_%d.pkl' % itr)
        joblib.dump(data, file_name, compress=3)


    def distill(self, samples, is_training=False):
        cleaned_obs = self.sampler.mask_teacher(samples["observations"], self.teacher_info)
        samples['observations'] = cleaned_obs
        log = self.il_trainer.distill(samples, source=self.source, is_training=is_training)
        return log

    def run_supervised(self):
        paths = self.sampler.obtain_samples(log=False, advance_curriculum=False, policy=self.il_trainer.acmodel,
                                            feedback_list=self.teacher_info, max_action=True,
                                            use_teacher=self.distill_with_teacher)
        samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix="Distill/")
        advance_curriculum, increase_dropout = self.check_advance_curriculum(samples_data)
        return advance_curriculum, increase_dropout

    def save_videos(self, step, policy, save_name='sample_video', num_rollouts=2, use_teacher=False, save_video=False):
        policy.eval()
        paths, accuracy = rollout(self.env, policy, max_path_length=200, reset_every=2, stochastic=True,
                                  batch_size=1, record_teacher=True, use_teacher=use_teacher, save_video=save_video,
                                  video_filename=self.exp_name + '/' + save_name + str(step) + '.mp4',
                                  num_rollouts=num_rollouts)
        print('Average Returns: ', np.mean([sum(path['rewards']) for path in paths]))
        print('Average Path Length: ', np.mean([path['env_infos'][-1]['episode_length'] for path in paths]))
        print('Average Success Rate: ', np.mean([path['env_infos'][-1]['success'] for path in paths]))
        return paths, accuracy

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        d = dict(itr=itr,
                        policy=self.policy,
                        env=self.env,
                        baseline=self.baseline,
                        config=self.config,
                        curriculum_step=self.curriculum_step,)
        if self.reward_predictor is not None:
            d['reward_predictor'] = self.reward_predictor
        if self.il_trainer.acmodel is not None:
            d['supervised_model'] = self.il_trainer.acmodel
        return d


    def log_supervised(self, samples_data):
        pred_actions, _ = self.il_trainer.acmodel.get_actions(samples_data['observations'])
        real_actions = samples_data['env_infos']['teacher_action']
        matches = pred_actions == real_actions
        log_prefix = "Supervised"
        logger.logkv(log_prefix + 'Accuracy', np.mean(matches))

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        # self.policy.log_diagnostics(paths, prefix)
        # self.baseline.log_diagnostics(paths, prefix)

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

        if positives.sum() > 0 and pred_positives.sum() > 0 and correct.sum() > 0:
            logger.logkv(log_prefix + 'F1', 2 * precision * recall / (precision + recall))
        else:
            logger.logkv(log_prefix + 'F1', -1)



    def log_rew_pred(self, r_discrete, rewards, env_infos):
        log_prefix = "RewPred/"

        # Flatten, trim out any which are just there for padding
        # Elements where step=0 are just padding on the end.
        curr_elements = 1 - (env_infos['step'].flatten() == 0)
        r_discrete = np.stack([data for data, curr_bool in zip(r_discrete.flatten(), curr_elements) if curr_bool])
        rewards = np.stack([data for data, curr_bool in zip(rewards.flatten(), curr_elements) if curr_bool])
        step = np.stack([data for data, curr_bool in zip(env_infos['step'].flatten(), curr_elements) if curr_bool])


        self._log_rew_pred(r_discrete, rewards, log_prefix)

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
            log_prefix_i = log_prefix + name + "-"
            r_discrete_i = np.stack([data for data, curr_bool in zip(r_discrete, mask) if curr_bool.all()])
            rewards_i = np.stack([data for data, curr_bool in zip(rewards, mask) if curr_bool.all()])
            self._log_rew_pred(r_discrete_i, rewards_i, log_prefix_i)
