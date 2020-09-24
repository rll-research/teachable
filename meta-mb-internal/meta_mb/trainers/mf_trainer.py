import torch
import numpy as np
from meta_mb.logger import logger
from meta_mb.samplers.utils import rollout
import os.path as osp
import joblib
import time
import psutil
import os
import copy

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
        use_rp_inner=False,
        use_rp_outer=False,
        success_threshold=0.95,
        accuracy_threshold=0.9,
        exp_name="",
        videos_every=5,
        curriculum_step=0,
        config=None,
        log_and_save=True,
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
        eval_every=25,
        save_every=100,
        log_every=10,
        save_videos_every=100,
        train_with_teacher=True,
        distill_with_teacher=False,
        supervised_model=None,
        reward_predictor=None,
        rp_trainer=None,
        advance_levels=True,
        is_debug=False
    ):
        self.algo = algo
        self.env = copy.deepcopy(env)
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.task = task
        self.use_rp_inner = use_rp_inner
        self.use_rp_outer = use_rp_outer
        self.success_threshold = success_threshold
        self.accuracy_threshold = accuracy_threshold
        self.curriculum_step = curriculum_step
        self.exp_name = exp_name
        self.videos_every = videos_every
        self.config = config
        self.log_and_save = log_and_save
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
        self.train_with_teacher = train_with_teacher
        self.distill_with_teacher = distill_with_teacher
        self.supervised_model = supervised_model
        self.reward_predictor = reward_predictor
        self.rp_trainer = rp_trainer
        self.advance_levels = advance_levels
        self.is_debug = is_debug
        if self.num_batches is not None:
            self.num_train_batches = (self.num_batches * 0.9)
            self.num_val_batches = self.num_batches - self.num_train_batches
            assert self.num_train_batches > 0

    def check_advance_curriculum(self, episode_logs, summary_logs):
        avg_success = np.mean(episode_logs["success_per_episode"])
        avg_accuracy = summary_logs['Accuracy']
        should_advance_curriculum = (avg_success >= self.success_threshold) and (avg_accuracy >= self.accuracy_threshold)
        return should_advance_curriculum

    def check_advance_curriculum_rollout(self, data):
        num_total_episodes = data['dones'].sum()
        num_successes = data['env_infos']['success'].sum()
        avg_success = num_successes / num_total_episodes
        # Episode length contains the timestep, starting at 1.  Padding values are 0.
        pad_steps = (data['env_infos']['episode_length'] == 0).sum()
        correct_actions = (data['actions'] == data['env_infos']['teacher_action']).sum() - pad_steps
        avg_accuracy = correct_actions / (np.prod(data['actions'].shape) - pad_steps)
        # We take the max since runs which end early will be 0-padded
        should_advance_curriculum = avg_success >= self.success_threshold
        return should_advance_curriculum, avg_success, avg_accuracy

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
        start_time = time.time()
        rollout_time = 0
        success_rate = 0
        itrs_on_level = 0

        for itr in range(self.start_itr, self.n_itr):

            # high_entropy = False
            # # At the very beginning of a new level, have high entropy
            # if itrs_on_level < 3:
            #     high_entropy = True
            # # At the very beginning, have high entropy
            # if self.curriculum_step == 0 and itrs_on_level < 20:
            #     high_entropy = True
            # # If the model never succeeds
            # if itrs_on_level > 20 and success_rate < .2:
            #     high_entropy = True
            # logger.logkv("HighE", high_entropy)
            logger.logkv("ItrsOnLEvel", itrs_on_level)
            itrs_on_level += 1

            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            """ -------------------- Sampling --------------------------"""

            logger.log("Obtaining samples...")
            time_env_sampling_start = time.time()
            samples_data, episode_logs = self.algo.collect_experiences(use_teacher=self.train_with_teacher)
            time_collection = time.time() - time_env_sampling_start
            time_training_start = time.time()
            # if high_entropy:
            #     entropy = self.algo.entropy_coef# * 10
            # else:
            #     entropy = self.algo.entropy_coef
            summary_logs = self.algo.optimize_policy(samples_data, use_teacher=self.train_with_teacher)
            time_training = time.time() - time_training_start
            self._log(episode_logs, summary_logs, tag="Train")
            logger.logkv('Curriculum Step', self.curriculum_step)
            advance_curriculum = self.check_advance_curriculum(episode_logs, summary_logs)
            logger.logkv('Train/AdvanceCurriculum', advance_curriculum)
            time_env_sampling = time.time() - time_env_sampling_start
            success_rate = np.mean(episode_logs['success_per_episode'])
            #
            # """ ------------------ Reward Predictor Splicing ---------------------"""
            rp_start_time = time.time()
            # samples_data = self.use_reward_predictor(samples_data)  # TODO: update
            rp_splice_time = time.time() - rp_start_time

            """ ------------------ End Reward Predictor Splicing ---------------------"""

            """ ------------------ Policy Update ---------------------"""

            logger.log("Optimizing policy...")
            # # This needs to take all samples_data so that it can construct graph for meta-optimization.
            time_rp_train_start = time.time()
            # self.train_rp(samples_data)
            time_rp_train = time.time() - time_rp_train_start

            """ ------------------ Distillation ---------------------"""
            if self.supervised_model is not None and advance_curriculum:
                time_distill_start = time.time()
                for _ in range(3):  # TODO: tune this!
                    distill_log = self.distill(samples_data, is_training=True)
                for k, v in distill_log.items():
                    logger.logkv(f"Distill/{k}_Train", v)
                distill_time = time.time() - time_distill_start
                advance_curriculum = distill_log['Accuracy'] >= self.accuracy_threshold
                logger.logkv('Distill/AdvanceCurriculum', advance_curriculum)
            else:
                distill_time = 0

            """ ------------------ Policy rollouts ---------------------"""
            run_policy_time = 0
            if advance_curriculum or (itr % self.eval_every == 0) or (itr == self.n_itr - 1):  # TODO: collect rollouts with and without the teacher
                train_advance_curriculum = advance_curriculum
                with torch.no_grad():
                    if self.supervised_model is not None:
                        # Distilled model
                        time_run_supervised_start = time.time()
                        self.sampler.supervised_model.reset(dones=[True] * len(samples_data.obs))
                        logger.log("Running supervised model")
                        advance_curriculum_sup = self.run_supervised(self.il_trainer.acmodel, self.distill_with_teacher, "DRollout/")
                        run_supervised_time = time.time() - time_run_supervised_start
                    else:
                        run_supervised_time = 0
                        advance_curriculum_sup = True
                    # Original Policy
                    time_run_policy_start = time.time()
                    # self.algo.acmodel.reset(dones=[True] * len(samples_data.obs))
                    logger.log("Running model with teacher")
                    # advance_curriculum_policy = self.run_supervised(self.algo.acmodel, self.train_with_teacher, "Rollout/")
                    run_policy_time = time.time() - time_run_policy_start

                    # advance_curriculum = advance_curriculum_policy and advance_curriculum_sup and train_advance_curriculum
                    print("ADvancing curriculum???", advance_curriculum)

                    logger.logkv('AdvanceCurriculum', advance_curriculum)
            else:
                run_supervised_time = 0
                advance_curriculum = False

            """ ------------------- Logging Stuff --------------------------"""
            logger.logkv('Itr', itr)
            logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

            logger.logkv('Time/Total', time.time() - start_time)
            logger.logkv('Time/Itr', time.time() - itr_start_time)

            logger.logkv('Curriculum Percent', self.curriculum_step / len(self.env.levels_list))

            process = psutil.Process(os.getpid())
            memory_use = process.memory_info().rss / float(2 ** 20)
            logger.logkv('Memory MiB', memory_use)

            logger.log(self.exp_name)

            logger.logkv('Time/Sampling', time_env_sampling)
            logger.logkv('Time/Training', time_training)
            logger.logkv('Time/Collection', time_collection)
            logger.logkv('Time/RPUse', rp_splice_time)
            logger.logkv('Time/RPTrain', time_rp_train)
            logger.logkv('Time/RunwTeacher', run_policy_time)
            logger.logkv('Time/Distillation', distill_time)
            logger.logkv('Time/RunDistilled', run_supervised_time)
            logger.logkv('Time/VidRollout', rollout_time)
            logger.dumpkvs()

            """ ------------------ Video Saving ---------------------"""

            should_save_video = (itr % self.save_videos_every == 0) or (itr == self.n_itr - 1) or advance_curriculum
            if should_save_video:
                time_rollout_start = time.time()
                if self.supervised_model is not None:
                    self.il_trainer.acmodel.reset(dones=[True])
                    self.save_videos(itr, self.il_trainer.acmodel, save_name='distilled_video_stoch',
                                   num_rollouts=5,
                                   use_teacher=self.distill_with_teacher,
                                   save_video=should_save_video, log_prefix="DVidRollout/Stoch", stochastic=True)
                    self.il_trainer.acmodel.reset(dones=[True])
                    self.save_videos(itr, self.il_trainer.acmodel, save_name='distilled_video_det',
                                     num_rollouts=5,
                                     use_teacher=self.distill_with_teacher,
                                     save_video=should_save_video, log_prefix="DVidRollout/Det", stochastic=False)

                self.algo.acmodel.reset(dones=[True])
                self.save_videos(self.curriculum_step, self.algo.acmodel, save_name='withTeacher_video_stoch',
                               num_rollouts=5,
                               use_teacher=self.train_with_teacher,
                               save_video=should_save_video, log_prefix="VidRollout/Stoch", stochastic=True)
                self.algo.acmodel.reset(dones=[True])
                self.save_videos(self.curriculum_step, self.algo.acmodel, save_name='withTeacher_video_det',
                                 num_rollouts=5,
                                 use_teacher=self.train_with_teacher,
                                 save_video=should_save_video, log_prefix="VidRollout/Det", stochastic=False)
                rollout_time = time.time() - time_rollout_start
            else:
                rollout_time = 0

            params = self.get_itr_snapshot(itr)
            step = self.curriculum_step

            if self.log_and_save:
                if (itr % self.save_every == 0) or (itr == self.n_itr - 1) or advance_curriculum:
                    logger.log("Saving snapshot...")
                    logger.save_itr_params(itr, step, params)
                    logger.log("Saved")

            if advance_curriculum and self.advance_levels:
                self.curriculum_step += 1
                self.sampler.advance_curriculum()
                self.algo.advance_curriculum()
                # self.algo.set_optimizer()
                itrs_on_level = 0


        logger.log("Training finished")

    def _log(self, episode_logs, summary_logs, tag=""):
        avg_return = np.mean(episode_logs['return_per_episode'])
        avg_path_length = np.mean(episode_logs['num_frames_per_episode'])
        avg_success = np.mean(episode_logs['success_per_episode'])

        logger.logkv(f"{tag}/Success", avg_success)
        logger.logkv(f"{tag}/Return", avg_return)
        logger.logkv(f"{tag}/PathLength", avg_path_length)
        for k, v in summary_logs.items():
            logger.logkv(f"{tag}/{k}", v)


    def use_reward_predictor(self, samples_data):
        pass
        # with torch.no_grad():
        #     r_discrete, logprobs = self.reward_predictor.get_actions(samples_data['env_infos']['next_obs_rewardfree'])
        #     if self.sparse_rewards:
        #         self.log_rew_pred(r_discrete[:, :, 0], samples_data['rewards'], samples_data['env_infos'])
        #     # Splice into the inference process
        #     if self.use_rp_inner:
        #         samples_data['observations'][:, :, -2] = r_discrete[:, :, 0]
        #     # Splice into the meta-learning process
        #     if self.use_rp_outer:
        #         samples_data['rewards'] = r_discrete[:, :, 0]
        #     if 'teacher_action' in samples_data['env_infos']:
        #         samples_data['env_infos']['teacher_action'] = samples_data['env_infos']['teacher_action'].astype(np.int32)
        #     return samples_data


    def save_data(self, data, itr):
        file_name = osp.join(self.exp_name, 'batch_%d.pkl' % itr)
        joblib.dump(data, file_name, compress=3)


    def train_rp(self, samples, is_training=False):
        log = self.rp_trainer.distill(samples, source=self.source, is_training=is_training)
        return log

    def distill(self, samples, is_training=False):
        log = self.il_trainer.distill(samples, source=self.source, is_training=is_training)
        return log

    def run_supervised(self, policy, use_teacher, tag):
        paths = self.sampler.obtain_samples(log=False, advance_curriculum=False, policy=policy,
                                            feedback_list=self.teacher_info, max_action=True,
                                            use_teacher=use_teacher)
        samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix=tag, log_teacher=self.train_with_teacher)
        advance_curriculum, avg_success, avg_accuracy = self.check_advance_curriculum_rollout(samples_data)
        logger.logkv(f"{tag}AdvanceCurriculum", advance_curriculum)
        logger.logkv(f"{tag}AvgSuccess", avg_success)
        logger.logkv(f"{tag}AvgAccuracy", avg_accuracy)
        return advance_curriculum

    def save_videos(self, step, policy, save_name='sample_video', num_rollouts=2, use_teacher=False, save_video=False,
                    log_prefix=None, stochastic=True):
        policy.eval()
        self.env.set_level_distribution(self.curriculum_step)
        save_wandb = (save_video and not self.is_debug)
        paths, accuracy = rollout(self.env, policy, max_path_length=200, reset_every=1, stochastic=stochastic,
                                  batch_size=1, record_teacher=True, use_teacher=use_teacher,
                                  video_directory=self.exp_name, video_name=save_name + str(self.curriculum_step),
                                  num_rollouts=num_rollouts, save_wandb=save_wandb, save_locally=False)
        if log_prefix is not None:
            logger.logkv(log_prefix + "Acc", accuracy)
            logger.logkv(log_prefix + "Reward", np.mean([sum(path['rewards']) for path in paths]))
            logger.logkv(log_prefix + "PathLength",
                         np.mean([path['env_infos'][-1]['episode_length'] for path in paths]))
            logger.logkv(log_prefix + "Success", np.mean([path['env_infos'][-1]['success'] for path in paths]))
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
                        optimizer=self.algo.optimizer.state_dict(),
                        curriculum_step=self.curriculum_step,)
        if self.reward_predictor is not None:
            d['reward_predictor'] = self.reward_predictor
        if self.il_trainer is not None:
            d['supervised_model'] = self.il_trainer.acmodel
        return d

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
        pass
        # TODO: Currently commented out since we compute this as if it was a binary classification, and we have a 3-way
        #   classification.  Still, we should rewrite this function to work with the new RP.
        # log_prefix = "RewPred/"
        #
        # # Flatten, trim out any which are just there for padding
        # # Elements where step=0 are just padding on the end.
        # curr_elements = 1 - (env_infos['step'].flatten() == 0)
        # r_discrete = np.stack([data for data, curr_bool in zip(r_discrete.flatten(), curr_elements) if curr_bool])
        # rewards = np.stack([data for data, curr_bool in zip(rewards.flatten(), curr_elements) if curr_bool])
        # step = np.stack([data for data, curr_bool in zip(env_infos['step'].flatten(), curr_elements) if curr_bool])
        #
        #
        # self._log_rew_pred(r_discrete, rewards, log_prefix)

        # # Log split by index in meta-task
        # unique_steps = np.unique(env_infos['step'])
        # for i in unique_steps:
        #     if i == 0:
        #         continue  # Remove 0, which is just filler
        #     curr_elements = step == i
        #     r_discrete_i = np.stack([data for data, curr_bool in zip(r_discrete, curr_elements) if curr_bool])
        #     rewards_i = np.stack([data for data, curr_bool in zip(rewards, curr_elements) if curr_bool])
        #     self._log_rew_pred(r_discrete_i, rewards_i, log_prefix + str(i) + "-")
