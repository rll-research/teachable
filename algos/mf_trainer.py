import torch
import numpy as np
from logger import logger
from envs.babyai.utils.buffer import Buffer
import time
import psutil
import os


class Trainer(object):
    def __init__(
        self,
        args=None,
        collect_policy=None,
        rl_policy=None,
        il_policy=None,
        relabel_policy=None,
        sampler=None,
        env=None,
        obs_preprocessor=None,
        log_dict={},
        log_fn=lambda w, x: None,
    ):
        self.args = args
        self.collect_policy = collect_policy
        self.rl_policy = rl_policy
        self.il_policy = il_policy
        self.relabel_policy = relabel_policy
        self.sampler = sampler
        self.env = env
        self.itr = args.start_itr
        self.obs_preprocessor = obs_preprocessor
        self.log_fn = log_fn
        self.buffer = None

        # Set run counters, or reinitialize if log_dict isn't empty (i.e. we're continuing a run).
        self.num_feedback_advice = log_dict.get('num_feedback_advice', 0)
        self.num_feedback_reward = log_dict.get('num_feedback_reward', 0)
        self.total_distillation_frames = log_dict.get('total_distillation_frames', 0)
        # Dict saying which teacher types the agent has access to for training.
        self.gave_feedback = log_dict.get('gave_feedback', {k: 0 for k in self.args.feedback_list})
        self.followed_feedback = log_dict.get('followed_feedback', {k: 0 for k in self.args.feedback_list})
        self.num_train_skip_itrs = log_dict.get('num_train_skip_itrs', 5)

        # Counters to determine early stopping and saving
        self.best_val_loss = float('inf')
        self.best_success = 0
        self.itrs_since_best = 0
        self.current_success = 0  # TODO: store these
        self.current_val_loss = float('inf')

    def save_model(self):
        params = self.get_itr_snapshot(self.itr)
        if (self.rl_policy is not None) and (self.il_policy is not None):
            assert not self.rl_policy.teacher == self.il_policy.teacher, "will overwrite if policies have same teacher"
        if self.rl_policy is not None:
            self.rl_policy.save(self.args.exp_dir)
        if self.il_policy is not None:
            self.il_policy.save(self.args.exp_dir)
        if self.rl_policy is not None and self.current_success >= self.best_success:
            self.rl_policy.save(self.args.exp_dir, save_name=f"best_{self.rl_policy.teacher}_model.pt")
        if self.il_policy is not None and self.current_val_loss <= self.best_val_loss:
            self.il_policy.save(self.args.exp_dir, save_name=f"best_{self.il_policy.teacher}_model.pt")
        logger.save_itr_params(self.itr, self.args.level, params)

    def log_rollouts(self):
        if self.args.feedback_from_buffer:
            num_feedback = self.buffer.num_feedback
        else:
            num_feedback = self.num_feedback_advice + self.num_feedback_reward
        self.log_fn(self.itr, num_feedback)

    def init_logs(self):
        self.all_time_training = 0
        self.all_time_collection = 0
        self.all_run_policy_time = 0
        self.all_distill_time = 0
        self.all_rollout_time = 0
        self.all_saving_time = 0
        self.all_unaccounted_time = 0
        self.start_time = time.time()
        self.rollout_time = 0
        self.saving_time = 0
        self.itr_start_time = time.time()
        self.last_success = 0
        self.last_accuracy = 0

    def update_logs(self, time_training, time_collection, distill_time, saving_time):
        logger.dumpkvs()

        time_itr = time.time() - self.itr_start_time
        time_unaccounted = time_itr - time_training - time_collection - distill_time - saving_time
        self.all_time_training += time_training
        self.all_time_collection += time_collection
        self.all_distill_time += distill_time
        self.all_saving_time += saving_time
        self.all_unaccounted_time += time_unaccounted

        logger.logkv('Itr', self.itr)

        time_total = time.time() - self.start_time
        self.itr_start_time = time.time()
        logger.logkv('Time/Total', time_total)
        logger.logkv('Time/Itr', time_itr)

        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / float(2 ** 20)
        logger.logkv('Memory MiB', memory_use)

        logger.logkv('Time/Training', time_training)
        logger.logkv('Time/Collection', time_collection)
        logger.logkv('Time/Distillation', distill_time)
        logger.logkv('Time/Saving', saving_time)
        logger.logkv('Time/Unaccounted', time_unaccounted)
        logger.logkv('Time/All_Training', self.all_time_training / time_total)
        logger.logkv('Time/All_Collection', self.all_time_collection / time_total)
        logger.logkv('Time/All_RunwTeacher', self.all_run_policy_time / time_total)
        logger.logkv('Time/All_Distillation', self.all_distill_time / time_total)
        logger.logkv('Time/All_Saving', self.all_saving_time / time_total)
        logger.logkv('Time/All_Unaccounted', self.all_unaccounted_time / time_total)

    def make_buffer(self):
        if not self.args.no_buffer:
            self.buffer = Buffer(self.args.buffer_name, self.args.buffer_capacity, val_prob=.1,
                                 successful_only=self.args.distill_successful_only)

    def relabel(self, batch):
        action, agent_dict = self.relabel_policy.act(batch.obs, sample=True)

        action = action.to(batch.action.dtype)
        assert type(action) == type(batch.action)
        assert action.dtype == batch.action.dtype, (action.dtype, batch.action.dtype)
        assert action.shape == batch.action.shape

        log_prob = agent_dict['dist'].log_prob(action).sum(-1).to(batch.log_prob.dtype)
        assert type(log_prob) == type(batch.log_prob)
        assert log_prob.dtype == batch.log_prob.dtype
        assert log_prob.shape == batch.log_prob.shape

        if 'argmax_action' in agent_dict and hasattr(batch, 'argmax_action'):
            argmax_action = agent_dict['argmax_action'].to(batch.argmax_action.dtype)
            assert type(argmax_action) == type(batch.argmax_action)
            assert argmax_action.dtype == batch.argmax_action.dtype
            assert argmax_action.shape == batch.argmax_action.shape
            batch.argmax_action = agent_dict['argmax_action'].to(batch.argmax_action.dtype).detach()

        batch.action = action.to(batch.action.dtype).detach()
        batch.log_prob = agent_dict['dist'].log_prob(action).sum(-1).to(batch.log_prob.dtype).detach()
        return batch

    def train(self):
        self.init_logs()
        self.make_buffer()

        for itr in range(self.itr, self.args.n_itr):
            self.itr = itr

            if self.num_feedback_advice + self.num_feedback_reward >= self.args.n_advice:
                self.log_rollouts()
                self.save_model()
                return

            if self.args.save_untrained:
                self.save_model()
                return

            if itr % self.args.log_interval == 0:
                self.log_rollouts()

            logger.log("\n ---------------- Iteration %d ----------------" % itr)

            """ -------------------- Sampling --------------------------"""

            logger.log("Obtaining samples...")
            time_env_sampling_start = time.time()
            self.should_collect = self.args.collect_teacher is not None
            self.should_train_rl = self.args.rl_teacher is not None
            if self.should_collect:
                logger.log("collect data...")
                # Collect if we are distilling OR if we're not skipping
                samples_data, episode_logs = self.sampler.collect_experiences(
                                                                           collect_with_oracle=self.args.collect_with_oracle,
                                                                           collect_reward=self.should_train_rl,
                                                                           train=self.should_train_rl)
                if self.relabel_policy is not None:
                    logger.log("relabel samples ...")
                    samples_data = self.relabel(samples_data)
                buffer_start = time.time()
                self.buffer.add_batch(samples_data, save=self.itr % 200  == 0)
                buffer_time = time.time() - buffer_start
                logger.logkv('Time/Buffer', buffer_time)
            else:
                episode_logs = None
                samples_data = None

            """ -------------------- Training --------------------------"""

            time_collection = time.time() - time_env_sampling_start
            time_training_start = time.time()
            if self.should_train_rl and itr > self.args.min_itr_steps:
                logger.log("RL Training...")
                for _ in range(self.args.epochs):
                    if self.args.on_policy:
                        logger.log("on_policy data...")
                        sampled_batch = samples_data
                    else:
                        sampled_batch = self.buffer.sample(total_num_samples=self.args.batch_size, split='train')
                    summary_logs = self.rl_policy.optimize_policy(sampled_batch, itr)
                if not self.args.on_policy:
                    val_batch = self.buffer.sample(total_num_samples=self.args.batch_size, split='val')
                    self.rl_policy.log_rl(val_batch)
            else:
                summary_logs = None
            time_training = time.time() - time_training_start
            self._log(episode_logs, summary_logs, samples_data, tag="Train")

            """ ------------------ Distillation ---------------------"""
            self.should_distill = self.args.distill_teacher is not None and self.itr >= self.args.min_itr_steps_distill
            if self.args.no_distill or (self.buffer is not None and self.buffer.counts_train == 0):
                self.should_distill = False
            if self.should_distill:
                logger.log("Distilling ...")
                time_distill_start = time.time()
                for dist_i in range(self.args.distillation_steps):
                    sampled_batch = self.buffer.sample(total_num_samples=self.args.batch_size, split='train')
                    self.total_distillation_frames += len(sampled_batch)
                    self.distill(sampled_batch, is_training=True)
                sampled_val_batch = self.buffer.sample(total_num_samples=self.args.batch_size, split='val')
                distill_log_val = self.distill(sampled_val_batch, is_training=False)

                val_loss = distill_log_val['Loss']
                self.current_val_loss = val_loss
                self.itrs_since_best = 0 if val_loss < self.best_val_loss else self.itrs_since_best + 1
                self.best_val_loss = min(self.best_val_loss, val_loss)
                distill_time = time.time() - time_distill_start
            else:
                distill_time = 0

            """ ------------------- Logging and Saving --------------------------"""
            logger.log(self.args.exp_dir)
            self.update_logs(time_training, time_collection, distill_time, self.saving_time)
            should_terminate = self.save_and_maybe_early_stop()
            if should_terminate:
                break

        # All done!
        self.log_rollouts()
        logger.log("Training finished")

    def save_and_maybe_early_stop(self):
        early_stopping = self.itrs_since_best > self.args.early_stop
        logger.logkv('Train/BestLoss', self.best_val_loss)
        logger.logkv('Train/ItrsSinceBest', self.itrs_since_best)
        if early_stopping or (self.itr % self.args.eval_interval == 0) or (self.itr == self.args.n_itr - 1):
            saving_time_start = time.time()
            logger.log("Saving snapshot...")
            self.save_model()
            logger.log("Saved")
            self.saving_time = time.time() - saving_time_start
        return early_stopping

    def _log(self, episode_logs, summary_logs, data, tag=""):
        logger.logkv('Level', self.args.level)
        counts_train = 0 if self.buffer is None else self.buffer.counts_train
        logger.logkv("BufferSize", counts_train)
        if episode_logs is not None:
            avg_return = np.mean(episode_logs['return_per_episode'])
            avg_path_length = np.mean(episode_logs['num_frames_per_episode'])
            avg_success = np.mean(episode_logs['success_per_episode'])
            avg_dist_to_goal = np.mean(episode_logs['dist_to_goal_per_episode'])
            avg_reward = np.mean(episode_logs["return_per_episode"])
            self.current_success = avg_success
            self.best_success = max(self.best_success, avg_success)
            logger.logkv(f"{tag}/Success", avg_success)
            logger.logkv(f"{tag}/DistToGoal", avg_dist_to_goal)
            logger.logkv(f"{tag}/Reward", avg_reward)
            logger.logkv(f"{tag}/Return", avg_return)
            logger.logkv(f"{tag}/PathLength", avg_path_length)

            if self.args.discrete:
                logger.logkv(f"{tag}/Accuracy", torch.eq(data.action, data.teacher_action).float().mean().item())
                logger.logkv(f"{tag}/Argmax_Accuracy", torch.eq(data.action_probs.argmax(dim=1).unsqueeze(1),
                                                                data.teacher_action).float().mean().item())

            self.num_feedback_advice += episode_logs['num_feedback_advice']
            self.num_feedback_reward += episode_logs['num_feedback_reward']
            for k in self.args.feedback_list:
                k_gave = f'gave_{k}'
                if k_gave in episode_logs:
                    self.gave_feedback[k] += episode_logs[k_gave]
                    logger.logkv(f"Feedback/Total_{k_gave}", self.gave_feedback[k])
                    k_followed = f'followed_{k}'
                    if k_followed in episode_logs:
                        self.followed_feedback[k] += episode_logs[k_followed]
                        logger.logkv(f"Feedback/Total_{k_followed}", self.followed_feedback[k])
                        logger.logkv(f"Feedback/Ratio_{k_followed}", episode_logs[k_followed] / episode_logs[k_gave])

            logger.logkv(f"{tag}/NumFeedbackAdvice", self.num_feedback_advice)
            logger.logkv(f"{tag}/NumFeedbackReward", self.num_feedback_reward)
            logger.logkv(f"{tag}/NumFeedbackTotal", self.num_feedback_advice + self.num_feedback_reward)
            logger.logkv(f"{tag}/num_feedback_reward", episode_logs['num_feedback_reward'])
            logger.logkv(f"{tag}/num_feedback_advice", episode_logs['num_feedback_advice'])
            for key in episode_logs:
                if 'followed_' in key or 'gave_' in key:
                    logger.logkv(f"Feedback/{key}", episode_logs[key])
        if summary_logs is not None:
            for k, v in summary_logs.items():
                if not k == 'Accuracy':
                    logger.logkv(f"{tag}/{k}", v)

    def distill(self, samples, is_training=False, source=None):
        if source is None:
            source = self.args.source
        log = self.il_policy.distill(samples, source=source, is_training=is_training)
        return log

    def get_itr_snapshot(self, itr):
        """ Saves training args (models are saved elsewhere) """
        d = dict(itr=itr,
                 env=self.env,
                 args=self.args,
                 log_dict={
                     'num_feedback_advice': self.num_feedback_advice,
                     'num_feedback_reward': self.num_feedback_reward,
                     'total_distillation_frames': self.total_distillation_frames,
                     'num_train_skip_itrs': self.num_train_skip_itrs,
                     'gave_feedback': self.gave_feedback,
                      'followed_feedback': self.followed_feedback,
                 })
        return d
