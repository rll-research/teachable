import torch
import numpy as np
from meta_mb.logger import logger
from babyai.utils.buffer import Buffer
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
        baseline (Baseline) : 
        policy (Policy) : 
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
    """

    def __init__(
        self,
        args={},
        collect_policy=None,
        rl_policy=None,
        il_policy=None,
        sampler=None,
        env=None,
        start_itr=0,
        buffer_name="",
        exp_name="",
        curriculum_step=0,
        eval_every=100,
        save_every=100,
        log_every=10,
        save_videos_every=1000,
        log_and_save=True,
        obs_preprocessor=None,
        log_dict={},
        eval_heldout=True,
        log_fn=lambda w, x, y, z: None,
        feedback_list=[],
    ):
        self.args = args
        self.collect_policy = collect_policy
        self.rl_policy = rl_policy
        self.il_policy = il_policy
        self.sampler = sampler
        self.env = copy.deepcopy(env)
        self.itr = start_itr
        self.buffer_name = buffer_name
        self.exp_name = exp_name
        self.curriculum_step = curriculum_step
        self.eval_every = eval_every
        self.save_every = save_every
        self.log_every = log_every
        self.save_videos_every = save_videos_every
        self.log_and_save = log_and_save
        self.introduced_teachers = log_dict.get('introduced_teachers', set())
        self.num_feedback_advice = log_dict.get('num_feedback_advice', 0)
        self.num_feedback_reward = log_dict.get('num_feedback_reward', 0)
        self.total_distillation_frames = log_dict.get('total_distillation_frames', 0)
        self.itrs_on_level = log_dict.get('itrs_on_level', 0)
        # Dict saying which teacher types the agent has access to for training.
        self.gave_feedback = log_dict.get('gave_feedback', {k: 0 for k in feedback_list})
        self.followed_feedback = log_dict.get('followed_feedback', {k: 0 for k in feedback_list})
        self.obs_preprocessor = obs_preprocessor
        self.next_train_itr = log_dict.get('next_train_itr', start_itr)
        self.num_train_skip_itrs = log_dict.get('num_train_skip_itrs', 5)
        self.eval_heldout = eval_heldout
        self.log_fn = log_fn
        self.advancement_count_threshold = getattr(args, 'advancement_count', 1)
        self.advancement_count = 0
        self.success_dict = {k: 0 for k in feedback_list}
        self.success_dict['none'] = 0
        self.best_train_perf = (0, 0, float('inf'))  # success, accuracy, loss
        self.itrs_since_best = 0
        self.buffer = None
        self.feedback_list = feedback_list

    def check_advance_curriculum_train(self, episode_logs, data):
        acc_threshold = self.args.accuracy_threshold_rl
        if episode_logs is None:
            return True, -1, -1
        if self.args.discrete:
            avg_accuracy = torch.eq(data.action_probs.argmax(dim=1),
                                    data.teacher_action).float().mean().item()
        else:
            avg_accuracy = 0
        avg_success = np.mean(episode_logs["success_per_episode"])
        should_advance_curriculum = (avg_success >= self.args.success_threshold_rl) \
                                    and (avg_accuracy >= acc_threshold)
        best_success, best_accuracy, best_loss = self.best_train_perf
        if avg_success > best_success or (avg_success == best_success and avg_accuracy > best_accuracy):
            if self.args.early_stop_metric == 'success':
                self.itrs_since_best = 0
            self.best_train_perf = (avg_success, avg_accuracy, best_loss)
        else:
            if self.args.early_stop_metric == 'success':
                self.itrs_since_best += 1
        return should_advance_curriculum, avg_success, avg_accuracy

    def save_model(self):
        params = self.get_itr_snapshot(self.itr)
        step = self.curriculum_step
        logger.save_itr_params(self.itr, step, params)

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
        logger.logkv('Train/SkipTrainRL', int(self.skip_training_rl))
        logger.logkv("ItrsOnLevel", self.itrs_on_level)

        time_total = time.time() - self.start_time
        self.itr_start_time = time.time()
        logger.logkv('Time/Total', time_total)
        logger.logkv('Time/Itr', time_itr)

        try:
            logger.logkv('Curriculum Percent', self.curriculum_step / len(self.env.train_levels))
        except:
            print("no curriculum")

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
            self.buffer = Buffer(self.buffer_name, self.args.buffer_capacity, self.args.prob_current, val_prob=.1,
                                 successful_only=self.args.distill_successful_only)

    def train(self):
        self.init_logs()
        self.make_buffer()

        for itr in range(self.itr, self.args.n_itr):
            self.itr = itr
            self.itrs_on_level += 1

            if self.args.save_untrained:
                self.save_model()
                return

            if itr % self.log_every == 0:
                self.log_rollouts()

            # If we're distilling, don't train the first time on the level in case we can zero-shot it
            self.skip_training_rl = self.args.reward_when_necessary and not self.next_train_itr == itr

            logger.log("\n ---------------- Iteration %d ----------------" % itr)

            """ -------------------- Sampling --------------------------"""

            logger.log("Obtaining samples...")
            time_env_sampling_start = time.time()
            self.should_collect = (self.args.collect_teacher is not None) and ((not self.skip_training_rl) or self.args.self_distill)
            self.should_train_rl = (self.args.rl_teacher is not None) and (not self.skip_training_rl)
            if self.should_collect:
                # Collect if we are distilling OR if we're not skipping
                samples_data, episode_logs = self.sampler.collect_experiences(
                                                                           collect_with_oracle=self.args.collect_with_oracle,
                                                                           collect_reward=self.should_train_rl,
                                                                           train=self.should_train_rl)
                self.buffer.add_batch(samples_data, self.curriculum_step)
            else:
                print("Not collecting")
                episode_logs = None
                raw_samples_data = None
                samples_data = None

            """ -------------------- Training --------------------------"""

            logger.log("RL Training...")

            time_collection = time.time() - time_env_sampling_start
            time_training_start = time.time()
            if self.should_train_rl:
                sampled_batch = self.buffer.sample(total_num_samples=self.args.batch_size, split='train')
                summary_logs = self.rl_policy.optimize_policy(sampled_batch, itr)
            else:
                summary_logs = None
            time_training = time.time() - time_training_start
            self._log(episode_logs, summary_logs, samples_data, tag="Train")
            advance_curriculum, avg_success, avg_accuracy = self.check_advance_curriculum_train(episode_logs,
                                                                                                samples_data)
            if (self.args.rl_teacher is None) or self.skip_training_rl:
                advance_curriculum = True
            else:
                # Decide whether to train RL next itr
                if advance_curriculum:
                    self.next_train_itr = itr + self.num_train_skip_itrs
                    self.num_train_skip_itrs += 5
                else:
                    self.next_train_itr = itr + 1
                    self.num_train_skip_itrs = 5

            logger.logkv('Train/Advance', int(advance_curriculum))

            """ ------------------ Distillation ---------------------"""
            self.should_distill = (self.args.distill_teacher is not None) and advance_curriculum and \
                                  self.itrs_on_level >= self.args.min_itr_steps_distill
            if self.args.yes_distill:
                self.should_distill = self.itrs_on_level >= self.args.min_itr_steps_distill
            if self.args.no_distill or (self.buffer is not None and sum(list(self.buffer.counts_train.values())) == 0):
                self.should_distill = False
            if self.should_distill:
                logger.log("Distilling ...")
                time_distill_start = time.time()
                time_sampling_from_buffer = 0
                time_train_distill = 0
                time_val_distill = 0
                for dist_i in range(self.args.distillation_steps):
                    sample_start = time.time()
                    sampled_batch = self.buffer.sample(total_num_samples=self.args.batch_size, split='train')
                    time_sampling_from_buffer += (time.time() - sample_start)
                    sample_start = time.time()
                    self.total_distillation_frames += len(sampled_batch)
                    self.distill_log = self.distill(sampled_batch, is_training=True)
                    time_train_distill += (time.time() - sample_start)
                sample_start = time.time()
                time_sampling_from_buffer += (time.time() - sample_start)
                sample_start = time.time()
                sampled_val_batch = self.buffer.sample(total_num_samples=self.args.batch_size,
                                                       split='val')
                distill_log_val = self.distill(sampled_val_batch, is_training=False)

                time_val_distill += (time.time() - sample_start)
                best_success, best_accuracy, best_loss = self.best_train_perf
                val_loss = distill_log_val['Loss']
                if val_loss < best_loss:
                    self.best_train_perf = (best_success, best_accuracy, val_loss)
                    if self.args.early_stop_metric == 'val_loss':
                        self.itrs_since_best = 0
                else:
                    if self.args.early_stop_metric == 'val_loss':
                        self.itrs_since_best += 1
                distill_time = time.time() - time_distill_start
                advance_curriculum = True
                acc = distill_log_val['Accuracy']
                if self.args.distill_teacher == 'none':
                    advance_teacher = acc >= self.args.accuracy_threshold_distill_no_teacher
                else:
                    advance_teacher = acc >= self.args.accuracy_threshold_distill_teacher
                logger.logkv(f'Distill/Advance_After_Distillation', int(advance_teacher))
                advance_curriculum = advance_curriculum and advance_teacher
            else:
                distill_time = 0

            """ ------------------- Logging Stuff --------------------------"""
            logger.log(self.exp_name)
            self.update_logs(time_training, time_collection, distill_time, self.saving_time)

            """ ------------------- Stop or advance --------------------------"""
            should_terminate = self.stop_or_advance(advance_curriculum)
            if should_terminate:
                break

        logger.log("Training finished")

    def stop_or_advance(self, advance_curriculum):
        if self.log_and_save:
            early_stopping = self.itrs_since_best > self.args.early_stop
            best_success, best_accuracy, best_loss = self.best_train_perf
            # early_stopping = early_stopping and best_success > .7
            logger.logkv('Train/BestSuccess', best_success)
            logger.logkv('Train/BestAccuracy', best_accuracy)
            logger.logkv('Train/BestLoss', best_loss)
            logger.logkv('Train/ItrsSinceBest', self.itrs_since_best)
            if early_stopping or (self.itr % self.save_every == 0) or (self.itr == self.args.n_itr - 1) or \
                (not self.args.single_level and advance_curriculum):
                saving_time_start = time.time()
                logger.log("Saving snapshot...")
                self.save_model()
                logger.log("Saved")
                self.saving_time = time.time() - saving_time_start
            if early_stopping:
                return True  # should_terminate

        if self.args.end_on_full_buffer:
            advance_curriculum = self.buffer.counts_train[self.curriculum_step] == self.buffer.train_buffer_capacity

        advance_curriculum = advance_curriculum and not self.args.single_level and self.itrs_on_level > self.args.min_itr_steps
        if advance_curriculum:
            self.advancement_count += 1
        else:
            self.advancement_count = 0

        if self.advancement_count >= self.advancement_count_threshold:
            self.advancement_count = 0
            self.success_dict = {k: 0 for k in self.success_dict.keys()}
            self.curriculum_step += 1
            if self.curriculum_step >= len(self.env.train_levels):
                return True  # should terminate
            try:
                self.algo.advance_curriculum()
            except NotImplementedError:
                # If we get a NotImplementedError b/c we ran out of levels, stop training
                return True  # should terminate
            self.itrs_on_level = 0
            self.next_train_itr = self.itr + 1
            self.num_train_skip_itrs = 5
        return False  # keep going

    def _log(self, episode_logs, summary_logs, data, tag=""):
        logger.logkv('Curriculum Step', self.curriculum_step)
        try:
            counts_train = self.buffer.counts_train[self.curriculum_step]
        except:
            counts_train = 0
        logger.logkv("BufferSize", counts_train)
        if episode_logs is not None:
            avg_return = np.mean(episode_logs['return_per_episode'])
            avg_path_length = np.mean(episode_logs['num_frames_per_episode'])
            avg_success = np.mean(episode_logs['success_per_episode'])
            avg_dist_to_goal = np.mean(episode_logs['dist_to_goal_per_episode'])
            avg_reward = np.mean(episode_logs["return_per_episode"])
            logger.logkv(f"{tag}/Success", avg_success)
            logger.logkv(f"{tag}/DistToGoal", avg_dist_to_goal)
            logger.logkv(f"{tag}/Reward", avg_reward)
            logger.logkv(f"{tag}/Return", avg_return)
            logger.logkv(f"{tag}/PathLength", avg_path_length)

            if self.args.discrete:
                logger.logkv(f"{tag}/Accuracy", torch.eq(data.action, data.teacher_action).float().mean().item())
                logger.logkv(f"{tag}/Argmax_Accuracy", torch.eq(data.action_probs.argmax(dim=1),
                                                                data.teacher_action).float().mean().item())

            self.num_feedback_advice += episode_logs['num_feedback_advice']
            self.num_feedback_reward += episode_logs['num_feedback_reward']
            for k in self.feedback_list:
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
        """
        Gets the current policy and env for storage
        """
        return {}  # TODO: MAKE THIS BETTER!
        d = dict(itr=itr,
                 policy=self.policy_dict,
                 env=self.env,
                 args=self.args,
                 optimizer=self.algo.optimizer_dict,
                 curriculum_step=self.curriculum_step,
                 il_optimizer=il_optimizer,
                 log_dict={
                     'num_feedback_advice': self.num_feedback_advice,
                     'num_feedback_reward': self.num_feedback_reward,
                     'total_distillation_frames': self.total_distillation_frames,
                     'itrs_on_level': self.itrs_on_level,
                     'next_train_itr': self.next_train_itr,
                     'num_train_skip_itrs': self.num_train_skip_itrs,
                     'gave_feedback': self.gave_feedback,
                     'followed_feedback': self.followed_feedback,
                     'introduced_teachers': self.introduced_teachers,
                 })
        return d
