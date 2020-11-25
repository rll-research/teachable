import torch
import numpy as np
from meta_mb.logger import logger
from meta_mb.samplers.utils import rollout
from babyai.utils.buffer import Buffer, trim_batch
import os.path as osp
import joblib
import time
import psutil
import os
import copy
import random

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
        args,
        algo,
        policy,
        env,
        sampler,
        sample_processor,
        start_itr=0,
        buffer_name="",
        exp_name="",
        curriculum_step=0,
        il_trainer=None,
        supervised_model=None,
        reward_predictor=None,
        rp_trainer=None,
        is_debug=False,
        eval_every=200,
        save_every=200,
        log_every=10,
        save_videos_every=200,
        log_and_save=True,
        teacher_schedule=lambda _: ({}, {}),
        obs_preprocessor=None,
    ):
        self.args = args
        self.algo = algo
        self.policy = policy
        self.env = copy.deepcopy(env)
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.start_itr = start_itr
        self.buffer_name = buffer_name
        self.exp_name = exp_name
        self.curriculum_step = curriculum_step
        self.il_trainer = il_trainer
        self.supervised_model = supervised_model
        self.reward_predictor = reward_predictor
        self.rp_trainer = rp_trainer
        self.is_debug = is_debug
        self.eval_every = eval_every
        self.save_every = save_every
        self.log_every = log_every
        self.save_videos_every = save_videos_every
        self.log_and_save = log_and_save
        self.train_with_teacher = args.feedback_type is not None
        self.num_feedback_advice = 0
        self.num_feedback_reward = 0
        # Dict saying which teacher types the agent has access to for training.
        self.teacher_schedule = teacher_schedule
        # Dict specifying no teacher provided.
        self.no_teacher_dict, _ = teacher_schedule(-1)
        self.obs_preprocessor = obs_preprocessor

    def check_advance_curriculum(self, episode_logs, summary_logs):
        if summary_logs is None or episode_logs is None:
            return True
        avg_success = np.mean(episode_logs["success_per_episode"])
        avg_accuracy = summary_logs['Accuracy']
        should_advance_curriculum = (avg_success >= self.args.success_threshold) \
                                    and (avg_accuracy >= self.args.accuracy_threshold)
        return should_advance_curriculum

    def check_advance_curriculum_rollout(self, data):
        num_total_episodes = data['dones'].sum()
        num_successes = data['env_infos']['success'].sum()
        avg_success = num_successes / num_total_episodes
        # Episode length contains the timestep, starting at 1.  Padding values are 0.
        pad_steps = (data['env_infos']['episode_length'] == 0).sum()
        correct_actions = (data['actions'] == data['env_infos']['teacher_action'][:, :, 0]).sum() - pad_steps
        avg_accuracy = correct_actions / (np.prod(data['actions'].shape) - pad_steps)
        # We take the max since runs which end early will be 0-padded
        should_advance_curriculum = avg_success >= self.args.success_threshold
        return should_advance_curriculum, avg_success, avg_accuracy

    def train(self):
        """
        Trains policy on env using algo
        """
        teacher_train_dict, teacher_distill_dict = self.teacher_schedule(self.curriculum_step)
        start_time = time.time()
        rollout_time = 0
        itrs_on_level = 0

        buffer = Buffer(self.buffer_name, self.args.buffer_capacity, self.args.prob_current, val_prob=.1)
        if self.args.use_dagger:
            dagger_buffer = Buffer(self.buffer_name, self.args.buffer_capacity, self.args.prob_current, val_prob=.1,
                                   buffer_name='dagger_buffer')
        else:
            dagger_buffer = None

        itr_start_time = time.time()

        all_time_training = 0
        all_time_collection = 0
        all_rp_splice_time = 0
        all_time_rp_train = 0
        all_run_policy_time = 0
        all_distill_time = 0
        all_run_supervised_time = 0
        all_rollout_time = 0
        all_unaccounted_time = 0


        total_distillation_frames = 0

        for itr in range(self.start_itr, self.args.n_itr):
            logger.logkv("ItrsOnLEvel", itrs_on_level)
            itrs_on_level += 1

            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            """ -------------------- Sampling --------------------------"""

            logger.log("Obtaining samples...")
            time_env_sampling_start = time.time()
            if not self.args.no_collect:
                samples_data, episode_logs = self.algo.collect_experiences(teacher_train_dict,
                                                                           collect_with_oracle=self.args.collect_with_oracle)
                raw_samples_data = copy.deepcopy(samples_data)
                buffer.add_batch(samples_data, self.curriculum_step)
                assert len(samples_data.action.shape) == 1, samples_data.action.shape

                if self.args.use_dagger:
                    dagger_samples_data, _ = self.algo.collect_experiences(teacher_train_dict, use_dagger=True,
                                                                           dagger_dict=self.no_teacher_dict)
                    dagger_buffer.add_batch(dagger_samples_data, self.curriculum_step)
                else:
                    dagger_samples_data = None

            else:
                episode_logs = None
                raw_samples_data = None
                dagger_samples_data = None
            time_collection = time.time() - time_env_sampling_start
            time_training_start = time.time()
            if not (self.args.no_collect or self.args.no_train_rl):
                summary_logs = self.algo.optimize_policy(samples_data, teacher_dict=teacher_train_dict)
            else:
                summary_logs = None
            time_training = time.time() - time_training_start
            self._log(episode_logs, summary_logs, tag="Train")
            logger.logkv('Curriculum Step', self.curriculum_step)
            advance_curriculum = self.check_advance_curriculum(episode_logs, summary_logs)
            logger.logkv('Train/Advance', int(advance_curriculum))

            # """ ------------------ Reward Predictor Splicing ---------------------"""
            rp_start_time = time.time()
            # samples_data = self.use_reward_predictor(samples_data)  # TODO: update
            rp_splice_time = time.time() - rp_start_time

            """ ------------------ Policy Update ---------------------"""

            logger.log("Optimizing policy...")
            # # This needs to take all samples_data so that it can construct graph for meta-optimization.
            time_rp_train_start = time.time()
            # self.train_rp(samples_data)
            time_rp_train = time.time() - time_rp_train_start

            """ ------------------ Distillation ---------------------"""
            if self.supervised_model is not None and advance_curriculum:
                time_distill_start = time.time()
                time_sampling_from_buffer = 0
                time_train_distill = 0
                time_val_distill = 0
                for _ in range(self.args.distillation_steps - 1):
                    sample_start = time.time()
                    sampled_batch = buffer.sample(total_num_trajs=self.args.batch_size, split='train')
                    time_sampling_from_buffer += (time.time() - sample_start)
                    sample_start = time.time()
                    total_distillation_frames += len(sampled_batch)
                    distill_log = self.distill(sampled_batch, is_training=True, teachers_dict=teacher_distill_dict)
                    time_train_distill += (time.time() - sample_start)

                    if self.args.use_dagger:
                        sampled_dagger_batch = dagger_buffer.sample(total_num_trajs=self.args.batch_size, split='train')
                        total_distillation_frames += len(sampled_dagger_batch)
                        self.distill(sampled_dagger_batch, is_training=True, teachers_dict=teacher_distill_dict)

                if raw_samples_data is not None:
                    sample_start = time.time()
                    total_distillation_frames += len(raw_samples_data)
                    distill_log = self.distill(trim_batch(raw_samples_data), is_training=True,
                                               teachers_dict=teacher_distill_dict)
                    time_train_distill += (time.time() - sample_start)
                if dagger_samples_data is not None:
                    total_distillation_frames += len(dagger_samples_data)
                    self.distill(trim_batch(dagger_samples_data), is_training=True,
                                 teachers_dict=teacher_distill_dict)
                for key_set, log_dict in distill_log.items():
                    key_set = '_'.join(key_set)
                    for k, v in log_dict.items():
                        logger.logkv(f"Distill/{key_set}{k}_Train", v)
                sample_start = time.time()
                sampled_val_batch = buffer.sample(total_num_trajs=self.args.batch_size, split='val')
                time_sampling_from_buffer += (time.time() - sample_start)
                sample_start = time.time()
                distill_log_val = self.distill(sampled_val_batch, is_training=False,
                                               teachers_dict=teacher_distill_dict)
                time_val_distill += (time.time() - sample_start)
                for key_set, log_dict in distill_log_val.items():
                    key_set = '_'.join(key_set)
                    for k, v in log_dict.items():
                        logger.logkv(f"Distill/{key_set}{k}_Val", v)
                distill_time = time.time() - time_distill_start
                try:
                    advance_curriculum = distill_log[()]['Accuracy'] >= self.args.accuracy_threshold
                except:
                    advance_curriculum = list(distill_log.values())[0]['Accuracy'] >= self.args.accuracy_threshold
                logger.logkv('Distill/Advance', int(advance_curriculum))
                logger.logkv('Distill/TotalFrames', total_distillation_frames)
                # print("DISTILLATION BREAKDOWN")
                # print("Sampling", time_sampling_from_buffer)
                # print("TRAINING", time_train_distill)
                # print("VAL", time_val_distill)

            else:
                distill_time = 0

            """ ------------------ Policy rollouts ---------------------"""
            run_policy_time = 0
            # TODO: put if advance_curriculum back in here
            if (itr % self.eval_every == 0) or (
                itr == self.args.n_itr - 1) or (advance_curriculum and itr % 10 == 0):
                train_advance_curriculum = advance_curriculum
                with torch.no_grad():
                    if self.supervised_model is not None:
                        # Distilled model
                        time_run_supervised_start = time.time()
                        logger.log("Running supervised model")
                        advance_curriculum_sup = self.run_supervised(self.il_trainer.acmodel, self.no_teacher_dict,
                                                                     "DRollout/")
                        run_supervised_time = time.time() - time_run_supervised_start
                    else:
                        run_supervised_time = 0
                        advance_curriculum_sup = True
                    if not self.args.no_train_rl:
                        # Original Policy
                        time_run_policy_start = time.time()
                        logger.log("Running model with teacher")
                        advance_curriculum_policy = self.run_supervised(self.algo.acmodel, teacher_train_dict,
                                                                        "Rollout/")
                        run_policy_time = time.time() - time_run_policy_start
                    else:
                        run_policy_time = 0
                        advance_curriculum_policy = True
                    advance_curriculum = advance_curriculum_policy and advance_curriculum_sup \
                                         and train_advance_curriculum
                    print("Advancing curriculum???", advance_curriculum)

                    logger.logkv('Advance', int(advance_curriculum))
            else:
                run_supervised_time = 0
                advance_curriculum = False

            """ ------------------- Logging Stuff --------------------------"""
            logger.logkv('Itr', itr)
            logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

            time_total = time.time() - start_time
            time_itr = time.time() - itr_start_time
            itr_start_time = time.time()
            logger.logkv('Time/Total', time_total)
            logger.logkv('Time/Itr', time_itr)

            logger.logkv('Curriculum Percent', self.curriculum_step / len(self.env.levels_list))

            process = psutil.Process(os.getpid())
            memory_use = process.memory_info().rss / float(2 ** 20)
            logger.logkv('Memory MiB', memory_use)

            logger.log(self.exp_name)

            logger.logkv('Time/Training', time_training)
            logger.logkv('Time/Collection', time_collection)
            logger.logkv('Time/RPUse', rp_splice_time)
            logger.logkv('Time/RPTrain', time_rp_train)
            logger.logkv('Time/RunwTeacher', run_policy_time)
            logger.logkv('Time/Distillation', distill_time)
            logger.logkv('Time/RunDistilled', run_supervised_time)
            logger.logkv('Time/VidRollout', rollout_time)
            time_unaccounted = time_itr - time_training - time_collection - \
                         rp_splice_time - time_rp_train - run_policy_time - distill_time - run_supervised_time - \
                         rollout_time
            logger.logkv('Time/Unaccounted', time_unaccounted)

            all_time_training += time_training
            all_time_collection += time_collection
            all_rp_splice_time += rp_splice_time
            all_time_rp_train += time_rp_train
            all_run_policy_time += run_policy_time
            all_distill_time += distill_time
            all_run_supervised_time += run_supervised_time
            all_rollout_time += rollout_time
            all_unaccounted_time += time_unaccounted

            logger.logkv('Time/All_Training', all_time_training / time_total)
            logger.logkv('Time/All_Collection', all_time_collection / time_total)
            logger.logkv('Time/All_RPUse', all_rp_splice_time / time_total)
            logger.logkv('Time/All_RPTrain', all_time_rp_train / time_total)
            logger.logkv('Time/All_RunwTeacher', all_run_policy_time / time_total)
            logger.logkv('Time/All_Distillation', all_distill_time / time_total)
            logger.logkv('Time/All_RunDistilled', all_run_supervised_time / time_total)
            logger.logkv('Time/All_VidRollout', all_rollout_time / time_total)
            logger.logkv('Time/All_Unaccounted', all_unaccounted_time / time_total)

            logger.dumpkvs()

            """ ------------------ Video Saving ---------------------"""

            should_save_video = (itr % self.save_videos_every == 0) or (
                itr == self.args.n_itr - 1) or advance_curriculum
            # If we're just collecting, don't log
            if (self.args.no_train_rl and self.supervised_model is None):
                should_save_video = False
            if should_save_video:
                time_rollout_start = time.time()
                if self.supervised_model is not None:
                    # distilled_det_advance = self.save_videos(self.il_trainer.acmodel,
                    #                                                             save_name='distilled_video_det',
                    #                                                             num_rollouts=10,
                    #                                                             teacher_dict=self.no_teacher_dict,
                    #                                                             save_video=should_save_video,
                    #                                                             log_prefix="DVidRollout/Det",
                    #                                                             stochastic=False)
                    distilled_stoch_advance = self.save_videos(self.il_trainer.acmodel,
                                                               save_name='distilled_video_stoch',
                                                               num_rollouts=10,
                                                               teacher_dict=self.no_teacher_dict,
                                                               save_video=should_save_video,
                                                               log_prefix="DVidRollout/Stoch",
                                                               stochastic=True)
                # teacher_det_advance = self.save_videos(self.algo.acmodel,
                #                                                           save_name='withTeacher_video_det',
                #                                                           num_rollouts=10,
                #                                                           teacher_dict=self.teacher_train_dict,
                #                                                           save_video=should_save_video,
                #                                                           log_prefix="VidRollout/Det", stochastic=False)
                teacher_stoch_advance = self.save_videos(self.algo.acmodel,
                                                         save_name='withTeacher_video_stoch',
                                                         num_rollouts=10,
                                                         teacher_dict=teacher_train_dict,
                                                         save_video=should_save_video,
                                                         log_prefix="VidRollout/Stoch",
                                                         stochastic=True)
                self.save_videos(self.algo.acmodel,
                                 save_name='oracle_video',
                                 num_rollouts=10,
                                 teacher_dict=self.no_teacher_dict,
                                 save_video=should_save_video,
                                 log_prefix="VidRollout/Oracle",
                                 stochastic=True,
                                 rollout_oracle=True)

                rollout_time = time.time() - time_rollout_start
            else:
                rollout_time = 0

            params = self.get_itr_snapshot(itr)
            step = self.curriculum_step

            if self.log_and_save:
                if (itr % self.save_every == 0) or (itr == self.args.n_itr - 1) or advance_curriculum:
                    logger.log("Saving snapshot...")
                    logger.save_itr_params(itr, step, params)
                    logger.log("Saved")

            if advance_curriculum and not self.args.single_level:
                if self.il_trainer is not None:
                    self.run_with_bad_teachers(buffer, teacher_train_dict)
                self.curriculum_step += 1
                self.sampler.advance_curriculum()
                self.algo.advance_curriculum()
                itrs_on_level = 0

        logger.log("Training finished")

    def _log(self, episode_logs, summary_logs, tag=""):
        if episode_logs is not None:
            avg_return = np.mean(episode_logs['return_per_episode'])
            avg_path_length = np.mean(episode_logs['num_frames_per_episode'])
            avg_success = np.mean(episode_logs['success_per_episode'])
            logger.logkv(f"{tag}/Success", avg_success)
            logger.logkv(f"{tag}/Return", avg_return)
            logger.logkv(f"{tag}/PathLength", avg_path_length)
        if summary_logs is not None:
            self.num_feedback_advice += summary_logs['num_feedback_advice']
            self.num_feedback_reward += summary_logs['num_feedback_reward']
            logger.logkv(f"{tag}/NumFeedbackAdvice", self.num_feedback_advice)
            logger.logkv(f"{tag}/NumFeedbackReward", self.num_feedback_reward)
            logger.logkv(f"{tag}/NumFeedbackTotal", self.num_feedback_advice + self.num_feedback_reward)
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
        log = self.rp_trainer.distill(samples, source=self.args.source, is_training=is_training)
        return log

    def run_with_bad_teachers(self, buffer, teachers_dict):
        sampled_batch = buffer.sample(total_num_trajs=self.args.batch_size, split='val')
        # We pass in teachers one at a time and see what success rate we'd get if we shuffle that teacher
        teacher_names = [teacher for teacher in teachers_dict.keys() if teachers_dict[teacher]]
        for teacher in teacher_names:
            teacher_subset_dict = {}
            for k in teachers_dict.keys():
                teacher_subset_dict[k] = False
            teacher_subset_dict[teacher] = True
            teacher_feedback = [obs_dict[teacher] for obs_dict in sampled_batch.obs]
            random.shuffle(teacher_feedback)
            for obs_dict, shuffled_obs in zip(sampled_batch.obs, teacher_feedback):
                obs_dict[teacher] = shuffled_obs
            log = self.il_trainer.distill(sampled_batch, source=self.args.source, is_training=False,
                                      teachers_dict=teacher_subset_dict, distill_target='all')
            log_dict = list(log.values())[0]
            for k, v in log_dict.items():
                logger.logkv(f"CheckTeachers/{teacher}{k}_Val", v)

    def distill(self, samples, is_training=False, teachers_dict=None):
        distill_target = 'powerset'
        if self.args.distill_all_teachers:
            distill_target = 'all'
        if self.args.distill_no_teachers:
            distill_target = 'none'
        log = self.il_trainer.distill(samples, source=self.args.source, is_training=is_training,
                                      teachers_dict=teachers_dict, distill_target=distill_target)
        return log

    def run_supervised(self, policy, teacher_dict, tag):
        policy.eval()
        paths = self.sampler.obtain_samples(log=False, advance_curriculum=False, policy=policy,
                                            teacher_dict=teacher_dict, max_action=False)
        samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix=tag,
                                                             log_teacher=self.train_with_teacher)
        advance_curriculum, avg_success, avg_accuracy = self.check_advance_curriculum_rollout(samples_data)
        logger.logkv(f"{tag}Advance", int(advance_curriculum))
        logger.logkv(f"{tag}AvgSuccess", avg_success)
        logger.logkv(f"{tag}AvgAccuracy", avg_accuracy)
        return advance_curriculum

    def save_videos(self, policy, save_name='sample_video', num_rollouts=2, teacher_dict={}, save_video=False,
                    log_prefix=None, stochastic=True, rollout_oracle=False):
        policy.eval()
        self.env.set_level_distribution(self.curriculum_step)
        save_wandb = (save_video and not self.is_debug)
        paths, accuracy, stoch_accuracy, det_accuracy = rollout(self.env, policy,
                                                                max_path_length=200,
                                                                reset_every=self.args.rollouts_per_meta_task,
                                                                stochastic=stochastic,
                                                                record_teacher=True, teacher_dict=teacher_dict,
                                                                video_directory=self.exp_name,
                                                                video_name=save_name + str(self.curriculum_step),
                                                                num_rollouts=num_rollouts, save_wandb=save_wandb,
                                                                save_locally=True,
                                                                num_save=5,
                                                                obs_preprocessor=self.obs_preprocessor,
                                                                rollout_oracle=rollout_oracle)
        if log_prefix is not None:
            logger.logkv(log_prefix + "Acc", accuracy)
            logger.logkv(log_prefix + "Stoch_Acc", stoch_accuracy)
            logger.logkv(log_prefix + "Det_Acc", det_accuracy)
            logger.logkv(log_prefix + "Reward", np.mean([sum(path['rewards']) for path in paths]))
            logger.logkv(log_prefix + "PathLength",
                         np.mean([path['env_infos'][-1]['episode_length'] for path in paths]))
            success = np.mean([path['env_infos'][-1]['success'] for path in paths])
            logger.logkv(log_prefix + "Success", success)
            advance = (accuracy >= self.args.accuracy_threshold) and (success >= self.args.success_threshold)
        return advance

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        d = dict(itr=itr,
                 policy=self.policy,
                 env=self.env,
                 args=self.args,
                 optimizer=self.algo.optimizer.state_dict(),
                 curriculum_step=self.curriculum_step, )
        if self.reward_predictor is not None:
            d['reward_predictor'] = self.reward_predictor
        if self.il_trainer is not None:
            d['supervised_model'] = self.il_trainer.acmodel
        return d

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
            logger.logkv(log_prefix + 'FalseNegative', -1)  # Indicates none available

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
