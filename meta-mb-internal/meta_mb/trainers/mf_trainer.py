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
        args,
        algo,
        policy,
        env,
        sampler,
        sample_processor,
        start_itr=0,
        exp_name="",
        curriculum_step=0,
        il_trainer=None,
        supervised_model=None,
        reward_predictor=None,
        rp_trainer=None,
        is_debug=False,
        eval_every=25,
        save_every=100,
        log_every=10,
        save_videos_every=10,
        log_and_save=True,
        teacher_train_dict={},
        obs_preprocessor=None,
    ):
        self.args = args
        self.algo = algo
        self.policy = policy
        self.env = copy.deepcopy(env)
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.start_itr = start_itr
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
        self.teacher_train_dict = teacher_train_dict  # TODO: write a function where we can adjust this over time
        # Dict saying which teacher types we should distill to.  Currently set to be identical, but could be different
        # if we were training on a single teacher but distilling to a set, or something like that.
        self.teacher_distill_dict = teacher_train_dict
        # Dict specifying no teacher provided.
        self.no_teacher_dict = copy.deepcopy(teacher_train_dict)
        for k in self.no_teacher_dict.keys():
             self.no_teacher_dict[k] = False
             self.obs_preprocessor = obs_preprocessor

    def check_advance_curriculum(self, episode_logs, summary_logs):
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
        correct_actions = (data['actions'] == data['env_infos']['teacher_action'][:,:,0]).sum() - pad_steps
        avg_accuracy = correct_actions / (np.prod(data['actions'].shape) - pad_steps)
        # We take the max since runs which end early will be 0-padded
        should_advance_curriculum = avg_success >= self.args.success_threshold
        return should_advance_curriculum, avg_success, avg_accuracy

    def train(self):
        """
        Trains policy on env using algo
        """
        start_time = time.time()
        rollout_time = 0
        itrs_on_level = 0

        for itr in range(self.start_itr, self.args.n_itr):
            logger.logkv("ItrsOnLEvel", itrs_on_level)
            itrs_on_level += 1

            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            """ -------------------- Sampling --------------------------"""

            logger.log("Obtaining samples...")
            time_env_sampling_start = time.time()
            samples_data, episode_logs = self.algo.collect_experiences(self.teacher_train_dict)
            assert len(samples_data.action.shape) == 1, (samples_data.action.shape)
            time_collection = time.time() - time_env_sampling_start
            time_training_start = time.time()
            summary_logs = self.algo.optimize_policy(samples_data, teacher_dict=self.teacher_train_dict)
            time_training = time.time() - time_training_start
            self._log(episode_logs, summary_logs, tag="Train")
            logger.logkv('Curriculum Step', self.curriculum_step)
            advance_curriculum = self.check_advance_curriculum(episode_logs, summary_logs)
            logger.logkv('Train/Advance', int(advance_curriculum))
            time_env_sampling = time.time() - time_env_sampling_start

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
                for _ in range(self.args.distillation_steps):
                    distill_log = self.distill(samples_data, is_training=True, teachers_dict=self.teacher_train_dict)
                for k, v in distill_log.items():
                    logger.logkv(f"Distill/{k}_Train", v)
                distill_time = time.time() - time_distill_start
                advance_curriculum = distill_log['Accuracy'] >= self.args.accuracy_threshold
                logger.logkv('Distill/Advance', int(advance_curriculum))
            else:
                distill_time = 0

            """ ------------------ Policy rollouts ---------------------"""
            run_policy_time = 0
            if advance_curriculum or (itr % self.eval_every == 0) or (
                itr == self.args.n_itr - 1):
                train_advance_curriculum = advance_curriculum
                with torch.no_grad():
                    if self.supervised_model is not None:
                        # Distilled model
                        time_run_supervised_start = time.time()
                        self.sampler.supervised_model.reset(dones=[True] * len(samples_data.obs))
                        logger.log("Running supervised model")
                        advance_curriculum_sup = self.run_supervised(self.il_trainer.acmodel, self.no_teacher_dict,
                                                                     "DRollout/")
                        run_supervised_time = time.time() - time_run_supervised_start
                    else:
                        run_supervised_time = 0
                        advance_curriculum_sup = True
                    # Original Policy
                    time_run_policy_start = time.time()
                    self.algo.acmodel.reset(dones=[True] * len(samples_data.obs))
                    logger.log("Running model with teacher")
                    advance_curriculum_policy = self.run_supervised(self.algo.acmodel, self.teacher_train_dict,
                                                                    "Rollout/")
                    run_policy_time = time.time() - time_run_policy_start

                    advance_curriculum = advance_curriculum_policy and advance_curriculum_sup \
                                         and train_advance_curriculum
                    print("ADvancing curriculum???", advance_curriculum)

                    logger.logkv('Advance', int(advance_curriculum))
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

            should_save_video = (itr % self.save_videos_every == 0) or (
                    itr == self.args.n_itr - 1) or advance_curriculum
            if should_save_video:
                time_rollout_start = time.time()
                if self.supervised_model is not None:
                    self.save_videos(itr, self.il_trainer.acmodel, save_name='distilled_video_det',
                                     num_rollouts=5,
                                     teacher_dict=self.no_teacher_dict,
                                     save_video=should_save_video, log_prefix="DVidRollout/Det", stochastic=False)
                    self.save_videos(itr, self.il_trainer.acmodel, save_name='distilled_video_stoch',
                                     num_rollouts=5,
                                     teacher_dict=self.teacher_train_dict,
                                     save_video=should_save_video, log_prefix="DVidRollout/Stoch", stochastic=True)
                    self.save_videos(itr, self.il_trainer.acmodel, save_name='distilled_video_stoch_oracle',
                                     num_rollouts=5,
                                     teacher_dict=self.teacher_train_dict,
                                     save_video=should_save_video, log_prefix="DVidRollout/OracleStoch",
                                     stochastic=True, rollout_oracle=True)

                self.save_videos(self.curriculum_step, self.algo.acmodel, save_name='withTeacher_video_stoch',
                                 num_rollouts=5,
                                 teacher_dict=self.teacher_train_dict,
                                 save_video=should_save_video, log_prefix="VidRollout/Stoch", stochastic=True)
                self.save_videos(self.curriculum_step, self.algo.acmodel, save_name='withTeacher_video_det',
                                 num_rollouts=5,
                                 teacher_dict=self.no_teacher_dict,
                                 save_video=should_save_video, log_prefix="VidRollout/Det", stochastic=False)
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
        self.num_feedback_advice += summary_logs['num_feedback_advice']
        self.num_feedback_reward += summary_logs['num_feedback_reward']
        logger.logkv(f"{tag}/NumFeedbackAdvice", self.num_feedback_advice)
        logger.logkv(f"{tag}/NumFeedbackReward", self.num_feedback_reward)
        logger.logkv(f"{tag}/NumFeedbackTotal", self.num_feedback_advice + self.num_feedback_reward)

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
        log = self.rp_trainer.distill(samples, source=self.args.source, is_training=is_training)
        return log

    def distill(self, samples, is_training=False, teachers_dict=None):
        log = self.il_trainer.distill(samples, source=self.args.source, is_training=is_training,
                                      teachers_dict=teachers_dict)
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

    def save_videos(self, step, policy, save_name='sample_video', num_rollouts=2, teacher_dict={}, save_video=False,
                    log_prefix=None, stochastic=True, rollout_oracle=False):
        policy.eval()
        self.env.set_level_distribution(self.curriculum_step)
        save_wandb = (save_video and not self.is_debug)
        paths, accuracy = rollout(self.env, policy,
                                  max_path_length=200,
                                  reset_every=self.args.rollouts_per_meta_task,
                                  stochastic=stochastic,
                                  record_teacher=True, teacher_dict=teacher_dict,
                                  video_directory=self.exp_name, video_name=save_name + str(self.curriculum_step),
                                  num_rollouts=num_rollouts, save_wandb=save_wandb, save_locally=True,
                                  obs_preprocessor=self.obs_preprocessor, rollout_oracle=rollout_oracle)
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
