import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
from babyai.evaluate import batch_evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.model import ACModel
import multiprocessing
import os
import json
import logging

logger = logging.getLogger(__name__)


class ImitationLearning(object):
    def __init__(self, model, env, args, distill_with_teacher, reward_predictor=False):
        self.args = args
        self.distill_with_teacher = distill_with_teacher
        self.reward_predictor = reward_predictor

        utils.seed(self.args.seed)
        self.env = env

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        # Define actor-critic model
        self.acmodel = model
        utils.save_model(self.acmodel, args.model)
        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE", self.device)

    def validate(self, episodes, verbose=True):
        # Seed needs to be reset for each validation, to ensure consistency
        utils.seed(self.args.val_seed)

        if verbose:
            logger.info("Validating the model")
        if getattr(self.args, 'multi_env', None):
            agent = utils.load_agent(self.env[0], model_name=self.args.model, argmax=True)
        else:
            agent = utils.load_agent(self.env, model_name=self.args.model, argmax=True)

        # Setting the agent model to the current model
        agent.model = self.acmodel

        agent.model.eval()
        logs = []

        for env_name in ([self.args.env] if not getattr(self.args, 'multi_env', None)
                         else self.args.multi_env):
            logs += [batch_evaluate(agent, env_name, self.args.val_seed, episodes)]
        agent.model.train()

        return logs

    def collect_returns(self):
        logs = self.validate(episodes=self.args.eval_episodes, verbose=False)
        mean_return = {tid: np.mean(log["return_per_episode"]) for tid, log in enumerate(logs)}
        return mean_return

    def transform_demos(self, demos, source='agent'):
        '''
        takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
        i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
        returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
        '''
        obs = demos.obs.detach().cpu().numpy()
        teacher_action = demos.teacher_action.detach().cpu().numpy()
        if self.reward_predictor:
            obs = demos.env_infos.next_obs_rewardfree
            reward = demos.reward
            action = torch.clamp(reward, 0, 2).long().detach().cpu().numpy()
        elif source == 'agent':
            action = demos.action.detach().cpu().numpy()
            assert len(action.shape) == 1, action.shape
        elif source == 'teacher':
            action = demos.teacher_action.detach().cpu().numpy()
            assert len(action.shape) == 1, action.shape
        else:
            raise NotImplementedError(source)
        done = (1 - demos.mask).detach().cpu().numpy()
        done = np.concatenate([done[1:], np.zeros_like(done[0:1])])
        split_indices = np.where(done == 1)[0]
        split_indices = np.concatenate([[0], split_indices, [len(obs)]], axis=0)
        new_demos = []
        for i in range(len(split_indices) - 1):
            o = obs[split_indices[i]:split_indices[i+1]]
            a = action[split_indices[i]:split_indices[i+1]]
            a_t = teacher_action[split_indices[i]:split_indices[i+1]][:]
            d = done[split_indices[i]:split_indices[i+1]]
            new_demos.append((o, a, d, a_t))
        return new_demos

    def obss_preprocessor(self, obs, device=None):
        obs_arr = np.stack(obs, 0)
        return torch.FloatTensor(obs_arr).to(device)

    def run_epoch_recurrence_one_batch(self, batch, is_training=False, source='agent'):

        if is_training:
            self.acmodel.train()
        else:
            self.acmodel.eval()

        batch_old = batch
        batch = self.transform_demos(batch, source)
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        obss = []
        action_true = []
        action_teacher = []
        done = []
        inds = [0]

        for demo in batch:
            obss.append(demo[0])
            action_true.append(demo[1])
            done.append(demo[2])
            action_teacher.append(demo[3])
            inds.append(inds[-1] + len(demo[0]))

        # (batch size * avg demo length , 3), where 3 is for (state, action, done)
        try:
            obss = np.concatenate(obss)
        except:
            print("?")
            import IPython
            IPython.embed()
        action_true = np.concatenate(action_true)
        assert len(action_true.shape) == 1
        done = np.concatenate(done)
        action_teacher = np.concatenate(action_teacher)
        inds = inds[:-1]
        num_frames = len(obss)

        mask = np.ones([len(obss)], dtype=np.float64)
        try:
            mask[inds] = 0
        except:
            print("???")
            print("BATCH LENGTH", len(batch))
            for demo in batch:
                print("Obs", demo[0]. shape)
                print("LEN", len(demo))
            print("INDS", inds, inds.shape)
            print("MASK", mask.shape)
            import IPython
            IPython.embed()
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)

        # Memory to be stored
        memories = torch.zeros([len(obss), self.acmodel.memory_size], device=self.device)
        episode_ids = np.zeros(len(obss))
        memory = torch.zeros([len(batch), self.acmodel.memory_size], device=self.device)

        preprocessed_first_obs = self.obss_preprocessor(obss[inds], self.device)
        instr = self.acmodel.get_instr(preprocessed_first_obs)
        instr_embedding = self.acmodel._get_instr_embedding(instr)

        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            preprocessed_obs = self.obss_preprocessor(obs, self.device)

            with torch.no_grad():
                # taking the memory till len(inds), as demos beyond that have already finished
                dist, info = self.acmodel(
                    preprocessed_obs,
                    memory[:len(inds), :], instr_embedding[:len(inds)], self.distill_with_teacher)
                new_memory = info['memory']

            memories[inds, :] = memory[:len(inds), :]
            memory[:len(inds), :] = new_memory
            episode_ids[inds] = range(len(inds))

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:int(len(inds) - sum(done_step))]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]

        # Here, actual backprop upto args.recurrence happens
        final_loss = 0
        per_token_correct = [0, 0, 0, 0, 0, 0, 0]
        per_token_teacher_correct = [0, 0, 0, 0, 0, 0, 0]
        per_token_count = [0, 0, 0, 0, 0, 0, 0]
        per_token_teacher_count = [0, 0, 0, 0, 0, 0, 0]
        per_token_agent_count = [0, 0, 0, 0, 0, 0, 0]
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0

        indexes = self.starting_indexes(num_frames)
        memory = memories[indexes]
        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence
        accuracy_list = []
        lengths_list = []
        for i in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)

            action_step = action_true[indexes]

            mask_step = mask[indexes]
            dist, info = self.acmodel(
                preprocessed_obs, memory * mask_step,
                instr_embedding[episode_ids[indexes]], self.distill_with_teacher)
            memory = info["memory"]

            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=False)[1]
            accuracy_list.append(float((action_pred == action_step).sum()))
            lengths_list.append((action_pred.shape, action_step.shape, indexes.shape))
            accuracy += float((action_pred == action_step).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss

            action_step = action_step.detach().cpu().numpy() # ground truth action
            action_pred = action_pred.detach().cpu().numpy() # action we took
            for j in range(len(per_token_count)):
                token_indices = np.where(action_step == j)[0]
                count = len(token_indices)
                correct = np.sum(action_step[token_indices] == action_pred[token_indices])
                per_token_correct[j] += correct
                per_token_count[j] += count

                action_teacher_index = action_teacher[indexes]
                teacher_token_indices = np.where(action_teacher_index == j)[0]
                teacher_count = len(teacher_token_indices)
                teacher_correct = np.sum(action_teacher_index[teacher_token_indices] == action_pred[teacher_token_indices])
                per_token_teacher_correct[j] += teacher_correct
                per_token_teacher_count[j] += teacher_count

                per_token_agent_count[j] += len(np.where(action_pred == j)[0])
            indexes += 1

        final_loss /= self.args.recurrence

        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        log = {}
        log["Entropy"] = float(final_entropy / self.args.recurrence)
        log["Loss"] = float(final_policy_loss / self.args.recurrence)
        log["Accuracy"] = float(accuracy)
        if not float(accuracy) <= 1:
            print("?")
            print("Accuracy List", len(accuracy_list), total_frames, len(indexes))
            print(accuracy_list)
            print(lengths_list)
            import IPython
            IPython.embed()
        assert float(accuracy) <= 1, float(accuracy)
        teacher_numerator = 0
        teacher_denominator = 0
        agent_numerator = 0
        agent_denominator = 0
        for i, (correct, count, teacher_correct, teacher_count) in enumerate(zip(per_token_correct, per_token_count, per_token_teacher_correct, per_token_teacher_count)):
            if count > 0:
                assert correct <= count, (correct, count)
                assert teacher_correct <= teacher_count, (teacher_correct, teacher_count)
                log[f'Accuracy_{i}'] = correct/count
                log[f'TeacherAccuracy_{i}'] = teacher_correct / teacher_count
                agent_numerator += correct
                agent_denominator += count
                teacher_numerator += teacher_correct
                teacher_denominator += teacher_count
        assert agent_denominator == teacher_denominator, (agent_denominator, teacher_denominator)
        assert abs(float(accuracy) - agent_numerator/agent_denominator) < .001, (accuracy, agent_numerator/agent_denominator)
        log["TeacherAccuracy"] = float(teacher_numerator / teacher_denominator)

        return log

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def distill(self, demo_batch, is_training=True, source='agent'):
        # Log is a dictionary with keys entropy, policy_loss, and accuracy
        log = self.run_epoch_recurrence_one_batch(demo_batch, is_training=is_training, source=source)
        if is_training:
            # Learning rate scheduler
            self.scheduler.step()
        return log
