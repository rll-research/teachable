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
    def __init__(self, model, env, args, ):
        self.args = args

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
        new_demos = []
        num_demos = len(demos['actions'])
        for i in range(num_demos):
            new_demo = []
            done = False
            t = 0
            while not done:
                obs = demos['observations'][i, t]
                if source == 'agent':
                    action = demos['actions'][i, t]
                elif source == 'teacher':
                    action = demos['env_infos']['teacher_action'][i, t]
                else:
                    raise NotImplementedError(source)
                done = demos['dones'][i, t]
                new_demo.append((obs, action, done))
                t += 1
            new_demos.append(new_demo)
        return new_demos

    def obss_preprocessor(self, obs, device=None):
        obs_arr = np.stack(obs, 0)
        return torch.FloatTensor(obs_arr).to(device)

    def run_epoch_recurrence_one_batch(self, batch, is_training=False, source='agent'):

        if is_training:
            self.acmodel.train()
        else:
            self.acmodel.eval()


        batch = self.transform_demos(batch, source)
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]

        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))

        # (batch size * avg demo length , 3), where 3 is for (state, action, done)
        flat_batch = np.array(flat_batch, dtype=object)
        inds = inds[:-1]
        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=self.device)
        episode_ids = np.zeros(len(flat_batch))
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
                new_memory = self.acmodel(
                    preprocessed_obs,
                    memory[:len(inds), :], instr_embedding[:len(inds)])[1]

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
        per_token_count = [0, 0, 0, 0, 0, 0, 0]
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0

        indexes = self.starting_indexes(num_frames)
        memory = memories[indexes]
        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence
        for i in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)

            action_step = action_true[indexes]
            action_step = action_step[:, 0]

            mask_step = mask[indexes]
            model_results = self.acmodel(
                preprocessed_obs, memory * mask_step,
                instr_embedding[episode_ids[indexes]])
            dist = model_results[2]
            memory = model_results[1]

            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=False)[1]
            accuracy += float((action_pred == action_step).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss
            indexes += 1

            action_step = action_step.detach().cpu().numpy()
            action_pred = action_pred.detach().cpu().numpy()
            for j in range(len(per_token_count)):
                token_indices = np.where(action_step == j)[0]
                count = len(token_indices)
                correct = np.sum(action_step[token_indices] == action_pred[token_indices])
                per_token_correct[j] += correct
                per_token_count[j] += count

        final_loss /= self.args.recurrence

        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        log = {}
        log["Entropy"] = float(final_entropy / self.args.recurrence)
        log["Loss"] = float(final_policy_loss / self.args.recurrence)
        log["Accuracy"] = float(accuracy)
        # log["Count"] = float(total_frames)
        for i, (correct, count) in enumerate(zip(per_token_correct, per_token_count)):
            if count > 0:
                log[f'Accuracy_{i}'] = correct/count
                # log[f'Count_{i}'] = count
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
