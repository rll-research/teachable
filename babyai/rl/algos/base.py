from abc import ABC, abstractmethod
import torch
import numpy as np
from meta_mb.logger import logger
import time

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv, SequentialEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info, parallel,
                 rollouts_per_meta_task=1):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        if parallel:
            self.env = ParallelEnv(envs, rollouts_per_meta_task)
        else:
            self.env = SequentialEnv(envs, rollouts_per_meta_task)
        self.policy_dict = acmodel
        for policy in self.policy_dict.values():
            policy.train()
        example_policy = list(self.policy_dict.values())[0]
        # self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info
        self.rollouts_per_meta_task = rollouts_per_meta_task

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], example_policy.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, example_policy.memory_size, device=self.device)
        self.dagger_memory = torch.zeros(shape[1], example_policy.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device).float()
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.action_probs = torch.zeros(*shape, envs[0].action_space.n, device=self.device, dtype=torch.float16)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.teacher_actions = torch.zeros(*shape, device=self.device)
        self.dones = torch.zeros(*shape, device=self.device)
        self.done_index = torch.zeros(self.num_procs, device=self.device)
        self.env_infos = [None] * len(self.dones)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_success = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        self.log_success = [0] * self.num_procs

    def collect_experiences(self, teacher_dict, use_dagger=False, dagger_dict={}, collect_with_oracle=False,
                            collect_reward=True):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        active_teachers = [k for k, v in teacher_dict.items() if v]
        assert len(active_teachers) >= 1
        teacher = 'none' if len(active_teachers) == 0 else active_teachers[0]
        acmodel = self.policy_dict[teacher]
        # TODO: Make this handle the case where the meta_rollout length > 1
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, teacher_dict)

            with torch.no_grad():
                dist, model_results = acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = None

            action = dist.sample()
            action_to_take = action.cpu().numpy()
            probs = dist.probs

            if collect_with_oracle:
                action_to_take = self.env.get_teacher_action()
            elif use_dagger:
                with torch.no_grad():
                    dagger_obs = self.preprocess_obss(self.obs, dagger_dict)
                    dagger_dist, dagger_model_results = acmodel(dagger_obs,
                                                                     self.dagger_memory * self.mask.unsqueeze(1))
                    self.dagger_memory = dagger_model_results['memory']
                    action_to_take = dagger_dist.sample().cpu().numpy()

            obs, reward, done, env_info = self.env.step(action_to_take)
            if not collect_reward:
                reward = [np.nan for _ in reward]

            # Update experiences values

            self.env_infos[i] = env_info
            self.obss[i] = self.obs
            self.obs = obs
            self.teacher_actions[i] = torch.FloatTensor([ei['teacher_action'][0] for ei in env_info]).to(self.device)

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            done_tensor = torch.FloatTensor(done).to(self.device)
            self.done_index = done_tensor + self.done_index

            done_meta = self.done_index == self.rollouts_per_meta_task
            self.done_index = torch.remainder(self.done_index, self.rollouts_per_meta_task)
            self.dones[i] = done_tensor
            self.mask = 1 - done_meta.to(torch.int32)
            self.actions[i] = action
            self.action_probs[i] = probs
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_success += torch.tensor([e['success'] for e in env_info], device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_success.append(self.log_episode_success[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.mask = self.mask.float()
            self.log_episode_return *= self.mask
            self.log_episode_success *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, teacher_dict)
        with torch.no_grad():
            next_value = acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))[1]['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        keys = list(env_info[0].keys())
        batch = len(env_info)
        timesteps = len(self.env_infos)
        env_info_dict = {}
        for k in keys:
            arr = []
            for b in range(batch):
                for t in range(timesteps):
                    arr.append(self.env_infos[t][b][k])
            env_info_dict[k] = np.stack(arr)
        env_info_dict = DictList(env_info_dict)
        exps.env_infos = env_info_dict
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.action_probs = self.action_probs.transpose(0, 1)
        exps.action_probs = exps.action_probs.reshape(self.action_probs.shape[0] * self.action_probs.shape[1], -1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        exps.teacher_action = self.teacher_actions.transpose(0, 1).reshape(-1)
        exps.done = self.dones.transpose(0, 1).reshape(-1)
        full_done = self.dones.transpose(0, 1)
        full_done[:, -1] = 1
        exps.full_done = full_done.reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "success_per_episode": self.log_success[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_success = self.log_success[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        num_feedback_advice = 0
        for key in exps.obs[0].keys():
            if 'gave_' in key:
                log[key] = np.sum([d[key] for d in exps.obs])
                num_feedback_advice += np.sum([d[key] for d in exps.obs])
        log["num_feedback_advice"] = num_feedback_advice
        log["num_feedback_reward"] = np.sum(exps.env_infos.gave_reward) if collect_reward else 0
        for key in exps.env_infos.keys():
            if 'followed_' in key:
                log[key] = np.sum(getattr(exps.env_infos, key))
        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
