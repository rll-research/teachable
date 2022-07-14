from abc import ABC, abstractmethod
import torch
import numpy as np

from algos.utils import to_np
from utils.dictlist import DictList, merge_dictlists
from utils.penv import ParallelEnv, SequentialEnv
from logger import logger


class DataCollector(ABC):
    """The collection class."""

    def __init__(self, collect_policy, envs, args, repeated_seed=None, device=None):

        if not args.sequential:
            self.env = ParallelEnv(envs, repeated_seed=repeated_seed)
        else:
            self.env = SequentialEnv(envs, repeated_seed=repeated_seed)
        self.policy = collect_policy
        self.args = args


        # Store helpers values
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.num_procs = len(envs)
        self.num_frames = self.args.frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.args.frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.mask = torch.ones(shape[1], device=self.device).float()
        self.masks = torch.zeros(*shape, device=self.device)
        try:
            action_shape = envs[0].action_space.n
        except:  # continuous
            action_shape = envs[0].action_space.shape[0]
        if self.args.discrete:
            self.actions = torch.zeros(*shape, 1, device=self.device, dtype=torch.int)
            self.teacher_actions = torch.zeros(*shape, 1, device=self.device, dtype=torch.int)
            self.action_probs = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
        else:
            self.actions = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
            self.teacher_actions = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
            self.argmax_action = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
        self.rewards = torch.zeros(*shape, device=self.device)
        if args.on_policy:
            self.values = torch.zeros(*shape, device=self.device)
            self.log_probs = torch.zeros(*shape, device=self.device)
        self.dones = torch.zeros(*shape, device=self.device)
        self.done_index = torch.zeros(self.num_procs, device=self.device)
        self.env_infos = [None] * len(self.dones)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_success = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []
        self.log_success = []
        self.log_dist_to_goal = []
        self.log_keep = 25

    def collect_experiences(self, collect_with_oracle=False, collect_reward=True, train=True):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.args.frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.args.frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        policy = self.policy
        policy.train(train)
        ncount = 0
        for i in range(self.args.frames_per_proc):
            ncount += 1
            with torch.no_grad():
                action, agent_dict = policy.act(list(self.obs), sample=True,
                                                instr_dropout_prob=self.args.collect_dropout_prob)

            action_to_take = action.cpu().numpy()
            if collect_with_oracle:
                action_to_take = self.env.get_teacher_action()
            
            if self.args.noise:
                if self.args.discrete:
                    if np.random.uniform() < .1:
                        action_to_take = np.random.randint(0, 5, size=np.array(action_to_take).shape)
                else:
                    if self.args.frames_per_proc <= 5:
                        print("Warning! Entirely noise.")
                    if i == 0:
                        repeated_action = np.random.uniform(-1, 1, size=np.array(action_to_take).shape)
                    if i < 5:
                        action_to_take = repeated_action

            obs, reward, done, env_info = self.env.step(action_to_take)
            if not collect_reward:
                reward = [np.nan for _ in reward]

            # Update experiences values
            self.env_infos[i] = env_info
            self.obss[i] = self.obs
            self.obs = obs
            try:
                self.teacher_actions[i] = torch.FloatTensor(np.stack([ei['teacher_action'] for ei in env_info])).to(self.device)
            except Exception as e:
                self.teacher_actions[i] = self.teacher_actions[i] * 0 - 1

            self.masks[i] = self.mask
            done_tensor = torch.FloatTensor(done).to(self.device)
            self.dones[i] = done_tensor
            self.mask = 1 - done_tensor.to(torch.int32)
            self.actions[i] = action
            if self.args.discrete:
                probs = agent_dict['dist'].probs
                self.action_probs[i] = probs
            else:
                self.argmax_action[i] = agent_dict['argmax_action']
            if self.args.on_policy:
                self.values[i] = agent_dict['value'].squeeze(1)
                if self.args.discrete:
                    self.log_probs[i] = agent_dict['dist'].log_prob(action[:, 0])
                else:
                    self.log_probs[i] = agent_dict['dist'].log_prob(action).sum(-1)
            self.rewards[i] = torch.tensor(reward, device=self.device)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_success += torch.tensor([e['success'] for e in env_info], device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            # print(f"self log_episode_success: {self.log_episode_success}")
            for i, done_ in enumerate(done):
                if done_:
                    # print(f"done on frame {ncount} index {i}")
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_success.append(self.log_episode_success[i].item())
                    # print(f"self log_success: {self.log_success}")
                    if 'dist_to_goal' in env_info[i]:
                        self.log_dist_to_goal.append(env_info[i]['dist_to_goal'].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.mask = self.mask.float()
            self.log_episode_return *= self.mask
            self.log_episode_success *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.args.frames_per_proc)]
        keys = list(env_info[0].keys())
        batch = len(env_info)
        timesteps = len(self.env_infos)
        env_info_dict = {}
        for k in keys:
            arr = []
            for b in range(batch):
                for t in range(timesteps):
                    arr.append(self.env_infos[t][b][k])
            if k == 'next_obs':
                exps.next_obs = arr
            else:
                env_info_dict[k] = np.stack(arr)
        env_info_dict = DictList(env_info_dict)
        exps.env_infos = env_info_dict
        # In commments below T is self.args.frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P -> P x T -> (P * T) x 1

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(self.actions.shape[0] * self.actions.shape[1], -1)
        exps.teacher_action = self.teacher_actions.transpose(0, 1)
        exps.teacher_action = exps.teacher_action.reshape(self.teacher_actions.shape[0] * self.actions.shape[1], -1)
        if self.args.discrete:
            exps.action_probs = self.action_probs.transpose(0, 1).reshape(
                self.action_probs.shape[0] * self.action_probs.shape[1], -1)
        else:
            exps.argmax_action = self.argmax_action.transpose(0, 1)
            exps.argmax_action = exps.argmax_action.reshape(self.argmax_action.shape[0] * self.argmax_action.shape[1], -1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.done = self.dones.transpose(0, 1).reshape(-1)
        full_done = self.dones.transpose(0, 1)
        full_done[:, -1] = 1
        exps.full_done = full_done.reshape(-1).int()

        if self.args.on_policy:
            self.advantages = torch.zeros(self.args.frames_per_proc, self.num_procs, device=self.device)
            # Add advantage and return to experiences
            with torch.no_grad():
                action, agent_dict = policy.act(list(self.obs), sample=True, instr_dropout_prob=self.args.collect_dropout_prob)
                next_value = agent_dict['value'].squeeze(1)

            for i in reversed(range(self.args.frames_per_proc)):
                next_mask = self.masks[i + 1] if i < self.args.frames_per_proc - 1 else self.mask
                next_value = self.values[i + 1] if i < self.args.frames_per_proc - 1 else next_value
                next_advantage = self.advantages[i + 1] if i < self.args.frames_per_proc - 1 else 0

                delta = self.rewards[i] + self.args.discount * next_value * next_mask - self.values[i]
                self.advantages[i] = delta + self.args.discount * self.args.gae_lambda * next_advantage * next_mask

            exps.value = self.values.transpose(0, 1).reshape(-1)
            exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
            exps.returnn = exps.value + exps.advantage
            exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
            logger.logkv("Train/Value", to_np(exps.value.mean()))
            logger.logkv("Train/Advantage", to_np(exps.advantage.mean()))
            logger.logkv("Train/Returnn", to_np(exps.returnn.mean()))

        # Log some values
        log_cutoff = min(self.args.num_envs, self.log_keep)
        log = {
            "return_per_episode": [] if len(self.log_return) < log_cutoff else self.log_return[-self.log_keep:],
            "success_per_episode": [] if len(self.log_success) < log_cutoff else self.log_success[-self.log_keep:],
            "dist_to_goal_per_episode": [] if len(self.log_dist_to_goal) < log_cutoff else self.log_dist_to_goal[-self.log_keep:],
            "reshaped_return_per_episode": [] if len(self.log_reshaped_return) < log_cutoff else self.log_reshaped_return[-self.log_keep:],
            "num_frames_per_episode": [] if len(self.log_num_frames) < log_cutoff else self.log_num_frames[-self.log_keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.log_keep:]
        self.log_success = self.log_success[-self.log_keep:]
        self.log_dist_to_goal = self.log_dist_to_goal[-self.log_keep:]
        self.log_reshaped_return = self.log_reshaped_return[-self.log_keep:]
        self.log_num_frames = self.log_num_frames[-self.log_keep:]

        num_feedback_advice = 0
        for key in exps.obs[0].keys():
            if 'gave_' in key:
                teacher_name = key[5:]
                if teacher_name == 'none':
                    continue
                log[key] = np.sum([d[key] for d in exps.obs])
                num_feedback_advice += np.sum([d[key] for d in exps.obs])
        log["num_feedback_advice"] = num_feedback_advice
        log["num_feedback_reward"] = np.sum(exps.env_infos.gave_reward) if collect_reward else 0
        for key in exps.env_infos.keys():
            if 'followed_' in key:
                log[key] = np.sum(getattr(exps.env_infos, key))
        return exps, log
