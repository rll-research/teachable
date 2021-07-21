from abc import ABC, abstractmethod
import torch
import numpy as np
from envs.babyai.rl.utils.dictlist import DictList, merge_dictlists
from envs.babyai.rl.utils.penv import ParallelEnv, SequentialEnv


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, policy, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, parallel,
                 rollouts_per_meta_task=1, instr_dropout_prob=.5, repeated_seed=None, reset_each_batch=False):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        collect_policy : torch.Module
            the model + rl training algo
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

        """
        # Store parameters

        if parallel:
            self.env = ParallelEnv(envs, rollouts_per_meta_task, repeated_seed=repeated_seed)
        else:
            self.env = SequentialEnv(envs, rollouts_per_meta_task, repeated_seed=repeated_seed)
        self.policy = policy
        self.policy.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss
        self.reshape_reward = reshape_reward
        self.rollouts_per_meta_task = rollouts_per_meta_task
        self.instr_dropout_prob = instr_dropout_prob
        self.reset_each_batch = reset_each_batch

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.mask = torch.ones(shape[1], device=self.device).float()
        self.masks = torch.zeros(*shape, device=self.device)
        try:
            action_shape = envs[0].action_space.n
        except:  # continuous
            action_shape = envs[0].action_space.shape[0]
        if self.discrete:
            self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
            self.teacher_actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
            self.action_probs = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
        else:
            self.actions = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
            self.teacher_actions = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
            self.argmax_action = torch.zeros(*shape, action_shape, device=self.device, dtype=torch.float16)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.dones = torch.zeros(*shape, device=self.device)
        self.done_index = torch.zeros(self.num_procs, device=self.device)
        self.env_infos = [None] * len(self.dones)

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
        self.log_dist_to_goal = [0] * self.num_procs

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
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        policy = self.policy
        policy.train(train)

        if self.reset_each_batch:
            self.obs = self.env.reset()

        # TODO: Make this handle the case where the meta_rollout length > 1
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            instr_dropout_prob = self.instr_dropout_prob
            preprocessed_obs = [self.preprocess_obss([o], policy.teacher,
                                                     show_instrs=np.random.uniform() > instr_dropout_prob)
                for o in self.obs]
            preprocessed_obs = merge_dictlists(preprocessed_obs)

            with torch.no_grad():
                action, agent_dict = policy.act(preprocessed_obs, sample=True)

            action_to_take = action.cpu().numpy()
            if collect_with_oracle:
                action_to_take = self.env.get_teacher_action()

            obs, reward, done, env_info = self.env.step(action_to_take)
            if not collect_reward:
                reward = [np.nan for _ in reward]

            # Update experiences values
            self.env_infos[i] = env_info
            self.obss[i] = self.obs
            self.obs = obs
            try:
                self.teacher_actions[i] = torch.FloatTensor([ei['teacher_action'] for ei in env_info]).to(self.device)
            except:
                self.teacher_actions[i] = self.teacher_actions[i] * 0 - 1  # TODO: compute teacher action for new envs

            self.masks[i] = self.mask
            done_tensor = torch.FloatTensor(done).to(self.device)
            self.done_index = done_tensor + self.done_index

            done_meta = self.done_index == self.rollouts_per_meta_task
            self.done_index = torch.remainder(self.done_index, self.rollouts_per_meta_task)
            self.dones[i] = done_tensor
            self.mask = 1 - done_meta.to(torch.int32)
            self.actions[i] = action
            if self.discrete:
                probs = agent_dict['probs']
                self.action_probs[i] = probs
            else:
                self.argmax_action[i] = agent_dict['argmax_action']
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)

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

        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        if self.discrete:
            exps.action_probs = self.action_probs.transpose(0, 1)
            exps.action_probs = exps.action_probs.reshape(self.action_probs.shape[0] * self.action_probs.shape[1], -1)
            exps.teacher_action = self.teacher_actions.transpose(0, 1).reshape(-1)
        else:
            exps.argmax_action = self.argmax_action.transpose(0, 1)
            exps.argmax_action = exps.argmax_action.reshape(self.argmax_action.shape[0] * self.argmax_action.shape[1], -1)

            exps.action = exps.action.reshape(self.actions.shape[0] * self.actions.shape[1], -1)

            exps.teacher_action = self.teacher_actions.transpose(0, 1)
            exps.teacher_action = exps.teacher_action.reshape(self.teacher_actions.shape[0] * self.actions.shape[1], -1)

        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.done = self.dones.transpose(0, 1).reshape(-1)
        full_done = self.dones.transpose(0, 1)
        full_done[:, -1] = 1
        exps.full_done = full_done.reshape(-1)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "success_per_episode": self.log_success[-keep:],
            "dist_to_goal_per_episode": self.log_dist_to_goal[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_success = self.log_success[-self.num_procs:]
        self.log_dist_to_goal = self.log_dist_to_goal[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

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

    @abstractmethod
    def update_parameters(self):
        pass
2