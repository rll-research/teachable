# Code found here: https://github.com/denisyarats/pytorch_sac

import copy

import numpy as np
import torch
import torch.nn.functional as F

from algos.agent import Agent, DoubleQCritic, DiagGaussianActor
from logger import logger

from algos import utils


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env,
                 device='cuda', discount=0.99,
                 init_temperature=0.1, alpha_lr=1e-4, alpha_betas=(0.9, 0.999),
                 actor_lr=1e-4, actor_betas=(0.9, 0.999), actor_update_frequency=1, critic_lr=1e-4,
                 critic_betas=(0.9, 0.999), critic_tau=0.005, critic_target_update_frequency=2,
                 batch_size=1024, learnable_temperature=True, control_penalty=0, repeat_advice=1):
        super().__init__(args, obs_preprocessor, teacher, env, device=device, discount=discount, batch_size=batch_size,
                         control_penalty=control_penalty)

        obs = env.reset()
        if args.discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]

        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature

        if args.image_obs:
            obs_dim = args.image_dim + len(obs[teacher]) * repeat_advice
        else:
            obs_dim = len(obs['obs'].flatten()) + len(obs[teacher]) * repeat_advice
        self.critic = DoubleQCritic(obs_dim, action_dim, hidden_dim=args.hidden_size).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim=args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim, action_dim, discrete=args.discrete, hidden_dim=args.hidden_size).to(
            self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def optimize_policy(self, batch, step):
        obs = batch.obs
        action = batch.action
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        reward = batch.reward.unsqueeze(1)
        next_obs = batch.next_obs
        not_done = 1 - batch.full_done.unsqueeze(1)
        obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        preprocessed_obs = copy.deepcopy(obs)
        next_obs = self.obs_preprocessor(next_obs, self.teacher, show_instrs=True)
        if self.image_encoder is not None:
            obs = self.image_encoder(obs)
            next_obs = self.image_encoder(next_obs)
        if self.instr_encoder is not None:
            obs = self.instr_encoder(obs)
            next_obs = self.instr_encoder(next_obs)
        obs = torch.cat([obs.obs.flatten(1)] + [obs.advice] * self.repeat_advice, dim=1).to(self.device)
        next_obs = torch.cat([next_obs.obs.flatten(1)] + [next_obs.advice] * self.repeat_advice, dim=1).to(self.device)

        logger.logkv('train/batch_reward', utils.to_np(reward.mean()))

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_frequency == 0:
            if self.image_encoder is not None:
                preprocessed_obs = self.image_encoder(preprocessed_obs)
            if self.instr_encoder is not None:
                preprocessed_obs = self.instr_encoder(preprocessed_obs)
            obs = torch.cat([preprocessed_obs.obs.flatten(1)] + [preprocessed_obs.advice] * self.repeat_advice, dim=1).to(self.device)
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)