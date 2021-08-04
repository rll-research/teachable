# Code found here: https://github.com/denisyarats/pytorch_sac

import copy

import numpy as np
import torch

from algos.agent import Agent, DoubleQCritic, DiagGaussianActor
from logger import logger

from algos import utils


class PPOAgent(Agent):
    """SAC algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env,
                 device='cuda', discount=0.99,
                 actor_lr=1e-4, actor_betas=(0.9, 0.999), actor_update_frequency=1, critic_lr=1e-4,
                 critic_betas=(0.9, 0.999),
                 batch_size=1024, control_penalty=0, repeat_advice=1):
        super().__init__(args, obs_preprocessor, teacher, env, device=device, discount=discount, batch_size=batch_size,
                         control_penalty=control_penalty, actor_update_frequency=actor_update_frequency)

        obs = env.reset()
        if args.discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]

        if args.image_obs:
            obs_dim = args.image_dim + len(obs[teacher]) * repeat_advice
        else:
            obs_dim = len(obs['obs'].flatten()) + len(obs[teacher]) * repeat_advice
        self.critic = utils.mlp(obs_dim, args.hidden_size, 1, 2).to(self.device)

        self.actor = DiagGaussianActor(obs_dim, action_dim, discrete=args.discrete, hidden_dim=args.hidden_size).to(
            self.device)

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.train()
        self.critic_target.train()

    def update_critic(self, obs, next_obs, batch, train=True, step=1):
        collected_value = batch.value
        collected_return = batch.returnn
        value = self.critic(obs)
        value_clipped = collected_value + torch.clamp(value - collected_value, -self.args.clip_eps, self.args.clip_eps)
        surr1 = (value - collected_return).pow(2)
        surr2 = (value_clipped - collected_return).pow(2)
        critic_loss = torch.max(surr1, surr2).mean()

        if train:
            logger.logkv('train_critic/loss', utils.to_np(critic_loss))

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        else:
            logger.logkv('val/critic_loss', utils.to_np(critic_loss))
            logger.logkv('val/V_mean', utils.to_np(value.mean()))
            logger.logkv('val/V_std', utils.to_np(value.std()))
            logger.logkv('val/obs_min', utils.to_np(obs.min()))
            logger.logkv('val/obs_max', utils.to_np(obs.max()))

    def update_actor(self, obs, batch):

        # control penalty
        dist = self.actor(obs)
        entropy = -dist.log_prob(dist.rsample()[0]).mean()
        action = batch.action
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        new_log_prob = dist.log_prob(action)
        ratio = torch.exp(new_log_prob - batch.log_prob)
        surrr1 = ratio * batch.advantage
        surrr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * batch.advantage
        control_penalty = action.float().norm(2, dim=-1).mean()
        policy_loss = -torch.min(surrr1, surrr2).mean()
        actor_loss = policy_loss \
                     - self.args.entropy_coef * entropy \
                     + self.control_penalty * control_penalty

        logger.logkv('train_actor/loss', utils.to_np(actor_loss))
        logger.logkv('train_actor/target_entropy', self.target_entropy)
        logger.logkv('train_actor/entropy', utils.to_np(entropy))
        logger.logkv('train_actor/V', utils.to_np(batch.value.mean()))
        if not self.args.discrete:
            logger.logkv('train_actor/abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
            logger.logkv('train_actor/std', utils.to_np(dist.scale.mean()))
        logger.logkv('train_actor/act_norm', utils.to_np(action.float().norm(2, dim=-1).mean()))

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
        self.actor_optimizer.step()

    def act(self, obs, sample=False):
        action, agent_dict = super().act(copy.deepcopy(obs), sample)
        if self.image_encoder is not None:
            obs = self.image_encoder(obs)
        if self.instr_encoder is not None:
            obs = self.instr_encoder(obs)
        obs = torch.cat([obs.obs.flatten(1)] + [obs.advice] * self.repeat_advice, dim=1).to(self.device)
        value = self.critic(obs)
        agent_dict['value'] = value
        return action, agent_dict