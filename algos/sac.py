# Code found here: https://github.com/denisyarats/pytorch_sac

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from algos.agent import Agent
from algos.utils import DoubleQCritic, DiagGaussianActor
from logger import logger

from algos import utils


class SACAgent(Agent, nn.Module):
    """SAC algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env,
                 device='cuda', discount=0.99,
                 init_temperature=0.1, alpha_lr=1e-4, alpha_betas=(0.9, 0.999),
                 actor_lr=1e-4, actor_betas=(0.9, 0.999), actor_update_frequency=1, critic_lr=1e-4,
                 critic_betas=(0.9, 0.999), critic_tau=0.005, critic_target_update_frequency=2,
                 batch_size=1024, learnable_temperature=True, control_penalty=0,advice_dim=128):
        obs = env.reset()
        if args.discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
        advice_size = 0 if teacher is 'none' else len(obs[teacher])
        #advice_dim = 0 if advice_size == 0 else args.advice_dim
        advice_dim = 0 if advice_size == 0 else advice_dim
        super().__init__(args, obs_preprocessor, teacher, env, device=device, advice_size=advice_size, advice_dim=advice_dim,
                         actor_update_frequency=actor_update_frequency)
        self.critic_tau = critic_tau
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature

        if args.image_obs:
            obs_dim = args.image_dim + advice_dim
        else:
            obs_dim = len(obs['obs'].flatten()) + advice_dim
        self.critic = DoubleQCritic(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim, action_dim, discrete=args.discrete, hidden_dim=args.hidden_dim).to(
            self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        embedding_params = []
        if self.state_encoder is not None:
            embedding_params += list(self.state_encoder.parameters())
        if self.task_encoder is not None:
            embedding_params += list(self.task_encoder.parameters())
        if self.advice_embedding is not None:
            embedding_params += list(self.advice_embedding.parameters())

        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()) + embedding_params,
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()) + embedding_params,
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

    def update_critic(self, obs, next_obs, batch, train=True, step=1):
        action = batch.action
        reward = batch.reward.unsqueeze(1)
        not_done = 1 - batch.full_done.unsqueeze(1)
        dist = self.actor(next_obs)
        if self.args.discrete:
            next_action, next_action_hard = dist.rsample(one_hot=True)
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        else:
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        if self.args.discrete:
            action = action.unsqueeze(1)
            action = F.one_hot(action[:, 0].long(), self.action_dim).float()
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        if train:
            logger.logkv('train_critic/loss', utils.to_np(critic_loss))

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        else:
            logger.logkv('val/critic_loss', utils.to_np(critic_loss))
            logger.logkv('val/Q_mean', utils.to_np(current_Q1.mean()))
            logger.logkv('val/Q_std', utils.to_np(current_Q1.std()))
            logger.logkv('val/entropy', utils.to_np(-log_prob.mean()))
            if self.args.discrete:
                action = torch.argmax(next_action_hard, dim=1)
                act_dim = next_action_hard.shape[-1]
                for i in range(act_dim):
                    prop_i = (action == i).float().mean()
                    logger.logkv(f'val/sampled_{i}', utils.to_np(prop_i))
            else:
                logger.logkv('val/abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
                logger.logkv('val/mean_std', utils.to_np(dist.loc.std()))
                logger.logkv('val/std', utils.to_np(dist.scale.mean()))
            logger.logkv('val/obs_min', utils.to_np(obs.min()))
            logger.logkv('val/obs_max', utils.to_np(obs.max()))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def update_actor(self, obs, batch):
        dist = self.actor(obs)
        if self.args.discrete:
            action, _ = dist.rsample(one_hot=True)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q + self.control_penalty * action.norm(2, dim=-1)).mean()

        logger.logkv('train_actor/loss', utils.to_np(actor_loss))
        logger.logkv('train_actor/target_entropy', self.target_entropy)
        logger.logkv('train_actor/entropy', utils.to_np(-log_prob.mean()))
        logger.logkv('train_actor/Q', utils.to_np(actor_Q.mean()))
        if not self.args.discrete:
            logger.logkv('train_actor/abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
            logger.logkv('train_actor/std', utils.to_np(dist.scale.mean()))
        logger.logkv('train_actor/act_norm', utils.to_np(action.norm(2, dim=-1).mean()))

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.logkv('train_alpha/loss', utils.to_np(alpha_loss))
            logger.logkv('train_alpha/value', utils.to_np(self.alpha))
            alpha_loss.backward()
            self.log_alpha_optimizer.step()