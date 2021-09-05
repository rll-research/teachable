# Code found here: https://github.com/denisyarats/pytorch_sac

import copy

import numpy as np
import torch
import torch.nn as nn

from algos.agent import Agent, DoubleQCritic, DiagGaussianActor, initialize_parameters
from logger import logger

from algos import utils


class PPOAgent(Agent, nn.Module):
    """PPO algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env,
                 device='cuda', discount=0.99,
                 actor_lr=1e-4, actor_betas=(0.9, 0.999), actor_update_frequency=1, critic_lr=1e-4,
                 critic_betas=(0.9, 0.999),
                 batch_size=1024, control_penalty=0, repeat_advice=50):
        super().__init__(args, obs_preprocessor, teacher, env, device=device, discount=discount, batch_size=batch_size,
                         control_penalty=control_penalty, actor_update_frequency=actor_update_frequency,
                         repeat_advice=repeat_advice)

        obs = env.reset()
        if args.discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]

        if args.image_obs:
            obs_dim = args.image_dim + args.advice_dim  # TODO: what if there's no advice?
        else:
            obs_dim = len(obs['obs'].flatten()) + args.advice_dim  # TODO: what if there's no advice?
        self.critic = utils.mlp(obs_dim, args.hidden_size, 1, 2).to(self.device)
        self.advice_embedding = nn.Sequential(
            nn.Linear(len(obs[teacher]), args.advice_dim),
            nn.Sigmoid(),
        ).to(self.device)

        self.actor = DiagGaussianActor(obs_dim, action_dim, discrete=args.discrete, hidden_dim=args.hidden_size).to(
            self.device)

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.advice_embedding.parameters())
        self.optimizer = torch.optim.Adam(params,
                                                lr=actor_lr,
                                                betas=actor_betas)
        self.actor_optimizer = self.optimizer
        self.critic_optimizer = self.optimizer

        model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Actor parameters:", params)

        model_parameters = filter(lambda p: p.requires_grad, self.critic.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Critic parameters:", params)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Total parameters:", params)
        
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
        #                                         lr=actor_lr,
        #                                         betas=actor_betas)
        #
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
        #                                          lr=critic_lr,
        #                                          betas=critic_betas)
        self.apply(initialize_parameters)

        self.train()

    def update_critic(self, obs, next_obs, batch, train=True, step=1):
        collected_value = batch.value
        collected_return = batch.returnn
        value = self.critic(obs)
        value_clipped = collected_value + torch.clamp(value - collected_value, -self.args.clip_eps, self.args.clip_eps)
        surr1 = (value - collected_return).pow(2)
        surr2 = (value_clipped - collected_return).pow(2)
        critic_loss = torch.max(surr1, surr2).mean()

        if train:
            tag = 'train_critic'

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
            for n, p in self.critic.named_parameters():
                param_norm = p.grad.detach().data.norm(2).cpu().numpy()
                logger.logkv(f'grads/critic{n}', param_norm)
            self.critic_optimizer.step()
        else:
            tag = 'val'
        logger.logkv(f'{tag}/critic_loss', utils.to_np(critic_loss))
        logger.logkv(f'{tag}/V_mean', utils.to_np(value.mean()))
        logger.logkv(f'{tag}/Return', utils.to_np(collected_return.mean()))
        logger.logkv(f'{tag}/Collected_value', utils.to_np(collected_value.mean()))
        logger.logkv(f'{tag}/V_std', utils.to_np(value.std()))
        logger.logkv(f'{tag}/obs_min', utils.to_np(obs.min()))
        logger.logkv(f'{tag}/obs_max', utils.to_np(obs.max()))

    def update_actor(self, obs, batch):

        # control penalty
        for n, p in self.actor.named_parameters():
            if p.isnan().sum() > 0:
                print("NAN in actor before anything else!")
        dist = self.actor(obs)
        entropy = -dist.log_prob(dist.rsample()).sum(-1).mean()
        entropy = torch.clamp(entropy, -10, 10) # Prevent this from blowing up
        action = batch.action
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        new_log_prob = dist.log_prob(action).sum(-1)
        if new_log_prob.isnan().sum() > 0:
            print("DIST")
            print(dist.scale.min(), dist.scale.max(), dist.scale.isnan().sum())
            print(dist.loc.min(), dist.loc.max(), dist.loc.isnan().sum())
        ratio = torch.exp(new_log_prob - batch.log_prob)
        surrr1 = ratio * batch.advantage
        surrr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * batch.advantage
        control_penalty = dist.rsample().float().norm(2, dim=-1).mean()
        policy_loss = -torch.min(surrr1, surrr2).mean()
        if policy_loss.isnan():
            print("DIST")
            print(dist.scale.min(), dist.scale.max(), dist.scale.isnan().sum())
            print(dist.loc.min(), dist.loc.max(), dist.loc.isnan().sum())

            print("bad policy loss!")
        if entropy.isnan():
            print("bad entropy!")
        if control_penalty.isnan():
            print("bad control_penalty!")

        actor_loss = policy_loss \
                     - self.args.entropy_coef * entropy \
                     + self.control_penalty * control_penalty

        logger.logkv('train_actor/loss', utils.to_np(actor_loss))
        logger.logkv('train_actor/target_entropy', self.target_entropy)
        logger.logkv('train_actor/entropy', utils.to_np(entropy))
        logger.logkv('Train/Entropy', utils.to_np(dist.entropy().mean()))
        logger.logkv('train_actor/V', utils.to_np(batch.value.mean()))
        logger.logkv('train_actor/policy_loss', utils.to_np(policy_loss))
        logger.logkv('train_actor/control_penalty', utils.to_np(control_penalty))
        if not self.args.discrete:
            logger.logkv('train_actor/abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
            logger.logkv('train_actor/std', utils.to_np(dist.scale.mean()))
        logger.logkv('train_actor/act_norm', utils.to_np(action.float().norm(2, dim=-1).mean()))

        # optimize the actor
        self.actor_optimizer.zero_grad()
        for n, p in self.actor.named_parameters():
            if p.isnan().sum() > 0:
                print("NAN in actor before backprop!")
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        for n, p in self.actor.named_parameters():
            param_norm = p.grad.detach().data.norm(2).cpu().numpy()
            logger.logkv(f'grads/actor{n}', param_norm)
        self.actor_optimizer.step()
        for n, p in self.actor.named_parameters():
            if p.isnan().sum() > 0:
                print("NAN in actor after backprop!")

    def act(self, obs, sample=False):
        if not 'advice' in obs:  # unpreprocessed
            obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        action, agent_dict = super().act(obs, sample)  # TODO: do we need deepcopy?
        if self.image_encoder is not None:
            obs = self.image_encoder(obs)
        if self.instr_encoder is not None:
            obs = self.instr_encoder(obs)
        if self.advice_embedding is not None:  # TODO: should it be None if no advice?
            advice = self.advice_embedding(obs.advice)
        else:
            advice = [obs.advice] * self.repeat_advice  # TODO: when do we hit this case?
        obs = torch.cat([obs.obs.flatten(1)] + advice, dim=1).to(self.device)
        value = self.critic(obs)
        agent_dict['value'] = value
        return action, agent_dict
