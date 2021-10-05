# Code found here: https://github.com/denisyarats/pytorch_sac

import torch

from algos.agent import Agent, DiagGaussianActor
from logger import logger

from algos import utils


class PPOAgent(Agent):
    """PPO algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env, device='cuda', discount=0.99,
                 lr=1e-4, betas=(0.9, 0.999), actor_update_frequency=1, batch_size=1024, control_penalty=0):
        obs = env.reset()
        if args.discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]

        advice_size = 0 if teacher is 'none' else len(obs[teacher])
        advice_dim = 0 if advice_size == 0 else args.advice_dim
        super().__init__(args, obs_preprocessor, teacher, env, device=device, discount=discount, batch_size=batch_size,
                         control_penalty=control_penalty, actor_update_frequency=actor_update_frequency,
                         advice_size=advice_size)
        if args.image_obs:
            obs_dim = args.image_dim + advice_dim
        else:
            obs_dim = len(obs['obs'].flatten()) + advice_dim
        self.critic = utils.mlp(obs_dim, args.hidden_size, 1, 2).to(self.device)
        self.actor = DiagGaussianActor(obs_dim, action_dim, discrete=args.discrete, hidden_dim=args.hidden_size).to(
            self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.image_encoder is not None:
            params += list(self.image_encoder.parameters())
        if self.instr_encoder is not None:
            params += list(self.instr_encoder.parameters())
        if self.advice_embedding is not None:
            params += list(self.advice_embedding.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=self.args.optim_eps)
        self.actor_optimizer = optimizer
        self.critic_optimizer = optimizer

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
        dist = self.actor(obs)
        entropy = dist.entropy().mean()
        action = batch.action
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        new_log_prob = dist.log_prob(action).sum(-1)
        ratio = torch.exp(new_log_prob - batch.log_prob)
        surrr1 = ratio * batch.advantage
        surrr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * batch.advantage
        control_penalty = dist.rsample().float().norm(2, dim=-1).mean()
        policy_loss = -torch.min(surrr1, surrr2).mean()
        actor_loss = policy_loss \
                     - self.args.entropy_coef * entropy \
                     + self.control_penalty * control_penalty

        logger.logkv('train_actor/loss', utils.to_np(actor_loss))
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
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        self.actor_optimizer.step()

    def act(self, obs, sample=False):
        if not 'advice' in obs:  # unpreprocessed
            obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        action, agent_dict = super().act(obs, sample)
        if self.image_encoder is not None:
            obs = self.image_encoder(obs)
        if self.instr_encoder is not None:
            obs = self.instr_encoder(obs)
        if self.advice_embedding is not None:
            advice = self.advice_embedding(obs.advice)
            obs = torch.cat([obs.obs.flatten(1), advice], dim=1)
        obs = obs.to(self.device)
        value = self.critic(obs)
        agent_dict['value'] = value
        return action, agent_dict
