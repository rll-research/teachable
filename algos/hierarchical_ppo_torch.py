# Code found here: https://github.com/denisyarats/pytorch_sac

import copy

import numpy as np
import torch

from algos.agent import Agent, DoubleQCritic, DiagGaussianActor
from algos.ppo_torch import PPOAgent
from logger import logger

from algos import utils


class HierarchicalPPOAgent(PPOAgent):
    """PPO algorithm, hard-coded to work with OffsetWaypoint subgoals."""

    def __init__(self, args, obs_preprocessor, teacher, env,
                 device='cuda', discount=0.99,
                 actor_lr=1e-4, actor_betas=(0.9, 0.999), actor_update_frequency=1, critic_lr=1e-4,
                 critic_betas=(0.9, 0.999),
                 batch_size=1024, control_penalty=0, repeat_advice=1):
        super().__init__(args, obs_preprocessor, teacher, env, device=device, discount=discount, batch_size=batch_size,
                         control_penalty=control_penalty, actor_update_frequency=actor_update_frequency, actor_lr=actor_lr,
                         actor_betas=actor_betas, critic_lr=critic_lr, critic_betas=critic_betas)

        obs = env.reset()
        if args.image_obs:
            no_advice_obs_dim = args.image_dim
        else:
            no_advice_obs_dim = len(obs['obs'].flatten())
        self.high_level = utils.mlp(no_advice_obs_dim, args.hidden_size, 2, 2).to(self.device)

        self.high_level_optimizer = torch.optim.Adam(self.high_level.parameters(),
                                                 lr=actor_lr,
                                                 betas=actor_betas)
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
            logger.logkv('train_critic/loss', utils.to_np(critic_loss))

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
            for n, p in self.critic.named_parameters():
                param_norm = p.grad.detach().data.norm(2).cpu().numpy()
                logger.logkv(f'grads/{n}', param_norm)
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
        entropy = -dist.log_prob(dist.rsample()).sum(-1).mean()
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
            logger.logkv(f'grads/{n}', param_norm)
        for n, p in self.actor.named_parameters():
            if p.isnan().sum() > 0:
                print("NAN in actor after backprop!")
        self.actor_optimizer.step()

    def get_high_level(self, obs, preprocessed=False):
        if not preprocessed:
            no_advice_obs = self.obs_preprocessor(obs, 'none', show_instrs=True)
        else:
            no_advice_obs = obs
        if self.image_encoder is not None:
            no_advice_obs = self.image_encoder(no_advice_obs)
        if self.instr_encoder is not None:
            no_advice_obs = self.instr_encoder(no_advice_obs)
        no_advice_obs = no_advice_obs.obs.flatten(1).to(self.device)
        return self.high_level(no_advice_obs)

    def update_high_level(self, obs):
        ground_truth = torch.FloatTensor(np.stack([o['OffsetWaypoint'] for o in obs])).to(self.device)
        assert len(ground_truth.shape) == 2 and ground_truth.shape[-1] == 2
        pred_advice = self.get_high_level(obs)
        assert ground_truth.shape == pred_advice.shape
        assert ground_truth.dtype == pred_advice.dtype
        assert ground_truth.requires_grad == False
        assert pred_advice.requires_grad == True
        loss = torch.abs(ground_truth - pred_advice).norm(2, dim=1).mean()
        self.high_level_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level.parameters(), .5)
        for n, p in self.high_level.named_parameters():
            param_norm = p.grad.detach().data.norm(2).cpu().numpy()
            logger.logkv(f'grads/high_level{n}', param_norm)
        self.high_level_optimizer.step()
        logger.logkv('train_high_level/loss', utils.to_np(loss))
        logger.logkv('train_high_level/gt_max_abs', utils.to_np(torch.abs(ground_truth).max()))
        logger.logkv('train_high_level/x_diff', utils.to_np(torch.abs(ground_truth - pred_advice)[:, 0].mean()))
        logger.logkv('train_high_level/y_diff', utils.to_np(torch.abs(ground_truth - pred_advice)[:, 1].mean()))

    def get_hierarchical_actions(self, obs):
        offset_waypoint = self.get_high_level(obs)
        obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        obs.advice = offset_waypoint
        action, agent_dict = self.act(obs, sample=True)
        return utils.to_np(action[0]), agent_dict


    def act(self, obs, sample=False, ):
        if (not 'advice' in obs):  # unpreprocessed
            obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        action, agent_dict = super().act(obs, sample)  # TODO: do we need deepcopy?
        if self.image_encoder is not None:
            obs = self.image_encoder(obs)
        if self.instr_encoder is not None:
            obs = self.instr_encoder(obs)
        obs = torch.cat([obs.obs.flatten(1)] + [obs.advice] * self.repeat_advice, dim=1).to(self.device)
        value = self.critic(obs)
        agent_dict['value'] = value
        return action, agent_dict

    def optimize_policy(self, batch, step):
        self.update_high_level(batch.obs)
        super().optimize_policy(batch, step)