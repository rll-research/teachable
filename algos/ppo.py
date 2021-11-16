# Code found here: https://github.com/denisyarats/pytorch_sac

import torch

from algos.agent import Agent
from algos.utils import DiagGaussianActor
from logger import logger

from algos import utils


class PPOAgent(Agent):
    """PPO algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env, device='cuda', advice_dim=128):
        obs = env.reset()
        self.action_dim = env.action_space.n if args.discrete else env.action_space.shape[0]
        self.action_shape = 1 if args.discrete else self.action_dim
        self.advice_size = 0 if teacher is 'none' else len(obs[teacher])
        self.advice_dim = 0 if self.advice_size == 0 else advice_dim
        super().__init__(args, obs_preprocessor, teacher, env, device=device, advice_size=self.advice_size,
                         advice_dim=128)
        obs_dim = (128 + self.advice_dim) if args.image_obs else (len(obs['obs'].flatten()) + self.advice_dim)
        self.critic = utils.mlp(obs_dim, args.hidden_dim, 1, 2).to(self.device)
        self.actor = DiagGaussianActor(obs_dim, self.action_dim, discrete=args.discrete, hidden_dim=args.hidden_dim).to(
            self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=self.args.optim_eps)
        self.train()

    def update_critic(self, obs, next_obs, batch, train=True, step=1):
        assert len(obs.shape) == 2
        assert obs.shape == next_obs.shape
        collected_value = batch.value
        collected_return = batch.returnn
        value = self.critic(obs)[:, 0]  # (batch, )
        assert value.shape == collected_value.shape == collected_return.shape, \
            (value.shape, collected_value.shape, collected_return.shape)
        assert value.shape == torch.Size((len(obs), )), (value.shape, len(obs))
        value_clipped = collected_value + torch.clamp(value - collected_value, -self.args.clip_eps, self.args.clip_eps)
        surr1 = (value - collected_return).pow(2)
        surr2 = (value_clipped - collected_return).pow(2)
        critic_loss = torch.max(surr1, surr2).mean()

        if train:
            tag = 'Train'
            # Optimize the critic
            self.optimizer.zero_grad()
            critic_loss.backward()
            grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.critic.parameters() if p.grad is not None) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
            self.optimizer.step()
        else:
            tag = 'Val'
            grad_norm = torch.zeros_like(critic_loss)
        clip = surr1 - surr2
        self.log_critic(tag, critic_loss, value, collected_value, collected_return, obs, grad_norm, clip)

    def log_critic(self, tag, critic_loss, value, collected_value, collected_return, obs, grad_norm, clip):
        logger.logkv(f'{tag}/Value_loss', utils.to_np(critic_loss))
        logger.logkv(f'{tag}/V_mean', utils.to_np(value.mean()))
        logger.logkv(f'{tag}/Return', utils.to_np(collected_return.mean()))
        logger.logkv(f'{tag}/Collected_value', utils.to_np(collected_value.mean()))
        logger.logkv(f'{tag}/V_std', utils.to_np(value.std()))
        logger.logkv(f'{tag}/obs_min', utils.to_np(obs.min()))
        logger.logkv(f'{tag}/obs_max', utils.to_np(obs.max()))
        logger.logkv('Train/Grad_norm_critic', utils.to_np(grad_norm))
        logger.logkv('Train/ValueClip', utils.to_np(clip.mean()))

    def log_actor(self, actor_loss, dist, value, policy_loss, control_penalty, action, log_prob, recon_loss,
                  grad_norm, clip, ratio):
        logger.logkv('Train/LogProb', utils.to_np(log_prob).mean())
        logger.logkv('Train/Loss', utils.to_np(actor_loss))
        logger.logkv('Train/Entropy', utils.to_np(dist.entropy().mean()))
        logger.logkv('Train/Entropy_Loss',  - self.args.entropy_coef * utils.to_np(dist.entropy().mean()))
        logger.logkv('Train/Entropy_loss',  - self.args.entropy_coef * utils.to_np(dist.entropy().mean()))
        logger.logkv('Train/V', utils.to_np(value.mean()))
        logger.logkv('Train/policy_loss', utils.to_np(policy_loss))
        logger.logkv('Train/Policy_loss', utils.to_np(policy_loss))
        logger.logkv('Train/control_penalty', utils.to_np(control_penalty))
        logger.logkv('Train/recon_loss', utils.to_np(recon_loss))
        logger.logkv('Train/Grad_norm_actor', utils.to_np(grad_norm))
        logger.logkv('Train/PolicyClip', utils.to_np(clip.mean()))
        logger.logkv('Train/Ratio', utils.to_np(ratio))
        if not self.args.discrete:
            logger.logkv('Train/Action_abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
            logger.logkv('Train/Action_std', utils.to_np(dist.scale.mean()))
        logger.logkv('Train/Action_magnitude', utils.to_np(action.float().norm(2, dim=-1).mean()))
        logger.logkv('Train/Action_magnitude_L1', utils.to_np(action.float().norm(1, dim=-1).mean()))
        logger.logkv('Train/Action_max', utils.to_np(action.float().max(dim=-1).mean()))

    def update_actor(self, obs, batch, advice=None, no_advice_obs=None):
        assert len(obs.shape) == 2
        batch_size = len(obs)
        # control penalty
        import time
        t = time.time()
        dist = self.actor(obs)
        logger.logkv('Time/B_actor_time', time.time() - t)
        t = time.time()
        entropy = dist.entropy().mean()
        action = batch.action
        assert action.shape == (batch_size, self.action_shape), (action.shape, batch_size, self.action_shape)
        if self.args.discrete:
            new_log_prob = dist.log_prob(action[:, 0])
        else:
            new_log_prob = dist.log_prob(action).sum(-1)
        assert batch.log_prob.shape == new_log_prob.shape == batch.advantage.shape == (batch_size, )
        ratio = torch.exp(new_log_prob - batch.log_prob)
        surrr1 = ratio * batch.advantage
        surrr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * batch.advantage
        if self.args.discrete:
            control_penalty = torch.zeros(1, device=ratio.device).mean()
        else:
            control_penalty = dist.rsample().float().norm(2, dim=-1).mean()
        policy_loss = -torch.min(surrr1, surrr2).mean()
        if self.args.recon_coef > 0:
            recon_loss = self.compute_recon_loss(dist, no_advice_obs, advice)
        else:
            recon_loss = torch.zeros(1, device=ratio.device).mean()
        actor_loss = policy_loss - self.args.entropy_coef * entropy + self.args.control_penalty * control_penalty + \
                     self.args.recon_coef * recon_loss
        logger.logkv('Time/B_compute_loss_time', time.time() - t)
        t = time.time()
        # optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.actor.parameters() if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        self.optimizer.step()
        logger.logkv('Time/B_update_time', time.time() - t)
        t = time.time()
        clip = surrr1 - surrr2
        self.log_actor(actor_loss, dist, batch.value, policy_loss, control_penalty, action, new_log_prob, recon_loss,
                       grad_norm, clip, ratio)
        logger.logkv('Time/B_log_time', time.time() - t)

    def act(self, obs, sample=False, instr_dropout_prob=0):
        obs, addl_obs = self.format_obs(obs, instr_dropout_prob=instr_dropout_prob)
        dist = self.actor(obs)
        argmax_action = dist.probs.argmax(dim=1) if self.args.discrete else dist.mean
        action = dist.sample() if sample else argmax_action
        value = self.critic(obs)
        agent_info = {'argmax_action': argmax_action, 'dist': dist, 'value': value, 'addl_obs': addl_obs}
        if len(action.shape) == 1:  # Make sure discrete envs still have an action_dim dimension
            action = action.unsqueeze(1)
        return action, agent_info

