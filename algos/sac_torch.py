# Code found here: https://github.com/denisyarats/pytorch_sac

import math
from torch import distributions as pyd
from torch import distributions

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from logger import logger

from algos import utils


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim=1024, hidden_depth=2, log_std_bounds=(-5, 2), discrete=False):
        super().__init__()

        self.discrete = discrete
        self.log_std_bounds = log_std_bounds
        if not discrete:
            action_dim *= 2
        self.trunk = utils.mlp(obs_dim, hidden_dim, action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        if self.discrete:
            logits = self.trunk(obs)
            dist = GumbelSoftmax(temperature=1e-3, logits=logits)
        else:
            mu, log_std = self.trunk(obs).chunk(2, dim=-1)

            # constrain log_std inside [log_std_min, log_std_max]
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                         1)

            std = log_std.exp()

            self.outputs['mu'] = mu
            self.outputs['std'] = std

            dist = SquashedNormal(mu, std)
        return dist


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim=1024, hidden_depth=2):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class SACAgent:
    """SAC algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env,
                 device='cuda', discount=0.99,
                 init_temperature=0.1, alpha_lr=1e-4, alpha_betas=(0.9, 0.999),
                 actor_lr=1e-4, actor_betas=(0.9, 0.999), actor_update_frequency=1, critic_lr=1e-4,
                 critic_betas=(0.9, 0.999), critic_tau=0.005, critic_target_update_frequency=2,
                 batch_size=1024, learnable_temperature=True):
        super().__init__()
        obs = env.reset()
        obs_dim = len(obs['obs'].flatten()) + len(obs[teacher])
        if args.discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
            self.action_range = (env.action_space.low, env.action_space.high)
        self.action_dim = action_dim
        self.args = args
        self.obs_preprocessor = obs_preprocessor
        self.teacher = teacher
        self.env = env
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = DoubleQCritic(obs_dim, action_dim).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim, action_dim, discrete=args.discrete).to(self.device)

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

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.cat([obs.obs.flatten(1), obs.advice], dim=1).to(self.device) / 10
        dist = self.actor(obs)
        if self.args.discrete:
            argmax_action = dist.probs.argmax(dim=1)
            action = dist.sample() if sample else argmax_action
        else:
            action = dist.sample() if sample else dist.mean
            min_val, max_val = self.action_range
            action = torch.maximum(action, torch.FloatTensor(min_val).unsqueeze(0).to(self.device))
            action = torch.minimum(action, torch.FloatTensor(max_val).unsqueeze(0).to(self.device))
            argmax_action = dist.mean
        agent_info = {
            'argmax_action': argmax_action,
            'dist': dist,
        }
        return action, agent_info

    def update_critic(self, obs, action, reward, next_obs, not_done, train=True):
        dist = self.actor(next_obs)
        next_action = dist.rsample(one_hot=True)
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        if self.args.discrete:
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
            logger.logkv('val/abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
            logger.logkv('val/mean_std', utils.to_np(dist.loc.std()))
            logger.logkv('val/std', utils.to_np(dist.scale.mean()))

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.logkv('train_actor/loss', utils.to_np(actor_loss))
        logger.logkv('train_actor/target_entropy', self.target_entropy)
        logger.logkv('train_actor/entropy', utils.to_np(-log_prob.mean()))
        logger.logkv('train_actor/Q', utils.to_np(actor_Q.mean()))
        logger.logkv('train_actor/abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
        logger.logkv('train_actor/std', utils.to_np(dist.scale.mean()))

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

    def log_rl(self, val_batch):
        self.train(False)
        with torch.no_grad():
            obs = val_batch.obs
            action = val_batch.action
            reward = val_batch.reward.unsqueeze(1)
            next_obs = val_batch.next_obs
            not_done = 1 - val_batch.full_done.unsqueeze(1)
            obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
            obs = torch.cat([obs.obs, obs.advice], dim=1).to(self.device) / 3
            next_obs = self.obs_preprocessor(next_obs, self.teacher, show_instrs=True)
            next_obs = torch.cat([next_obs.obs, next_obs.advice], dim=1).to(self.device) / 3
            self.update_critic(obs, action, reward, next_obs, not_done, train=False)


    def optimize_policy(self, batch, step):
        obs = batch.obs
        action = batch.action
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        reward = batch.reward.unsqueeze(1)
        next_obs = batch.next_obs
        not_done = 1 - batch.full_done.unsqueeze(1)
        obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        obs = torch.cat([obs.obs.flatten(1), obs.advice], dim=1).to(self.device) / 3
        next_obs = self.obs_preprocessor(next_obs, self.teacher, show_instrs=True)
        next_obs = torch.cat([next_obs.obs.flatten(1), next_obs.advice], dim=1).to(self.device) / 3

        logger.logkv('train/batch_reward', utils.to_np(reward.mean()))

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}_actor.pt')
        torch.save(self.critic.state_dict(), f'{model_dir}_critic.pt')

    def load(self, model_dir):
        self.actor.load_state_dict(torch.load(f'{model_dir}_actor.pt'))
        self.critic.load_state_dict(torch.load(f'{model_dir}_critic.pt'))

    def reset(self, *args, **kwargs):
        pass

    def get_actions(self, obs, training=False):
        self.train(training=training)
        action, agent_dict = self.act(obs, sample=True)
        return utils.to_np(action[0]), agent_dict

    def log_distill(self, action_true, action_teacher, entropy, policy_loss, loss, dist, train):
        if self.args.discrete:
            action_pred = dist.probs.max(1, keepdim=False)[1]  # argmax action
            avg_mean_dist = -1
            avg_std = -1
        else:  # Continuous env
            action_pred = dist.mean
            avg_std = dist.scale.mean()
            avg_mean_dist = torch.abs(action_pred - action_true).mean()
        log = {
            'Loss': float(policy_loss),
            'Accuracy': float((action_pred == action_true).sum()) / len(action_pred),
        }
        train_str = 'Train' if train else 'Val'
        logger.logkv(f"Distill/Entropy_Loss_{train_str}", float(entropy * self.args.entropy_coef))
        logger.logkv(f"Distill/Entropy_{train_str}", float(entropy))
        logger.logkv(f"Distill/Loss_{train_str}", float(policy_loss))
        logger.logkv(f"Distill/TotalLoss_{train_str}", float(loss))
        logger.logkv(f"Distill/Accuracy_{train_str}", float((action_pred == action_true).sum()) / len(action_pred))
        logger.logkv(f"Distill/Mean_Dist_{train_str}", float(avg_mean_dist))
        logger.logkv(f"Distill/Std_{train_str}", float(avg_std))
        return log

    def preprocess_distill(self, batch, source):
        obss = batch.obs
        if source == 'teacher':
            action_true = batch.teacher_action
            if len(action_true.shape) == 2 and action_true.shape[1] == 1:
                action_true = action_true[:, 0]
        elif source == 'agent':
            action_true = batch.action
        elif source == 'agent_probs':
            if self.args.discrete == False:
                raise NotImplementedError('Action probs not implemented for continuous envs.')
            action_true = batch.argmax_action
        if not source == 'agent_probs':
            dtype = torch.long if self.args.discrete else torch.float32
            action_true = torch.tensor(action_true, device=self.device, dtype=dtype)
        if hasattr(batch, 'teacher_action'):
            action_teacher = batch.teacher_action
        else:
            action_teacher = torch.zeros_like(action_true)
        if len(action_teacher.shape) == 2 and action_teacher.shape[1] == 1:
            action_teacher = action_teacher[:, 0]
        return obss, action_true, action_teacher

    def distill(self, batch, is_training=False, source='agent'):
        ### SETUP ###
        self.train(is_training)

        # Obtain batch and preprocess
        obss, action_true, action_teacher = self.preprocess_distill(batch, source)
        # Dropout instructions with probability instr_dropout_prob,
        # unless we have no teachers present in which case we keep the instr.
        instr_dropout_prob = 0 if self.teacher == 'none' else self.args.instr_dropout_prob
        obss = self.obs_preprocessor(obss, self.teacher, show_instrs=np.random.uniform() > instr_dropout_prob)

        ### RUN MODEL, COMPUTE LOSS ###
        _, info = self.act(obss, sample=True)
        dist = info['dist']
        if len(action_true.shape) == 3:  # Has an extra dimension
            action_true = action_true.squeeze(1)
        policy_loss = -dist.log_prob(action_true).mean()
        # entropy = dist.entropy().mean()
        entropy = 0
        # TODO: consider re-adding recon loss
        loss = policy_loss# - self.args.entropy_coef * entropy

        ### LOGGING ###
        log = self.log_distill(action_true, action_teacher, entropy, policy_loss, loss, dist, is_training)

        ### UPDATE ###
        if is_training:
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        return log


# https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size(), one_hot=False):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        argmax = torch.argmax(noisy_logits, dim=-1, keepdim=True)
        if one_hot:
            argmax = F.one_hot(argmax[:, 0].long(), self.logits.shape[-1]).float()
        return argmax

    def rsample(self, sample_shape=torch.Size(), one_hot=False):
        return self.sample(sample_shape, one_hot)

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value[:, 0].long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)