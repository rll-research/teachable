# Code found here: https://github.com/denisyarats/pytorch_sac

import numpy as np
import torch
from torch import nn

from algos.utils import ImageEmbedding, InstrEmbedding
from logger import logger

from algos import utils
from utils.dictlist import merge_dictlists


class Agent(nn.Module):
    """SAC algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env, device='cuda', advice_size=0, advice_dim=128,
                 actor_update_frequency=1):
        super().__init__()
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
        self.actor_update_frequency = actor_update_frequency

        # Create encoders or dummy encoders for each piece of our input
        if args.image_obs:
            self.state_encoder = ImageEmbedding().to(self.device)
        else:
            self.state_encoder = None
        if not args.no_instr:
            self.task_encoder = InstrEmbedding(args, env).to(self.device)
        else:
            self.task_encoder = None
        if teacher == 'none':
            self.advice_embedding = None
        else:
            self.advice_embedding = nn.Sequential(
                nn.Linear(advice_size, advice_dim),
                nn.Sigmoid(),
            ).to(self.device)
        self.apply(utils.weight_init)

    def train(self, training=True):
        self.actor.train(training)
        self.critic.train(training)
        if self.state_encoder is not None:
            self.state_encoder.train(training)
        if self.task_encoder is not None:
            self.task_encoder.train(training)
        if self.advice_embedding is not None:
            self.advice_embedding.train(training)

    def act(self, obs, sample=False, instr_dropout_prob=0):
        obs = self.format_obs(obs, instr_dropout_prob=instr_dropout_prob)
        dist = self.actor(obs)
        argmax_action = dist.probs.argmax(dim=1) if self.args.discrete else dist.mean
        action = dist.sample() if sample else argmax_action
        agent_info = {'argmax_action': argmax_action, 'dist': dist}
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        return action, agent_info

    def log_rl(self, val_batch):
        self.train(False)
        with torch.no_grad():
            obs = self.format_obs(val_batch.obs)
            next_obs = self.format_obs(val_batch.next_obs)
            self.update_critic(obs, next_obs, val_batch, train=False)

    def format_obs(self, obs, instr_dropout_prob=0):
        obs = [self.obs_preprocessor([o], self.teacher, show_instrs=np.random.uniform() > instr_dropout_prob)
               for o in obs]
        obs = merge_dictlists(obs)
        if self.state_encoder is not None:
            obs = self.state_encoder(obs)
        if self.task_encoder is not None:
            obs = self.task_encoder(obs)
        if self.advice_embedding is not None:
            advice = self.advice_embedding(obs.advice)
            obs = torch.cat([obs.obs.flatten(1), advice], dim=1).to(self.device)
        else:
            obs = obs.obs.flatten(1)
        return obs

    def optimize_policy(self, batch, step):
        reward = batch.reward.unsqueeze(1)
        logger.logkv('train/batch_reward', utils.to_np(reward.mean()))

        obs = self.format_obs(batch.obs)
        next_obs = self.format_obs(batch.next_obs)
        self.update_critic(obs, next_obs, batch, step=step)

        if step % self.actor_update_frequency == 0:
            obs = self.format_obs(batch.obs)  # Recompute with updated params
            self.update_actor(obs, batch)

    def update_critic(self, obs, next_obs, batch, train=True, step=1):
        raise NotImplementedError('update_critic should be defined in child class')

    def update_actor(self, obs, batch):
        raise NotImplementedError('update_actor should be defined in child class')

    def save(self, model_dir, save_name=None):
        if save_name is None:
            save_name = f"{self.teacher}_model.pt"
        torch.save(self.state_dict(), f'{model_dir}/{save_name}')

    def load(self, model_dir):
        self.load_state_dict(torch.load(f'{model_dir}/{self.teacher}_model.pt'))

    def get_actions(self, obs, training=False, instr_dropout_prob=0):
        self.train(training=training)
        action, agent_dict = self.act(obs, sample=True, instr_dropout_prob=instr_dropout_prob)
        return utils.to_np(action[0]), agent_dict

    def log_distill(self, action_true, action_teacher, policy_loss, loss, dist, train):
        if self.args.discrete:
            action_pred = dist.probs.max(1, keepdim=False)[1]  # argmax action
            avg_mean_dist, avg_std = -1, -1
        else:  # Continuous env
            action_pred = dist.mean
            avg_std = dist.scale.mean()
            avg_mean_dist = torch.abs(action_pred - action_true).mean()
        log = {
            'Loss': float(policy_loss),
            'Accuracy': float((action_pred == action_true).sum()) / len(action_pred),
        }
        train_str = 'Train' if train else 'Val'
        logger.logkv(f"Distill/Loss_{train_str}", float(policy_loss))
        logger.logkv(f"Distill/Entropy_{train_str}", float(dist.entropy().mean()))
        logger.logkv(f"Distill/TotalLoss_{train_str}", float(loss))
        logger.logkv(f"Distill/Accuracy_{train_str}", float((action_pred == action_true).sum()) / len(action_pred))
        logger.logkv(f"Distill/Label_Accuracy_{train_str}", float((action_teacher == action_true).sum()) / len(action_pred))
        logger.logkv(f"Distill/Mean_Dist_{train_str}", float(avg_mean_dist))
        logger.logkv(f"Distill/Std_{train_str}", float(avg_std))
        return log

    def preprocess_distill(self, batch, source):
        obss = batch.obs
        if source == 'teacher':
            action_true = batch.teacher_action
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
        if self.args.discrete:  # remove extra dim, since it messes with log_prob function
            if len(action_true.shape) == 2:
                action_true = action_true[:, 0]
                action_teacher = action_teacher[:, 0]
        return obss, action_true, action_teacher

    def distill(self, batch, is_training=False, source='agent'):
        ### SETUP ###
        self.train(is_training)

        # Obtain batch and preprocess
        obss, action_true, action_teacher = self.preprocess_distill(batch, source)
        if self.args.discrete:
            assert len(action_true.shape) == 1
        else:
            assert len(action_true.shape) == 2
        assert action_true.shape == action_teacher.shape, (action_true.shape, action_teacher.shape)

        # Dropout instructions with probability instr_dropout_prob,
        # unless we have no teachers present in which case we keep the instr.
        instr_dropout_prob = 0 if self.teacher == 'none' else self.args.distill_dropout_prob
        ### RUN MODEL, COMPUTE LOSS ###
        _, info = self.act(obss, sample=True, instr_dropout_prob=instr_dropout_prob)
        dist = info['dist']
        policy_loss = -dist.log_prob(action_true).mean()
        entropy_loss = -dist.entropy().mean()
        # TODO: consider re-adding recon loss
        loss = policy_loss + self.args.distill_entropy_coef * entropy_loss

        ### LOGGING ###
        log = self.log_distill(action_true, action_teacher, policy_loss, loss, dist, is_training)

        ### UPDATE ###
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return log
