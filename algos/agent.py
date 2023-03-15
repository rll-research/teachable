# Code found here: https://github.com/denisyarats/pytorch_sac
import random

import numpy as np
import torch
from torch import nn
import os

from algos.utils import ImageEmbedding, InstrEmbedding
from logger import logger

from algos import utils
from utils import agent_saver
from utils.dictlist import merge_dictlists

from algos.utils import WaypointToDirectional as wtd


class Agent(nn.Module):

    def __init__(self, args, obs_preprocessor, teacher, env, device=None, advice_size=0, advice_dim=128,
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
        if device is None:
           device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.actor_update_frequency = actor_update_frequency

        self.use_input_converter = self.args.use_input_converter

        if self.use_input_converter:
            self.converter = wtd()
            if os.path.isfile('../trained_input_converter.pth'):
                self.converter.trunk.load_state_dict(torch.load('../trained_input_converter.pth'))

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
            self.advice_embedding = utils.mlp(advice_size, None, advice_dim, 0, output_mod=nn.Sigmoid()).to(self.device)
            if self.use_input_converter:
                self.advice_embedding.load_state_dict(torch.load("../advice_embedder.pth"))

        # TODO: these next 2 lines are in place since I want to run models collected before recon_coef was added.
        # If you're not doing this, feel free to remove.
        if not hasattr(args, 'recon_coef'):
            args.recon_coef = 0

        if args.recon_coef:
            obs = env.reset()
            obs_dim = 128 if args.image_obs else len(obs['obs'].flatten())
            act_dim = self.action_dim if args.discrete else 2 * self.action_dim
            out_dim = 2 * self.advice_size
            self.reconstructor = utils.mlp(obs_dim + act_dim, 64, out_dim, 1).to(self.device)
        else:
            self.reconstructor = None

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
        if self.reconstructor is not None:
            self.reconstructor.train(training)
        self.training = training

    def act(self, obs, sample=False, instr_dropout_prob=0):
        obs, addl_obs = self.format_obs(obs, instr_dropout_prob=instr_dropout_prob)
        dist = self.actor(obs)
        argmax_action = dist.probs.argmax(dim=1) if self.args.discrete else dist.mean
        action = dist.sample() if sample else argmax_action
        agent_info = {'argmax_action': argmax_action, 'dist': dist, 'addl_obs': addl_obs}
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        return action, agent_info

    def log_rl(self, val_batch):
        return
        self.train(False)
        with torch.no_grad():
            obs, _ = self.format_obs(val_batch.obs)
            next_obs, _ = self.format_obs(val_batch.next_obs)
            self.update_critic(obs, next_obs, val_batch, train=False)

    def format_obs(self, obs, instr_dropout_prob=0):
        # TODO: understand
        cutoff = int(instr_dropout_prob * len(obs))
        without_obs = [] if cutoff == 0 else [self.obs_preprocessor(obs[:cutoff], self.teacher, show_instrs=False)]
        with_obs = [] if cutoff == len(obs) else [self.obs_preprocessor(obs[cutoff:], self.teacher, show_instrs=True)]
        obs = without_obs + with_obs
        obs = merge_dictlists(obs)
        if self.state_encoder is not None:
            obs = self.state_encoder(obs)
        if self.task_encoder is not None:
            obs = self.task_encoder(obs)
        no_advice_obs = obs.obs.flatten(1).to(self.device)
        unprocessed_advice = obs.advice
        if self.advice_embedding is not None:
            # TODO: understand why we need 2 -> 128 mlp
            # TODO: where is the mlp learning
            advice = self.advice_embedding(obs.advice)
            obs_temp = torch.cat([obs.obs.flatten(1), advice], dim=1).to(self.device)
            if self.use_input_converter:
                advice = self.advice_embedding(self.converter.forward(obs_temp))
                obs = torch.cat([obs.obs.flatten(1), advice], dim=1).to(self.device)
            else:
                obs = obs_temp
        else:
            obs = no_advice_obs
        return obs, (unprocessed_advice, no_advice_obs)

    def optimize_policy(self, batch, step):
        import time
        t = time.time()
        reward = batch.reward.unsqueeze(1)
        logger.logkv('train/batch_reward', utils.to_np(reward.mean()))

        next_obs, _ = self.format_obs(batch.next_obs)
        logger.logkv('Time/B_Original_Format_Time', time.time() - t)
        t = time.time()
        critic_time = time.time() - t


        if step % self.actor_update_frequency == 0:
            t = time.time()
            obs, (advice, no_advice_obs) = self.format_obs(batch.obs)  # Recompute with updated params
            actor_time = time.time() - t
            logger.logkv('Time/B_Format_Time', actor_time)
            t = time.time()
            self.update_actor(obs, batch, advice=advice, no_advice_obs=no_advice_obs, next_obs=next_obs)
            actor_time = time.time() - t
            logger.logkv('Time/Actor_Time', actor_time)
        logger.logkv('Time/Critic_Time', critic_time)

    def update_critic(self, obs, next_obs, batch, actor_loss=0):
        raise NotImplementedError('update_actor should be defined in child class')

    def update_actor(self, obs, batch, advice=None, no_advice_obs=None, critic_loss=0):
        raise NotImplementedError('update_actor should be defined in child class')

    def save(self, model_dir, save_name=None):
        if save_name is None:
            save_name = f"{self.teacher}_model.pt"
        torch.save(self.state_dict(), f'{model_dir}/{save_name}')

    def load(self, model_dir):
        self.load_state_dict(torch.load(f'{model_dir}/{self.teacher}_model.pt', map_location=self.device))

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
        if self.args.discrete:
            tokens = torch.unique(action_true)
            for t in tokens.tolist():
                locations = action_true == t
                pred_token = action_pred[locations]
                true_token = action_true[locations]
                acc = float((pred_token == true_token).sum()) / len(true_token)
                logger.logkv(f"Distill/Accz_{t}_{train_str}", acc)

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

    def compute_recon_loss(self, dist, no_advice_obs, advice):
        if self.args.discrete:
            output = dist.probs
        else:
            output = torch.cat([dist.loc, dist.scale], dim=1)
        recon_embedding = torch.cat([no_advice_obs, output], dim=1)  # Output = (B, 4); obs = (B, 383)
        pred_advice = self.reconstructor(recon_embedding)  # 259 total (255 + 4 from action
        full_advice_size = int(len(pred_advice[0]) / 2)
        mean = pred_advice[:, :full_advice_size]
        var = torch.exp(pred_advice[:, full_advice_size:])
        recon_loss = torch.nn.GaussianNLLLoss()(advice, mean, var)  # TODO: consider just reconstructing embedding
        return recon_loss

    def distill(self, batch, is_training=False, source='agent'):
        import time
        ### SETUP ###
        self.train(is_training)

        # Obtain batch and preprocess
        t = time.time()
        obss, action_true, action_teacher = self.preprocess_distill(batch, source)
        logger.logkv(f"Time/Q_Preprocess", time.time() - t)
        t = time.time()
        if self.args.discrete:
            assert len(action_true.shape) == 1
        else:
            assert len(action_true.shape) == 2
        assert action_true.shape == action_teacher.shape, (action_true.shape, action_teacher.shape)

        # Dropout instructions with probability instr_dropout_prob,
        # unless we have no teachers present in which case we keep the instr.
        instr_dropout_prob = 0 if self.teacher == 'none' else self.args.distill_dropout_prob
        ### RUN MODEL, COMPUTE LOSS ###
        # TODO: forward pass. Probabilities for different actions
        _, info = self.act(obss, sample=True, instr_dropout_prob=instr_dropout_prob)
        logger.logkv(f"Time/Q_Act", time.time() - t)
        t = time.time()
        dist = info['dist']
        # action true -> the actual action that was taken
        policy_loss = -dist.log_prob(action_true).mean()
        entropy_loss = -dist.entropy().mean()
        (advice, no_advice_obs) = info['addl_obs']
        if self.args.recon_coef > 0:
            recon_loss = self.compute_recon_loss(dist, no_advice_obs, advice)
        else:
            recon_loss = torch.zeros_like(policy_loss)
        # Entropy loss is to favor
        loss = policy_loss + self.args.distill_entropy_coef * entropy_loss + self.args.recon_coef * recon_loss

        ### LOGGING ###
        log = self.log_distill(action_true, action_teacher, policy_loss, loss, dist, is_training)
        logger.logkv(f"Time/Q_Log", time.time() - t)
        t = time.time()

        ### UPDATE ###
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.logkv(f"Time/Q_Update", time.time() - t)
        t = time.time()
        return log
