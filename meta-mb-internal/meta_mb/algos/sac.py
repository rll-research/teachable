from numbers import Number

import numpy as np
import tensorflow as tf
from collections import OrderedDict
from .base import Algo
"""===============added==========start============"""
from meta_mb.utils import create_feed_dict
from meta_mb.logger import logger

from tensorflow.python.training import training_util


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(Algo):
    """Soft Actor-Critic (SAC)
    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            policy,
            Qs,
            env,
            Q_targets=None,
            discount=0.99,
            name="sac",
            learning_rate=3e-4,
            target_entropy='auto',
            # evaluation_environment,
            # samplers,
            # pool,
            # plotter=None,
            reward_scale=1.0,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=True,
            buffer_size=100000,
            sampler_batch_size=64,
            # save_full_state=False,
            # n_epochs=1000,
            # train_every_n_steps=1,
            # n_train_repeat=1,
            # max_train_repeat_per_timestep=5,
            # n_initial_exploration_steps=0,
            # initial_exploration_policy=None,
            # epoch_length=1000,
            # eval_n_episodes=10,
            # eval_deterministic=True,
            # eval_render_kwargs=None,
            # video_save_frequency=0,
            session=None,
            **kwargs
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(policy)

        """===============added==========start============"""
        self.name = name
        self.policy = policy
        self.discount = discount
        # self.recurrent = getattr(self.policy, 'recurrent', False)
        self.recurrent = False
        self.training_environment = env
        self.target_entropy = (-np.prod(self.training_environment.action_space.shape)
                               if target_entropy == 'auto' else target_entropy)
        self.learning_rate = learning_rate
        self.policy_lr = learning_rate
        self.Q_lr = learning_rate
        self.tau = tau
        self._dataset = None
        self.buffer_size = buffer_size
        self.sampler_batch_size = sampler_batch_size
        # self.sampler = sampler
        # self._n_epochs = n_epochs
        # self._n_train_repeat = n_train_repeat
        # self._max_train_repeat_per_timestep = max(max_train_repeat_per_timestep, n_train_repeat)
        # self._train_every_n_steps = train_every_n_steps
        # self._epoch_length = epoch_length
        # self._n_initial_exploration_steps = n_initial_exploration_steps
        # self._initial_exploration_policy = initial_exploration_policy
        # self._eval_n_episodes = eval_n_episodes
        # self._eval_deterministic = eval_deterministic
        # self._video_save_frequency = video_save_frequency
        # self._eval_render_kwargs = eval_render_kwargs or {}
        self.session = session or tf.keras.backend.get_session()
        self._squash = True
        # self._epoch = 0
        # self._timestep = 0
        # self._num_train_steps = 0
        """===============added==========end============"""
        self.Qs = Qs
        if Q_targets is None:
            self.Q_targets = Qs
        else:
            self.Q_targets = Q_targets
        self.reward_scale = reward_scale
        self.target_update_interval = target_update_interval
        self.action_prior = action_prior
        self.reparameterize = reparameterize
        self._optimization_keys = ['observations', 'actions',
                                   'next_observations', 'dones', 'rewards']

        self.build_graph()

    def build_graph(self):
        self.training_ops = {}
        self._init_global_step()
        obs_ph, action_ph, next_obs_ph, terminal_ph, all_phs_dict = self._make_input_placeholders('',
                                                                                                  recurrent=False,
                                                                                                  next_obs=True)
        self.op_phs_dict = all_phs_dict
        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self.training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _make_input_placeholders(self, prefix='', recurrent=False, next_obs=False):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable
        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task,
            and for convenience, a list containing all placeholders created
        """
        dist_info_specs = self.policy.distribution.dist_info_specs
        all_phs_dict = OrderedDict()

        # observation ph
        obs_shape = [None, self.policy.obs_dim]
        obs_ph = tf.placeholder(tf.float32, shape=obs_shape, name=prefix + 'obs')
        all_phs_dict['%s%s' % (prefix, 'observations')] = obs_ph

        # action ph
        action_shape = [None, self.policy.action_dim]
        action_ph = tf.placeholder(dtype=tf.float32, shape=action_shape, name=prefix + 'action')
        all_phs_dict['%s%s' % (prefix, 'actions')] = action_ph

        """add the placeholder for terminal here"""
        terminal_shape = [None] if not recurrent else [None, None]
        terminal_ph = tf.placeholder(dtype=tf.bool, shape=terminal_shape, name=prefix + 'dones')
        all_phs_dict['%s%s' % (prefix, 'dones')] = terminal_ph

        rewards_shape = [None] if not recurrent else [None, None]
        rewards_ph = tf.placeholder(dtype=tf.float32, shape=rewards_shape, name=prefix + 'rewards')
        all_phs_dict['%s%s' % (prefix, 'rewards')] = rewards_ph

        if not next_obs:
            return obs_ph, action_ph, all_phs_dict

        else:
            obs_shape = [None, self.policy.obs_dim]
            next_obs_ph = tf.placeholder(dtype=np.float32, shape=obs_shape, name=prefix + 'obs')
            all_phs_dict['%s%s' % (prefix, 'next_observations')] = next_obs_ph

        return obs_ph, action_ph, next_obs_ph, terminal_ph, all_phs_dict

    def _get_q_target(self):
        next_observations_ph = self.op_phs_dict['next_observations']
        dist_info_sym = self.policy.distribution_info_sym(next_observations_ph)
        next_actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
        next_log_pis_var = self.policy.distribution.log_likelihood_sym(next_actions_var, dist_info_sym)
        next_log_pis_var = tf.expand_dims(next_log_pis_var, axis=-1)

        input_q_fun = tf.concat([next_observations_ph, next_actions_var], axis=-1)
        next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Q_targets]

        min_next_Q = tf.reduce_min(next_q_values, axis=0)
        next_values_var = min_next_Q - self.alpha * next_log_pis_var

        dones_ph = tf.cast(self.op_phs_dict['dones'], next_values_var.dtype)
        dones_ph = tf.expand_dims(dones_ph, axis=-1)
        rewards_ph = self.op_phs_dict['rewards']
        rewards_ph = tf.expand_dims(rewards_ph, axis=-1)
        self.q_target = td_target(
            reward=self.reward_scale * rewards_ph,
            discount=self.discount,
            next_value=(1 - dones_ph) * next_values_var)

        return tf.stop_gradient(self.q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.
        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        q_target = self._get_q_target()
        assert q_target.shape.as_list() == [None, 1]
        observations_ph = self.op_phs_dict['observations']
        actions_ph = self.op_phs_dict['actions']
        input_q_fun = tf.concat([observations_ph, actions_ph], axis=-1)

        q_values_var = self.q_values_var = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
        q_losses = self.q_losses = [tf.losses.mean_squared_error(labels=q_target, predictions=q_value, weights=0.5)
                                    for q_value in q_values_var]

        self.q_optimizers = [tf.train.AdamOptimizer(
                                                    learning_rate=self.Q_lr,
                                                    name='{}_{}_optimizer'.format(Q.name, i)
                                                    )
                             for i, Q in enumerate(self.Qs)]

        q_training_ops = [
            q_optimizer.minimize(loss=q_loss, var_list=list(Q.vfun_params.values()))
            for i, (Q, q_loss, q_optimizer)
            in enumerate(zip(self.Qs, q_losses, self.q_optimizers))]

        self.training_ops.update({'Q': tf.group(q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.
        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations_ph = self.op_phs_dict['observations']
        dist_info_sym = self.policy.distribution_info_sym(observations_ph)
        actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
        log_pis_var = self.policy.distribution.log_likelihood_sym(actions_var, dist_info_sym)
        log_pis_var = tf.expand_dims(log_pis_var, axis=1)

        assert log_pis_var.shape.as_list() == [None, 1]

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self.target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis_var + self.target_entropy))
            self.log_pis_var = log_pis_var

            self.alpha_optimizer = tf.train.AdamOptimizer(self.policy_lr, name='alpha_optimizer')
            self.alpha_train_op = self.alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])

            self.training_ops.update({
                'temperature_alpha': self.alpha_train_op
            })

        self.alpha = alpha

        if self.action_prior == 'normal':
            raise NotImplementedError
        elif self.action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        input_q_fun = tf.concat([observations_ph, actions_var], axis=-1)
        next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
        min_q_val_var = tf.reduce_min(next_q_values, axis=0)

        if self.reparameterize:
            policy_kl_losses = (self.alpha * log_pis_var - min_q_val_var - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self.policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self.policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.policy_lr,
            name="policy_optimizer")

        policy_train_op = self.policy_optimizer.minimize(
            loss=policy_loss,
            var_list=list(self.policy.policy_params.values()))

        self.training_ops.update({'policy_train_op': policy_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self.q_values_var),
            ('Q_loss', self.q_losses),
            ('policy_loss', self.policy_losses),
            ('alpha', self.alpha),
            ('Q_targets', self.q_target),
            ('scaled_rewards', self.reward_scale * self.op_phs_dict['rewards']),
            ('log_pis', self.log_pis_var)
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            # ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))
        self.diagnostics_ops = OrderedDict([
            ("%s-%s"%(key,metric_name), metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _update_target(self, tau=None):
        tau = tau or self.tau
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            source_params = Q.get_param_values()
            target_params = Q_target.get_param_values()
            Q_target.set_params(OrderedDict([
                (param_name, tau * source + (1.0 - tau) * target_params[param_name])
                for param_name, source in source_params.items()
            ]))

    def optimize_policy(self, buffer, timestep, grad_steps, log=True):
        sess = tf.get_default_session()
        for i in range(grad_steps):
            feed_dict = create_feed_dict(placeholder_dict=self.op_phs_dict,
                                         value_dict=buffer.random_batch(self.sampler_batch_size))
            sess.run(self.training_ops, feed_dict)
            if log:
                diagnostics = sess.run({**self.diagnostics_ops}, feed_dict)
                for k, v in diagnostics.items():
                    logger.logkv(k, v)
            if timestep % self.target_update_interval == 0:
                self._update_target()
