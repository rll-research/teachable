from meta_mb.utils.networks.mlp import create_rnn
from meta_mb.policies.distributions.discrete import Discrete
from meta_mb.policies.base import Policy
from meta_mb.utils import Serializable
from meta_mb.utils.utils import remove_scope_from_name
from meta_mb.logger import logger

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class DiscreteRNNPolicy(Policy):
    """
    Discrete multi-layer perceptron policy
    Provides functions for executing and updating policy parameters
    A container for storing the current pre and post update policies

    Args:
        obs_dim (int): dimensionality of the observation space -> specifies the input size of the policy
        action_dim (int): dimensionality of the action space -> specifies the output size of the policy
        name (str): name of the policy used as tf variable scope
        hidden_sizes (tuple): tuple of integers specifying the hidden layer sizes of the MLP
        hidden_nonlinearity (tf.op): nonlinearity function of the hidden layers
        output_nonlinearity (tf.op or None): nonlinearity function of the output layer
        learn_std (boolean): whether the standard_dev / variance is a trainable or fixed variable
        init_std (float): initial policy standard deviation
        min_std( float): minimal policy standard deviation

    """

    def __init__(self, *args, init_std=1., min_std=1e-6, cell_type='lstm', **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Policy.__init__(self, *args, **kwargs)

        self.init_policy = None
        self.policy_params = None
        self.obs_var = None
        self.probs_var = None
        self.action_var = None
        self._dist = None
        self._hidden_state = None
        self.recurrent = True
        self._cell_type = cell_type

        self.build_graph()
        self._zero_hidden = self.cell.zero_state(1, tf.float32)

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        with tf.variable_scope(self.name):
            # build the actual policy network
            rnn_outs = create_rnn(name='probs_network',
                                  cell_type=self._cell_type,
                                  output_dim=self.action_dim,
                                  hidden_sizes=self.hidden_sizes,
                                  hidden_nonlinearity=self.hidden_nonlinearity,
                                  output_nonlinearity=tf.nn.softmax,
                                  input_dim=(None, None, self.obs_dim,),
                                  )

            self.obs_var, self.hidden_var, self.probs_var, self.next_hidden_var, self.cell = rnn_outs

            # symbolically define sampled action and distribution
            self._dist = Discrete(self.action_dim)

            # save the policy's trainable variables in dicts
            current_scope = tf.get_default_graph().get_name_scope()
            trainable_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.policy_params = OrderedDict([(remove_scope_from_name(var.name, current_scope), var) for var in trainable_policy_vars])
            
    def get_action(self, observation):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observation = np.expand_dims(observation, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[0], dict(probs=agent_infos[0][0]['probs'])
        return action, agent_infos

    def weighted_sample_n(self, prob_matrix, items):
        s = prob_matrix.cumsum(axis=1)
        min_s = np.min(s[:, -1])
        r = np.random.rand(prob_matrix.shape[0])
        max_r = np.max(r)
        k = (s < r.reshape((-1, 1))).sum(axis=1)
        min_val = 0
        max_val = s.shape[1] - 1
        k = np.clip(k, min_val, max_val)
        return np.asarray(k)

    def get_actions(self, observations):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """
        observations = np.array(observations)
        assert observations.shape[-1] == self.obs_dim
        if observations.ndim == 2:
            observations = np.expand_dims(observations, 1)
        elif observations.ndim == 3:
            pass
        else:
            raise AssertionError

        sess = tf.get_default_session()
        probs, self._hidden_state = sess.run([self.probs_var, self.next_hidden_var],
                                                     feed_dict={self.obs_var: observations,
                                                                self.hidden_var: self._hidden_state})

        assert probs.ndim == 3 and probs.shape[-1] == self.action_dim
        
        action_idx = np.repeat(np.arange(self.action_dim)[None, :], probs.shape[0], axis=0)
        if probs.shape[1] != 1:
            actions = []
            for idx in range(probs.shape[1]):  
                actions_curr = self.weighted_sample_n(probs[:, idx, :], action_idx)[:, None, None]
                actions.append(actions_curr)
            actions = np.array(actions)[:, :, :, 0]
            actions = np.transpose(actions, [1, 0, 2])
            agent_infos = probs
        else:
            actions = self.weighted_sample_n(np.squeeze(probs, axis=1), action_idx)[:, None, None]
            probs = probs[:, 0, :]
            # assert actions.shape == (observations.shape[0], 1, 1)
            agent_infos = [[dict(probs=prob)] for prob in probs]
        return actions, agent_infos

    def log_diagnostics(self, paths, prefix=''):
        """
        Log extra information per iteration based on the collected paths
        """
        # TODO add entropy here
        pass

    def load_params(self, policy_params):
        """
        Args:
            policy_params (ndarray): array of policy parameters for each task
        """
        raise NotImplementedError

    @property
    def distribution(self):
        """
        Returns this policy's distribution

        Returns:
            (Distribution) : this policy's distribution
        """
        return self._dist

    def distribution_info_sym(self, obs_var, params=None):
        """
        Return the symbolic distribution information about the actions.

        Args:
            obs_var (placeholder) : symbolic variable for observations
            params (dict) : a dictionary of placeholders or vars with the parameters of the MLP

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        assert params is None
        with tf.variable_scope(self.name, reuse=True):
            rnn_outs = create_rnn(name="probs_network",
                                  output_dim=self.action_dim,
                                  hidden_sizes=self.hidden_sizes,
                                  hidden_nonlinearity=self.hidden_nonlinearity,
                                  output_nonlinearity=tf.nn.softmax,
                                  input_var=obs_var,
                                  cell_type=self._cell_type,
                                  )
            obs_var, hidden_var, probs_var, next_hidden_var, cell = rnn_outs

        return dict(probs=probs_var), hidden_var, next_hidden_var

    def distribution_info_keys(self, obs, state_infos):
        """
        Args:
            obs (placeholder) : symbolic variable for observations
            state_infos (dict) : a dictionary of placeholders that contains information about the
            state of the policy at the time it received the observation

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        raise ["probs"]

    def reset(self, dones=None):
        sess = tf.get_default_session()
        if dones is None:
            dones = [True]
        _hidden_state = sess.run(self._zero_hidden)
        if self._hidden_state is None:
            self._hidden_state = sess.run(self.cell.zero_state(len(dones), tf.float32))
        else:
            if isinstance(self._hidden_state, tf.contrib.rnn.LSTMStateTuple):
                self._hidden_state.c[dones] = _hidden_state.c
                self._hidden_state.h[dones] = _hidden_state.h
            else:
                self._hidden_state[dones] = _hidden_state

    def get_zero_state(self, batch_size):
        sess = tf.get_default_session()
        _hidden_state = sess.run(self._zero_hidden)
        if isinstance(_hidden_state, tf.contrib.rnn.LSTMStateTuple):
            hidden_c = np.concatenate([_hidden_state.c] * batch_size)
            hidden_h = np.concatenate([_hidden_state.h] * batch_size)
            hidden = tf.contrib.rnn.LSTMStateTuple(hidden_c, hidden_h)
            return hidden
        else:
            return np.concatenate([_hidden_state] * batch_size)
