from meta_mb.utils.networks.mlp import create_rnn
from meta_mb.policies.distributions.discrete import Discrete
from meta_mb.policies.base import Policy
from meta_mb.utils import Serializable
from meta_mb.utils.utils import remove_scope_from_name
from meta_mb.logger import logger

import tensorflow as tf
import tensorflow.nn as nn
import numpy as np
from collections import OrderedDict
import copy

# # Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
# class ExpertControllerFiLM(nn.Module):
#     def __init__(self, in_features, out_features, in_channels, imm_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(imm_channels)
#         self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
#         self.bn2 = nn.BatchNorm2d(out_features)
#
#         self.weight = nn.Linear(in_features, out_features)
#         self.bias = nn.Linear(in_features, out_features)
#
#
#     def forward(self, x, y):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.conv2(x)
#         out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
#         out = self.bn2(out)
#         out = F.relu(out)
#         return out

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

            instr_embedding = self._get_instr_embedding(obs.instr)
            x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
            x = x.reshape(x.shape[0], -1)

            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)

            embedding = torch.cat((embedding, instr_embedding), dim=1)
            x = self.actor(embedding)
            dist = Categorical(logits=F.log_softmax(x, dim=1))





            # memory_rnn = tf.nn.rnn_cell.LSTMCell(self.memory_dim)  # TODO: set these

            rnn_outs = create_rnn(name='probs_network',
                                  cell_type='lstm',
                                  output_dim=self.action_dim,
                                  hidden_sizes=self.hidden_sizes,
                                  hidden_nonlinearity=self.hidden_nonlinearity,
                                  output_nonlinearity=tf.nn.softmax,
                                  input_dim=(None, None, self.obs_dim,),
                                  )

            # obs_var, hidden_var, probs_var, next_hidden_var, cell = create_rnn(name='probs_network2',
            #                       cell_type='lstm',
            #                       output_dim=self.action_dim,
            #                       hidden_sizes=self.hidden_sizes,
            #                       hidden_nonlinearity=self.hidden_nonlinearity,
            #                       output_nonlinearity=tf.nn.softmax,
            #                       input_dim=(None, None, self.obs_dim,),
            #                       )

            from tensorflow.keras import datasets, layers, models
            import matplotlib.pyplot as plt

            # x = "INPUT"
            # input_conv = nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
            #
            # model = models.Sequential()
            # model.add(layers.Conv2D(128, (2,2), padding='SAME', input_shape=(8, 8, 3)))
            # model.add(layers.BatchNormalization())
            # model.add(layers.ReLU())
            # model.add(layers.MaxPooling2D((2, 2), strides=2))
            # model.add(layers.Conv2D(128, (3, 3), padding='SAME'))
            # model.add(layers.BatchNormalization())
            # model.add(layers.ReLU())
            # model.add(layers.MaxPooling2D((2, 2), strides=2))
            # model.compile('rmsprop', 'mse')

            film_pool = layers.MaxPooling2D((2, 2), strides=2)
            word_embedding = layers.Embedding(obs_space["instr"], self.instr_dim)  # TODO: get this!
            gru_dim = self.instr_dim  # TODO: set this
            gru_dim //= 2
            instr_rnn = tf.keras.layers.GRUCell(
                self.instr_dim, gru_dim, batch_first=True,
                bidirectional=True)
            self.final_instr_dim = self.instr_dim

            memory_rnn = tf.keras.layers.LSTMCell(self.image_dim, self.memory_dim)  # TODO: set these

            # Resize image embedding
            embedding_size = self.semi_memory_size  # TODO: set this
            # if self.use_instr and not "filmcnn" in arch:
            #     self.embedding_size += self.final_instr_dim  # TODO: consider keepint this!

            num_module = 2
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module - 1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim, out_features=self.image_dim,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)
            #
            # Define actor's model
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )







            # rnn_outs = create_rnn(name='probs_network',
            #                       cell_type=self._cell_type,
            #                       output_dim=self.action_dim,
            #                       hidden_sizes=self.hidden_sizes,
            #                       hidden_nonlinearity=self.hidden_nonlinearity,
            #                       output_nonlinearity=tf.nn.softmax,
            #                       input_dim=(None, None, self.obs_dim,),
            #                       )

            self.obs_var, self.hidden_var, self.probs_var, self.next_hidden_var, self.cell = rnn_outs
            self.probs_var = (self.probs_var + probs_var) / 2

            # symbolically define sampled action and distribution
            self._dist = Discrete(self.action_dim)

            # save the policy's trainable variables in dicts
            current_scope = tf.get_default_graph().get_name_scope()
            trainable_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.policy_params = OrderedDict(
                [(remove_scope_from_name(var.name, current_scope), var) for var in trainable_policy_vars])

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
        assert observations.shape[-1] == self.obs_dim, (observations.shape, self.obs_dim)
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

    def set_hidden_state(self, _hidden_state):
        self._hidden_state = _hidden_state

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
            elif isinstance(self._hidden_state, tuple) and isinstance(self._hidden_state[0],
                                                                      tf.contrib.rnn.LSTMStateTuple):
                for layer_state, new_layer_state in zip(self._hidden_state, _hidden_state):
                    layer_state.c[dones] = new_layer_state.c
                    layer_state.h[dones] = new_layer_state.h
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
        elif isinstance(_hidden_state, tuple) and isinstance(_hidden_state[0], tf.contrib.rnn.LSTMStateTuple):
            hidden_states = []
            for layer_state in _hidden_state:
                hidden_c = np.concatenate([layer_state.c] * batch_size)
                hidden_h = np.concatenate([layer_state.h] * batch_size)
                hidden = tf.contrib.rnn.LSTMStateTuple(hidden_c, hidden_h)
                hidden_states.append(hidden)
            hidden_states = tuple(hidden_states)
            return hidden_states
        else:
            return np.concatenate([_hidden_state] * batch_size)
