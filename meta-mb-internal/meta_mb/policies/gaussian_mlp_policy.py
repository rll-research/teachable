from meta_mb.utils.networks.mlp import create_mlp, forward_mlp
from meta_mb.policies.distributions.diagonal_gaussian import DiagonalGaussian
from meta_mb.policies.base import Policy
from meta_mb.utils import Serializable
from meta_mb.utils.utils import remove_scope_from_name
from meta_mb.logger import logger

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class GaussianMLPPolicy(Policy):
    """
    Gaussian multi-layer perceptron policy (diagonal covariance matrix)
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

    def __init__(self, *args, init_std=1., min_std=1e-6, **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Policy.__init__(self, *args, **kwargs)

        self.min_log_std = np.log(min_std)
        self.init_log_std = np.log(init_std)

        self.init_policy = None
        self.policy_params = None
        self.obs_var = None
        self.mean_var = None
        self.log_std_var = None
        self.action_var = None
        self._dist = None

        self.build_graph()

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # build the actual policy network
            self.obs_var, self.mean_var = create_mlp(name='mean_network',
                                                     output_dim=self.action_dim,
                                                     hidden_sizes=self.hidden_sizes,
                                                     hidden_nonlinearity=self.hidden_nonlinearity,
                                                     output_nonlinearity=self.output_nonlinearity,
                                                     input_dim=(None, self.obs_dim,),
                                                     )

            with tf.variable_scope("log_std_network", reuse=tf.AUTO_REUSE):
                log_std_var = tf.get_variable(name='log_std_var',
                                              shape=(1, self.action_dim,),
                                              dtype=tf.float32,
                                              initializer=tf.constant_initializer(self.init_log_std),
                                              trainable=self.learn_std,
                                              )

                self.log_std_var = tf.maximum(log_std_var, self.min_log_std, name='log_std')

            # symbolically define sampled action and distribution
            self.action_var = self.mean_var + tf.random_normal(shape=tf.shape(self.mean_var)) * tf.exp(log_std_var)
            self._dist = DiagonalGaussian(self.action_dim)

            # save the policy's trainable variables in dicts
            # current_scope = tf.get_default_graph().get_name_scope()
            current_scope = self.name
            trainable_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.policy_params = OrderedDict([(remove_scope_from_name(var.name, current_scope), var) for var in trainable_policy_vars])

            self.policy_params_ph = self._create_placeholders_for_vars(scope=self.name + "/mean_network")
            log_std_network_phs = self._create_placeholders_for_vars(scope=self.name + "/log_std_network")
            self.policy_params_ph.update(log_std_network_phs)
            self.policy_params_keys = self.policy_params_ph.keys()

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
        action, agent_infos = action[0], dict(mean=agent_infos[0]['mean'], log_std=agent_infos[0]['log_std'])
        return action, agent_infos

    def get_actions(self, observations):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """
        observations = np.array(observations)
        assert observations.ndim == 2 and observations.shape[1] == self.obs_dim

        sess = tf.get_default_session()
        actions, means, log_stds = sess.run([self.action_var, self.mean_var, self.log_std_var],
                                             feed_dict={self.obs_var: observations})

        agent_infos = [dict(mean=mean, log_std=log_stds[0]) for mean in means]
        return actions, agent_infos

    # def get_actions(self, observations):
    #     """
    #     Runs each set of observations through each task specific policy
    #
    #     Args:
    #         observations (ndarray) : array of observations - shape: (batch_size, obs_dim)
    #
    #     Returns:
    #         (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
    #     """
    #     meta_batch_size = len(observations)
    #     batch_size = observations[0].shape[0]
    #     assert all([obs.shape[0] == batch_size for obs in observations])
    #     observations = np.stack(observations)
    #     assert observations.ndim == 2 and observations.shape[1] == self.obs_dim
    #
    #     sess = tf.get_default_session()
    #     actions, means, log_stds = sess.run([self.action_var, self.mean_var, self.log_std_var],
    #                                          feed_dict={self.obs_var: observations})
    #
    #     assert actions.shape == (observations.shape[0], self.action_dim)
    #     actions, means = np.split(actions, meta_batch_size, axis=0), np.split(means, meta_batch_size, axis=0)
    #     agent_infos = [[dict(mean=mean, log_std=log_stds[0]) for mean in means[idx]] for idx in
    #                    range(meta_batch_size)]
    #     return actions, agent_infos

    def log_diagnostics(self, paths, prefix=''):
        """
        Log extra information per iteration based on the collected paths
        """
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        means = np.vstack([path["agent_infos"]["mean"] for path in paths])
        logger.logkv(prefix+'AveragePolicyStd', np.mean(np.exp(log_stds)))
        logger.logkv(prefix + 'AverageAbsPolicyMean', np.mean(np.abs(means)))

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
        if params is None:
            with tf.variable_scope(self.name):
                obs_var, mean_var = create_mlp(name='mean_network',
                                               output_dim=self.action_dim,
                                               hidden_sizes=self.hidden_sizes,
                                               hidden_nonlinearity=self.hidden_nonlinearity,
                                               output_nonlinearity=self.output_nonlinearity,
                                               input_var=obs_var,
                                               reuse=True,
                                               )
                log_std_var = self.log_std_var
        else:
            mean_network_params = OrderedDict()
            log_std_network_params = []
            for name, param in params.items():
                if 'log_std_network' in name:
                    log_std_network_params.append(param)
                else:# if 'mean_network' in name:
                    mean_network_params[name] = param

            assert len(log_std_network_params) == 1
            obs_var, mean_var = forward_mlp(output_dim=self.action_dim,
                                            hidden_sizes=self.hidden_sizes,
                                            hidden_nonlinearity=self.hidden_nonlinearity,
                                            output_nonlinearity=self.output_nonlinearity,
                                            input_var=obs_var,
                                            mlp_params=mean_network_params,
                                            )

            log_std_var = log_std_network_params[0]

        return dict(mean=mean_var, log_std=log_std_var)

    def distribution_info_keys(self, obs, state_infos):
        """
        Args:
            obs (placeholder) : symbolic variable for observations
            state_infos (dict) : a dictionary of placeholders that contains information about the
            state of the policy at the time it received the observation

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        raise ["mean", "log_std"]

    """
    should not overwrite parent
    def __getstate__(self):
        # state = LayersPowered.__getstate__(self)
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['policy_params'] = self.get_param_values()
        return state

    def __setstate__(self, state):
        # LayersPowered.__setstate__(self, state)
        Serializable.__setstate__(self, state['init_args'])
        '''
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.Session()
        '''
        tf.get_default_session().run(tf.variables_initializer(self.get_params().values()))

        self.set_params(state['policy_params'])
    """

    def get_shared_param_values(self):
        state = dict()
        state['network_params'] = self.get_param_values()
        return state

    def set_shared_params(self, state):
        self.set_params(state['network_params'])

