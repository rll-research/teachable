from meta_mb.utils.networks.mlp import create_mlp, forward_mlp
from meta_mb.utils import Serializable
from meta_mb.utils.utils import remove_scope_from_name
from meta_mb.logger import logger

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class ValueFunction(Serializable):
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

    def __init__(self,
                 obs_dim,
                 action_dim,
                 name='v_fun',
                 hidden_sizes=(256, 256),
                 hidden_nonlinearity=tf.tanh,
                 output_nonlinearity=None,
                 **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.name = name
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.vfun_params = None
        self.input_var = None
        self.qval_var = None
        self.log_std_var = None
        self.action_var = None
        self._assign_ops = None

        self.build_graph()

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # build the actual policy network
            self.input_var, self.output_var = create_mlp(name='v_network',
                                                     output_dim=1,
                                                     hidden_sizes=self.hidden_sizes,
                                                     hidden_nonlinearity=self.hidden_nonlinearity,
                                                     output_nonlinearity=self.output_nonlinearity,
                                                     input_dim=(None, self.obs_dim + self.action_dim,),
                                                     )

            # save the policy's trainable variables in dicts
            # current_scope = tf.get_default_graph().get_name_scope()
            current_scope = self.name
            trainable_vfun_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.vfun_params = OrderedDict([(remove_scope_from_name(var.name, current_scope), var) for var in trainable_vfun_vars])

            self.vfun_params_ph = self._create_placeholders_for_vars(scope=self.name + "/v_network")
            self.vfun_params_keys = self.vfun_params_ph.keys()

    def log_diagnostics(self, paths, prefix=''):
        """
        Log extra information per iteration based on the collected paths
        """
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        means = np.vstack([path["agent_infos"]["mean"] for path in paths])
        logger.logkv(prefix+'AveragePolicyStd', np.mean(np.exp(log_stds)))
        logger.logkv(prefix + 'AverageAbsPolicyMean', np.mean(np.abs(means)))

    def value_sym(self, input_var, params=None):
        """
        Return the symbolic distribution information about the actions.

        Args:
            obs_var (placeholder) : symbolic variable for observations
            params (dict) : a dictionary of placeholders or vars with the parameters of the MLP

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        if params is None:
            with tf.variable_scope(self.name, reuse=True):
                input_var, output_var = create_mlp(name='v_network',
                                               output_dim=1,
                                               hidden_sizes=self.hidden_sizes,
                                               hidden_nonlinearity=self.hidden_nonlinearity,
                                               output_nonlinearity=self.output_nonlinearity,
                                               input_var=input_var,
                                               reuse=True,
                                               )
        else:
            input_var, output_var = forward_mlp(output_dim=1,
                                                hidden_sizes=self.hidden_sizes,
                                                hidden_nonlinearity=self.hidden_nonlinearity,
                                                output_nonlinearity=self.output_nonlinearity,
                                                input_var=input_var,
                                                mlp_params=params,
                                                )

        return output_var

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

    def _create_placeholders_for_vars(self, scope, graph_keys=tf.GraphKeys.TRAINABLE_VARIABLES):
        var_list = tf.get_collection(graph_keys, scope=scope)
        placeholders = []
        for var in var_list:
            var_name = remove_scope_from_name(var.name, scope.split('/')[0])
            placeholders.append((var_name, tf.placeholder(tf.float32, shape=var.shape, name="%s_ph" % var_name)))
        return OrderedDict(placeholders)

    def get_shared_param_values(self):
        state = dict()
        state['network_params'] = self.get_param_values()
        return state

    def set_shared_params(self, state):
        self.set_params(state['network_params'])

    def get_params(self):
        """
        Get the tf.Variables representing the trainable weights of the network (symbolic)

        Returns:
            (dict) : a dict of all trainable Variables
        """
        return self.vfun_params

    def get_param_values(self):
        """
        Gets a list of all the current weights in the network (in original code it is flattened, why?)

        Returns:
            (list) : list of values for parameters
        """
        param_values = tf.get_default_session().run(self.vfun_params)
        return param_values

    def set_params(self, vfun_params):
        """
        Sets the parameters for the graph

        Args:
            policy_params (dict): of variable names and corresponding parameter values
        """
        # from pdb import set_trace as st
        # print(self.get_params().keys(), vfun_params.keys())
        # st()
        assert all([k1 == k2 for k1, k2 in zip(self.get_params().keys(), vfun_params.keys())]), \
            "parameter keys must match with variable"

        if self._assign_ops is None:
            assign_ops, assign_phs = [], []
            for var in self.get_params().values():
                assign_placeholder = tf.placeholder(dtype=var.dtype)
                assign_op = tf.assign(var, assign_placeholder)
                assign_ops.append(assign_op)
                assign_phs.append(assign_placeholder)
            self._assign_ops = assign_ops
            self._assign_phs = assign_phs
        feed_dict = dict(zip(self._assign_phs, vfun_params.values()))
        tf.get_default_session().run(self._assign_ops, feed_dict=feed_dict)

    def __getstate__(self):
        state = {
            'init_args': Serializable.__getstate__(self),
            'network_params': self.get_param_values()
        }
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        # tf.get_default_session().run(tf.global_variables_initializer())
        self.set_params(state['network_params'])
