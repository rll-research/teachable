import numpy as np
from meta_mb.logger import logger
from meta_mb.algos.base import Algo
import tensorflow as tf
from collections import OrderedDict
from meta_mb.utils import create_feed_dict


class VGModel(Algo):
    """
    Algorithm for PPO MAML

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for the meta-objective
        exploration (bool): use exploration / pre-update sampling term / E-MAML term
        inner_type (str): inner optimization objective - either log_likelihood or likelihood_ratio
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(
            self,
            policy,
            dynamics_model,
            value_function,
            tf_reward,
            discount,
            name="svg_1",
            learning_rate=1e-3,
            buffer_size=100000,
            max_grad_norm=5.,
            batch_size=64,
            num_grad_steps=100,
            kl_penalty=0.01,
            **kwargs
            ):
        super(VGModel, self).__init__(policy)

        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.tf_reward = tf_reward
        self.discount = discount
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.num_grad_steps = num_grad_steps
        self.kl_penalty = kl_penalty

        self.recurrent = getattr(self.policy, 'recurrent', False)
        self._optimization_keys = ['observations', 'actions', 'next_observations', 'advantages', 'agent_infos']
        self.name = name
        self._train_op = None
        self._loss = None
        self._dataset = None

        self.build_graph()

    def build_graph(self):
        """
        Creates the computation graph

        Notes:
            Pseudocode:
            for task in meta_batch_size:
                make_vars
                init_init_dist_sym
            for step in num_inner_grad_steps:
                for task in meta_batch_size:
                    make_vars
                    update_init_dist_sym
            set objectives for optimizer
        """

        """ Create Variables """

        """ ----- Build graph for the meta-update ----- """
        self.op_phs_dict = OrderedDict()
        obs_ph, action_ph, next_obs_ph, adv_ph, dist_info_old_ph, all_phs_dict = \
            self._make_input_placeholders('train', recurrent=False, next_obs=True)

        self.op_phs_dict.update(all_phs_dict)

        distribution_info_vars = self.policy.distribution_info_sym(obs_ph)
        act_ev = distribution_info_vars['mean']

        model_distribution_info_vars = self.dynamics_model.distribution_info_sym(obs_ph, act_ev)
        obs_ev = model_distribution_info_vars['mean']

        value_fun_dist_info_vars = self.value_function.distribution_info_sym(obs_ev)
        value_ev = value_fun_dist_info_vars['mean']

        surr_obj = tf.reduce_mean(self.tf_reward(obs_ph, act_ev, obs_ev) + value_ev)

        # KL-penalty
        last_policy_info_vars = self.policy.distribution_info_sym(obs_ph, params=self.policy.policy_params_ph)
        surr_obj -= self.kl_penalty * self.policy.distribution.kl_sym(distribution_info_vars, last_policy_info_vars)

        self._train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-surr_obj,
                                                                             var_list=self.policy.get_params())
        self._loss = surr_obj

    def optimize_policy(self, samples_data, log=True):
        sess = tf.get_default_session()
        input_dict = self._extract_input_dict(samples_data, self._optimization_keys, prefix='train')
        if self._dataset is None:
            self._dataset = input_dict

        else:
            for k, v in input_dict.items():
                n_new_samples = len(v)
                n_max = self.buffer_size - n_new_samples
                self._dataset[k] = np.concatenate([self._dataset[k][-n_max:], v], axis=0)

        num_elements = len(list(self._dataset.values())[0])
        policy_params_dict = self.policy.policy_params_feed_dict
        for _ in range(self.num_grad_steps):
            idxs = np.random.randint(0, num_elements, size=self.batch_size)
            batch_dict = self._get_indices_from_dict(self._dataset, idxs)

            feed_dict = create_feed_dict(placeholder_dict=self.op_phs_dict, value_dict=batch_dict)
            feed_dict.update(policy_params_dict)

            _ = sess.run(self._train_op, feed_dict=feed_dict)

    def _get_indices_from_dict(self, d, idxs):
        _d = OrderedDict()
        for k, v in d.items():
            _d[k] = v[idxs]
        return _d
