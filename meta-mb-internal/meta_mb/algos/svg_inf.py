from meta_mb.logger import logger
from meta_mb.algos.base import Algo
from meta_mb.optimizers.rl2_first_order_optimizer import RL2FirstOrderOptimizer
from meta_mb.optimizers.first_order_optimizer import FirstOrderOptimizer

import tensorflow as tf
from collections import OrderedDict


class SVGInf(Algo):
    """

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
            tf_reward,
            name="svg_inf",
            learning_rate=1e-3,
            max_epochs=5,
            **kwargs
            ):
        super(PPO, self).__init__(policy)
        self.dynamics_model = dynamics_model
        self.tf_reward = tf_reward

        self.recurrent = getattr(self.policy, 'recurrent', False)
        if self.recurrent:
            backprop_steps = kwargs.get('backprop_steps', 32)
            self.optimizer = RL2FirstOrderOptimizer(learning_rate=learning_rate, max_epochs=max_epochs,
                                                    backprop_steps=backprop_steps)
        else:
            self.optimizer = FirstOrderOptimizer(learning_rate=learning_rate, max_epochs=max_epochs)
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self._clip_eps = clip_eps

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

        # TODO: I need the full trajectory here! So I need to reshape or concat the data or sth so the distribution info_vars makes sense
        self.op_phs_dict.update(all_phs_dict)

        distribution_info_vars = self.policy.distribution_info_sym(obs_ph)
        hidden_ph, next_hidden_var = None, None

        for t in range(sel.horizon, 0, -1):
            v = self.tf_reward(obs_ph[t-1], action_ph[t], obs_ph[t])
            


        surr_obj = - tf.reduce_mean(clipped_obj)

        self.optimizer.build_graph(
            loss=surr_obj,
            target=self.policy,
            input_ph_dict=self.op_phs_dict,
            hidden_ph=hidden_ph,
            next_hidden_var=next_hidden_var
        )

    def optimize_policy(self, samples_data, log=True):
        """
        Performs MAML outer step

        Args:
            samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        input_dict = self._extract_input_dict(samples_data, self._optimization_keys, prefix='train')

        if log: logger.log("Optimizing")
        loss_before = self.optimizer.optimize(input_val_dict=input_dict)

        if log: logger.log("Computing statistics")
        loss_after = self.optimizer.loss(input_val_dict=input_dict)

        if log:
            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)