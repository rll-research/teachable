from meta_mb.optimizers.first_order_optimizer import RNNFirstOrderOptimizer
import tensorflow as tf


class RL2FirstOrderOptimizer(RNNFirstOrderOptimizer):
    def __init__(self, *args ,**kwargs):
        super(RL2FirstOrderOptimizer, self).__init__(*args, **kwargs)

    def compute_loss_variations(self, input_dict, entropy_loss, reward_loss, log_values=None):
        # Log different loss components
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_dict)
        batch_size, seq_len, *_ = list(input_dict.values())[0].shape
        hidden_batch = self._target.get_zero_state(batch_size)
        feed_dict[self._hidden_ph] = hidden_batch
        entropy_loss = sess.run(entropy_loss, feed_dict=feed_dict)
        reward_loss = sess.run(reward_loss, feed_dict=feed_dict)
        if log_values is not None:
            log_values = sess.run(log_values, feed_dict=feed_dict)
        return entropy_loss, reward_loss


class RL2PPOOptimizer(RNNFirstOrderOptimizer):
    """
    Adds inner and outer kl terms to first order optimizer  #TODO: (Do we really need this?)

    """
    def __init__(self, *args, **kwargs):
        # Todo: reimplement minibatches
        super(RL2PPOOptimizer, self).__init__(*args, **kwargs)
        self._inner_kl = None
        self._outer_kl = None

    def build_graph(self, loss, target, input_ph_dict, inner_kl=None, outer_kl=None):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf.Tensor) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
            inner_kl (list): list with the inner kl loss for each task
            outer_kl (list): list with the outer kl loss for each task
        """
        super(RL2PPOOptimizer, self).build_graph(loss, target, input_ph_dict)
        assert inner_kl is not None

        self._inner_kl = inner_kl
        self._outer_kl = outer_kl

    def compute_stats(self, input_val_dict):
        """
        Computes the value the loss, the outer KL and the inner KL-divergence between the current policy and the
        provided dist_info_data

        Args:
           inputs (list): inputs needed to compute the inner KL
           extra_inputs (list): additional inputs needed to compute the inner KL

        Returns:
           (float): value of the loss
           (ndarray): inner kls - numpy array of shape (num_inner_grad_steps,)
           (float): outer_kl
        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss, inner_kl, outer_kl = sess.run([self._loss, self._inner_kl, self._outer_kl], feed_dict=feed_dict)
        return loss, inner_kl, outer_kl