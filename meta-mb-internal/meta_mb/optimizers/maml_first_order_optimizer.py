import tensorflow as tf
from meta_mb.optimizers.first_order_optimizer import FirstOrderOptimizer


class MAMLFirstOrderOptimizer(FirstOrderOptimizer):
    def __init__(self, *args, **kwargs):
        super(MAMLFirstOrderOptimizer, self).__init__(*args, **kwargs)


class MAMLPPOOptimizer(FirstOrderOptimizer):
    """
    Adds inner and outer kl terms to first order optimizer  #TODO: (Do we really need this?)

    """
    def __init__(self, *args, **kwargs):
        # Todo: reimplement minibatches
        super(MAMLPPOOptimizer, self).__init__(*args, **kwargs)
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
        super(MAMLPPOOptimizer, self).build_graph(loss, target, input_ph_dict)
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



