from meta_mb.logger import logger
from meta_mb.optimizers.base import Optimizer
from meta_mb.utils import Serializable
import numpy as np
import tensorflow as tf


class FirstOrderOptimizer(Optimizer, Serializable):
    """
    Optimizer for first order methods (SGD, Adam)

    Args:
        tf_optimizer_cls (tf.train.optimizer): desired tensorflow optimzier for training
        tf_optimizer_args (dict or None): arguments for the optimizer
        learning_rate (float): learning rate
        max_epochs: number of maximum epochs for training
        tolerance (float): tolerance for early stopping. If the loss fucntion decreases less than the specified tolerance
        after an epoch, then the training stops.
        num_minibatches (int): number of mini-batches for performing the gradient step. The mini-batch size is
        batch size//num_minibatches.
        verbose (bool): Whether to log or not the optimization process

    """

    def __init__(
            self,
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            num_minibatches=1,
            verbose=False,
    ):

        Serializable.quick_init(self, locals())
        self._target = None
        if tf_optimizer_args is None:
            tf_optimizer_args = dict()
        tf_optimizer_args['learning_rate'] = learning_rate

        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._verbose = verbose
        self._num_minibatches = num_minibatches
        self._all_inputs = None
        self._train_op = None
        self._loss = None
        self._input_ph_dict = None

    def build_graph(self, loss, target, input_ph_dict, *args, **kwargs):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
        """
        assert isinstance(loss, tf.Tensor)
        assert hasattr(target, 'get_params')
        assert isinstance(input_ph_dict, dict)

        self._target = target
        self._input_ph_dict = input_ph_dict
        self._loss = loss
        self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params())

    def loss(self, input_val_dict):
        """
        Computes the value of the loss for given inputs

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float): value of the loss

        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss = sess.run(self._loss, feed_dict=feed_dict)
        return loss

    def optimize(self, input_val_dict):
        """
        Carries out the optimization step

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float) loss before optimization

        """

        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)

        # Todo: reimplement minibatches

        loss_before_opt = None
        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % epoch)

            loss, _ = sess.run([self._loss, self._train_op], feed_dict)

            if not loss_before_opt: loss_before_opt = loss

        return loss_before_opt



class RNNFirstOrderOptimizer(Optimizer):
    """
    Optimizer for first order methods (SGD, Adam)

    Args:
        tf_optimizer_cls (tf.train.optimizer): desired tensorflow optimzier for training
        tf_optimizer_args (dict or None): arguments for the optimizer
        learning_rate (float): learning rate
        max_epochs: number of maximum epochs for training
        tolerance (float): tolerance for early stopping. If the loss fucntion decreases less than the specified tolerance
        after an epoch, then the training stops.
        num_minibatches (int): number of mini-batches for performing the gradient step. The mini-batch size is
        batch size//num_minibatches.
        verbose (bool): Whether to log or not the optimization process

    """

    def __init__(
            self,
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            num_minibatches=1,
            backprop_steps=32,
            verbose=False,
            grad_clip_threshold=None,
    ):
        self._target = None
        if tf_optimizer_args is None:
            tf_optimizer_args = dict()
        tf_optimizer_args['learning_rate'] = learning_rate

        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._num_minibatches = num_minibatches  # Unused
        self._verbose = verbose
        self._all_inputs = None
        self._train_op = None
        self._loss = None
        self._next_hidden_var = None
        self._hidden_ph = None
        self._input_ph_dict = None
        self._backprop_steps = backprop_steps
        self._grad_clip_threshold = grad_clip_threshold

    def build_graph(self, loss, target, input_ph_dict, hidden_ph, next_hidden_var):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
        """
        assert isinstance(loss, tf.Tensor)
        assert hasattr(target, 'get_params')
        assert isinstance(input_ph_dict, dict)

        self._target = target
        self._input_ph_dict = input_ph_dict
        self._loss = loss
        self._hidden_ph = hidden_ph
        self._next_hidden_var = next_hidden_var
        params = list(target.get_params().values())
        self._gradients_var = tf.gradients(loss, params)
        self._gradients_ph = [tf.placeholder(shape=param.shape, dtype=tf.float32) for param in params]
        applied_gradients = zip(self._gradients_ph, params)
        self._train_op = self._tf_optimizer.apply_gradients(applied_gradients)

    def loss(self, input_val_dict):
        """
        Computes the value of the loss for given inputs

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float): value of the loss

        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        batch_size, seq_len, *_ = list(input_val_dict.values())[0].shape
        hidden_batch = self._target.get_zero_state(batch_size)
        feed_dict[self._hidden_ph] = hidden_batch
        loss = sess.run(self._loss, feed_dict=feed_dict)
        return loss

    def optimize(self, input_val_dict):
        """
        Carries out the optimization step

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float) loss before optimization

        """

        sess = tf.get_default_session()
        batch_size, seq_len, *_ = list(input_val_dict.values())[0].shape

        loss_before_opt = None
        full_losses = []
        for epoch in range(self._max_epochs):
            hidden_batch = self._target.get_zero_state(batch_size)
            if self._verbose:
                logger.log("Epoch %d" % epoch)
            # run train op
            loss = []
            all_grads = []

            for i in range(0, seq_len, self._backprop_steps):
                n_i = i + self._backprop_steps
                feed_dict = dict([(self._input_ph_dict[key], input_val_dict[key][:, i:n_i]) for key in
                                  self._input_ph_dict.keys()])
                feed_dict[self._hidden_ph] = hidden_batch

                try:
                    batch_loss, grads, hidden_batch = sess.run([self._loss, self._gradients_var, self._next_hidden_var],
                                                                feed_dict=feed_dict)
                except Exception as e:
                    import IPython
                    IPython.embed()
                loss.append(batch_loss)
                all_grads.append(grads)

            grads = [np.mean(grad, axis=0) for grad in zip(*all_grads)]
            grad_mag = [np.mean(np.sqrt(grad ** 2)) for grad in grads]

            if self._grad_clip_threshold is not None:
                clipped_grad = []
                for grad, grad_mag in zip(grads, grad_mag):
                    if grad_mag > self._grad_clip_threshold:
                        print("clipping grad", grad_mag)
                        grad = grad / grad_mag * self._grad_clip_threshold
                    clipped_grad.append(grad)
            else:
                clipped_grad = grads
            feed_dict = dict(zip(self._gradients_ph, clipped_grad))
            _ = sess.run(self._train_op, feed_dict=feed_dict)
            full_losses.append(np.mean(np.asarray(loss)))
            if not loss_before_opt: loss_before_opt = np.mean(loss)

            # if self._verbose:
            #     logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            #
            # if abs(last_loss - new_loss) < self._tolerance:
            #     break
            # last_loss = new_loss
        return loss_before_opt
