import tensorflow as tf
import numpy as np
from meta_mb.policies.distributions.base import Distribution

TINY = 1e-8

class Discrete(Distribution):
    """
    General methods for a diagonal gaussian distribution of this size
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    # Check correctness and shape
    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Computes the symbolic representation of the KL divergence of two multivariate
        Gaussian distribution with diagonal covariance matrices
        Args:
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor
        Returns:
            (tf.Tensor) : Symbolic representation of kl divergence (tensorflow op)
        """
        old_prob_var = old_dist_info_vars["probs"]
        new_prob_var = new_dist_info_vars["probs"]
        # Assume layout is N * T * A
        return tf.reduce_sum(
            old_prob_var * (tf.log(old_prob_var + TINY) - np.log(new_prob_var + TINY)),
            axis=2
        )

    # Check correctness and shape
    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
       Args:
            old_dist_info (dict): dict of old distribution parameters as numpy array
            new_dist_info (dict): dict of new distribution parameters as numpy array
        Returns:
            (numpy array): kl divergence of distributions
        """
        old_prob = old_dist_info["probs"]
        new_prob = new_dist_info["probs"]
        # Assume layout is N * T * A
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=2
        )

    # DONE
    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        """
        Symbolic likelihood ratio p_new(x)/p_old(x) of two distributions
        Args:
            x_var (tf.Tensor): variable where to evaluate the likelihood ratio p_new(x)/p_old(x)
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor
        Returns:
            (tf.Tensor): likelihood ratio
        """
        with tf.variable_scope("log_li_new"):
            logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        with tf.variable_scope("log_li_old"):
            logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old)

    # Check correctness and shape
    def log_likelihood_sym(self, x_var, dist_info_vars):
        """
        Symbolic log likelihood log p(x) of the distribution
        Args:
            x_var (tf.Tensor): variable where to evaluate the log likelihood
            dist_info_vars (dict) : dict of distribution parameters as tf.Tensor
        Returns:
             (numpy array): log likelihood
        """
        probs = dist_info_vars["probs"]
        # Assume layout is N * T * A
        a_dim = probs.shape[-1]
        prob_shape = tf.shape(probs)
        probs = tf.reshape(probs, (-1, a_dim))
        x_var = tf.reshape(x_var, (-1,))
        x_var = tf.one_hot(x_var, self.dim)
        flat_logli = tf.log(tf.reduce_sum(probs * tf.cast(x_var, 'float32'), axis=-1) + TINY)
        return tf.reshape(flat_logli, prob_shape[:2])

    # Check correctness and shape
    def log_likelihood(self, xs, dist_info):
        probs = dist_info["probs"]
        # Assume layout is N * T * A
        a_dim = probs.shape[-1]
        probs = probs.reshape((-1, a_dim))
        xs = xs.reshape((-1, 1))
        xs_oh = np.zeros((xs.shape[0], self.dim))
        xs_oh[xs.astype(np.int32)] = 1.0
        xs = xs_oh
        flat_logli = np.log(np.sum(probs * np.cast(xs, 'float32'), axis=-1) + TINY)
        return flat_logli.reshape(dist_info_vars["probs"].shape[:2])

    # TODO: Check size and numerical
    def entropy_sym(self, dist_info_vars):
        """
        Symbolic entropy of the distribution
        Args:
            dist_info (dict) : dict of distribution parameters as tf.Tensor
        Returns:
            (tf.Tensor): entropy
        """
        probs = dist_info_vars['probs']
        entropy = tf.reduce_sum(-probs * tf.log(probs), -1)
        return entropy

    # TODO: Check size and numerical
    def entropy(self, dist_info):
        """
        Compute the entropy of the distribution
        Args:
            dist_info (dict) : dict of distribution parameters as numpy array
        Returns:
          (numpy array): entropy
        """
        probs = dist_info['probs']
        entropy = np.reduce_sum(-probs * np.log(probs), -1)
        return entropy

    # TODO: Check size
    def sample(self, dist_info):
        """
        Draws a sample from the distribution
        Args:
           dist_info (dict) : dict of distribution parameter instantiations as numpy array
        Returns:
           (obj): sample drawn from the corresponding instantiation
        """
        probs = dist_info['probs']
        return np.random.multinomial(1, probs)

    # TODO: Check size
    def sample_sym(self, dist_info):
        """
        Draws a sample from the distribution
        Args:
           dist_info (dict) : dict of distribution parameter instantiations as numpy array
        Returns:
           (obj): sample drawn from the corresponding instantiation
        """
        probs = dist_info['probs']
        return tf.compat.v1.multinomial(
                    probs, 1, seed=None, name=None, output_dtype=None
                )

    @property
    def dist_info_specs(self):
        return [("probs", (self.dim,))]
