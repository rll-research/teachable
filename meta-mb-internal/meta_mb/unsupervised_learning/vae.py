import random
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from meta_mb.logger import logger
import pickle
import joblib
from meta_mb.utils.serializable import Serializable
from meta_mb.utils.utils import remove_scope_from_name
from collections import OrderedDict


class VAE(Serializable):

    def __init__(self, latent_dim, img_size=(64, 64), channels=3, lr=1e-4, step=0, batch_size=32):
        """
        VAE Class

        Args:
            ds (int): dimension of the latent space
            img_size (tuple (int, int)): size of the image
            channels (int): number of channels [3 for rgb, 1 for grayscale]
            sess (tf.Session): tf.Session
            lr (float): learning rate
            out_dir (Path): output of the data directory
            step (int): initial training step
            batch_size (int): batch size
        """
        Serializable.quick_init(self, locals())

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.n_channels = channels
        self.do = img_size[0] * img_size[1] * channels
        self.batch_shape = [-1, img_size[0], img_size[1], channels]
        self.lr = lr
        self.batch_size = batch_size

        self._assign_ops = None
        self._assign_phs = None

        self.writer = tf.summary.FileWriter(logger.get_dir())
        with tf.variable_scope('vae', reuse=tf.AUTO_REUSE):

            self.initialize_placeholders()
            self.initialize_objective()
            self.global_step = step

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                self.z = tf.placeholder(tf.float32, [None, self.latent_dim])
                self.decoder = self.decode_sym(self.z).probs

            current_scope = tf.get_default_graph().get_name_scope()
            trainable_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.vae_params = OrderedDict(
                [(remove_scope_from_name(var.name, current_scope), var) for var in trainable_policy_vars])

    def decode_sym(self, z):
        net = tf.layers.dense(z, 256, activation=tf.nn.relu)
        net = tf.reshape(net, [-1, 1, 1, 256])
        net = tf.layers.conv2d_transpose(net, filters=64, kernel_size=4, strides=1, activation=tf.nn.relu, padding='valid')
        net = tf.layers.conv2d_transpose(net, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d_transpose(net, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d_transpose(net, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d_transpose(net, filters=self.n_channels, kernel_size=4, strides=2, padding='same')
        net = tf.contrib.layers.flatten(net)
        return tf.distributions.Bernoulli(logits=net)

    def encode_sym(self, inputs):
        net = tf.reshape(inputs, self.batch_shape)
        net = tf.layers.conv2d(net, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d(net, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d(net, filters=256, kernel_size=4, activation=tf.nn.relu, padding='valid')
        net = tf.layers.flatten(net)
        mean = tf.layers.dense(net, self.latent_dim)
        std = tf.exp(tf.layers.dense(net, self.latent_dim))
        return tf.contrib.distributions.MultivariateNormalDiag(mean, scale_diag=std)

    def initialize_placeholders(self):
        self.O = tf.placeholder(tf.float32, [None, self.do])
        self.beta = tf.placeholder(tf.float32, [])

    def initialize_objective(self):
        self.q_S = self.encode_sym(self.O)
        self.q_S_mean = self.q_S.mean()
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.q_O = self.decode_sym(self.q_S.sample())
        p_S = tf.contrib.distributions.MultivariateNormalDiag(
            loc=tf.zeros([self.batch_size, self.latent_dim]),
            scale_diag=tf.ones([self.batch_size, self.latent_dim])
        )
        self.kl = tf.distributions.kl_divergence(self.q_S, p_S)
        self.log_likelihood = tf.reduce_sum(self.q_O.log_prob(self.O), axis=1)
        self.elbo = tf.reduce_mean(self.log_likelihood - self.kl)
        self.beta_elbo = tf.reduce_mean(self.log_likelihood - self.beta * self.kl)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.beta_elbo)

        tf.summary.scalar('KL', tf.reduce_mean(self.kl))
        tf.summary.scalar('Log Likelihood', tf.reduce_mean(self.log_likelihood))
        tf.summary.scalar('ELBO', self.elbo)
        tf.summary.scalar('Beta ELBO', self.beta_elbo)
        img, rec = tf.reshape(self.O, self.batch_shape), tf.reshape(self.q_O.probs, self.batch_shape)
        tf.summary.image('State', img[7:10])
        tf.summary.image('Reconstruction', rec[7:10])
        self.summary = tf.summary.merge_all()

    def process_data(self, data):
        return data.reshape([-1, self.do])

    def train_step(self, data, beta=1.):
        O = self.process_data(data)
        sess = tf.get_default_session()
        _, summary = sess.run([self.train_op, self.summary], {
            self.O: O,
            self.beta: beta,
        })
        self.writer.add_summary(summary, self.global_step)
        self.global_step += 1

    def encode(self, imgs):
        flat_imgs = np.reshape(imgs, (-1, self.do))
        sess = tf.get_default_session()
        z = sess.run(self.q_S_mean, feed_dict={self.O: flat_imgs})
        return z

    def decode(self, z):
        sess = tf.get_default_session()
        flat_imgs = sess.run(self.decoder, feed_dict={self.z: z})
        imgs = np.reshape(flat_imgs, self.batch_shape)
        return imgs

    def get_params(self):
        """
        Get the tf.Variables representing the trainable weights of the network (symbolic)

        Returns:
            (dict) : a dict of all trainable Variables
        """
        return self.vae_params

    def get_param_values(self):
        """
        Gets a list of all the current weights in the network (in original code it is flattened, why?)

        Returns:
            (list) : list of values for parameters
        """
        sess = tf.get_default_session()
        uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
        sess.run(tf.variables_initializer(uninit_vars))
        param_values = sess.run(self.vae_params)
        return param_values

    def set_params(self, vae_params):
        """
        Sets the parameters for the graph

        Args:
            policy_params (dict): of variable names and corresponding parameter values
        """
        assert all([k1 == k2 for k1, k2 in zip(self.get_params().keys(), vae_params.keys())]), \
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
        feed_dict = dict(zip(self._assign_phs, vae_params.values()))
        tf.get_default_session().run(self._assign_ops, feed_dict=feed_dict)

    def __getstate__(self):
        state = {
            'init_args': Serializable.__getstate__(self),
            'network_params': self.get_param_values()
        }
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        tf.get_default_session().run(tf.global_variables_initializer())
        self.set_params(state['network_params'])
