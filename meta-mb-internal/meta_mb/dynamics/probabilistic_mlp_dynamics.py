from meta_mb.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.dynamics.mlp_dynamics import MLPDynamicsModel
from meta_mb.dynamics.utils import denormalize


class ProbMLPDynamics(MLPDynamicsModel):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity='swish',
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 weight_normalization=False,  # Doesn't work
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 buffer_size=50000,
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = .0
        min_logvar = -10

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name = name
        self._dataset_train = None
        self._dataset_test = None

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]

        self.hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = self._activations[output_nonlinearity]
        self.hidden_sizes = hidden_sizes

        """ computation graph for training and simple inference """
        with tf.variable_scope(name):
            self.max_logvar = tf.Variable(np.ones([1, obs_space_dims]) * max_logvar, dtype=tf.float32,
                                          name="max_logvar")
            self.min_logvar = tf.Variable(np.ones([1, obs_space_dims]) * min_logvar, dtype=tf.float32,
                                          name="min_logvar")


            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            self._create_stats_vars()

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # create MLP
            delta_preds = []
            var_preds = []
            self.obs_next_pred = []
            with tf.variable_scope('dynamics_model'):
                mlp = MLP(name,
                          output_dim=2 * obs_space_dims,
                          hidden_sizes=self.hidden_sizes,
                          hidden_nonlinearity=self.hidden_nonlinearity,
                          output_nonlinearity=self.output_nonlinearity,
                          input_var=self.nn_input,
                          input_dim=obs_space_dims+action_space_dims,
                          )

            mean, logvar = tf.split(mlp.output_var, 2,  axis=-1)
            logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
            var = tf.exp(logvar)

            self.delta_pred = mean
            self.var_pred = var

            # define loss and train_op
            self.loss = tf.reduce_mean((self.delta_ph - self.delta_pred)**2/self.var_pred + tf.log(self.var_pred))
            self.loss += 0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar)
            self.optimizer = optimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_delta_pred = compile_function([self.obs_ph, self.act_ph], self.delta_pred)
            self.f_var_pred = compile_function([self.obs_ph, self.act_ph], self.var_pred)

        """ computation graph for inference where each of the models receives a different batch"""
        self._networks = [mlp]

    def predict(self, obs, act):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta = np.array(self.f_delta_pred(obs, act))
            var = np.array(self.f_var_pred(obs, act))
            delta = np.random.normal(delta, np.sqrt(var))
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta = np.array(self.f_delta_pred(obs, act))
            var = np.array(self.f_var_pred(obs, act))
            delta = np.random.normal(delta, np.sqrt(var))

        assert delta.shape == obs.shape

        pred_obs = obs_original + delta
        return pred_obs

    # FIXME: use predicy_sym instead
    def distribution_info_sym(self, obs_var, act_var):
        with tf.variable_scope(self.name + '/dynamics_model', reuse=True):
            in_obs_var = (obs_var - self._mean_obs_var) / (self._std_obs_var + 1e-8)
            in_act_var = (act_var - self._mean_act_var) / (self._std_act_var + 1e-8)
            input_var = tf.concat([in_obs_var, in_act_var], axis=1)
            mlp = MLP(self.name,
                      output_dim=2 * self.obs_space_dims,
                      hidden_sizes=self.hidden_sizes,
                      hidden_nonlinearity=self.hidden_nonlinearity,
                      output_nonlinearity=self.output_nonlinearity,
                      input_var=input_var,
                      input_dim=self.obs_space_dims + self.action_space_dims,
                      )
        mean, log_std = tf.split(mlp.output_var, 2, axis=-1)
        mean = mean * self._std_delta_var + self._mean_delta_var + obs_var
        log_std = log_std + tf.log(self._std_delta_var)
        return dict(mean=mean, log_std=log_std)


