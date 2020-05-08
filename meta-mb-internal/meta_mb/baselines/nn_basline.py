# from meta_mb.core import MLP
from meta_mb.dynamics.layers import MLP
from meta_mb.dynamics.utils import normalize, denormalize
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.logger import logger
from collections import OrderedDict


class NNValueFun(Serializable):
    """
    Class for MLP continous dynamics model
    """
    _activations = {
        None: tf.identity,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 # policy,
                 hidden_sizes=(500, 500),
                 hidden_nonlinearity="tanh",
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 weight_normalization=True,
                 normalize_input=False,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 buffer_size=100000,
                 ):

        Serializable.quick_init(self, locals())

        # self.policy = policy
        self.normalization = None
        self.normalize_input = normalize_input
        self.buffer_size = buffer_size
        self.name = name
        self.hidden_sizes = hidden_sizes
        self.next_batch = None

        self._dataset_train = None
        self._dataset_test = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency
        self.hidden_nonlinearity = hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = output_nonlinearity = self._activations[output_nonlinearity]

        with tf.variable_scope(name):
            self.batch_size = batch_size
            self.learning_rate = learning_rate

            # determine dimensionality of state and action space
            self.obs_space_dims = env.observation_space.shape[0]
            self.action_space_dims = env.action_space.shape[0]

            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_space_dims))
            self.ret_ph = tf.placeholder(tf.float32, shape=(None,))

            self._create_stats_vars()



            # concatenate action and observation --> NN input
            # create MLP
            with tf.variable_scope('value_function'):
                mlp = MLP(name,
                          output_dim=1,
                          hidden_sizes=hidden_sizes,
                          hidden_nonlinearity=hidden_nonlinearity,
                          output_nonlinearity=output_nonlinearity,
                          input_var=self.obs_ph,
                          input_dim=self.obs_space_dims,
                          weight_normalization=weight_normalization)

            self.value_pred = tf.reshape(mlp.output_var, shape=(-1,))

            # define loss and train_op
            self.loss = tf.reduce_mean((self.ret_ph - self.value_pred)**2)
            self.optimizer = optimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_value_pred = compile_function([self.obs_ph], self.value_pred)

        self._networks = [mlp]
        # LayersPowered.__init__(self, [mlp.output_layer])

    def fit(self, obs, ret,
            epochs=1000, compute_normalization=True,
            verbose=False, valid_split_ratio=None,
            rolling_average_persitency=None,
            log_tabular=False):
        """
        Fits the NN dynamics model
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after taking action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param compute_normalization: boolean indicating whether normalization shall be (re-)computed given the data
        :param valid_split_ratio: relative size of validation split (float between 0.0 and 1.0)
        :param verbose: logging verbosity
        """
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, ret)

        if self.normalize_input:
            # normalize data
            obs, ret = self._normalize_data(obs, ret)
            assert obs.ndim == 2 and ret.ndim == 1

        # split into valid and test set

        obs_train, ret_train, obs_test, ret_test = train_test_split(obs, ret, test_split_ratio=valid_split_ratio)

        if self._dataset_test is None:
            self._dataset_test = dict(obs=obs_test, ret=ret_test)
            self._dataset_train = dict(obs=obs_train, ret=ret_train)

        else:
            # n_test_new_samples = len(obs_test)
            # n_max_test = self.buffer_size - n_test_new_samples
            # n_train_new_samples = len(obs_train)
            # n_max_train = self.buffer_size - n_train_new_samples
            # self._dataset_test['obs'] = np.concatenate([self._dataset_test['obs'][-n_max_test:], obs_test])
            # self._dataset_test['ret'] = np.concatenate([self._dataset_test['ret'][-n_max_test:], ret_test])
            #
            # self._dataset_train['obs'] = np.concatenate([self._dataset_train['obs'][-n_max_train:], obs_train])
            # self._dataset_train['ret'] = np.concatenate([self._dataset_train['ret'][-n_max_train:], ret_train])
            # FIXME: Hack so it always has on-policy samples

            self._dataset_test = dict(obs=obs_test, ret=ret_test)
            self._dataset_train = dict(obs=obs_train, ret=ret_train)

        # Create data queue
        if self.next_batch is None:
            self.next_batch, self.iterator = self._data_input_fn(self._dataset_train['obs'],
                                                                 self._dataset_train['ret'],
                                                                 batch_size=self.batch_size,
                                                                 buffer_size=self.buffer_size)

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(self._dataset_train['obs'], self._dataset_train['ret'])

        if self.normalize_input:
            # Normalize data
            obs_train, ret_train = self._normalize_data(self._dataset_train['obs'], self._dataset_train['ret'])
            assert obs.ndim == 2 and ret.ndim == 1
        else:
            obs_train, ret_train = self._dataset_train['obs'], self._dataset_train['ret']

        valid_loss_rolling_average = None

        # Training loop
        for epoch in range(epochs):

            # initialize data queue
            sess.run(self.iterator.initializer,
                     feed_dict={self.obs_dataset_ph: obs_train,
                                self.ret_dataset_ph: ret_train})

            batch_losses = []
            while True:
                try:
                    obs_batch, rest_batch = sess.run(self.next_batch)

                    # run train op
                    batch_loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.obs_ph: obs_batch,
                                                                                    self.ret_ph: rest_batch})
                    batch_losses.append(batch_loss)

                except tf.errors.OutOfRangeError:

                    if self.normalize_input:
                        # normalize data
                        obs_test, ret_test = self._normalize_data(self._dataset_test['obs'], self._dataset_test['ret'])
                    else:
                        obs_test, ret_test = self._dataset_test['obs'], self._dataset_test['ret']

                        # compute validation loss
                    valid_loss = sess.run(self.loss, feed_dict={self.obs_ph: obs_test,
                                                                self.ret_ph: ret_test})

                    if valid_loss_rolling_average is None:
                        valid_loss_rolling_average = 1.5 * valid_loss # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev = 2.0 * valid_loss

                    valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average + (1.0-rolling_average_persitency)*valid_loss

                    if verbose:
                        logger.log("Training NNDynamicsModel - finished epoch %i -- train loss: %.4f  valid loss: %.4f  valid_loss_mov_avg: %.4f"%(epoch, float(np.mean(batch_losses)), valid_loss, valid_loss_rolling_average))
                    break

            if valid_loss_rolling_average_prev < valid_loss_rolling_average:
                logger.log('Stopping DynamicsEnsemble Training since valid_loss_rolling_average decreased')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average

    def predict(self, obs):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: pred_obs_next: predicted batch of next observations (n_samples, ndim_obs)
        """
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims

        if self.normalize_input:
            obs = self._normalize_data(obs)
            values = np.array(self.f_value_pred(obs))
            values = denormalize(values, self.normalization['ret'][0], self.normalization['ret'][1])
        else:
            values = np.array(self.f_value_pred(obs))
        return values

    def compute_normalization(self, obs, ret):
        """
        Computes the mean and std of the data and saves it in a instance variable
        -> the computed values are used to normalize the data at train and test time
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after takeing action - numpy array of shape (n_samples, ndim_obs)
        """
        assert obs.ndim == 2 and obs.shape[0] == ret.shape[0]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=0), np.std(obs, axis=0))
        self.normalization['ret'] = (np.mean(ret, axis=0), np.std(ret, axis=0))
        sess = tf.get_default_session()
        sess.run(self._assignations, feed_dict={self._mean_input_ph: self.normalization['obs'][0],
                                                self._std_input_ph: self.normalization['obs'][1],
                                                self._mean_output_ph: self.normalization['ret'][0],
                                                self._std_output_ph: self.normalization['ret'][1]})


    def _data_input_fn(self, obs, ret, batch_size=500, buffer_size=5000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert obs.ndim == 2, "inputs must have 2 dims"
        assert obs.shape[0] == ret.shape[0], "inputs must have same length along axis 0"

        self.obs_dataset_ph = tf.placeholder(tf.float32, (None, obs.shape[1]))
        self.ret_dataset_ph = tf.placeholder(tf.float32, (None,))

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.obs_dataset_ph, self.ret_dataset_ph)
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator

    def _normalize_data(self, obs, ret):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])
        ret_normalized = normalize(ret, self.normalization['ret'][0], self.normalization['ret'][1])

        return obs_normalized, ret_normalized

    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)

        sess.run(tf.variables_initializer(uninit_variables))

    def reinit_model(self):
        sess = tf.get_default_session()
        if '_reinit_model_op' not in dir(self):
            self._reinit_model_op = [tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=self.name + '/dynamics_model'))]
        sess.run(self._reinit_model_op)

    def distribution_info_sym(self, obs_var):
        with tf.variable_scope(self.name + '/value_function', reuse=True):
            input_var = (obs_var - self._mean_input_var)/(self._std_input_var + 1e-8)
            mlp = MLP(self.name,
                      output_dim=1,
                      hidden_sizes=self.hidden_sizes,
                      hidden_nonlinearity=self.hidden_nonlinearity,
                      output_nonlinearity=self.output_nonlinearity,
                      input_var=input_var,
                      input_dim=self.obs_space_dims)
            output_var = tf.reshape(mlp.output_var, shape=(-1,))
            output_var = output_var * self._std_output_var + self._mean_output_var
        return dict(mean=output_var)

    def _create_stats_vars(self):
        self._mean_input_var = tf.get_variable('mean_input', shape=(self.obs_space_dims,),
                                               dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
        self._mean_output_var = tf.get_variable('mean_ouput', shape=tuple(),
                                                dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
        self._std_input_var = tf.get_variable('std_input', shape=(self.obs_space_dims,),
                                              dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)
        self._std_output_var = tf.get_variable('std_output', shape=tuple(),
                                               dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)

        self._mean_input_ph = tf.placeholder(tf.float32, shape=(self.obs_space_dims,))
        self._std_input_ph = tf.placeholder(tf.float32, shape=(self.obs_space_dims,))
        self._mean_output_ph = tf.placeholder(tf.float32, shape=tuple())
        self._std_output_ph = tf.placeholder(tf.float32, shape=tuple())

        self._assignations = [tf.assign(self._mean_input_var, self._mean_input_ph),
                              tf.assign(self._std_input_var, self._std_input_ph),
                              tf.assign(self._mean_output_var, self._mean_output_ph),
                              tf.assign(self._std_output_var, self._std_output_ph)]


    def __getstate__(self):
        # state = LayersPowered.__getstate__(self)
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['normalization'] = self.normalization
        state['networks'] = [nn.__getstate__() for nn in self._networks]
        return state

    def __setstate__(self, state):
        # LayersPowered.__setstate__(self, state)
        Serializable.__setstate__(self, state['init_args'])
        self.normalization = state['normalization']
        for i in range(len(self._networks)):
            self._networks[i].__setstate__(state['networks'][i])


def train_test_split(obs, ret, test_split_ratio=0.2):
    assert obs.shape[0] == ret.shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    return obs[idx_train, :], ret[idx_train], obs[idx_test, :], ret[idx_test]


