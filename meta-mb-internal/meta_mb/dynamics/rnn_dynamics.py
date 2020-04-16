from meta_mb.dynamics.layers import RNN
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.logger import logger
from collections import OrderedDict
from meta_mb.dynamics.utils import normalize, denormalize, train_test_split


class RNNDynamicsModel(Serializable):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(512,),
                 cell_type='lstm',
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 batch_size=500,
                 step_size=0.001,
                 weight_normalization=True,
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 backprop_steps=50,
                 ):

        Serializable.quick_init(self, locals())

        self.recurrent = True

        self.normalization = None
        self.normalize_input = normalize_input

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency
        self.backprop_steps = backprop_steps
        self._dataset_train = None
        self._dataset_test = None

        self.batch_size = batch_size
        self.step_size = step_size
        with tf.variable_scope(name):
            self.obs_space_dims = env.observation_space.shape[0]
            self.action_space_dims = env.action_space.shape[0]

            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, None, self.obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, None, self.action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, None, self.obs_space_dims))

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=-1)
            rnn = RNN(name,
                      output_dim=self.obs_space_dims,
                      hidden_sizes=hidden_sizes,
                      hidden_nonlinearity=hidden_nonlinearity,
                      output_nonlinearity=output_nonlinearity,
                      input_var=self.nn_input,
                      input_dim=self.obs_space_dims + self.action_space_dims,
                      weight_normalization=weight_normalization,
                      cell_type=cell_type,
                      )
            self.hidden_state_ph = rnn.state_var
            self.next_hidden_state_var = rnn.next_state_var
            self.cell = rnn.cell

            self.delta_pred = rnn.output_var

            # define loss and train_op
            self.loss = tf.reduce_mean((self.delta_ph - self.delta_pred)**2)
            self.optimizer = optimizer(self.step_size)
            params = list(rnn.get_params().values())
            gradients_ph = [tf.placeholder(shape=param.shape, dtype=tf.float32) for param in params]
            self._gradients_vars = tf.gradients(self.loss, params)
            self._gradients_phs = gradients_ph
            applied_gradients = zip(gradients_ph, params)
            self.train_op = optimizer(self.step_size).apply_gradients(applied_gradients)

            # tensor_utils
            self.f_delta_pred = compile_function([self.obs_ph, self.act_ph, self.hidden_state_ph],
                                                              [self.delta_pred, self.next_hidden_state_var])
        self._networks = [rnn]

    def fit(self, obs, act, obs_next, epochs=1000,
            compute_normalization=True, verbose=False, valid_split_ratio=None, rolling_average_persitency=None):
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
        assert obs.ndim == 3 and obs.shape[2] == self.obs_space_dims
        assert obs_next.ndim == 3 and obs_next.shape[2] == self.obs_space_dims
        assert act.ndim == 3 and act.shape[2] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        delta = obs_next - obs
        obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                             test_split_ratio=valid_split_ratio)

        if self._dataset_test is None:
            self._dataset_test = dict(obs=obs_test, act=act_test, delta=delta_test)
        else:
            self._dataset_test['obs'] = np.concatenate([self._dataset_test['obs'], obs_test])
            self._dataset_test['act'] = np.concatenate([self._dataset_test['act'], act_test])
            self._dataset_test['delta'] = np.concatenate([self._dataset_test['delta'], delta_test])
        # create data queue
        next_batch, iterator = self._data_input_fn(obs_train, act_train, delta_train, batch_size=self.batch_size)

        valid_loss_rolling_average = None

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(self._dataset_train['obs'], self._dataset_train['act'],
                                       self._dataset_train['delta'])

        if self.normalize_input:
            # normalize data
            obs_train, act_train, delta_train = self._normalize_data(self._dataset_train['obs'],
                                                                     self._dataset_train['act'],
                                                                     self._dataset_train['delta'])
            assert obs_train.ndim == act_train.ndim == delta_train.ndim == 3
        else:
            obs_train = self._dataset_train['obs']
            act_train = self._dataset_train['act']
            delta_train = self._dataset_train['delta']

        # Training loop
        for epoch in range(epochs):

            # initialize data queue
            sess.run(iterator.initializer,
                     feed_dict={self.obs_dataset_ph: obs_train,
                                self.act_dataset_ph: act_train,
                                self.delta_dataset_ph: delta_train})

            batch_losses = []
            while True:
                try:
                    obs_batch, act_batch, delta_batch = sess.run(next_batch)

                    hidden_batch = sess.run(self.cell.zero_state(obs_batch.shape[0], tf.float32))
                    seq_len = obs_batch.shape[1]

                    # run train op
                    all_grads = []
                    for i in range(0, seq_len, self.backprop_steps):
                        n_i = i + self.backprop_steps
                        batch_loss, grads, hidden_batch = sess.run([self.loss, self._gradients_vars, self.next_hidden_state_var],
                                                                feed_dict={self.obs_ph: obs_batch[:, i:n_i, :],
                                                                self.act_ph: act_batch[:, i:n_i, :],
                                                                self.delta_ph: delta_batch[:, i:n_i, :],
                                                                self.hidden_state_ph: hidden_batch})

                        batch_losses.append(batch_loss)
                        all_grads.append(grads)
                    grads = [np.mean(grad, axis=0) for grad in zip(*all_grads)]
                    feed_dict = dict(zip(self._gradients_phs, grads))
                    _ = sess.run(self.train_op, feed_dict=feed_dict)

                except tf.errors.OutOfRangeError:

                    hidden_test = self.get_initial_hidden(obs_test.shape[0])
                    # compute validation loss
                    valid_loss = sess.run(self.loss, feed_dict={self.obs_ph: obs_test,
                                                                self.act_ph: act_test,
                                                                self.delta_ph: delta_test,
                                                                self.hidden_state_ph: hidden_test})
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

    def predict(self, obs, act, hidden_state):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param hidden_state: hidden_state - numpy array of shape (n_samples, hidden_sizes)
        :return: pred_obs_next: predicted batch of next observations (n_samples, ndim_obs)
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        obs, act = np.expand_dims(obs, 1), np.expand_dims(act, 1)

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta, next_hidden_state = self.f_delta_pred(obs, act, hidden_state)
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta, next_hidden_state = self.f_delta_pred(obs, act, hidden_state)

        delta = delta[:, 0, :]

        pred_obs = obs_original + delta
        return pred_obs, next_hidden_state

    def get_initial_hidden(self, batch_size):
        sess = tf.get_default_session()
        return sess.run(self.cell.zero_state(batch_size, tf.float32))

    def compute_normalization(self, obs, act, obs_next):
        """
        Computes the mean and std of the data and saves it in a instance variable
        -> the computed values are used to normalize the data at train and test time
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after takeing action - numpy array of shape (n_samples, ndim_obs)
        """

        assert obs.shape[0] == obs_next.shape[0] == act.shape[0]
        assert obs.shape[1] == obs_next.shape[1] == act.shape[1]
        delta = obs_next - obs

        assert delta.ndim == 3 and delta.shape[2] == obs_next.shape[2] == obs.shape[2]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=(0, 1)), np.std(obs, axis=(0,1)))
        self.normalization['delta'] = (np.mean(delta, axis=(0, 1)), np.std(delta, axis=(0,1)))
        self.normalization['act'] = (np.mean(act, axis=(0, 1)), np.std(act, axis=(0,1)))

    def _data_input_fn(self, obs, act, delta, batch_size=500, buffer_size=100000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert obs.ndim == act.ndim == delta.ndim, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == act.shape[1] == delta.shape[1], "inputs must have same length along axis 1"
        assert obs.shape[2] == delta.shape[2], "obs and obs_next must have same length along axis 2 "

        self.obs_dataset_ph = tf.placeholder(tf.float32, (None, None, obs.shape[2]))
        self.act_dataset_ph = tf.placeholder(tf.float32, (None, None, act.shape[2]))
        self.delta_dataset_ph = tf.placeholder(tf.float32, (None, None, delta.shape[2]))

        dataset = tf.data.Dataset.from_tensor_slices((self.obs_dataset_ph, self.act_dataset_ph, self.delta_dataset_ph))
        if self._dataset_train is None:
            self._dataset_train = dataset
        else:
            dataset = dataset.concatenate(self._dataset_train)
        dataset = dataset.batch(batch_size)
        self._dataset_train = dataset.shuffle(buffer_size=buffer_size)
        iterator = self._dataset_train.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator

    def _normalize_data(self, obs, act, obs_next=None):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])
        actions_normalized = normalize(act, self.normalization['act'][0], self.normalization['act'][1])

        if obs_next is not None:
            delta = obs_next - obs
            deltas_normalized = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized

    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)

        sess.run(tf.variables_initializer(uninit_variables))

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['normalization'] = self.normalization
        state['networks'] = [nn.__getstate__() for nn in self._networks]
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.normalization = state['normalization']
        for i in range(len(self._networks)):
            self._networks[i].__setstate__(state['networks'][i])


