import os

import tensorflow as tf
import numpy as np


class ValueFunctionDQN:
    def __init__(self, scope="MyValueFunctionEstimator", state_dim=2, n_actions=3, train_batch_size=64,
                 learning_rate=1e-4, hidden_layers_size=None, decay_lr=False, huber_loss=False, summaries_path=None,
                 reset_default_graph=False, checkpoints_dir=None, apply_wis=False):
        # Input check
        if hidden_layers_size is None:
            hidden_layers_size = [128, 64]  # Default ANN architecture
        assert len(hidden_layers_size) >= 1, "At least one hidden layer must be specified."

        # Support variables
        self.scope = scope
        self.layers_size = [state_dim] + hidden_layers_size + [n_actions]  # Size of all layers (including in & out)
        self.weights = []
        self.biases = []
        self.weights_old = []
        self.biases_old = []
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.n_train_epochs = 0
        self.summaries_path = summaries_path
        self.train_writer = None
        self.checkpoints_dir = checkpoints_dir

        # Apply Weighted Importance Sampling. See "Weighted importance sampling for off-policy learning with linear
        # function approximation". In Advances in Neural Information Processing Systems, pp. 3014–3022, 2014
        #   https://pdfs.semanticscholar.org/f8ef/8d1c31ae97c8acdd2d758dd2c0fe4e4bd6d7.pdf
        self.apply_wis = apply_wis

        if reset_default_graph:
            tf.reset_default_graph()

        # Build Tensorflow graph
        with tf.variable_scope(self.scope):
            # Inputs, weights, biases and targets of the ANN
            self.x = tf.placeholder(tf.float32, shape=(None, state_dim), name="x")
            self.train_targets = tf.placeholder(tf.float32, shape=(None, n_actions), name="train_targets")

            for l in range(len(self.layers_size) - 1):
                self.weights.append(tf.get_variable(name="w" + str(l), shape=[self.layers_size[l],
                                                                              self.layers_size[l + 1]],
                                                    initializer=tf.contrib.layers.xavier_initializer()))
                self.biases.append(tf.get_variable(name="b" + str(l), shape=[self.layers_size[l + 1]],
                                                   initializer=tf.constant_initializer(0.0)))

                self.weights_old.append(tf.get_variable(name="w-" + str(l),
                                                        initializer=self.weights[l].initialized_value()))
                self.biases_old.append(tf.get_variable(name="b-" + str(l),
                                                       initializer=self.biases[l].initialized_value()))

            if summaries_path is not None:
                with tf.name_scope('params_summaries'):
                    for l in range(len(self.layers_size) - 1):
                        self.variable_summaries(self.weights[l], "w" + str(l), histogram=True)
                        self.variable_summaries(self.biases[l], "b" + str(l), histogram=True)

            # Interconnection of the various ANN nodes
            self.prediction = self.model(self.x)
            self.prediction_with_old_params = self.model(self.x, use_old_params=True)

            # Training calculations
            if huber_loss:
                self.loss = self.huber_loss(self.train_targets, self.prediction)
            else:
                self.E = tf.subtract(self.train_targets, self.prediction, name="Error")
                self.SE = tf.square(self.E, name="SquaredError")

                if self.apply_wis:
                    self.rho = tf.placeholder(tf.float32, shape=(train_batch_size, n_actions), name="wis_weights")
                    self.loss = tf.reduce_mean(tf.multiply(self.rho, self.SE), name="loss")
                else:
                    self.loss = tf.reduce_mean(self.SE, name="loss")

            self.global_step = tf.Variable(0, trainable=False)
            if decay_lr:
                self.learning_rate = tf.train.exponential_decay(1e-4, self.global_step, 3000 * 200, 1e-5 / 1e-4)
            self.opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.opt_op.minimize(self.loss, global_step=self.global_step)

            self.init_op = tf.global_variables_initializer()

            # Operations to update the target Q network
            self.update_ops = []
            for l in range(len(self.layers_size) - 1):
                self.update_ops.append(self.weights_old[l].assign(self.weights[l]))
                self.update_ops.append(self.biases_old[l].assign(self.biases[l]))

            if self.summaries_path is not None:
                self.variable_summaries(self.loss, "loss", scalar_only=True)
                self.variable_summaries(self.learning_rate, "learning_rate", scalar_only=True)

        if self.checkpoints_dir is not None:
            var_list = []
            for l in range(len(self.layers_size) - 1):
                var_list.append(self.weights[l])
                var_list.append(self.biases[l])
            self.saver = tf.train.Saver(var_list, pad_step_number=True)

        if self.summaries_path is not None:
            self.merged_summaries = tf.summary.merge_all()
            self.summaries_path += "_{}".format(self.scope)
            if not os.path.exists(self.summaries_path):
                os.makedirs(self.summaries_path)
            self.train_writer = tf.summary.FileWriter(self.summaries_path, graph=tf.get_default_graph())
        else:
            self.merged_summaries = None

        self.session = None

    def model(self, x, use_old_params=False):
        z = []
        hidden = [x]
        for l in range(len(self.layers_size)-2):
            if use_old_params:
                z.append(tf.matmul(hidden[l], self.weights_old[l]) + self.biases_old[l])
            else:
                z.append(tf.matmul(hidden[l], self.weights[l]) + self.biases[l])
            hidden.append(tf.nn.relu(z[l], name="hidden_" + str(l + 1)))
        if use_old_params:
            z.append(tf.matmul(hidden[-1], self.weights_old[-1]) + self.biases_old[-1])
        else:
            z.append(tf.matmul(hidden[-1], self.weights[-1]) + self.biases[-1])

        if not use_old_params:
            if self.summaries_path is not None:
                with tf.name_scope('layers_summaries'):
                    for l in range(len(self.layers_size) - 1):
                        self.variable_summaries(z[l], "z" + str(l))
                        self.variable_summaries(hidden[l], "hidden" + str(l))

        return z[-1]  # Output layer has Identity units.

    @staticmethod
    def huber_loss(targets, predictions):
        error = targets - predictions
        fn_choice_maker1 = (tf.to_int32(tf.sign(error + 1)) + 1) / 2
        fn_choice_maker2 = (tf.to_int32(tf.sign(-error + 1)) + 1) / 2
        choice_maker_sqr = tf.to_float(tf.multiply(fn_choice_maker1, fn_choice_maker2))
        sqr_contrib = tf.multiply(choice_maker_sqr, tf.square(error)*0.5)
        abs_contrib = tf.abs(error)-0.5 - tf.multiply(choice_maker_sqr, tf.abs(error)-0.5)
        loss = tf.reduce_mean(sqr_contrib + abs_contrib)
        return loss

    def init_tf_session(self):
        if self.session is None:
            self.session = tf.Session()
            self.session.run(self.init_op)  # Global Variables Initializer (init op)

    def predict(self, states, use_old_params=False):
        self.init_tf_session()  # Make sure the Tensorflow session exists

        feed_dict = {self.x: states}
        if use_old_params:
            q = self.session.run(self.prediction_with_old_params, feed_dict=feed_dict)
        else:
            q = self.session.run(self.prediction, feed_dict=feed_dict)

        return q

    def train(self, states, targets, w=None):
        self.init_tf_session()  # Make sure the Tensorflow session exists

        feed_dict = {self.x: states, self.train_targets: targets}
        if self.apply_wis:
            feed_dict[self.rho] = np.transpose(np.tile(w, (self.layers_size[-1], 1)))

        if self.summaries_path is not None and self.n_train_epochs % 2000 == 0:
            fetches = [self.loss, self.train_op, self.E, self.merged_summaries]
        else:
            fetches = [self.loss, self.train_op, self.E]

        values = self.session.run(fetches, feed_dict=feed_dict)

        if self.summaries_path is not None and self.n_train_epochs % 2000 == 0:
            self.train_writer.add_summary(values[3], global_step=self.n_train_epochs)

        if self.checkpoints_dir is not None and self.n_train_epochs % 40000 == 0:
            self.saver.save(self.session, self.checkpoints_dir, global_step=self.global_step)

        self.n_train_epochs += 1
        return values[0], values[2]

    @staticmethod
    def variable_summaries(var, name, histogram=False, scalar_only=False):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        if scalar_only:
            tf.summary.scalar(name, var)
        else:
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name+'_mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar(name+'_stddev', stddev)
            tf.summary.scalar(name+'_max', tf.reduce_max(var))
            tf.summary.scalar(name+'_min', tf.reduce_min(var))
            if histogram:
                tf.summary.histogram(name+'_histogram', var)

    def update_old_params(self):
        self.init_tf_session()  # Make sure the Tensorflow session exists
        self.session.run(self.update_ops)

    def close_summary_file(self):
        if self.summaries_path is not None:
            self.train_writer.close()
