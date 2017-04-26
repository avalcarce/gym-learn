import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np


class DumbValueFunction:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.scope = "dumb"
        pass

    def update_old_params(self):
        pass

    def predict(self, states, use_old_params=False):
        return np.zeros(shape=(len(states), self.n_actions))

    def train(self, states, targets, w=None):
        errors = np.zeros(shape=(len(states), self.n_actions))
        loss = 0
        return loss, errors

    def update_summarizables(self, reward, epsilon):
        pass

    def save(self):
        pass


class ValueFunctionDQN:
    def __init__(self, scope="MyValueFunctionEstimator", state_dim=2, n_actions=3, train_batch_size=64,
                 learning_rate=1e-4, hidden_layers_size=None, decay_lr=False, learning_rate_end=None, decay_steps=1,
                 huber_loss=False, summaries_path=None, reset_default_graph=False, checkpoints_dir=None,
                 apply_wis=False, checkpoint_save_period_epochs=40000, restoration_checkpoint=None, n_embeddings=0,
                 epsilon0=0.0, summarize_internal_excitations=False):
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
        self.trained_w = []
        self.trained_b = []
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.n_train_epochs = 0
        self.summaries_path = summaries_path
        self.train_writer = None
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint_pathname = os.path.join(self.checkpoints_dir, self.scope)
        self.checkpoint_save_period_epochs = checkpoint_save_period_epochs
        self.restoration_checkpoint = restoration_checkpoint
        self.n_embeddings = n_embeddings
        self.summarize_internal_excitations = summarize_internal_excitations
        self.global_step = 0

        # Apply Weighted Importance Sampling. See "Weighted importance sampling for off-policy learning with linear
        # function approximation". In Advances in Neural Information Processing Systems, pp. 3014–3022, 2014
        #   https://pdfs.semanticscholar.org/f8ef/8d1c31ae97c8acdd2d758dd2c0fe4e4bd6d7.pdf
        self.apply_wis = apply_wis

        if reset_default_graph:
            tf.reset_default_graph()

        self.graph = tf.get_default_graph()

        # Build Tensorflow graph
        with tf.variable_scope(self.scope):
            # Inputs, weights, biases and targets of the ANN
            self.x = tf.placeholder(tf.float32, shape=(None, state_dim), name="x")
            self.train_targets = tf.placeholder(tf.float32, shape=(None, n_actions), name="train_targets")

            self.__define_summarizables(epsilon0)

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

                # Operations to read the trained variables
                self.trained_w.append(tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope=self.scope + "/w" + str(l)))
                self.trained_b.append(tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope=self.scope + "/b" + str(l)))

            if summaries_path is not None:
                with tf.name_scope('params_summaries'):
                    for l in range(len(self.layers_size) - 1):
                        self.__variable_summaries(self.weights[l], "w" + str(l), histogram=True, collections=['train'])
                        self.__variable_summaries(self.biases[l], "b" + str(l), histogram=True, collections=['train'])

            # Interconnection of the various ANN nodes
            self.prediction, last_hidden = self.__model(self.x)
            self.prediction_with_old_params, _ = self.__model(self.x, use_old_params=True)

            self.__define_loss(train_batch_size, n_actions, huber_loss)

            if decay_lr:
                lr0 = self.learning_rate
                self.learning_rate = tf.train.exponential_decay(lr0, self.global_step, decay_steps,
                                                                learning_rate_end / lr0)
                # self.learning_rate = tf.train.polynomial_decay(lr0, self.global_step, 300000, learning_rate_end)
            self.opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.opt_op.minimize(self.loss)

            self.__create_embedding_ops(last_hidden)

            self.init_op = tf.global_variables_initializer()

            # Operations to update the target Q network
            self.update_ops = []
            for l in range(len(self.layers_size) - 1):
                self.update_ops.append(self.weights_old[l].assign(self.weights[l]))
                self.update_ops.append(self.biases_old[l].assign(self.biases[l]))

            if self.summaries_path is not None:
                self.__variable_summaries(self.loss, "loss", scalar_only=True, collections=['train'])
                self.__variable_summaries(self.reward, "reward", scalar_only=True, collections=['progress'])
                for i in range(state_dim):
                    self.__variable_summaries(tf.squeeze(tf.slice(self.x, [0, i], [1, 1])),
                                              "observation_"+str(i), scalar_only=True, collections=['state'])
                self.__variable_summaries(self.epsilon, "epsilon", scalar_only=True, collections=['progress'])
                self.__variable_summaries(self.learning_rate, "learning_rate", scalar_only=True, collections=['progress'])

        self.__create_checkpoints_saver()

        if self.summaries_path is not None:
            self.train_summaries = tf.summary.merge_all(key='train')
            self.state_summaries = tf.summary.merge_all(key='state')
            self.progress_summaries = tf.summary.merge_all(key='progress')
            self.summaries_path += "_{}".format(self.scope)
            if not os.path.exists(self.summaries_path):
                os.makedirs(self.summaries_path)
            self.train_writer = tf.summary.FileWriter(self.summaries_path, graph=self.graph)
        else:
            self.train_summaries = None
            self.state_summaries = None
            self.progress_summaries = None

        self.session = None

    def __define_summarizables(self, epsilon0=0.0):
        # Properties to be saved as summaries for later Tensorboard visualization
        self.reward_ph = tf.placeholder(tf.float32, name="reward_placeholder")
        self.epsilon_ph = tf.placeholder(tf.float32, name="epsilon_placeholder")
        self.reward = tf.Variable(0.0, name="reward")  # Current reward
        self.epsilon = tf.Variable(epsilon0, name="epsilon")  # Current epsilon, as per Epsilon-greedy policy
        self.reward_update_op = self.reward.assign(self.reward_ph)
        self.epsilon_update_op = self.epsilon.assign(self.epsilon_ph)

    def __define_loss(self, train_batch_size, n_actions, huber_loss=False):
        if huber_loss:
            self.loss = self.__huber_loss(self.train_targets, self.prediction)
        else:
            self.E = tf.subtract(self.train_targets, self.prediction, name="Error")
            self.SE = tf.square(self.E, name="SquaredError")

            if self.apply_wis:
                self.rho = tf.placeholder(tf.float32, shape=(train_batch_size, n_actions), name="wis_weights")
                self.loss = tf.reduce_mean(tf.multiply(self.rho, self.SE), name="loss")
            else:
                self.loss = tf.reduce_mean(self.SE, name="loss")

    def __create_checkpoints_saver(self):
        if self.checkpoints_dir is not None or self.restoration_checkpoint is not None:
            var_list = []
            for l in range(len(self.layers_size) - 1):
                var_list.append(self.weights[l])
                var_list.append(self.biases[l])
            self.saver = tf.train.Saver(var_list, pad_step_number=True)

    def __create_embedding_ops(self, last_hidden):
        if self.n_embeddings > 0:  # Preallocate memory to save embeddings
            self.embedding_var = tf.Variable(tf.zeros([self.n_embeddings, self.layers_size[-2]]), name='representation')
            self.next_embedding = tf.Variable(tf.zeros([1], dtype=tf.int32), name="next_embedding_counter")
            self.save_embedding_op = tf.scatter_update(self.embedding_var, self.next_embedding, last_hidden)
            self.increment_next_embedding_op = self.next_embedding.assign_add(tf.constant([1]))
            self.embeddings_saver = tf.train.Saver([self.embedding_var])

    def __model(self, x, use_old_params=False):
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
            if self.summaries_path is not None and self.summarize_internal_excitations:
                with tf.name_scope('layers_summaries'):
                    for l in range(len(self.layers_size) - 1):
                        self.__variable_summaries(z[l], "z" + str(l), collections=["state"])
                        self.__variable_summaries(hidden[l], "hidden" + str(l), collections=["state"])

        return z[-1], tf.reshape(hidden[-1], [1, self.layers_size[-2]])  # Output layer has Identity units.

    @staticmethod
    def __huber_loss(targets, predictions):
        error = targets - predictions
        fn_choice_maker1 = (tf.to_int32(tf.sign(error + 1)) + 1) / 2
        fn_choice_maker2 = (tf.to_int32(tf.sign(-error + 1)) + 1) / 2
        choice_maker_sqr = tf.to_float(tf.multiply(fn_choice_maker1, fn_choice_maker2))
        sqr_contrib = tf.multiply(choice_maker_sqr, tf.square(error)*0.5)
        abs_contrib = tf.abs(error)-0.5 - tf.multiply(choice_maker_sqr, tf.abs(error)-0.5)
        loss = tf.reduce_mean(sqr_contrib + abs_contrib)
        return loss

    def __init_tf_session(self):
        if self.session is None:
            self.session = tf.Session(graph=self.graph)
            self.session.run(self.init_op)  # Global Variables Initializer (init op)

            if self.restoration_checkpoint is not None:
                self.saver.restore(self.session, self.restoration_checkpoint)
                print("Model restored from checkpoint at {}.".format(self.restoration_checkpoint))

    def predict(self, states, global_step=None, use_old_params=False, saveembedding=False, summaries_to_save=[]):
        self.__init_tf_session()  # Make sure the Tensorflow session exists
        if global_step is not None:
            self.global_step = global_step

        feed_dict = {self.x: states}
        if use_old_params:
            fetches = [self.prediction_with_old_params]
        else:
            fetches = [self.prediction]

        if saveembedding and self.n_embeddings > 0:
            fetches.append([self.save_embedding_op])

        if self.summaries_path is not None and summaries_to_save:
            if "progress" in summaries_to_save:
                fetches.append(self.progress_summaries)
            if "state" in summaries_to_save:
                fetches.append(self.state_summaries)

        q = self.session.run(fetches, feed_dict=feed_dict)

        if saveembedding and self.n_embeddings > 0:
            self.increment_next_embedding_op.eval(session=self.session)

        if self.summaries_path is not None and summaries_to_save:
            for k in range(len(summaries_to_save)):
                if summaries_to_save[k] in ["progress", "state"]:
                    self.train_writer.add_summary(q[-k-1], global_step=self.global_step)

        return q[0]

    def train(self, states, targets, w=None, summaries_to_save=[]):
        self.__init_tf_session()  # Make sure the Tensorflow session exists

        feed_dict = {self.x: states, self.train_targets: targets}
        if self.apply_wis:
            feed_dict[self.rho] = np.transpose(np.tile(w, (self.layers_size[-1], 1)))

        fetches = [self.loss, self.train_op, self.E]

        if self.summaries_path is not None and "train" in summaries_to_save:
            fetches.append(self.train_summaries)

        values = self.session.run(fetches, feed_dict=feed_dict)

        if self.summaries_path is not None and "train" in summaries_to_save:
            self.train_writer.add_summary(values[-1], global_step=self.global_step)

        if self.checkpoints_dir is not None:
            if self.n_train_epochs > 0 and self.n_train_epochs % self.checkpoint_save_period_epochs == 0:
                self.save()

        self.n_train_epochs += 1
        return values[0], values[2]

    def save(self):
        self.saver.save(self.session, self.checkpoint_pathname, global_step=self.global_step)

    def save_embeddings(self, log_dir=None, metadata_filename=None, sprite_path=None, embedding_thumbnail_w=0,
                        embedding_thumbnail_h=0):
        if self.n_embeddings > 0:
            if metadata_filename is not None:
                config = projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = self.embedding_var.name
                embedding.metadata_path = os.path.abspath(os.path.join(log_dir, metadata_filename))
                if sprite_path is not None:
                    embedding.sprite.image_path = os.path.abspath(sprite_path)
                    embedding.sprite.single_image_dim.extend([embedding_thumbnail_w, embedding_thumbnail_h])
                summary_writer = tf.summary.FileWriter(log_dir)
                projector.visualize_embeddings(summary_writer, config)

            self.embeddings_saver.save(self.session, os.path.join(self.summaries_path, "embeddings"))
            n_saved_embeddings = self.next_embedding.eval(self.session)
            print("{} embeddings have been saved.".format(n_saved_embeddings[0]))

    def read_learned_weights_and_biases(self):
        self.__init_tf_session()  # Make sure the Tensorflow session exists

        fetches = []
        for l in range(len(self.layers_size) - 1):
            fetches.append(self.trained_w[l])
            fetches.append(self.trained_b[l])
        return self.session.run(fetches)

    @staticmethod
    def __variable_summaries(var, name, histogram=False, scalar_only=False, collections=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        if scalar_only:
            tf.summary.scalar(name, var, collections=collections)
        else:
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name+'_mean', mean, collections=collections)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar(name+'_stddev', stddev, collections=collections)
            tf.summary.scalar(name+'_max', tf.reduce_max(var), collections=collections)
            tf.summary.scalar(name+'_min', tf.reduce_min(var), collections=collections)
            if histogram:
                tf.summary.histogram(name+'_histogram', var, collections=collections)

    def update_old_params(self):
        self.__init_tf_session()  # Make sure the Tensorflow session exists
        self.session.run(self.update_ops)

    def close_summary_file(self):
        if self.summaries_path is not None:
            self.train_writer.close()

    def update_summarizables(self, reward, epsilon):
        self.__init_tf_session()  # Make sure the Tensorflow session exists

        feed_dict = {self.reward_ph: reward, self.epsilon_ph: epsilon}
        fetches = [self.reward_update_op, self.epsilon_update_op]

        self.session.run(fetches, feed_dict=feed_dict)


class DuelingNetwork:
    def __init__(self):
        pass
        # TODO
