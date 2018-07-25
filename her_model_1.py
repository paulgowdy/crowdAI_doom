import tensorflow as tf
import numpy as np

class HER_Model:

    def __init__(self, session, input_shape, available_actions_count):

        self.session = session

        channels = 3
        self.frame_shape = input_shape

        self.learning_rate = 0.0001

        self.s1_ = tf.placeholder(tf.float32, [None,input_shape[0], input_shape[1], channels * 2], name="State")
        self.a_ = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        self.conv1 = tf.contrib.layers.convolution2d(self.s1_, num_outputs=64, kernel_size=[6, 6], stride=[3, 3],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))

        self.conv2 = tf.contrib.layers.convolution2d(self.conv1, num_outputs=64, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))

        self.conv2_flat = tf.contrib.layers.flatten(self.conv2)

        self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, num_outputs=64, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        self.q = tf.contrib.layers.fully_connected(self.fc1, num_outputs=available_actions_count, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
        self.best_a = tf.argmax(self.q, 1)

        self.loss = tf.losses.mean_squared_error(self.q, self.target_q_)

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        # Update the parameters according to the computed gradient using RMSProp.
        self.train_step = self.optimizer.minimize(self.loss)



    def function_get_best_action(self, state):
        return self.session.run(self.best_a, feed_dict={self.s1_: state})

    def policy(self, state):

        return self.function_get_best_action(state.reshape([1, self.frame_shape[0], self.frame_shape[1], 6]))[0]

    def get_q_values(self, state):

        return self.session.run(self.q, feed_dict={self.s1_: state})

    def learn(self, s1, target_q):

        feed_dict = {self.s1_: s1, self.target_q_: target_q}
        l, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)

        return l
