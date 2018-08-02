from random import choice, sample, random, randint
import tensorflow as tf
import numpy as np
import skimage.color, skimage.transform
from collections import deque
from vizdoom import *

class ReplayMemory:
    def __init__(self, capacity, state_shape):

        self.s1 = np.zeros((capacity, state_shape[0], state_shape[1], state_shape[2]), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_shape[0], state_shape[1], state_shape[2]), dtype=np.float32)

        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        # This is where I would implement prioritzed replay
        # Although I would need another class variable for the importance
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

class DQN:

    def __init__(self, tf_session, action_count, state_shape, learning_rate, name_scope):

        self.session = tf_session
        self.state_shape = state_shape # Rows x Cols x Channels
        self.learning_rate = learning_rate
        self.name_scope = name_scope

        with tf.variable_scope(self.name_scope):

            self.s1_ = tf.placeholder(tf.float32, [None] + list(state_shape), name="State")
            self.a_ = tf.placeholder(tf.int32, [None], name="Action")
            self.target_q_ = tf.placeholder(tf.float32, [None, action_count], name="TargetQ")

            # Add 2 convolutional layers with ReLu activation
            self.conv1 = tf.contrib.layers.convolution2d(self.s1_, num_outputs=64, kernel_size=[7, 7], stride=[3, 3],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))

            self.conv2 = tf.contrib.layers.convolution2d(self.conv1, num_outputs=128, kernel_size=[3, 3], stride=[2, 2],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))

            self.conv2_flat = tf.contrib.layers.flatten(self.conv2)


            self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, num_outputs=64, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            '''
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=32, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            '''

            self.q = tf.contrib.layers.fully_connected(self.fc1, num_outputs=action_count, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.constant_initializer(0.1))
            self.best_a = tf.argmax(self.q, 1)

            self.loss = tf.losses.mean_squared_error(self.q, self.target_q_)

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

            self.train_step = self.optimizer.minimize(self.loss)

    def function_learn(self, s1, target_q):
        feed_dict = {self.s1_: s1, self.target_q_: target_q}
        l, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
        return l

    def get_loss(self, s1, target_q):
        feed_dict = {self.s1_: s1, self.target_q_: target_q}
        l = self.session.run([self.loss], feed_dict=feed_dict)
        return l

    def function_get_q_values(self, state):
        return self.session.run(self.q, feed_dict={self.s1_: state})

    def function_get_best_action(self, state):
        return self.session.run(self.best_a, feed_dict={self.s1_: state})

    def function_simple_get_best_action(self, state):
        return self.function_get_best_action(state.reshape([1, self.state_shape[0], self.state_shape[1], self.state_shape[2]]))[0]

class DoomAgent:

    def __init__(self, state_shape, replay_memory_size = 25000):

        # fixed epsilon, for now
        self.epsilon = 0.1

        self.actions = [

            #[0,0,0,0,0,0,0,0],

            #[1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],

            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,0,1],
            [0,0,0,0,1,0,1,0],
            [0,0,0,0,1,0,0,1],

            #[1,0,0,0,0,1,0,0],
            #[1,0,0,0,1,0,0,0],

            #[1,0,0,1,0,1,0,0],
            #[1,0,0,1,1,0,0,0],

            #[1,0,0,0,0,1,1,0],
            #[1,0,0,0,0,1,0,1],
            #[1,0,0,0,1,0,1,0],
            #[1,0,0,0,1,0,0,1],

            [0,1,0,0,0,1,0,0],
            [0,1,0,0,1,0,0,0],

            [0,1,0,0,0,1,1,0],
            [0,1,0,0,0,1,0,1],
            [0,1,0,0,1,0,1,0],
            [0,1,0,0,1,0,0,1],

            #[1,0,0,0,0,0,0,1],
            #[1,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,1],
            [0,1,0,0,0,0,1,0]
        ]

        self.session = tf.Session()

        self.state_shape = state_shape
        learning_rate = 0.001

        # Create DQNs

        save_model = False
        load_model = False

        DEFAULT_MODEL_SAVEFILE = ""
        DEFAULT_MODEL_LOADFILE = ""

        # DQN __init__(self, tf_session, action_count, state_shape, learning_rate)

        self.action_net_scope = 'action_net'
        self.target_net_scope = 'target_net'

        self.action_network = DQN(self.session, len(self.actions), self.state_shape, learning_rate, self.action_net_scope)
        self.target_network = DQN(self.session, len(self.actions), self.state_shape, learning_rate, self.target_net_scope)

        # Load model check here

        self.saver = tf.train.Saver()

        if load_model:

            print("Loading model from: ", DEFAULT_MODEL_LOADFILE)
            self.saver.restore(self.session, DEFAULT_MODEL_LOADFILE)

        else:

            init = tf.global_variables_initializer()
            self.session.run(init)

        # Create replay memory

        self.memory = ReplayMemory(replay_memory_size, self.state_shape)

        self.row_coords, self.col_coords = self.create_coord_channels()
        self.frame_q_length = 10
        self.frame_grabs = [-1,-2,-5,-10]

        self.discount_factor = 0.99

        self.train_batch_size = 64

    def preprocess(self, img):

        img = np.rollaxis(img, 0, 3)
        img = img[:,:,0] #should be red channel
        img = img[80:380, 40:600]
        img = skimage.transform.resize(img, (50, 50))
        img = img.astype(np.float32)

        img = 2.0 * img - 1.0

        return img

    def create_coord_channels(self):
        # run this just once at beginning, then stack each training step

        rows, cols = self.state_shape[0], self.state_shape[1]

        row_coords = np.arange(cols)
        row_coords = 2*row_coords/cols
        row_coords = row_coords - 1.0
        row_coords = np.tile(row_coords, [rows,1])

        col_coords = np.arange(rows)
        col_coords = 2*col_coords/rows
        col_coords = col_coords - 1.0
        col_coords = np.tile(col_coords, [cols,1])
        col_coords = np.transpose(col_coords)

        return row_coords, col_coords

    def dist_from_goal(self, current_pos, starting_pos, goal_pos):

        x_disp = starting_pos[0] - current_pos[0]
        y_disp = starting_pos[1] - current_pos[1]

        x_dist_from_goal = x_disp - goal_pos[0]
        y_dist_from_goal = y_disp - goal_pos[1]

        return np.sqrt((x_dist_from_goal ** 2) + (y_dist_from_goal ** 2)), x_disp, y_disp

    def compute_intrinsic_reward(self, game, extrinsic_reward, distance_from_exit, isterminal):

        dist_reward = -1 * distance_from_exit

        # Do i want to titrate this, rather than just a cut off...
        if distance_from_exit < 250:

            # don't hardcode timeout
            if isterminal and game.get_total_reward() > -4000:

                print('Found the Exit!!')

                return 5000000

            else:

                return extrinsic_reward

        # Don't hardcode death penalty
        elif extrinsic_reward < -50000:

            print('DEATH!')

            return dist_reward + extrinsic_reward

        else:

            return dist_reward

    def update_target_graph(self):

        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.action_net_scope)

        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_net_scope)

        op_holder = []

        # Update our target_network parameters with DQNNetwork parameters
        for from_var,to_var in zip(from_vars,to_vars):

            op_holder.append(to_var.assign(from_var))

        return op_holder

    def learn_from_memory(self):

        if self.memory.size > self.train_batch_size:

            s1, a, s2, isterminal, r = self.memory.get_sample(self.train_batch_size)

            # old way, just using the target network to get the next q frame, pick the max
            #q2 = np.max(self.target_network.function_get_q_values(s2), axis=1)
            # but now we're doing 'double'

            # pick the s2 action using current action network
            #a2 = self.action_network.function_simple_get_best_action(s2)
            a2 = self.action_network.function_get_best_action(s2)
            q2 = self.target_network.function_get_q_values(s2)
            q2 = q2[np.arange(q2.shape[0]), a2]

            target_q = self.target_network.function_get_q_values(s1)

            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r

            target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2

            self.action_network.function_learn(s1, target_q)

    def play_episode_train(self, environment):

        environment.game.init()
        print('Doom Initialized')

        environment.game.new_episode()

        initial_pos = [environment.game.get_game_variable(GameVariable.POSITION_X),
                       environment.game.get_game_variable(GameVariable.POSITION_Y)]

        # goal position for a given episode (really for a given wad)
        # must come from doom environment

        goal_position = environment.goal_position

        frame_queue = deque([np.zeros((self.state_shape[0], self.state_shape[1]))] * self.frame_q_length)

        while not environment.game.is_episode_finished():

            s1 = self.preprocess(environment.game.get_state().screen_buffer)

            frame_queue.append(s1)
            frame_queue.popleft()

            frames = [frame_queue[x] for x in self.frame_grabs]

            frames.append(self.row_coords)
            frames.append(self.col_coords)

            state_1 = np.dstack(frames)

            if random() <= self.epsilon:

                a = randint(0, len(self.actions) - 1)

            else:

                a = self.action_network.function_simple_get_best_action(state_1)

            extrinsic_reward = environment.game.make_action(self.actions[a])

            current_pos = [environment.game.get_game_variable(GameVariable.POSITION_X),
                           environment.game.get_game_variable(GameVariable.POSITION_Y)]

            dist, x, y = self.dist_from_goal(current_pos, initial_pos, goal_position)

            isterminal = environment.game.is_episode_finished()

            intrinsic_reward = self.compute_intrinsic_reward(environment.game, extrinsic_reward, dist, isterminal)

            if not isterminal:

                s2 = self.preprocess(environment.game.get_state().screen_buffer)

            else:

                s2 = np.zeros((self.state_shape[0], self.state_shape[1]))

            # This should be a function, i do it twice in this episode train loop
            frame_queue.append(s2)
            frame_queue.popleft()

            frames = [frame_queue[x] for x in self.frame_grabs]

            frames.append(self.row_coords)
            frames.append(self.col_coords)

            state_2 = np.dstack(frames)

            self.memory.add_transition(state_1, a, state_2, isterminal, intrinsic_reward)

            # now its time to learn from memory!
            self.learn_from_memory()

        # Update the weights of the target network after every episode!
        # Or do this periodically within an episode...

        self.session.run(self.update_target_graph())
        print('Model updated')
        environment.game.close()
        print('Closed env')
        self.saver.save(self.session, 'saved_models/doom_agent_3/doom_agent_3')
        print('saved model')

        return dist
