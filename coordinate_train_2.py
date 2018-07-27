import itertools as it
from random import sample, randint, random, choice
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
import glob
from vizdoom import *

import matplotlib.pyplot as plt

# Q-learning settings
learning_rate = 0.001
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 30
learning_steps_per_epoch = 600
replay_memory_size = 250000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 5

# Other parameters
frame_repeat = 12
resolution = (39, 52)
episodes_to_watch = 10

# coordinates of exit relative to the start
goal_position = (-47, -1167)


# TODO move to argparser
save_model = True
load_model = True
skip_learning = False

# Configuration file path
DEFAULT_MODEL_SAVEFILE = "saved_models/model_1"
DEFAULT_MODEL_LOADFILE = "saved_models/model_1"

def dist_from_goal(current_pos, starting_pos, goal_pos):

    x_disp = starting_pos[0] - current_pos[0]
    y_disp = starting_pos[1] - current_pos[1]

    x_dist_from_goal = x_disp - goal_pos[0]
    y_dist_from_goal = y_disp - goal_pos[1]

    return np.sqrt((x_dist_from_goal ** 2) + (y_dist_from_goal ** 2)), x_disp, y_disp

def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=64, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=64, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv2_flat = tf.contrib.layers.flatten(conv2)


    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=32, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=32, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))


    q = tf.contrib.layers.fully_connected(fc2, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def get_loss(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l = session.run([loss], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action

def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)

def perform_learning_step(epoch, initial_position):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 0.5
        end_eps = 0.05
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)
    #print(s1.shape)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)

    # Extrinsic reward
    reward = game.make_action(actions[a], frame_repeat)

    # insert if statement if its close enough turn off coords
    # if
    # intrinsic position reward
    current_pos = [game.get_game_variable(GameVariable.POSITION_X),
                   game.get_game_variable(GameVariable.POSITION_Y)]



    reward, x, y = dist_from_goal(current_pos, initial_pos, goal_position)

    reward = -1 * reward

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()

    return reward, x, y

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path, wad_file_path, difficulty):

    print("Initializing doom...")
    game = vzd.DoomGame()

    game.load_config(config_file_path)
    game.set_doom_scenario_path(wad_file_path)

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.set_doom_skill(difficulty)

    game.init()
    print("Doom initialized.")

    return game

config = 'configs/standard_config.cfg'

wad = 'wad_files/oblige_multi_1.wad'

# Create Doom instance
game = initialize_vizdoom(config, wad, 1)

# Action = which buttons are pressed
#n = game.get_available_buttons_size()
actions = [

[0,0,0,0,0,0,0,0],

[1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0],
#[0,0,1,0,0,0,0,0],
#[0,0,0,1,0,0,0,0],
[0,0,0,0,1,0,0,0],
[0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,1],

#[0,0,1,0,0,1,0,0],
#[0,0,1,0,1,0,0,0],
#[0,0,0,1,0,1,0,0],
#[0,0,0,1,1,0,0,0],

[0,0,0,0,0,1,1,0],
[0,0,0,0,0,1,0,1],
[0,0,0,0,1,0,1,0],
[0,0,0,0,1,0,0,1],


[1,0,0,0,0,1,0,0],
[1,0,0,0,1,0,0,0],
#[1,0,0,1,0,0,0,0],
#[1,0,1,0,0,0,0,0],

#[1,0,1,0,0,1,0,0],
#[1,0,1,0,1,0,0,0],
[1,0,0,1,0,1,0,0],
[1,0,0,1,1,0,0,0],

[1,0,0,0,0,1,1,0],
[1,0,0,0,0,1,0,1],
[1,0,0,0,1,0,1,0],
[1,0,0,0,1,0,0,1],


[0,1,0,0,0,1,0,0],
[0,1,0,0,1,0,0,0],
#[0,1,0,1,0,0,0,0],
#[0,1,1,0,0,0,0,0],

#[0,1,1,0,0,1,0,0],
#[0,1,1,0,1,0,0,0],
#[0,1,0,1,0,1,0,0],
#[0,1,0,1,1,0,0,0],

[0,1,0,0,0,1,1,0],
[0,1,0,0,0,1,0,1],
[0,1,0,0,1,0,1,0],
[0,1,0,0,1,0,0,1],

[1,0,0,0,0,0,0,1],
[1,0,0,0,0,0,1,0],
[0,1,0,0,0,0,0,1],
[0,1,0,0,0,0,1,0]
]

# Create replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)

session = tf.Session()
learn, get_q_values, get_best_action = create_network(session, len(actions))
saver = tf.train.Saver()

if load_model:
    print("Loading model from: ", DEFAULT_MODEL_LOADFILE)
    saver.restore(session, DEFAULT_MODEL_LOADFILE)
else:
    init = tf.global_variables_initializer()
    session.run(init)
print("Starting the training!")

time_start = time()

game = initialize_vizdoom(config, wad, 1)


#multi_wads = ['wad_files/door_training_1.wad', 'wad_files/door_training_2a.wad', 'wad_files/door_training_2b.wad', 'wad_files/door_training_2c.wad']
#multi_wads = glob.glob('wad_files/oblige/multi/*.wad')
#multi_wads = multi_wads[6:7]
#print('wad count:', len(multi_wads))

reward_collector = []
plt.figure()

if not skip_learning:

    for epoch in range(epochs):

        m_wad = wad#choice(multi_wads)
        #print(m_wad)

        game = initialize_vizdoom(config, m_wad, 1)

        initial_pos = [game.get_game_variable(GameVariable.POSITION_X),
                       game.get_game_variable(GameVariable.POSITION_Y)]

        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")

        game.new_episode()

        ep_scores = []

        maps = []
        ep_map_x = []
        ep_map_y = []

        for learning_step in trange(learning_steps_per_epoch, leave=False):

            r,x,y = perform_learning_step(epoch, initial_pos)

            ep_scores.append(r)

            ep_map_x.append(x)
            ep_map_y.append(y)


            if game.is_episode_finished():

                #score = game.get_total_reward()
                #train_scores.append(score)
                game.new_episode()
                train_scores.append(ep_scores)
                maps.append([ep_map_x, ep_map_y])
                ep_scores = []
                ep_map_x = []
                ep_map_y = []

                train_episodes_finished += 1

        print("%d training episodes played." % train_episodes_finished)

        #train_scores = np.array(train_scores)

        #print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
        #      "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        #reward_collector.append(train_scores.mean())
        plt.clf()

        plt.xlim([-1000, 1000])
        plt.ylim([-1000, 1000])

        for es in maps:
            plt.plot(es[0], es[1], '.r-')

        plt.show()

        '''
        print("\nTesting...")
        test_episode = []
        test_scores = []
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.new_episode()
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                best_action_index = get_best_action(state)
                game.make_action(actions[best_action_index], frame_repeat)


            r = game.get_total_reward()


            test_scores.append(r)
        test_scores = np.array(test_scores)
        print("Results: mean: %.1f±%.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
              "max: %.1f" % test_scores.max())

        '''
        print("Saving the network weigths to:", DEFAULT_MODEL_SAVEFILE)
        saver.save(session, DEFAULT_MODEL_SAVEFILE)

        print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

game.close()


print("======================================")
print("Training finished. It's time to watch!")

# Reinitialize the game with window visible
game.set_window_visible(True)
game.set_mode(vzd.Mode.ASYNC_PLAYER)
game.init()

for _ in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().screen_buffer)
        best_action_index = get_best_action(state)

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.set_action(actions[best_action_index])
        for _ in range(frame_repeat):
            game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)
