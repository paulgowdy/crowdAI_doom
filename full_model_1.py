import itertools as it
from random import sample, randint, random, choice, shuffle
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
import glob
from vizdoom import *
import pickle
import matplotlib.pyplot as plt
#from doom_utils import *
from collections import deque


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


def dist_from_goal(current_pos, starting_pos, goal_pos):

    x_disp = starting_pos[0] - current_pos[0]
    y_disp = starting_pos[1] - current_pos[1]

    x_dist_from_goal = x_disp - goal_pos[0]
    y_dist_from_goal = y_disp - goal_pos[1]

    return np.sqrt((x_dist_from_goal ** 2) + (y_dist_from_goal ** 2)), x_disp, y_disp

def preprocess(img):

    img = np.rollaxis(img, 0, 3)
    img = img[:,:,0] #should be red channel
    img = img[80:380, 40:600]
    img = skimage.transform.resize(img, (60,108))
    img = img.astype(np.float32)

    img = 2.0 * img - 1.0

    return img

def create_coord_channels(img):
    # run this just once at beginning, then stack each training step

    rows, cols = img.shape

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

class ReplayMemory:
    def __init__(self, capacity, channels, resolution):
        #channels =
        state1_shape = (capacity, resolution[0], resolution[1], channels)
        #print(state1_shape)
        #state2_shape = (capacity, resolution[0], resolution[1], 1)
        self.s1 = np.zeros(state1_shape, dtype=np.float32)
        self.s2 = np.zeros(state1_shape, dtype=np.float32)
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
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

def create_network(session, available_actions_count, resolution, channels, learning_rate):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [channels], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=128, kernel_size=[8, 8], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=128, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv2_flat = tf.contrib.layers.flatten(conv2)


    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
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

    def function_simple_get_best_action(state,channels):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], channels]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action

def learn_from_memory(memory):
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

def perform_learning_step_stack(memory, epoch, initial_position, frame_queue, r, c, frame_grabs):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    channels = 5

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 0.4
        end_eps = 0.05
        const_eps_epochs = 0.05 * epochs  # 10% of learning time
        eps_decay_epochs = 0.5 * epochs  # 80% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    #if epoch % 3 == 0:

    #    print(s1.shape)

    frame_queue.append(s1)
    frame_queue.popleft()

    frames = [frame_queue[x] for x in frame_grabs]
    frames.append(r)
    frames.append(c)

    state_1 = np.dstack(frames)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)

    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(state_1, channels)

    # Extrinsic reward
    extrinsic_reward = game.make_action(actions[a], frame_repeat)

    # insert if statement if its close enough turn off coords
    # if
    # intrinsic position reward
    current_pos = [game.get_game_variable(GameVariable.POSITION_X),
                   game.get_game_variable(GameVariable.POSITION_Y)]



    dist, x, y = dist_from_goal(current_pos, initial_pos, goal_position)

    dist_reward = -1 * dist

    if dist < 200:

        reward = extrinsic_reward

    elif extrinsic_reward < -50000:

        #print('Death!')

        reward = dist_reward + extrinsic_reward

    else:

        reward = dist_reward

    isterminal = game.is_episode_finished()

    fin = 0

    if isterminal:

        #print('total reward:', game.get_total_reward())

        #if extrinsic_reward < -40000:

            #print('DEAD!')

        if game.get_total_reward() > -1 * episode_timeout:

            fin = 1

            reward = 10000

            print('found the exit!!')

    if not isterminal:

        s2 = preprocess(game.get_state().screen_buffer)

    else:

        s2 = np.zeros((60,108))

    frame_queue.append(s2)
    frame_queue.popleft()

    frames = [frame_queue[x] for x in frame_grabs]
    frames.append(r)
    frames.append(c)

    state_2 = np.dstack(frames)

    #print(state_1.shape)
    #print(s2.shape)

    # Remember the transition that was just experienced.
    memory.add_transition(state_1, a, state_2, isterminal, reward)

    print(reward)

    plt.figure()

    plt.subplot(141)
    plt.imshow(state_1[:,:,-1])

    plt.subplot(142)
    plt.imshow(state_1[:,:,-2])

    plt.subplot(143)
    plt.imshow(state_1[:,:,-3])

    plt.subplot(144)
    plt.imshow(state_1[:,:,0])

    plt.show()


    learn_from_memory(memory)
    return reward, x, y, fin, frame_queue

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path, wad_file_path, difficulty):

    print("Initializing doom...")
    game = vzd.DoomGame()

    game.load_config(config_file_path)
    game.set_doom_scenario_path(wad_file_path)

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    #game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.set_doom_skill(difficulty)


    game.init()
    print("Doom initialized.")

    return game


learning_rate = 0.001
discount_factor = 0.99

epochs = 50
episodes_per_epoch = 50

episode_timeout = 4000

replay_memory_size = 25000

batch_size = 64

frame_repeat = 12

resolution = (60,108)

frame_q_length = 10
frame_grabs = [-1,-3,-8]

channels = len(frame_grabs) + 2
# +2 for r and c!


goal_dictionary = {
'oblige_29_no_mon': (1040,1220)
}

training_wads = ['oblige_29_no_mon.wad']

wad_prefix = 'wad_files/'

save_model = True
load_model = True

DEFAULT_MODEL_SAVEFILE = "saved_models/model_1/model_1"
DEFAULT_MODEL_LOADFILE = "saved_models/model_1/model_1"

config = 'configs/standard_config.cfg'

init_wad = wad_prefix + training_wads[0]

game = initialize_vizdoom(config, init_wad, 1)


memory = ReplayMemory(replay_memory_size, channels, resolution)

session = tf.Session()
learn, get_q_values, get_best_action = create_network(session, len(actions), resolution, channels, learning_rate)
saver = tf.train.Saver()

if load_model:

    print("Loading model from: ", DEFAULT_MODEL_LOADFILE)
    saver.restore(session, DEFAULT_MODEL_LOADFILE)

else:

    init = tf.global_variables_initializer()
    session.run(init)


game.add_game_args("+freelook 1")
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_window_visible(True)
game.set_mode(vzd.Mode.SPECTATOR)

game.init()

game.new_episode()

init_state = game.get_state()
init_screen = init_state.screen_buffer
init_screen = preprocess(init_screen)

# Need to initialize queue every episode!
# To prevent the exit frames from going into the beginning of the next one



row_coords, col_coords = create_coord_channels(init_screen)

#frame_queue = deque([np.zeros(init_screen.shape)] * frame_q_length)


exit_percent_collector = []

plt.figure()

for epoch in range(epochs):

    print('Epoch:', str(epoch + 1), 'out of', epochs)


    train_episodes_finished = 0
    exit_counter = 0

    shuffle(training_wads)

    for current_wad in training_wads:

        wad_file = current_wad

        m_wad = wad_prefix + wad_file

        goal_position = goal_dictionary[wad_file[:-4]]

        print(m_wad, goal_position)

        game = initialize_vizdoom(config, m_wad, 1)

        initial_pos = [game.get_game_variable(GameVariable.POSITION_X),
                       game.get_game_variable(GameVariable.POSITION_Y)]


        game.new_episode()
        frame_queue = deque([np.zeros(init_screen.shape)] * frame_q_length)

        current_ep = 0

        while current_ep < episodes_per_epoch:

            while not game.is_episode_finished():

                r,x,y,fin,frame_queue = perform_learning_step_stack(memory, epoch, initial_pos, frame_queue, row_coords, col_coords, frame_grabs)

            if game.is_episode_finished():

                #print('episode complete!')

                game.new_episode()
                frame_queue = deque([np.zeros(init_screen.shape)] * frame_q_length)

                current_ep += 1

                exit_counter += fin

                train_episodes_finished += 1




        game.close()

    print('')
    exit_percent_collector.append(float(exit_counter) /train_episodes_finished)

    plt.clf()
    plt.plot(exit_percent_collector)
    plt.pause(0.1)


    print("Saving the network weigths to:", DEFAULT_MODEL_SAVEFILE)
    saver.save(session, DEFAULT_MODEL_SAVEFILE)

plt.show()





'''

#plt.figure()

while not game.is_episode_finished():

    state = game.get_state()
    time = game.get_episode_time()

    game.advance_action()
    last_action = game.get_last_action()
    reward = game.get_last_reward()

    screen = state.screen_buffer
    screen = preprocess(screen)

    z = np.dstack((screen, r, c))

    frame_queue.append(screen)
    frame_queue.popleft()

    frames = [frame_queue[x] for x in frame_grabs]
    frames.append(r)
    frames.append(c)
    print(len(frames))

    frames = np.dstack(frames)

    print(frames.shape)

    #plt.imshow(frames[:,:,-1] - frames[:,:,-2])
    #plt.pause(0.01)
'''
