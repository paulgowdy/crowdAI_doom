import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

from random import sample

from doom_utils import *
from her_model_1 import *
#from memory_utils import *

def get_sample(s1, a, s2, r, sample_size):

    i = sample(range(0, s1.shape[0]), sample_size)

    return s1[i], a[i], s2[i], r[i]

episodes = 2

# Going to assume always RGB
frame_shape = (30,40)

additional_goals = 3

discount_factor = 0.99

# From Doom Utils
action_count = len(actions)

config_file = 'config_files/standard_config.cfg'
wad_files = [glob.glob('wad_files/oblige/*.wad')[0]]

session = tf.Session()
model = HER_Model(session, frame_shape, action_count)
#Model - class with methods
#policy(takes state, gives action), learn_from_buffer(buffer, mini_batch_size)
saver = tf.train.Saver()

init = tf.global_variables_initializer()
session.run(init)


for M in range(episodes):

    #Initialize replay buffer
    #memory = HER_Replay_Buffer
    # Do I need memory buffer, can I just concatenate to the s,a,s arrays?
    # seems rough to just have them free floating
    # But I don't know how to size the buffer beforehand, its going to be randomly sized each time...
    # No memory_buffer for now

    print('Episode:', str(M+1))

    # Select wad file from list of random wads...
    #wad_file = 'wad_files/door_training_1.wad'
    wad_file = np.random.choice(wad_files)

    print(wad_file)

    game = initialize_vizdoom(config_file, wad_file, difficulty = 1)

    initial_goal = np.zeros((frame_shape[0], frame_shape[1], 3))

    # observe episode of the game - concatenate initial goal with the states...
    # go until game end: death, time out,

    s1_gs, actions, s2_gs, rewards = observe_episode(game, model, initial_goal, frame_shape) # doom utils


    # create goal for each frame where the goal is just that frames s2

    #print(s1_gs.shape)
    #print(actions.shape)
    #print(s2_gs.shape)
    #print(rewards.shape)

    # preallocate more space

    steps = s1_gs.shape[0]

    s1_gs = np.concatenate((s1_gs, np.zeros(s1_gs.shape)), axis = 0)
    s2_gs = np.concatenate((s2_gs, np.zeros(s2_gs.shape)), axis = 0)
    actions = np.concatenate((actions, np.zeros(actions.shape)), axis = 0)
    rewards = np.concatenate((rewards, np.zeros(rewards.shape)), axis = 0)

    for t in range(steps):

        g = s2_gs[t,:,:,:3]

        s1_new_g = s1_gs[t,:,:,:]
        s1_new_g[:,:,3:] = g

        s2_new_g = s2_gs[t,:,:,:]
        s2_new_g[:,:,3:] = g

        a = actions[t]
        r = 0

        # replace all this appending with prealocation
        #s1_gs = np.append(s1_gs, [s1_new_g], axis=0)
        #s2_gs = np.append(s2_gs, [s2_new_g], axis=0)
        #actions = np.append(actions, [a], axis=0)
        #rewards = np.append(rewards, [r], axis=0)
        s1_gs[steps + t] = s1_new_g
        s2_gs[steps + t] = s2_new_g
        actions[steps + t] = a
        rewards[steps + t] = r

    #print('')
    #print(s1_gs.shape)
    #print(actions.shape)
    #print(s2_gs.shape)
    #print(rewards.shape)

    #print(game.get_total_reward())
    #print(np.sum(rewards))
    print('')

    print('training...')

    minibatches = int(s1_gs.shape[0] / 16) * 3

    # This is where I would go sequentially in the case of LSTM model
    for mb in range(minibatches):

        s1, a, s2, r = get_sample(s1_gs, actions, s2_gs, rewards, 16)

        a = a.astype(int)

        q2 = np.max(model.get_q_values(s2), axis=1)

        target_q = model.get_q_values(s1)

        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * q2

        model.learn(s1, target_q)







    # see if this learns at all with just this
    # then add HER tomorrow



    '''
    plt.figure()
    plt.imshow(s2_gs[-1,:,:,:3])

    plt.figure()
    plt.imshow(s2_gs[-1,:,:,3:])
    plt.show()
    '''

    # truncate these in the observe function
    # correct the rewards in the observe function

    # so have triplets:
    # s1||g, a, s2||g

    # get the actual rewards from the episode:
    # -1 live step
    # 10000 for completing but not dead
    # -50000 for dead

    # if timeout, flip the last reward to 0 (from -1)

    # So this is a replay memory buffer: s1||g, a, s2||g, r
    # could train DQN with LSTM on these, sequentially (compute qs at each step as before, bellman)
    # Do this second, after training non sequentially, with random batches from memory buffer

    # Or dump these into a memory buffer, train randomly on them, no LSTM!
    # start with this one since its simpler...

    #add these to replay buffer

    # cycle through each time step
    # pick that frames actual s2 as a goal
    # which yields the experience:
    # s1||s2, a, s2||s2, r=0 (don't want this to include the last one which should have reward 10000, if s2 = 0)
    '''
    for time_step in range(len()):

        new experience with the actual s2
        reward check (reward will be 0)
        -add to buffer

        then pick additional_goals

        and create new replay experiences
        reward check (closeness to the actual s2, usually -1, but maybe 0) be careful about the last frame!
        -add each to buffer

    # additionally pick 3-5 more s2s
    # make an experience for each select goal using the current frame
    # probably should check the epislon ball closeness here, if its close (or equal), reward = 0
    # else reward = -1

    # so this adds 3-4X the original episode length to the replay buffer

    Now Ive got the memory buffer loaded up, do minibatch training
    # now do minibatch training of the model on the buffer

    Print train validation score
    # that's the whole training loop
    # so after each: do game.get_total_reward()
    # watch this thing decrease, overfitting on a single wad first
    # then expanding to random wads, still training
    # then expand to out-of-sample validation during training
    '''
