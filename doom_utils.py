import vizdoom as vzd
import skimage.color, skimage.transform
import numpy as np
from random import random, randint

def preprocess(img, frame_shape):

    img = skimage.transform.resize(img, frame_shape)
    img = img.astype(np.float32)
    return img

'''
def observe_episode(game, frame_shape, episode_buffer_size):

    # Need a parameter for the policy, currently random
    # actions should be in the same class...

    death_penalty = -50000

    c = 3

    # Make these numpy arrays
    s1 = np.zeros((episode_buffer_size, frame_shape[0], frame_shape[1], c))
    a = np.zeros(episode_buffer_size)
    s2 = np.zeros((episode_buffer_size, frame_shape[0], frame_shape[1], c))
    rewards = np.zeros(episode_buffer_size)

    step = 0


    while not game.is_episode_finished():

        action = np.random.randint(0, len(actions))

        state_before = game.get_state().screen_buffer
        state_before = np.rollaxis(state_before, 0, 3)

        s1[step] = preprocess(state_before, frame_shape)
        a[step] = action

        rewards[step] = game.make_action(actions[action], 1)

        if not game.is_episode_finished():

            state_after = game.get_state().screen_buffer
            state_after = np.rollaxis(state_after, 0, 3)

            s2[step] = preprocess(state_after, frame_shape)

            if rewards[step] < death_penalty:

                rewards[step] = death_penalty

            if rewards[step] == 0:

                rewards[step] = 10000

        step += 1

    return s1, np.array(a), s2, rewards, step
'''

def observe_episode(game, model, goal, frame_shape):

    # epsilon greedy policy
    epsilon = 0.8
    # want to implement epsilon decay

    #These come from the config...
    death_penalty = -50000
    timeout_steps = 777

    frame_repeat = 1

    max_buffer_size = 10000

    s1_qs = np.zeros((max_buffer_size, frame_shape[0], frame_shape[1], 6))
    action_collector = np.zeros(max_buffer_size)
    s2_qs = np.zeros((max_buffer_size, frame_shape[0], frame_shape[1], 6))
    reward_collector = np.zeros(max_buffer_size)

    game.new_episode()

    step = 0

    while not game.is_episode_finished():

        state_before = game.get_state().screen_buffer
        state_before = np.rollaxis(state_before, 0, 3)
        state_before = preprocess(state_before, frame_shape)

        sq_before = np.concatenate([state_before, goal], -1)

        if random() < epsilon:

            action_ind = randint(0, len(actions) - 1)

        else:

            action_ind = model.policy(sq_before)



        action = actions[action_ind]

        reward = game.make_action(action, frame_repeat)



        isterminal = game.is_episode_finished()

        if not isterminal:

            state_after = game.get_state().screen_buffer
            state_after = np.rollaxis(state_after, 0, 3)
            state_after = preprocess(state_after, frame_shape)

            sq_after = np.concatenate([state_after, goal], -1)

            s2_qs[step,:,:,:] = sq_after

        # Reward modifications here

        #If you die, just make the total reward for the episode the death death_penalty
        # basically don't penalize dying later
        #print(isterminal, reward)
        if isterminal and reward == (death_penalty - 1):

            #print('died', np.sum(reward_collector))

            reward = reward - np.sum(reward_collector)

        if isterminal and step < (timeout_steps - 2) and reward > death_penalty:

            reward = 10000

        if isterminal and step == (timeout_steps - 1):

            reward = 0

        s1_qs[step,:,:,:] = sq_before
        action_collector[step] = action_ind
        reward_collector[step] = reward

        step += 1

    #truncate
    s1_qs = s1_qs[:step,:,:,:]
    s2_qs = s2_qs[:step,:,:,:]
    action_collector = action_collector[:step]
    reward_collector = reward_collector[:step]

    return s1_qs, action_collector, s2_qs, reward_collector






    # observe episode of the game - concatenate initial goal with the states...
    # go until game end: death, time out,

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

    return s1_gs, actions, s2_gs, rewards

def initialize_vizdoom(config_file_path, wad_file_path, difficulty = 1):

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

actions = [

[0,0,0,0,0,0,0,0],

[1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0],
[0,0,0,1,0,0,0,0],
[0,0,0,0,1,0,0,0],
[0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,1],

[0,0,1,0,0,1,0,0],
[0,0,1,0,1,0,0,0],
[0,0,0,1,0,1,0,0],
[0,0,0,1,1,0,0,0],

[0,0,0,0,0,1,1,0],
[0,0,0,0,0,1,0,1],
[0,0,0,0,1,0,1,0],
[0,0,0,0,1,0,0,1],


[1,0,0,0,0,1,0,0],
[1,0,0,0,1,0,0,0],
[1,0,0,1,0,0,0,0],
[1,0,1,0,0,0,0,0],

[1,0,1,0,0,1,0,0],
[1,0,1,0,1,0,0,0],
[1,0,0,1,0,1,0,0],
[1,0,0,1,1,0,0,0],

[1,0,0,0,0,1,1,0],
[1,0,0,0,0,1,0,1],
[1,0,0,0,1,0,1,0],
[1,0,0,0,1,0,0,1],


[0,1,0,0,0,1,0,0],
[0,1,0,0,1,0,0,0],
[0,1,0,1,0,0,0,0],
[0,1,1,0,0,0,0,0],

[0,1,1,0,0,1,0,0],
[0,1,1,0,1,0,0,0],
[0,1,0,1,0,1,0,0],
[0,1,0,1,1,0,0,0],

[0,1,0,0,0,1,1,0],
[0,1,0,0,0,1,0,1],
[0,1,0,0,1,0,1,0],
[0,1,0,0,1,0,0,1]

]
