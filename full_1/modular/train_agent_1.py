from environment import *
from doom_agent import *

import matplotlib.pyplot as plt

config_file = 'configs/standard_config.cfg'
wad_files = ['wad_files/oblige_29_no_mon.wad']

epochs = 50
episodes_per_epoch = 2

# need number of actions, and state size

agent = DoomAgent((60,108,6))
environment = DoomEnv(config_file)

reward_collector = []
plt.figure()

for epoch in range(epochs):

    print ('Epoch:', str(epoch + 1), 'of', epochs)

    for wad_file in wad_files:

        environment.initalize_game(wad_file)

        for episode in range(episodes_per_epoch):

            print('Episode', str(episode + 1), 'of', episodes_per_epoch)

            ep_reward = agent.play_episode_train(environment)
            reward_collector.append(ep_reward)

            plt.clf()
            plt.plot(reward_collector)
            plt.pause(0.1)

        #environment.game.close()
        print('')

plt.show()
