from environment import *
from doom_agent_5 import *

config_file = 'standard_config.cfg'
wad_files = ['oblige_multi_1.wad']

epochs = 3
episodes_per_epoch = 3

# need number of actions, and state size

agent = DoomAgent((60,108,6))
environment = DoomEnv(config_file)


for epoch in range(epochs):

    print ('Epoch:', str(epoch + 1), 'of', epochs)

    for wad_file in wad_files:

        environment.initalize_game(wad_file)

        for episode in range(episodes_per_epoch):

            print('Episode', str(episode + 1), 'of', episodes_per_epoch)

            agent.play_episode_train(environment)
