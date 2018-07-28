from time import sleep
import os
import numpy as np
import vizdoom as vzd
from vizdoom import *

config = 'configs/standard_config.cfg'
wad = 'wad_files/handmade/full_level_1.wad'
seed = 1337

game = vzd.DoomGame()
# Use your config
game.load_config(config)
game.set_doom_map("map01")
game.set_doom_skill(2)

'''
# Create Doom Level Generator instance and set optional seed.
generator = oblige.DoomLevelGenerator()
generator.set_seed(seed)

# Set generator configs, specified keys will be overwritten.
generator.set_config({
    "size": "micro",
    "health": "more",
    "weapons": "sooner"})
'''

def dist_from_goal(current_pos, starting_pos, goal_pos):

    x_disp = starting_pos[0] - current_pos[0]
    y_disp = starting_pos[1] - current_pos[1]

    x_dist_from_goal = x_disp - goal_pos[0]
    y_dist_from_goal = y_disp - goal_pos[1]

    return np.sqrt((x_dist_from_goal ** 2) + (y_dist_from_goal ** 2)) #x_dist_from_goal, y_dist_from_goal

# There are few predefined sets of settings already defined in Oblige package, like test_wad and childs_play_wad
#generator.set_config(oblige.childs_play_wad)

# Tell generator to generate few maps (options for "length": "single", "few", "episode", "game").
#generator.set_config({"length": "few"})

# Generate method will return number of maps inside wad file.
'''
wad_path = args.output_file
print("Generating {} ...".format(wad_path))
num_maps = generator.generate(wad_path, verbose=args.verbose)
print("Generated {} maps.".format(num_maps))

if args.exit:
    exit(0)
'''

# Set Scenario to the new generated WAD
game.set_doom_scenario_path(wad)

# Sets up game for spectator (you)
game.add_game_args("+freelook 1")
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_window_visible(True)
game.set_mode(vzd.Mode.SPECTATOR)

game.init()

# Play as many episodes as maps in the new generated WAD file.
episodes = 1 #num_maps

# Play until the game (episode) is over.
for i in range(1, episodes + 1):
    print(i)

    # Update map name
    print("Map {}/{}".format(i, episodes))
    map = "map{:02}".format(i)
    game.set_doom_map(map)
    game.new_episode()

    #x = game.get_game_variable(GameVariable.PlayerX)

    initial_pos = [game.get_game_variable(GameVariable.POSITION_X),
                   game.get_game_variable(GameVariable.POSITION_Y)]

    time = 0
    while not game.is_episode_finished():
        state = game.get_state()
        time = game.get_episode_time()

        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()

        #print("State #" + str(state.number))
        #print("Game variables: ", state.game_variables)
        #print("Action:", last_action)
        #print("Reward:", reward)
        #print("=====================")

        pos = [game.get_game_variable(GameVariable.POSITION_X),
                       game.get_game_variable(GameVariable.POSITION_Y)]

        print(initial_pos[0] - pos[0], initial_pos[1] - pos[1])
        print(dist_from_goal(pos, initial_pos, (-60, -1167)))




    print("Episode finished!")
    print("Total reward:", game.get_total_reward())
    print("Kills:", game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    print("Items:", game.get_game_variable(vzd.GameVariable.ITEMCOUNT))
    print("Secrets:", game.get_game_variable(vzd.GameVariable.SECRETCOUNT))
    print("Time:", time / 35, "s")
    print("************************")
    sleep(2.0)

game.close()
