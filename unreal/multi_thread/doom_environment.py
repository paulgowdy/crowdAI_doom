from multiprocessing import Process, Pipe
import numpy as np
import cv2
#import gym
import vizdoom as vzd
from vizdoom import *
import random
import glob

import environment
#from environment import Environment

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

def dist_from_goal(current_pos, starting_pos, goal_pos):

  x_disp = starting_pos[0] - current_pos[0]
  y_disp = starting_pos[1] - current_pos[1]

  x_dist_from_goal = x_disp - goal_pos[0]
  y_dist_from_goal = y_disp - goal_pos[1]

  return np.sqrt((x_dist_from_goal ** 2) + (y_dist_from_goal ** 2)), x_disp, y_disp



def initialize_vizdoom(config_file_path = 'config/standard_config.cfg', wad_file_path = 'wad_files/oblige_530_no_mon.wad', difficulty = 1):

    #wad_file_path = random.choice(glob.glob('wad_files/*.wad'))
    #wad_file_path = 'wad_files/door_training_2a.wad'

    print("Initializing doom...")
    print(wad_file_path)
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


def preprocess_frame(observation):
  # observation shape = (210, 160, 3)

  # Put crop in here!
  #print(observation.shape)
  observation = np.rollaxis(observation, 0, 3)
  observation = observation[:,:,0] #should be red channel
  observation = observation[80:380, 40:600]
  #observation = np.expand_dims(observation, -1)

  #print(observation.shape)



  observation = observation.astype(np.float32)
  resized_observation = cv2.resize(observation, (84, 84))
  resized_observation = np.expand_dims(resized_observation, -1)

  #print(resized_observation.shape)

  resized_observation = resized_observation / 255.0

  return resized_observation

def worker(conn, env_name):

  goal_position = (1150,1303)

  #env = gym.make(env_name)
  env = initialize_vizdoom()

  initial_pos = [env.get_game_variable(GameVariable.POSITION_X),
                 env.get_game_variable(GameVariable.POSITION_Y)]

  #print('initial pos:', initial_pos)

  #env.reset()
  conn.send(0)

  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      #obs = env.reset()
      env.new_episode()

      obs = env.get_state().screen_buffer
      state = preprocess_frame(obs)

      conn.send(state)

    elif command == COMMAND_ACTION:

      reward = 0

      for i in range(4):



        #obs, r, terminal, _ = env.step(arg)

        # Arg needs to be the button vector, not just a number
        r = env.make_action(arg, 1)


        terminal = env.is_episode_finished()

        if terminal:
          print('TERMINAL!!')
          break

        #print(terminal)

        obs = env.get_state().screen_buffer

        current_pos = [env.get_game_variable(GameVariable.POSITION_X),
                       env.get_game_variable(GameVariable.POSITION_Y)]

        dist, x, y = dist_from_goal(current_pos, initial_pos, goal_position)

        reward += -1 * dist

        #print(dist)



        #reward += r

      #print('')
      state = preprocess_frame(obs)

      conn.send([state, reward, terminal])

    elif command == COMMAND_TERMINATE:

      break

    else:

      print("bad command: {}".format(command))

  env.close()

  conn.send(0)
  conn.close()


class DoomEnvironment(environment.Environment):
  @staticmethod
  def get_action_size(env_name):
    #env = gym.make(env_name)
    #action_size = env.action_space.n
    #env.close()
    action_size = 4 # MF, TL, TR, USE
    return action_size

  def __init__(self, env_name):
    environment.Environment.__init__(self)

    #initial_pos = [environment.game.get_game_variable(GameVariable.POSITION_X),
    #               environment.game.get_game_variable(GameVariable.POSITION_Y)]

    print('initializing doom env')

    #self.goal_position = (1150,1303)

    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn, env_name))
    self.proc.start()
    self.conn.recv()
    self.reset()

  def reset(self):
    print('resetting!')
    self.conn.send([COMMAND_RESET, 0])
    self.last_state = self.conn.recv()

    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    print("doom environment stopped")

  def process(self, action):

    # process action into buttons here

    actions = [
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
    ]

    self.conn.send([COMMAND_ACTION, actions[action]])
    state, reward, terminal = self.conn.recv()

    #dist, x, y = self.dist_from_goal(,self.goal_position)

    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change
