# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
#import deepmind_lab
import vizdoom as vzd
import skimage.color, skimage.transform

from environment import Environment

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

class innerDoomEnv():

	def __init__(self, WAD_FILE = 'wad_files/test_6.wad', visible = False):

		CONFIG = "standard_config.cfg"
		#WAD_FILE

		self.game = vzd.DoomGame()

		self.game.load_config(CONFIG)
		self.game.set_doom_map("map01")
		self.game.set_doom_skill(2)

		# This line connects to the actual wad file we just generated
		self.game.set_doom_scenario_path(WAD_FILE)

		# Sets up game for spectator (you)
		# Do I want this for RL env, need to look into it
		#self.game.add_game_args("+freelook 1")
		self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
		self.game.set_window_visible(visible)
		#self.game.set_mode(vzd.Mode.SPECTATOR)
		self.game.set_mode(vzd.Mode.PLAYER)

		self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
		self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)

		self.game.init()

		self.game.set_doom_map("map{:02}".format(1))
		self.game.new_episode()

		self.max_ep_steps = 2000

		self.action_list = [

				[0,0,0,0,0,0,0,0],

				[1,0,0,0,0,0,0,0],
				[0,1,0,0,0,0,0,0],
				[0,0,0,0,1,0,0,0],
				[0,0,0,0,0,1,0,0],
				[0,0,0,0,0,0,1,0],
				[0,0,0,0,0,0,0,1],

				[0,0,0,0,0,1,1,0],
				[0,0,0,0,0,1,0,1],
				[0,0,0,0,1,0,1,0],
				[0,0,0,0,1,0,0,1],

				[1,0,0,0,0,1,0,0],
				[1,0,0,0,1,0,0,0],

				[1,0,0,1,0,1,0,0],
				[1,0,0,1,1,0,0,0],

				[1,0,0,0,0,1,1,0],
				[1,0,0,0,0,1,0,1],
				[1,0,0,0,1,0,1,0],
				[1,0,0,0,1,0,0,1],

				[0,1,0,0,0,1,0,0],
				[0,1,0,0,1,0,0,0],

				[0,1,0,0,0,1,1,0],
				[0,1,0,0,0,1,0,1],
				[0,1,0,0,1,0,1,0],
				[0,1,0,0,1,0,0,1],

				[1,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0],
				[0,1,0,0,0,0,0,1],
				[0,1,0,0,0,0,1,0]
			]

	def process_frame_to_state(self, img):

		# Comes in as 3 x 480 x 680

		img = np.rollaxis(img, 0, 3)
		#img = img[:,:,0] #should be red channel
		img = img[80:380, 40:600, :]
		#img = skimage.transform.resize(img, (60,108))
		img = skimage.transform.resize(img, (84,84))
		img = img.astype(np.float32)

		#img = 2.0 * img - 1.0
		#img = img.flatten()

		return img

	def state(self):

		frame = self.game.get_state().screen_buffer

		state = self.process_frame_to_state(frame)

		return state

	def reset(self):

		self.game.new_episode()

		s = self.state()

		return s

	def step(self, action):

		#print('Action:', action)
		#reward = self.game.make_action(self.action_list[action])

		state_number = self.game.get_state().number

		reward = self.game.make_action(action)

		done = self.game.is_episode_finished()

		if not done:

			new_state = self.state()
			#state_number = self.game.get_state().number

		else:
			#print('done')
			new_state = 0
			#state_number = 2000

			if state_number < 1999:

				reward += 1000



		return reward, done, new_state



# this is where the env actually lives...
def worker(conn, env_name):
	level = env_name

	env = innerDoomEnv()



	'''
	env = deepmind_lab.Lab(
	level,
	['RGB_INTERLACED'],
	config={
	  'fps': str(60),
	  'width': str(84),
	  'height': str(84)
	})
	'''


	conn.send(0)

	while True:

		command, arg = conn.recv()

		if command == COMMAND_RESET:

			obs = env.reset()

			#obs = env.observations()['RGB_INTERLACED']

			conn.send(obs)

		elif command == COMMAND_ACTION:

			#reward = env.step(arg, num_steps=4)

			reward, terminal, obs = env.step(arg)

			'''
			terminal = not env.is_running()

			if not terminal:

			obs = env.observations()['RGB_INTERLACED']

			else:

			obs = 0
			'''

			conn.send([obs, reward, terminal])

		elif command == COMMAND_TERMINATE:

			break

		else:

			print("bad command: {}".format(command))

	env.close()
	conn.send(0)
	conn.close()


def _action(*entries):
	return np.array(entries, dtype=np.intc)


class DoomEnvironment(Environment):


	ACTION_LIST = [

              [0,0,0,0,0,0,0,0],

              [1,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0],
              [0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,1],

              [0,0,0,0,0,1,1,0],
              [0,0,0,0,0,1,0,1],
              [0,0,0,0,1,0,1,0],
              [0,0,0,0,1,0,0,1],

              [1,0,0,0,0,1,0,0],
              [1,0,0,0,1,0,0,0],

              [1,0,0,1,0,1,0,0],
              [1,0,0,1,1,0,0,0],

              [1,0,0,0,0,1,1,0],
              [1,0,0,0,0,1,0,1],
              [1,0,0,0,1,0,1,0],
              [1,0,0,0,1,0,0,1],

              [0,1,0,0,0,1,0,0],
              [0,1,0,0,1,0,0,0],

              [0,1,0,0,0,1,1,0],
              [0,1,0,0,0,1,0,1],
              [0,1,0,0,1,0,1,0],
              [0,1,0,0,1,0,0,1],

              [1,0,0,0,0,0,0,1],
              [1,0,0,0,0,0,1,0],
              [0,1,0,0,0,0,0,1],
              [0,1,0,0,0,0,1,0]
          ]


	@staticmethod
	def get_action_size(env_name):
		return len(DoomEnvironment.ACTION_LIST)

	def __init__(self, env_name):
		Environment.__init__(self)

		self.conn, child_conn = Pipe()
		self.proc = Process(target=worker, args=(child_conn, env_name))
		self.proc.start()
		self.conn.recv()
		self.reset()

	def reset(self):
		self.conn.send([COMMAND_RESET, 0])
		obs = self.conn.recv()

		self.last_state = self._preprocess_frame(obs)
		self.last_action = 0
		self.last_reward = 0

	def stop(self):
		self.conn.send([COMMAND_TERMINATE, 0])
		ret = self.conn.recv()
		self.conn.close()
		self.proc.join()
		print("doom environment stopped")

	def _preprocess_frame(self, image):
		image = image.astype(np.float32)
		image = image / 255.0
		return image

	def process(self, action):
		real_action = DoomEnvironment.ACTION_LIST[action]

		self.conn.send([COMMAND_ACTION, real_action])
		obs, reward, terminal = self.conn.recv()

		if not terminal:
			state = self._preprocess_frame(obs)
		else:
			state = self.last_state

		pixel_change = self._calc_pixel_change(state, self.last_state)
		self.last_state = state
		self.last_action = action
		self.last_reward = reward

		return state, reward, terminal, pixel_change
