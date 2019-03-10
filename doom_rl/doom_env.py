import numpy as np
import vizdoom as vzd
import skimage.color, skimage.transform

class doom_env:

    # wrapper for vizdoom game

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
        img = img[:,:,0] #should be red channel
        img = img[80:380, 40:600]
        img = skimage.transform.resize(img, (60,108))
        img = img.astype(np.float32)

        #img = 2.0 * img - 1.0
        img = img.flatten()

        return img

    def state(self):

        frame = self.game.get_state().screen_buffer

        state = self.process_frame_to_state(frame)

        return state

    def reset(self):

        self.game.new_episode()

        s = self.state()

        return s

    def position(self):

        x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)

        return (x, y)

    def step(self, action):

        # take action as an int (index into list), or as the actual list of game_actions
        # also, want positive reward for reaching the end of the level...
        # there's just a living penalty and a huge death penalty in the config
        # don't see a flag for 'end of level reward'
        # seems like vizdoom wasn't really designed for this kind of exploration problem

        # Need to think carefully about the terminal state
        # some things expect there to always be a 'next state'
        old_state = self.state()

        reward = self.game.make_action(self.action_list[action])

        done = self.game.is_episode_finished()

        if not done:

            new_state = self.state()
            #state_number = self.game.get_state().number

        else:
            #print('done')
            new_state = old_state
            #state_number = 2000

        # Determine if agent found the door!
        '''
        if done and state_number < self.max_ep_steps - 1:

            print('Found the exit!')
            reward = 10000

        info = state_number
        '''
        #print(info)
        info = False

        return new_state, reward, done, info
