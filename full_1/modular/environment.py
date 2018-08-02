import vizdoom as vzd



class DoomEnv:

    def __init__(self, config_file):

        self.config = config_file
        #self.map_list = map_list

        self.goal_pos_dict = {
        'wad_files/oblige_29_no_mon.wad':  (1040,1220),
        'wad_files/oblige_530_no_mon.wad':  (1150,1303)

        }

    def initalize_game(self, wad_file):

        difficulty = 1

        print("Initializing Doom")
        print('Using wad file:', wad_file)
        self.game = vzd.DoomGame()

        self.game.load_config(self.config)
        self.game.set_doom_scenario_path(wad_file)

        self.game.set_window_visible(True)
        self.game.set_mode(vzd.Mode.PLAYER)
        #game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

        self.game.set_doom_skill(difficulty)

        #self.game.init()
        #print("Doom initialized.")

        #self.game = game

        self.goal_position = self.goal_pos_dict[wad_file]
