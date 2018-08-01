import vizdoom as vzd



class DoomEnv:

    def __init__(self, config_file):

        self.config = config_file
        #self.map_list = map_list

        self.goal_pos_dict = {
        'oblige_multi_1.wad': (-47, -1167)
        }

    def initalize_game(self, wad_file):

        difficulty = 1

        print("Initializing Doom")
        print('Using wad file:', wad_file)
        game = vzd.DoomGame()

        game.load_config(self.config)
        game.set_doom_scenario_path(wad_file)

        game.set_window_visible(True)
        game.set_mode(vzd.Mode.PLAYER)
        #game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

        game.set_doom_skill(difficulty)

        game.init()
        print("Doom initialized.")

        self.game = game

        self.goal_position = self.goal_pos_dict[wad_file]

    #def start_episode(self):
