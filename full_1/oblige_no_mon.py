import vizdoom as vzd
import oblige

'''
config = 'standard_config.cfg'

game = vzd.DoomGame()
    # Use your config
game.load_config(args.config)
game.set_doom_map("map01")
game.set_doom_skill(3)
'''

wad_path = 'wad_files/oblige/oblige_530_no_mon.wad'
seed = 530

# Create Doom Level Generator instance and set optional seed.
generator = oblige.DoomLevelGenerator()
generator.set_seed(seed)

# Set generator configs, specified keys will be overwritten.
generator.set_config({
    "size": "micro",
    "health": "more",
    "weapons": "sooner"})

# There are few predefined sets of settings already defined in Oblige package, like test_wad and childs_play_wad
generator.set_config(oblige.childs_play_wad)

# Tell generator to generate few maps (options for "length": "single", "few", "episode", "game").
generator.set_config({"length": "single"})
generator.set_config({"mons": "none"})

# Generate method will return number of maps inside wad file.
#wad_path = output_file
print("Generating {} ...".format(wad_path))
num_maps = generator.generate(wad_path, verbose=True)
print("Generated {} maps.".format(num_maps))
