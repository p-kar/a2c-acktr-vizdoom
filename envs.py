from vizdoom import *

def make_env(worker_id, config_file_path=None):
    print("Initializing doom environment", worker_id, "...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.init()
    return game

def make_visual_env(config_file_path=None):
    print("Initializing doom environment...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    return game

