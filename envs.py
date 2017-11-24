from vizdoom import *

def configure_game_basic(config_file_path):
    # reward divide by 100.0
    # no other change to reward
    game = DoomGame()
    game.set_doom_scenario_path("/scratch/cluster/pkar/a2c-acktr-vizdoom/scenarios/basic.wad") #This corresponds to the simple task we will pose our agent
    game.set_doom_map("map01")
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.add_available_game_variable(GameVariable.HEALTH)
    game.set_episode_timeout(300)
    game.set_episode_start_time(14)                             # after the agent unholsters the gun
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_living_reward(-1)
    game.set_mode(Mode.PLAYER)
    return game

def configure_game_defend_line(config_file_path):
    # kill reward 1.0
    # shoot reward -0.05
    # living reward -0.01
    # death reward -1.0
    # receive damage reward -0.1
    game = DoomGame()
    game.set_doom_scenario_path("/scratch/cluster/pkar/a2c-acktr-vizdoom/scenarios/defend_the_line_shoot_penalty.wad") #This corresponds to the simple task we will pose our agent
    game.set_doom_map("map01")
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.add_available_game_variable(GameVariable.HEALTH)
    game.set_episode_timeout(1000)
    game.set_episode_start_time(14)                             # after the agent unholsters the gun
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_living_reward(-1)
    game.set_death_penalty(100)
    game.set_mode(Mode.PLAYER)
    return game

def configure_game(config_file_path=None):
    game = configure_game_defend_line(config_file_path)
    return game


def make_env(worker_id, config_file_path=None):
    print("Initializing doom environment", worker_id, "...")
    game = configure_game(config_file_path)
    game.init()
    return game

def make_visual_env(config_file_path=None):
    print("Initializing doom environment...")
    game = configure_game(config_file_path)
    game.set_window_visible(True)
    game.init()
    return game
