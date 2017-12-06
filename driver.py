from envs import make_visual_env, make_env
from vec_env import VecEnv
from time import sleep
from random import choice
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vis', type=int, default=0)
args = parser.parse_args()
num_envs = 1

if args.vis:
    envs = VecEnv([make_visual_env('./scenarios/deathmatch_maze.cfg') for i in range(num_envs)])
else:
    envs = VecEnv([make_env(0, './scenarios/deathmatch_maze.cfg') for i in range(num_envs)])


# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.
# actions = [[True, False, False], [False, True, False], [False, False, True]]
actions = range(envs.action_space_shape)
episode_num = 0

while True:
    print ('Episode #', episode_num)
    for j in range(1000):
        action_array = [choice(actions) for i in range(num_envs)]
        # print (action_array)
        obs, reward, done, info = envs.step(action_array)
        if done:
            game_vars = envs.get_game_variables(0)
            print('Kills : ', game_vars[2])
            episode_num += 1
            break
        # print ('Reward:', reward)
        sleep(0.01)
    envs.reset()
    sleep(0.1)