from envs import make_visual_env, make_env
from vec_env import VecEnv
from time import sleep
from random import choice

num_envs = 1

envs = VecEnv([make_visual_env('./scenarios/health_gathering.cfg') for i in range(num_envs)])

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
        print (action_array)
        obs, reward, done, info = envs.step(action_array)
        print ('Reward:', reward)
        sleep(0.01)
    envs.reset()
    sleep(0.1)
