import argparse
import os

import torch
from torch.autograd import Variable

from envs import make_visual_env, make_env
from vec_env import VecEnv
from time import sleep


def evaluate(envs, model, opts):

    obs_shape = envs.observation_space_shape
    obs_shape = (obs_shape[0] * opts['num_stack'], *obs_shape[1:])
    current_obs = torch.zeros(envs.num_envs, *obs_shape)
    
    if opts['cuda']:
        current_obs = current_obs.cuda()

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space_shape[0]
        obs = torch.from_numpy(obs).float()
        if opts['num_stack'] > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    num_episodes = opts['num_episodes']
    total_reward = 0.0
    episode_cnt = 0
    episode_reward = 0.0
    total_kills = 0.0

    while episode_cnt < num_episodes:
        # sleep(0.01)
        # print(model.get_probs(Variable(current_obs, volatile=True)))
        value, action = model.act(Variable(current_obs, volatile=True),
                                            deterministic=True)
        cpu_actions = action.data.cpu().numpy()

        # print('Action:', [cpu_actions[0]])

        # Obser reward and next obs
        obs, reward, done, _ = envs.step(cpu_actions)
        episode_reward += reward[0]

        episode_end = done if envs.num_envs == 1 else done[0]
        if episode_end:
            total_reward += episode_reward
            episode_cnt += 1
            episode_reward = 0.0
            episode_game_variables = envs.get_game_variables(0)
            total_kills += episode_game_variables[2]
            obs = envs.reset()

        update_current_obs(obs)

    # print ('Avg reward:', round(total_reward / num_episodes))
    # print ('Avg kills:', (total_kills/num_episodes))
    return round(total_reward / num_episodes), (total_kills/num_episodes)

