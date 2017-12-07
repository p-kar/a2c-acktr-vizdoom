import argparse
import os

import torch
from torch.autograd import Variable

from model import CNNPolicy
from envs import make_visual_env
from vec_env import VecEnv
from time import sleep

from eval import *


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | acktr')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='VizDoom',
                    help='environment to train on (default: VizDoom)')
parser.add_argument('--config-path', default='./scenarios/basic.cfg',
                        help='vizdoom configuration file path (default: ./scenarios/basic.cfg)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--log-dir', default='/tmp/doom/',
                    help='directory to save agent logs (default: /tmp/doom)')
parser.add_argument('--vis', default=0, type=int)
parser.add_argument('--num_episodes', default=10, type=int)
args = parser.parse_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

if args.vis == 1:
    envs = VecEnv([make_visual_env(args.config_path)])
else:
    envs = VecEnv([make_env(0, args.config_path)])

# actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
obs_shape = envs.observation_space_shape
actor_critic = CNNPolicy(obs_shape[0], envs.action_space_shape)
actor_critic.eval()


reward, kills = evaluate(envs, actor_critic, {'num_episodes': args.num_episodes, 'num_stack': args.num_stack, 'cuda' : False})

print('Reward : %f, Kills : %f\n' % (reward, kills))

envs.close()

