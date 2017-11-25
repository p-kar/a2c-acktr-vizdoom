import argparse
import os

import torch
from torch.autograd import Variable

from envs import make_visual_env
from vec_env import VecEnv
from time import sleep


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | acktr')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
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
args = parser.parse_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

envs = VecEnv([make_visual_env(args.config_path)])

actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
actor_critic.eval()

obs_shape = envs.observation_space_shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_obs = torch.zeros(1, *obs_shape)


def update_current_obs(obs):
    shape_dim0 = envs.observation_space_shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

obs = envs.reset()
update_current_obs(obs)

while True:
    sleep(0.01)
    value, action = actor_critic.act(Variable(current_obs, volatile=True),
                                        deterministic=True)
    cpu_actions = action.data.cpu().numpy()

    print ('Action:', [cpu_actions[0]])

    # Obser reward and next obs
    obs, _, done, _ = envs.step([cpu_actions[0]])

    if done:
        obs = envs.reset()
        actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
        actor_critic.eval()

    update_current_obs(obs)
