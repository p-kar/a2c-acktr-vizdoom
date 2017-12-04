import numpy as np
from multiprocessing import Process, Pipe
import scipy.misc
import os

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    # prev_agent_health = 0
    # prev_agent_ammo = 0
    log_file = None
    if env_fn_wrapper.log_file != "":
        log_file = open(env_fn_wrapper.log_file, 'w')
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    total_reward = 0.0
    episode_reward = 0.0
    episode_cnt = 0.0
    total_episode_cnt = 0
    while True:
        cmd, data = remote.recv()
        if data is None:
            import random
            data = random.randint(0, 2**env.get_available_buttons_size() - 1)
        action = [True if i == '1' else False for i in get_bin(data, env.get_available_buttons_size())]
        
        if cmd == 'step':
            reward = env.make_action(action)
            if not env.is_episode_finished():
                ob = process_frame(env.get_state().screen_buffer)
                # agent_health = env.get_state().game_variables[0]
                # agent_ammo = env.get_state().game_variables[1]
                # if prev_agent_health > agent_health:                # we add a penalty if the agent is hit
                #     reward = reward - 0                             # having a penalty doesn't seem to help
                # if prev_agent_ammo > agent_ammo:                    # we add a penalty if the agent fires
                #     reward = reward - 0
                # prev_agent_health = agent_health
                # prev_agent_ammo = agent_ammo
            reward = reward / 100.0                                 # normalizing the reward
            episode_reward += reward
            done = env.is_episode_finished()
            if done:
                env.new_episode()
                ob = process_frame(env.get_state().screen_buffer)
                total_reward += episode_reward
                episode_cnt += 1
                total_episode_cnt += 1
                episode_reward = 0.0
            remote.send((ob, reward, done, 0.0))
        elif cmd == 'log':
            if log_file is None:
                continue
            if episode_cnt == 0.0:
                continue
            avg_reward = round(total_reward / episode_cnt, 5)
            log_file.write(str(total_episode_cnt) + ', ' + str(avg_reward) + '\n')
            log_file.flush()
            total_reward = 0.0
            episode_cnt = 0.0
        elif cmd == 'reset':
            env.new_episode()
            ob = process_frame(env.get_state().screen_buffer)
            remote.send(ob)
        elif cmd == 'reset_task':
            print ('reset_task: Not implemented')
            raise NotImplementedError
        elif cmd == 'close':
            print ('Terminating doom environment')
            if log_file is not None:
                log_file.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((2**env.get_available_buttons_size(), (1, 84, 84)))
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x, log_file=""):
        self.x = x
        self.log_file = log_file
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv():
    def __init__(self, env_fns, logging=False, log_dir='/tmp/vizdoom/'):
        """
        envs: list of vizdoom game environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.log_files = [os.path.join(log_dir, 'worker_' + str(i) + '.log') for i in range(nenvs)]
        if logging is False:
            self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        else:
            self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn, log_file)))
                for (work_remote, remote, env_fn, log_file) in zip(self.work_remotes, self.remotes, env_fns, self.log_files)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space_shape, self.observation_space_shape = self.remotes[0].recv()


    def step(self, actions):
        cumul_rewards = None
        cumul_dones = None
        for _ in range(4):
            for remote, action in zip(self.remotes, actions):
                remote.send(('step', action))
            results = [remote.recv() for remote in self.remotes]
            obs, rews, dones, infos = zip(*results)
            if cumul_rewards is None:
                cumul_rewards = np.stack(rews)
            else:
                cumul_rewards += np.stack(rews)
            if cumul_dones is None:
                cumul_dones = np.stack(dones)
            else:
                cumul_dones |= np.stack(dones)
        return np.stack(obs), cumul_rewards, cumul_dones, infos

    def log(self):
        for remote in self.remotes:
            remote.send(('log', None))
        return

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)
