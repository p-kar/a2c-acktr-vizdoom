import numpy as np
from multiprocessing import Process, Pipe
import scipy.misc

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        action = [True if i == data else False for i in range(env.get_available_buttons_size())]
        if cmd == 'step':
            info = 0.0
            reward = env.make_action(action) / 100.0
            if not env.is_episode_finished():
                ob = process_frame(env.get_state().screen_buffer)
            done = env.is_episode_finished()
            if done:
                # print ('Restarting worker node')
                env.new_episode()
                ob = process_frame(env.get_state().screen_buffer)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            env.new_episode()
            ob = process_frame(env.get_state().screen_buffer)
            remote.send(ob)
        elif cmd == 'reset_task':
            print ('reset_task: Not implemented')
            raise NotImplementedError
            # ob = env.reset_task()
            # remote.send(ob)
        elif cmd == 'close':
            print ('close: vizdoom game object')
            # remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.get_available_buttons_size(), (1, 84, 84)))
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv():
    def __init__(self, env_fns):
        """
        envs: list of vizdoom game environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space_shape, self.observation_space_shape = self.remotes[0].recv()


    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

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