import numpy as np
import os


class CircularBuffer:
    """
    implementation of a circular buffer
    """

    def __init__(self, shape, limit=1000000):
        self.start = 0
        self.data_shape = shape
        self.size = 0
        self.limit = limit
        self.data = np.zeros((self.limit,) + shape)

    def append(self, data):
        if self.size < self.limit:
            self.size += 1
        else:
            self.start = (self.start + 1) % self.limit

        self.data[(self.start + self.size - 1) % self.limit] = data

    def get_batch(self, idxs):

        return self.data[(self.start + idxs) % self.limit]

    def __len__(self):
        return self.size


def flatten_shape(shape):
    num = 1
    for dim in shape:
        num *= dim

    return num


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def normalize_angle(theta):
    # normalize a given angle so that the angle lies in [-pi, pi)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def sample_from_unit_ball(dim):
    # uniformly sample a point from the l^2 unit ball
    x = np.random.randn(dim)
    x_size = np.linalg.norm(x)
    x = x / (x_size + 1e-8)
    r = np.random.rand() ** (1.0 / dim)

    return r * x


def get_env_spec(env):
    print('environment : ' + env.unwrapped.spec.id)
    print('obs dim : ', env.observation_space.shape, '/ ctrl dim : ', env.action_space.shape)
    dimS = env.observation_space.shape[0]
    dimA = env.action_space.shape[0]
    ctrl_range = env.action_space.high
    max_ep_len = env._max_episode_steps
    print('-' * 80)

    print('dt : {}'.format(env.dt))
    print('ctrl range : ({})'.format(ctrl_range))
    print('max_ep_len : ', max_ep_len)
    print('-' * 80)

    return dimS, dimA, env.dt, ctrl_range, max_ep_len


def set_log_dir(env_id):
    log_pth = './log/' + env_id + '/'
    os.makedirs(log_pth, exist_ok=True)
    os.makedirs('./checkpoints/' + env_id + '/', exist_ok=True)
    return
