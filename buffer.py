import numpy as np
from typing import Dict, Any


class SemiMDPReplayBuffer:
    def __init__(self, state_dim, limit):
        self.state_mem = Memory(shape=(state_dim,), limit=limit)
        self.action_mem = Memory(shape=(1,), limit=limit, dtype=np.int)
        self.reward_mem = Memory(shape=(1,), limit=limit)
        self.next_state_mem = Memory(shape=(state_dim,), limit=limit)
        self.terminal_flag_mem = Memory(shape=(1,), limit=limit)
        self.operating_time_mem = Memory(shape=(1,), limit=limit)

        self.limit = limit
        self.size = 0

    def append(self, s, a, r, s_next, d, dt):
        self.state_mem.append(s)
        self.action_mem.append(a)
        self.reward_mem.append(r)
        self.next_state_mem.append(s_next)
        self.terminal_flag_mem.append(d)
        self.operating_time_mem.append(dt)

        self.size = len(self.state_mem)

    def sample_batch(self, batch_size: int) -> Dict[str, Any]:
        rng = np.random.default_rng()
        idxs = rng.choice(self.size, batch_size)

        # get batch from each buffer
        states = self.state_mem.get_batch(idxs)
        actions = self.action_mem.get_batch(idxs)
        rewards = self.reward_mem.get_batch(idxs)
        next_states = self.next_state_mem.get_batch(idxs)
        terminal_flags = self.terminal_flag_mem.get_batch(idxs)
        dts = self.operating_time_mem.get_batch(idxs)

        batch = {'state': states,
                 'action': actions,
                 'reward': rewards,
                 'next_state': next_states,
                 'done': terminal_flags,
                 'dt': dts
                 }

        return batch

    def __len__(self):
        return len(self.state_mem)

class Memory:
    """
    implementation of a circular buffer
    """
    def __init__(self, shape, limit=1000000, dtype=np.float):
        self.start = 0
        self.data_shape = shape
        self.size = 0
        self.dtype = dtype
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
