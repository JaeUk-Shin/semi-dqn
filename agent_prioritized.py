import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
import gym
from utils import freeze
from model import Critic, DoubleCritic
from typing import Callable, List
import random
from replay import PrioritizedTransitionReplay, Transition


class SemiDQNAgent:
    def __init__(self,
                 dimS,
                 nA,
                 action_map: Callable[..., List[int]],
                 gamma,
                 hidden1,
                 hidden2,
                 lr,
                 tau,
                 buffer_size,
                 batch_size,
                 priority_exponent,
                 normalize_weights,
                 uniform_sample_prob,
                 anneal_schedule: Callable,
                 clipped=False,
                 device='cpu',
                 render=False):

        arg_dict = locals()
        print('agent spec')
        print('-' * 80)
        print(arg_dict)
        print('-' * 80)

        self.dimS = dimS
        self.nA = nA
        self.clipped = clipped
        # set networks
        if clipped:
            self.Q = DoubleCritic(dimS, nA, hidden_size1=hidden1, hidden_size2=hidden2).to(device)
        else:
            self.Q = Critic(dimS, nA, hidden_size1=hidden1, hidden_size2=hidden2).to(device)
        self.target_Q = copy.deepcopy(self.Q).to(device)
        freeze(self.target_Q)

        self.optimizer = Adam(self.Q.parameters(), lr=lr)
        # discount factor & polyak constant
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        replay_structure = Transition(s_tm1=None, a_tm1=None, r_t=None, s_t=None, dt=None, d=None)

        # replay buffer for experience replay in semi-MDP
        # prioritized experience replay for semi-DQN
        self.replay = PrioritizedTransitionReplay(capacity=buffer_size,
                                                  structure=replay_structure,
                                                  priority_exponent=priority_exponent,
                                                  importance_sampling_exponent=anneal_schedule,
                                                  uniform_sample_probability=uniform_sample_prob,
                                                  normalize_weights=normalize_weights,
                                                  random_state=np.random.RandomState(1),
                                                  encoder=None,
                                                  decoder=None)
        self.max_seen_priority = 1.
        self.schedule = anneal_schedule

        # function which returns the set of executable actions at a given state
        # expected return type : numpy array when 2nd arg = True / list when False
        self.action_map = action_map
        self.render = render
        self.device = device
        return

    def target_update(self):
        for p, target_p in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)

        return

    def get_action(self, state, eps):
        dimS = self.dimS
        possible_actions = self.action_map(state)  # return a set of indices instead of a mask vector
        u = np.random.rand()
        if u < eps:
            # random selection among executable actions
            a = random.choice(possible_actions)
            # print('control randomly selected : ', a)
        else:
            m = mask(possible_actions)
            # greedy selection among executable actions
            # non-admissible actions are not considered since their value corresponds to -inf
            s = torch.tensor(state, dtype=torch.float).view(1, dimS).to(self.device)
            if self.clipped:
                q = self.Q.Q1(s)
            else:
                q = self.Q(s)
            a = np.argmax(q.cpu().data.numpy() + m)
            # print('control greedily selected : ', a)
            # print('value function : ', q.cpu().data.numpy() + self.marker(state))
        return a

    def train(self):
        device = self.device
        gamma = self.gamma
        # batch = self.buffer.sample_batch(self.batch_size)

        # transition samples with importance sampling weights
        transitions, indices, weights = self.replay.sample(self.batch_size)
        # TODO : unroll transitions
        state = transitions[0]

        m = np.vstack([mask(self.action_map(state[i])) for i in range(self.batch_size)])

        # unroll batch
        # each sample : (s, a, r, s^\prime, \Delta t)
        with torch.no_grad():
            s = torch.tensor(transitions[0], dtype=torch.float).to(device)
            a = torch.unsqueeze(torch.tensor(transitions[1], dtype=torch.long).to(device), 1)  # action type : discrete
            r = torch.tensor(transitions[2], dtype=torch.float).to(device)
            s_next = torch.tensor(transitions[3], dtype=torch.float).to(device)
            d = torch.tensor(transitions[4], dtype=torch.float).to(device)
            dt = torch.tensor(transitions[5], dtype=torch.float).to(device)

            m = torch.tensor(m, dtype=torch.float).to(device)
            w = torch.tensor(weights, dtype=torch.float).to(device)
            # compute $\max_{a^\prime} Q (s^\prime, a^\prime)$
            # note that the maximum MUST be taken over the set of admissible actions
            # this can be done via masking invalid entries
            # double DQN
            # Be careful of shape of each tensor!
            if self.clipped:
                a_inner = torch.unsqueeze(torch.max(self.Q.Q1(s_next) + m, 1)[1], 1)
                q_next1, q_next2 = self.target_Q(s_next)
                q_next = torch.min(torch.squeeze(q_next1.gather(1, a_inner)), torch.squeeze(q_next2.gather(1, a_inner)))
            else:
                a_inner = torch.unsqueeze(torch.max(self.Q(s_next) + m, 1)[1], 1)
                q_next = torch.squeeze(self.target_Q(s_next).gather(1, a_inner))
            # target construction in semi-MDP case
            # see [Puterman, 1994] for introduction to the theory of semi-MDPs
            # $r\Delta t + \gamma^{\Delta t} \max_{a^\prime} Q (s^\prime, a^\prime)$
            target = r + gamma ** dt * (1. - d) * q_next

        # loss construction & parameter update
        if self.clipped:
            a1_vector, a2_vector = self.Q(s)
            out1 = torch.squeeze(a1_vector.gather(1, a))
            out2 = torch.squeeze(a2_vector.gather(1, a))
            td_errors = target - out1
            loss = .5 * ((w * (target - out1)) ** 2).sum() + .5 * ((w * (target - out2)) ** 2).sum()
        else:
            out = torch.squeeze(self.Q(s).gather(1, a))
            td_errors = target - out
            loss = .5 * ((w * td_errors) ** 2).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # priority update
        new_priorities = np.abs(np.squeeze(td_errors.cpu().data.numpy()))
        max_priority = np.max(new_priorities)
        self.max_seen_priority = max([self.max_seen_priority, max_priority])
        self.replay.update_priorities(indices=indices, priorities=new_priorities)

        # soft target update
        self.target_update()

        return

    def eval(self, test_env, T=14400, eval_num=3):
        """
        evaluation of agent
        during evaluation, agent execute noiseless actions
        """
        print('evaluating on 24 hrs data...', end=' ')
        reward_log = np.zeros(eval_num)
        num_log = np.zeros(eval_num, dtype=int)
        for ep in range(eval_num):
            state = test_env.reset()
            step_count = 0
            ep_reward = 0.
            t = 0.
            # done = False
            info = None
            # while not done:
            while t < T:
                # half hr evaluation
                if self.render and ep == 0:
                    test_env.render()

                action = self.get_action(state, 0.)  # noiseless evaluation
                next_state, reward, done, info = test_env.step(action)

                step_count += 1
                state = next_state
                ep_reward += self.gamma ** t * reward
                t = info['elapsed_time']


            # save carried quantity at the end of the episode
            carried = test_env.operation_log['carried']
            reward_log[ep] = ep_reward
            num_log[ep] = carried

            if self.render and ep == 0:
                test_env.close()
        avg = np.mean(reward_log)
        num_avg = np.mean(num_log)

        print('average reward : {:.4f} | carried : {}'.format(avg, num_avg))

        return [avg, num_avg]


def mask(actions: List[int]) -> np.ndarray:
    # generate a mask representing the set
    m = np.full(30, -np.inf)
    # 0 if admissible, -inf else
    m[actions] = 0.
    return m

