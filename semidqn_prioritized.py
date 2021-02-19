import csv
import time
import argparse
import gym
from agent_prioritized import SemiDQNAgent
from replay import Transition
import torch
import gym_lifter
import datetime
import os


def run_prioritized(env_id='Lifter-v0',
                    gamma=0.999,
                    lr=0.00025 / 4,
                    polyak=1e-3,
                    hidden1=256,
                    hidden2=256,
                    num_ep=int(1e3),
                    buffer_size=int(1e6),
                    fill_buffer=20000,
                    batch_size=32,
                    train_interval=50,
                    start_train=10000,
                    eval_interval=20,
                    priority_exponent=.6,
                    importance_sampling_exponent_begin=.4,
                    importance_sampling_exponent_end=1.,
                    uniform_sample_prob=1e-3,
                    normalize_weights=True,
                    device='cuda',
                    pth=None,
                    render=False):

    arg_dict = locals()
    num_ep = int(num_ep)
    buffer_size = int(buffer_size)
    env = gym.make(env_id)

    dimS = env.observation_space.shape[0]   # dimension of state space
    nA = env.action_space.n                 # number of actions

    # (physical) length of the time horizon of each truncated episode
    # each episode run for t \in [0, T)
    # set for RL in semi-MDP setting
    T = 300
    max_epsilon = 1.
    min_epsilon = 0.02

    # linearly scheduled $\epsilon$
    exploration_schedule = LinearSchedule(begin_t=0,
                                          end_t=num_ep,
                                          begin_value=max_epsilon,
                                          end_value=min_epsilon)
    # linearly scheduled importance sampling weight exponent
    anneal_schedule = LinearSchedule(begin_t=0,
                                     end_t=num_ep * 125,
                                     begin_value=importance_sampling_exponent_begin,
                                     end_value=importance_sampling_exponent_end)

    agent = SemiDQNAgent(
                         dimS=dimS,
                         nA=nA,
                         action_map=env.action_map,
                         gamma=gamma,
                         hidden1=hidden1,
                         hidden2=hidden2,
                         lr=lr,
                         tau=polyak,
                         buffer_size=buffer_size,
                         batch_size=batch_size,
                         priority_exponent=priority_exponent,
                         anneal_schedule=anneal_schedule,
                         uniform_sample_prob=uniform_sample_prob,
                         normalize_weights=normalize_weights,
                         device=device,
                         render=render
                         )

    if pth is None:
        # default location of directory for training log
        pth = './log' + env_id + '/'

    os.makedirs(pth, exist_ok=True)
    current_time = time.strftime("%m_%d-%H%_M_%S")
    log_file = open(pth + 'prioritized_' + current_time + '.csv',
                    'w',
                    encoding='utf-8',
                    newline='')
    logger = csv.writer(log_file)

    with open(pth + 'prioritized_' + current_time + '.txt', 'w') as f:
        for key, val in arg_dict.items():
            print(key, '=', val, file=f)

    # start environment roll-out

    carried = None
    for i in range(num_ep):
        s = env.reset()
        t = 0.  # physical elapsed time of the present episode
        ep_reward = 0.
        epsilon = exploration_schedule(i)
        while t < T:
            # print('t = {:.4f}'.format(t))
            a = agent.get_action(s, epsilon)

            s_next, r, d, info = env.step(a)
            ep_reward += gamma ** t * r
            dt = info['dt']
            carried = info['carried']
            t = info['elapsed time']
            transition = Transition(s_tm1=s, a_tm1=a, r_t=r, s_t=s_next, dt=dt, d=d)

            agent.replay.add(item=transition, priority=agent.max_seen_priority)
            agent.train()
            s = s_next                          # please don't forget this...please...
        log_time = datetime.datetime.now(tz=None).strftime("%Y-%m-%d %H:%M:%S")
        replay_size = agent.replay.size

        # TODO : improve logging
        print('{} (episode {} / epsilon = {:.2f}) reward = {:.4f} | carried = {} | max_seen_priority = {:.2f} | replay size = {}'.format(log_time,
              i, epsilon, ep_reward, carried, agent.max_seen_priority, replay_size))
        logger.writerow([i, ep_reward, carried])

    log_file.close()

    """
        if i % eval_interval == 0:
            eval_score = eval_agent(agent, env_id, render=False)
            log = [i, eval_score]
            print('step {} : {:.4f}'.format(i, eval_score))
            eval_logger.writerow(log)
    """
    return


"""
def eval_agent(agent, env_id, eval_num=5, render=False):
    log = []
    for ep in range(eval_num):
        env = gym.make(env_id)

        state = env.reset()
        step_count = 0
        ep_reward = 0
        done = False

        while not done:
            if render and ep == 0:
                env.render()

            action = agent.get_action(state, 0.0)
            next_state, reward, done, _ = env.step(action)
            step_count += 1
            state = next_state
            ep_reward += reward

        if render and ep == 0:
            env.close()
        log.append(ep_reward)

    avg = sum(log) / eval_num

    return avg
"""


class LinearSchedule:
    """Linear schedule, used for exploration epsilon in DQN agents."""
    # taken from https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/parts.py
    def __init__(self,
                 begin_value,
                 end_value,
                 begin_t,
                 end_t=None,
                 decay_steps=None):
        if (end_t is None) == (decay_steps is None):
            raise ValueError('Exactly one of end_t, decay_steps must be provided.')
        self._decay_steps = decay_steps if end_t is None else end_t - begin_t
        self._begin_t = begin_t
        self._begin_value = begin_value
        self._end_value = end_value

    def __call__(self, t):
        """Implements a linear transition from a begin to an end value."""
        frac = min(max(t - self._begin_t, 0), self._decay_steps) / self._decay_steps
        return (1 - frac) * self._begin_value + frac * self._end_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument('--env', required=True)
    parser.add_argument('--max_iter', required=False, default=2e3, type=float)
    parser.add_argument('--eval_interval', required=False, default=1000, type=int)
    parser.add_argument('--render', required=False, default=False, type=bool)
    parser.add_argument('--tau', required=False, default=1e-3, type=float)
    parser.add_argument('--q_lr', required=False, default=1e-3, type=float)
    parser.add_argument('--hidden1', required=False, default=256, type=int)
    parser.add_argument('--hidden2', required=False, default=256, type=int)
    parser.add_argument('--train_interval', required=False, default=50, type=int)
    parser.add_argument('--start_train', required=False, default=1000, type=int)
    parser.add_argument('--fill_buffer', required=False, default=1000, type=int)
    parser.add_argument('--batch_size', required=False, default=32, type=int)
    parser.add_argument('--device', required=False, default=default_device, type=str)

    args = parser.parse_args()

    run_prioritized(args.env,
                    gamma=0.999,
                    lr=args.q_lr,
                    polyak=args.tau,
                    hidden1=args.hidden1,
                    hidden2=args.hidden2,
                    num_ep=args.max_iter,
                    buffer_size=int(1e6),
                    fill_buffer=args.fill_buffer,
                    batch_size=args.batch_size,
                    train_interval=args.train_interval,
                    start_train=args.start_train,
                    eval_interval=args.eval_interval,
                    device=args.device,
                    render=args.render)
