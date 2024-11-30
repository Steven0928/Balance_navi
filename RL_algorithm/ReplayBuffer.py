import collections
from gc import collect
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.distributions import Normal
import time
# from noisegen import Noisegen
from RL_algorithm.Actor import Actor
from RL_algorithm.Critic import Critic

'''
class ReplayBuffer:
    def __init__(self, capacity = 200000):
        self.buffer = collections.deque(maxlen=capacity) 

    def push(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 
        if len(self.buffer) >= 199000:
            self.buffer = sorted(self.buffer, key=lambda x:x[2], reverse=True)
           
            self.buffer = self.buffer[:50000]


    def sample(self, batch_size): 
        # transitions = random.sample(self.buffer, batch_size)
        transitions = self.buffer
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)
    def clean(self):
        self.buffer.clear()
        self.buffer = collections.deque(maxlen=100000)
'''


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

    def clean(self):
        self.buffer.clear()
        self.buffer = collections.deque(maxlen=50000)


class ReplayBuffer_LSTM:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
        self.cur_sample_state = []
        self.cur_sample_action = []
        self.cur_sample_reward = []
        self.cur_sample_next_state = []
        self.cur_sample_done = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample_flag = 0
        for sample_flag in range(0, batch_size // 64):
            sample_start = random.randint(0, (len(self.buffer) - 1) - batch_size)
            transitions = [self.buffer[i] for i in range(sample_start, sample_start + 64)]
            state, action, reward, next_state, done = zip(*transitions)
            self.cur_sample_state.append(state)
            self.cur_sample_action.append(action)
            self.cur_sample_reward.append(reward)
            self.cur_sample_next_state.append(next_state)
            self.cur_sample_done.append(done)
        return np.array(self.cur_sample_state), np.array(self.cur_sample_action), self.cur_sample_reward, np.array(
            self.cur_sample_next_state), self.cur_sample_done

        # sample_start = random.randint(0, (len(self.buffer)-1)-batch_size)
        # transitions = [self.buffer[i] for i in range(sample_start, sample_start+batch_size)]
        # state, action, reward, next_state, done = zip(*transitions)
        # return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

    def clean(self):
        self.cur_sample_state = []
        self.cur_sample_action = []
        self.cur_sample_reward = []
        self.cur_sample_next_state = []
        self.cur_sample_done = []

        # self.buffer.clear()
        # self.buffer = collections.deque(maxlen=100000)
