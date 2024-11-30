from math import e, exp, tanh
import math
import random
import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from RL_algorithm.Actor import Actor
from RL_algorithm.Critic import Critic

from RL_algorithm.Actor import Actor_LSTM
from RL_algorithm.Critic import Critic_LSTM

from RL_algorithm.ReplayBuffer import ReplayBuffer as Memory
from RL_algorithm.ReplayBuffer import ReplayBuffer_LSTM as Memory_LSTM


class DDPG:
    def __init__(self, n_states, n_actions, hidden_dim_1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = Critic(n_states, n_actions, hidden_dim_1).to(self.device)
        self.actor = Actor(n_states, n_actions, hidden_dim_1).to(self.device)
        self.target_critic = Critic(n_states, n_actions, hidden_dim_1).to(self.device)
        self.target_actor = Actor(n_states, n_actions, hidden_dim_1).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.memory = Memory()
        self.batch_size = 512  # 1024
        self.soft_tau = 0.01
        self.gamma = 0.9  # 0.9
        self.sigma = 0.08
        self.action_dim = n_actions

        self.critic_loss = 0.0
        self.actor_loss = 0.0
        self.total_loss = 0.0

    def choose_balance_action(self, if_train, state):
        action = 0
        if if_train:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.actor(state)
            action = action.cpu().detach().numpy()[0]
            action = action + self.sigma * random.uniform(-1, 1)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.actor(state).item()
            action = action
        return action * 2

    def update(self, transition_dict):

        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action'], dtype=torch.float).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        # displacement = torch.tensor(transition_dict['displacement'], dtype=torch.float).view(-1, 1).to(self.device)
        done = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)
        next_q_value = self.target_critic(next_state, self.target_actor(next_state))
        q_targets = reward + self.gamma * next_q_value * (1.0 - done)
        critic_loss = torch.mean(F.mse_loss(self.critic(state, action), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(state, self.actor(state)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_loss = critic_loss.item()
        self.actor_loss = actor_loss.item()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save(self, path):
        torch.save(self.actor, path + '.pth')
        # input_tensor = torch.randn(7, 2, 64)
        # torch.onnx.export(ex_actor, input_tensor, path + '.onnx', verbose=True)

    def load_from_onnx(self, path):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        self.actor.load_state_dict = onnx.load(path)

        self.actor.eval()

    def load(self, path):
        self.actor = torch.load(path)
        # self.actor.eval()
        self.actor.train()
        self.target_actor = torch.load(path)
        self.target_actor.eval()


class DDPG_LSTM:
    def __init__(self, n_states, n_actions, hidden_size, num_layers):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic_lstm = Critic_LSTM(n_states, n_actions, hidden_size, num_layers, 1).to(self.device)
        # self.actor_lstm = Actor_LSTM(n_states, n_actions, hidden_size, 3, 2).to(self.device)
        self.actor_lstm = Actor_LSTM(n_states, hidden_size, num_layers, 2).to(self.device)

        self.target_critic_lstm = Critic_LSTM(n_states, n_actions, hidden_size, num_layers, 1).to(self.device)
        # self.target_actor_lstm = Actor_LSTM(n_states, n_actions, hidden_size, 3, 2).to(self.device)
        self.target_actor_lstm = Actor_LSTM(n_states, hidden_size, num_layers, 2).to(self.device)

        self.target_critic_lstm.load_state_dict(self.critic_lstm.state_dict())
        self.target_actor_lstm.load_state_dict(self.actor_lstm.state_dict())
        # self.critic_optimizer = optim.Adam(self.critic_lstm.parameters(), lr=3e-3)
        # self.actor_optimizer = optim.Adam(self.actor_lstm.parameters(), lr=3e-4)
        self.memory = Memory_LSTM()
        self.batch_size = 256
        self.soft_tau = 0.01
        self.gamma = 0.99  # 0.9
        self.sigma = 0.08
        self.action_dim = n_actions

        self.critic_loss = 0.0
        self.actor_loss = 0.0
        self.total_loss = 0.0

        self.sequence_length = 64

        # for lstm
        # 测试时候输入为（1，self.sequence_length，35）的张量，self.sequence_length为序列长度，35为state输入
        # 训练时候为（batch_size,self.sequence_length,35）的张量
        self.test_queue = torch.empty(1, 1, n_states)  # 35   测试时候输入为（1，self.sequence_length，35）的张量，self.sequence_length为序列长度，35为state输入

    def update(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action'], dtype=torch.float).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        # displacement = torch.tensor(transition_dict['displacement'], dtype=torch.float).view(-1, 1).to(self.device)
        # done = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

        # 将[512, 35]的state
        state = state.view(-1, self.sequence_length, 35)
        action = action.view(-1, self.sequence_length, 2)

        reward = reward.view(-1, self.sequence_length, 1)
        next_state = next_state.view(-1, self.sequence_length, 35)

        next_q_value = self.target_critic_lstm(next_state, self.target_actor_lstm(next_state))
        q_targets = reward + self.gamma * next_q_value  # * (1.0 - done)
        critic_loss = torch.mean(F.mse_loss(self.critic_lstm(state, action), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic_lstm(state, self.actor_lstm(state)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_loss = critic_loss.item()
        self.actor_loss = actor_loss.item()

        self.soft_update(self.actor_lstm, self.target_actor_lstm)  # 软更新策略网络
        self.soft_update(self.critic_lstm, self.target_critic_lstm)

    def choose_navi_action(self, if_train, state):
        action = 0
        if if_train:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            state = state.unsqueeze(1)  # 由[1,35]扩张为[1,1,35]
            self.test_queue = self.test_queue.to('cuda:0')
            state = state.to('cuda:0')
            self.test_queue = torch.cat((self.test_queue, state), dim=1)
            if self.test_queue.size(1) >= self.sequence_length:
                # 取最新的self.sequence_length个元素
                window = self.test_queue[:self.sequence_length, :]
                action = self.actor_lstm(window, False)
                action = action.cpu().detach().numpy()[0]
                action = action + self.sigma * random.uniform(-1, 1)
                # 如果宽度大于或等于self.sequence_length，我们只保留最后的self.sequence_length个时间步长
                self.test_queue = self.test_queue[:, -self.sequence_length:]
            else:
                action = torch.empty(1, 2)
                action = action.cpu().detach().numpy()[0]

        # TODO: fix this
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.actor(state).item()
            action = action

        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save(self, path):
        torch.save(self.actor_lstm, path + '.pth')

        # torch.onnx.export(ex_actor, input_tensor, path + '.onnx', verbose=True)

    def load(self, path):
        self.actor_lstm = torch.load(path)
        # self.actor.eval()
        self.actor_lstm.train()
        self.target_actor_lstm = torch.load(path)
        self.target_actor_lstm.train()
