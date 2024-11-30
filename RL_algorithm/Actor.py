import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim_1, init_w=1e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim_1)
        self.linear2 = nn.Linear(hidden_dim_1, hidden_dim_1)
        self.linear4 = nn.Linear(hidden_dim_1, n_actions)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear4(x)

        x = torch.tanh(x)
        return x


class Actor_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, init_w=1e-4):
        super(Actor_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.fc.weight.data.uniform_(-init_w, init_w)
        self.fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, if_train=True):
        # 初始化隐藏状态h0, c0为全0向量
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 将输入x和隐藏状态(h0, c0)传入LSTM网络
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出作为LSTM网络的输出
        if if_train:
            out = self.fc(out)
        else:
            out = self.fc(out[:, -1, :])
        out = torch.tanh(out)
        return out
