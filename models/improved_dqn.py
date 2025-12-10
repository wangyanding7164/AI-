# models/improved_dqn.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import heapq


class PrioritizedReplayBuffer:
    """优先级经验回放"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, valid_moves):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.pos] = (state, action, reward, next_state, done, valid_moves)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """采样经验"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 根据概率采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class ImprovedDQN(nn.Module):
    """改进的DQN网络"""

    def __init__(self, board_size=9, action_size=81):
        super(ImprovedDQN, self).__init__()

        # 残差块
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                residual = x
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.bn2(self.conv2(x))
                x += residual
                return F.relu(x)

        # 网络结构
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # 价值流
        self.value_conv = nn.Conv2d(128, 1, 1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # 优势流
        self.advantage_conv = nn.Conv2d(128, action_size, 1)
        self.advantage_fc = nn.Linear(board_size * board_size, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))

        # Dueling DQN: 价值流 + 优势流
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)

        advantage = F.relu(self.advantage_conv(x))
        advantage = advantage.view(advantage.size(0), -1)
        advantage = self.advantage_fc(advantage)

        # 合并: Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values