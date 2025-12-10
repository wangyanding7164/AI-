import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from models.base_ai import BaseAI


class DynamicDQN(nn.Module):
    """支持可变棋盘尺寸的DQN网络"""

    def __init__(self, board_size=9, action_size=81):
        super(DynamicDQN, self).__init__()
        self.board_size = board_size
        self.action_size = action_size

        # 动态计算卷积输出尺寸
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        # 全连接层
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def preprocess_state(self, board, current_player):
        """预处理状态为网络输入"""
        board_np = np.array(board, dtype=np.float32)
        n = board_np.shape[0]

        # 创建三个通道
        state_tensor = np.zeros((3, n, n), dtype=np.float32)
        state_tensor[0] = (board_np == current_player).astype(np.float32)  # 当前玩家
        state_tensor[1] = (board_np == (3 - current_player)).astype(np.float32)  # 对手
        state_tensor[2] = np.ones((n, n), dtype=np.float32) * 0.5  # 当前玩家指示

        return torch.FloatTensor(state_tensor).unsqueeze(0)


class DQNAgent(BaseAI):
    """DQN智能体，支持可变棋盘尺寸"""

    def __init__(self, board_size=9, player=1, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super().__init__(player, "DQNAgent", board_size)

        self.action_size = board_size * board_size
        self.policy_net = DynamicDQN(board_size, self.action_size)
        self.target_net = DynamicDQN(board_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.steps_done = 0

    def get_move(self, game_state, valid_moves):
        """ε-贪婪策略选择动作"""
        state_tensor = self.policy_net.preprocess_state(game_state, self.player)

        if np.random.random() < self.epsilon:
            # 随机探索
            valid_indices = np.where(valid_moves == 1)[0]
            return np.random.choice(valid_indices) if len(valid_indices) > 0 else None

        # 利用策略
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
            # 掩码非法动作
            for i in range(self.action_size):
                if valid_moves[i] == 0:
                    q_values[i] = -float('inf')
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done, valid_moves):
        """保存经验"""
        self.memory.append((state, action, reward, next_state, done, valid_moves))

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, valid_moves_list = zip(*batch)

        # 转换为张量
        states = torch.cat([self.policy_net.preprocess_state(s, self.player) for s in states])
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)

        # 当前Q值
        current_q = self.policy_net(states).gather(1, actions)

        # 目标Q值
        with torch.no_grad():
            next_q_values = []
            for i, (next_state, done) in enumerate(zip(next_states, dones)):
                if done:
                    next_q_values.append(torch.zeros(1, 1))
                else:
                    next_state_tensor = self.policy_net.preprocess_state(next_state, self.player)
                    q_values = self.target_net(next_state_tensor)[0]
                    next_q_values.append(torch.max(q_values).unsqueeze(0).unsqueeze(0))

            next_q = torch.cat(next_q_values)
            target_q = rewards + self.gamma * next_q

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_done += 1
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'board_size': self.board_size
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.board_size = checkpoint['board_size']
        self.action_size = self.board_size * self.board_size