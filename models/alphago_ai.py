# models/alphago_ai.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ResidualBlock(nn.Module):
    """残差块"""

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


class PolicyValueNetwork(nn.Module):
    """AlphaGo风格的策略价值网络"""

    def __init__(self, board_size=9, action_size=81, num_channels=128, num_blocks=10):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # 输入层 (17个历史平面)
        self.conv_input = nn.Conv2d(17, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # 残差塔
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 共同特征提取
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.residual_tower(x)

        # 策略输出
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # 价值输出
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def preprocess_state(self, state_history, current_player):
        """
        预处理状态为17个平面
        平面0-7: 当前玩家最近8步历史
        平面8-15: 对手最近8步历史
        平面16: 当前玩家颜色 (全1或全0)
        """
        board_size = self.board_size
        state = np.zeros((17, board_size, board_size), dtype=np.float32)

        # 填充历史平面
        for i in range(min(8, len(state_history))):
            board = state_history[-(i + 1)]
            # 当前玩家历史
            state[i] = (board == current_player).astype(np.float32)
            # 对手历史
            state[8 + i] = (board == (3 - current_player)).astype(np.float32)

        # 当前玩家颜色平面
        state[16] = np.ones((board_size, board_size), dtype=np.float32)

        return torch.FloatTensor(state).unsqueeze(0)