# models/dqn_ai_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from models.base_ai import BaseAI


class DynamicDQN(nn.Module):
    """最终版DQN网络"""

    def __init__(self, board_size=9, action_size=81):
        super(DynamicDQN, self).__init__()
        self.board_size = board_size
        self.action_size = action_size

        # 网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

    def preprocess_state(self, board, current_player):
        """预处理状态"""
        board_np = np.array(board, dtype=np.float32)
        n = board_np.shape[0]

        # 创建三个通道
        state_tensor = np.zeros((3, n, n), dtype=np.float32)

        # 通道0: 当前玩家的棋子
        state_tensor[0] = (board_np == current_player).astype(np.float32)

        # 通道1: 对手的棋子
        state_tensor[1] = (board_np == (3 - current_player)).astype(np.float32)

        # 通道2: 当前玩家标识
        state_tensor[2] = np.ones((n, n), dtype=np.float32)

        return torch.FloatTensor(state_tensor).unsqueeze(0)


class DQNAgent(BaseAI):
    """最终版DQN智能体"""

    def __init__(self, board_size=9, player=1, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 target_update=100, memory_size=20000, device='auto'):
        super().__init__(player, "DQNAgent_Final", board_size)

        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.action_size = board_size * board_size

        # 网络初始化
        self.policy_net = DynamicDQN(board_size, self.action_size).to(self.device)
        self.target_net = DynamicDQN(board_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        # 训练参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        # 经验回放
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 64
        self.steps_done = 0
        self.losses = []

    def get_move(self, game_state, valid_moves, training=False):
        """ε-贪婪策略选择动作"""
        state_tensor = self.policy_net.preprocess_state(game_state, self.player)
        state_tensor = state_tensor.to(self.device)

        if training and np.random.random() < self.epsilon:
            # 随机探索
            valid_indices = np.where(valid_moves == 1)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
                return action
            return None

        # 利用策略
        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)[0]
            self.policy_net.train()

            # 掩码非法动作
            q_values_masked = q_values.clone()
            for i in range(self.action_size):
                if valid_moves[i] == 0:
                    q_values_masked[i] = -float('inf')

            return torch.argmax(q_values_masked).item()

    def remember(self, state, action, reward, next_state, done, valid_moves, player):
        """保存经验"""
        self.memory.append((state, action, reward, next_state, done, valid_moves, player))

    def replay(self):
        """修复版经验回放"""
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)

        # 过滤掉无效的样本
        valid_batch = []
        for sample in batch:
            state, action, reward, next_state, done, valid_moves, player = sample
            # 确保所有必要的元素都存在
            if state is not None and action is not None and next_state is not None:
                valid_batch.append(sample)

        if len(valid_batch) < 32:  # 如果有效样本太少，跳过本次训练
            return 0

        # 使用有效样本
        states, actions, rewards, next_states, dones, valid_moves_list, players = zip(*valid_batch)
        actual_batch_size = len(valid_batch)

        try:
            # 预处理状态
            state_tensors = []
            for state, player in zip(states, players):
                state_tensor = self.policy_net.preprocess_state(state, player)
                state_tensors.append(state_tensor)

            states_tensor = torch.cat(state_tensors).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # 当前Q值
            current_q = self.policy_net(states_tensor).gather(1, actions)

            # 计算目标Q值
            with torch.no_grad():
                next_state_tensors = []
                for next_state, player in zip(next_states, players):
                    next_state_tensor = self.policy_net.preprocess_state(next_state, player)
                    next_state_tensors.append(next_state_tensor)

                next_states_tensor = torch.cat(next_state_tensors).to(self.device)
                next_q_values = self.target_net(next_states_tensor)

                # 应用有效动作掩码
                for i, valid_moves in enumerate(valid_moves_list):
                    for j in range(self.action_size):
                        if j < len(valid_moves) and valid_moves[j] == 0:
                            next_q_values[i, j] = -float('inf')

                next_max_q = next_q_values.max(1)[0].unsqueeze(1)

                # 确保张量维度匹配
                if next_max_q.size(0) != rewards.size(0):
                    # 如果维度不匹配，调整到最小尺寸
                    min_size = min(next_max_q.size(0), rewards.size(0))
                    next_max_q = next_max_q[:min_size]
                    rewards = rewards[:min_size]
                    dones = dones[:min_size]

                # 计算目标Q值
                target_q = rewards + self.gamma * next_max_q * (1 - dones)

            # 计算损失
            loss = F.smooth_l1_loss(current_q, target_q)
            self.losses.append(loss.item())

            # 优化
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.steps_done += 1

            # 更新目标网络
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return loss.item()

        except Exception as e:
            print(f"⚠️ 经验回放出错: {e}")
            # 清空有问题的记忆
            self.memory.clear()
            return 0

    def get_valid_moves_mask(self, state, player):
        """获取合法动作掩码"""
        n = self.board_size
        valid_moves = np.zeros(self.action_size, dtype=int)

        for y in range(n):
            for x in range(n):
                if state[y][x] == 0:  # 空位置
                    action = y * n + x
                    valid_moves[action] = 1

        return valid_moves

    def compute_reward(self, game_state, done, winner, player):
        """计算奖励"""
        if not done:
            return 0.0

        if winner == 0:  # 平局
            return 0.1
        elif winner == player:  # 获胜
            return 1.0
        else:  # 失败
            return -1.0

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'board_size': self.board_size,
            'losses': self.losses,
            'player': self.player
        }, path)
        print(f"✅ 模型保存到: {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.board_size = checkpoint.get('board_size', 9)
        self.player = checkpoint.get('player', 1)
        self.losses = checkpoint.get('losses', [])

        self.action_size = self.board_size * self.board_size

        self.policy_net.eval()
        self.target_net.eval()

        print(f"✅ 模型加载成功: {path}")
        print(f"   步数: {self.steps_done}, 探索率: {self.epsilon:.4f}")

    def get_training_info(self):
        """获取训练信息"""
        return {
            'steps': self.steps_done,
            'epsilon': self.epsilon,
            'loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'memory_size': len(self.memory)
        }

