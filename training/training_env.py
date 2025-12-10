# training/training_env.py
import numpy as np
import sys
import os


class TrainingGomokuEnv:
    """支持可变棋盘尺寸的训练环境"""

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.board = None
        self.current_player = 1
        self.done = False
        self.winner = None
        self.reset()

    def reset(self):
        """重置环境"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def is_valid_move(self, action):
        """检查移动是否合法"""
        if action is None:
            return False
        x, y = action % self.board_size, action // self.board_size
        return (0 <= x < self.board_size and 0 <= y < self.board_size and
                self.board[y][x] == 0)

    def step(self, action):
        """执行一步动作 - 强化学习标准接口"""
        if not self.is_valid_move(action) or self.done:
            return self.get_state(), 0, True, {}

        x, y = action % self.board_size, action // self.board_size
        self.board[y][x] = self.current_player

        # 检查游戏状态
        if self.check_win(y, x):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif np.all(self.board != 0):  # 平局
            self.done = True
            self.winner = 0
            reward = 0.1
        else:
            self.current_player = 3 - self.current_player
            reward = 0.0

        return self.get_state(), reward, self.done, {}

    def check_win(self, y, x):
        """检查是否获胜"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player = self.board[y][x]

        for dy, dx in directions:
            count = 1
            # 正向检查
            for i in range(1, 5):
                ny, nx = y + dy * i, x + dx * i
                if 0 <= ny < self.board_size and 0 <= nx < self.board_size and self.board[ny][nx] == player:
                    count += 1
                else:
                    break
            # 反向检查
            for i in range(1, 5):
                ny, nx = y - dy * i, x - dx * i
                if 0 <= ny < self.board_size and 0 <= nx < self.board_size and self.board[ny][nx] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def get_state(self):
        """获取当前状态"""
        return self.board.copy()

    def get_valid_moves(self):
        """获取合法移动掩码"""
        return np.array([1 if self.is_valid_move(i) else 0
                         for i in range(self.board_size * self.board_size)])

    def get_reward(self):
        """获取奖励"""
        if not self.done:
            return 0
        if self.winner == 1:
            return 1
        elif self.winner == 2:
            return -1
        else:
            return 0.1