import pygame
import sys
import numpy as np
import os
class GomokuGame:
    def __init__(self, board_size=9):
        self.n = board_size
        self.board = None
        self.current_player = None
        self.done = False
        self.winner = None
        self.ai_player = None
        self.human_player = None
        self.reset()

    def set_players(self, human_is_black=True):
        """设置人类和AI的角色"""
        if human_is_black:
            self.human_player = 1  # 人类黑棋
            self.ai_player = 2  # AI白棋
        else:
            self.human_player = 2  # 人类白棋
            self.ai_player = 1  # AI黑棋
        print(f"人类执{'黑' if human_is_black else '白'}棋, AI执{'白' if human_is_black else '黑'}棋")

    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.current_player = 1  # 黑棋先手
        self.done = False
        self.winner = None
        return self.get_state()

    def is_valid_move(self, action):
        """判断落子是否合法"""
        x, y = action % self.n, action // self.n
        return (0 <= x < self.n and 0 <= y < self.n and self.board[y][x] == 0)

    def make_move(self, action):
        """执行落子"""
        if not self.is_valid_move(action) or self.done:
            raise ValueError("非法移动或游戏已结束")

        x, y = action % self.n, action // self.n
        self.board[y][x] = self.current_player

        if self.check_win(y, x):
            self.done = True
            self.winner = self.current_player
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
        else:
            self.current_player = 3 - self.current_player

        return self.get_state(), self.get_reward(), self.done, {}

    def check_win(self, y, x):
        """检查是否获胜"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player = self.board[y][x]

        for dy, dx in directions:
            count = 1
            for i in range(1, 5):
                ny, nx = y + dy * i, x + dx * i
                if 0 <= ny < self.n and 0 <= nx < self.n and self.board[ny][nx] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                ny, nx = y - dy * i, x - dx * i
                if 0 <= ny < self.n and 0 <= nx < self.n and self.board[ny][nx] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def get_state(self):
        return self.board.copy()

    def get_reward(self):
        if not self.done:
            return 0
        if self.winner == 1:
            return 1
        elif self.winner == 2:
            return -1
        else:
            return 0.1

    def get_valid_moves(self):
        return np.array([1 if self.is_valid_move(i) else 0 for i in range(self.n * self.n)])

    def get_current_player_type(self):
        """获取当前玩家类型"""
        if self.ai_player is None or self.human_player is None:
            return 'human'  # 默认
        return 'human' if self.current_player == self.human_player else 'ai'
