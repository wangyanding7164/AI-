# models/random_ai.py
import numpy as np


class RandomAI:
    """完全随机落子的AI"""

    def __init__(self, player=1):
        self.player = player

    def get_move(self, game_state, valid_moves):
        """
        参数:
            game_state: 棋盘状态 (n, n) 数组
            valid_moves: 合法移动掩码 (n*n,) 数组
        返回:
            action: 选择的落子位置 (0 到 n*n-1)
        """
        # 获取所有合法移动
        valid_indices = [i for i, valid in enumerate(valid_moves) if valid]

        if not valid_indices:
            return None

        # 完全随机选择
        return np.random.choice(valid_indices)

    def train_step(self, *args, **kwargs):
        """随机AI不需要训练"""
        pass