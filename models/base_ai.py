import numpy as np
import abc


class BaseAI(abc.ABC):
    """AI基类，支持可变棋盘尺寸"""

    def __init__(self, player=1, name="BaseAI", board_size=9):
        self.player = player
        self.name = name
        self.opponent = 3 - player
        self.board_size = board_size
        self.action_size = board_size * board_size

    @abc.abstractmethod
    def get_move(self, game_state, valid_moves):
        """获取移动"""
        pass

    def reset(self):
        """重置AI状态"""
        pass

    def __str__(self):
        return f"{self.name}(Player={self.player}, BoardSize={self.board_size})"