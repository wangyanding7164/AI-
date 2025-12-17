# models/weak_rule_ai.py
import numpy as np
import random
from models.base_ai import BaseAI


class LearningFriendlyAI(BaseAI):
    """
    学习友好型AI - 专门为DQN学习设计
    特点：从最简单开始，逐步展示更复杂的概念
    """

    def __init__(self, player=1, board_size=9, skill_level=0.3, debug=False):
        """
        player: 玩家编号
        skill_level: 技能等级 (0.1-1.0) - 为了保持接口兼容性
                    0.1: 概念1级 (只理解中心)
                    0.3: 概念2级 (理解连接)
                    0.5: 概念3级 (理解防守)
                    0.8: 概念4级 (理解攻击)
        debug: 调试模式
        """
        # 将skill_level映射到concept_level
        if skill_level <= 0.2:
            concept_level = 1
        elif skill_level <= 0.4:
            concept_level = 2
        elif skill_level <= 0.6:
            concept_level = 3
        else:
            concept_level = 4

        super().__init__(player, f"LearnAI_L{concept_level}", board_size)
        self.concept_level = concept_level
        self.skill_level = skill_level  # 保持兼容性
        self.debug = debug

        if debug:
            print(f"[学习友好AI] 等级{concept_level}, 技能等级{skill_level:.1f}, 玩家{player}")

    def get_move(self, game_state, valid_moves):
        """根据概念等级选择动作"""
        n = self.board_size

        # 获取所有合法动作
        valid_actions = np.where(valid_moves == 1)[0]
        if len(valid_actions) == 0:
            return None

        # 1级概念：中心优先
        if self.concept_level == 1:
            return self._concept_level_1(game_state, valid_actions)

        # 2级概念：连接优先
        elif self.concept_level == 2:
            return self._concept_level_2(game_state, valid_actions)

        # 3级概念：阻止对手
        elif self.concept_level == 3:
            return self._concept_level_3(game_state, valid_actions)

        # 4级概念：简单攻击
        else:  # level 4
            return self._concept_level_4(game_state, valid_actions)

    def _concept_level_1(self, board, valid_actions):
        """等级1：只理解'好位置' - 中心优先"""
        n = self.board_size
        center = n // 2

        # 计算每个位置的中心得分
        scores = []
        for action in valid_actions:
            x, y = action % n, action // n
            # 距离中心越近，分数越高
            distance = abs(x - center) + abs(y - center)
            score = (n * 2) - distance  # 距离越近，分数越高

            # 随机波动，让AI不总是下相同位置
            score *= random.uniform(0.9, 1.1)
            scores.append(score)

        # 选择最高分
        best_idx = np.argmax(scores)
        return valid_actions[best_idx]

    def _concept_level_2(self, board, valid_actions):
        """等级2：理解'连接' - 靠近自己的棋子"""
        n = self.board_size

        scores = []
        for action in valid_actions:
            x, y = action % n, action // n
            score = 0

            # 1. 中心偏好
            center = n // 2
            distance = abs(x - center) + abs(y - center)
            score += (n * 2) - distance

            # 2. 连接奖励：靠近自己的棋子
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if board[ny][nx] == self.player:
                            score += 20  # 靠近自己棋子有大奖励
                        elif board[ny][nx] == self.opponent:
                            score += 5  # 靠近对手棋子有小奖励

            # 随机波动
            score *= random.uniform(0.8, 1.2)
            scores.append(score)

        best_idx = np.argmax(scores)
        return valid_actions[best_idx]

    def _concept_level_3(self, board, valid_actions):
        """等级3：理解'阻止' - 阻止对手形成三个连子"""
        n = self.board_size

        scores = []
        for action in valid_actions:
            x, y = action % n, action // n
            score = 0

            # 1. 中心偏好
            center = n // 2
            distance = abs(x - center) + abs(y - center)
            score += (n * 2) - distance

            # 2. 连接奖励
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if board[ny][nx] == self.player:
                            score += 15
                        elif board[ny][nx] == self.opponent:
                            score += 3

            # 3. 防守检查：这个位置是否阻止对手？
            defense_score = self._check_defense_value(board, x, y)
            score += defense_score

            # 随机波动
            score *= random.uniform(0.7, 1.3)
            scores.append(score)

        best_idx = np.argmax(scores)
        return valid_actions[best_idx]

    def _concept_level_4(self, board, valid_actions):
        """等级4：理解'攻击' - 尝试形成三个连子"""
        n = self.board_size

        scores = []
        for action in valid_actions:
            x, y = action % n, action // n
            score = 0

            # 1. 中心偏好
            center = n // 2
            distance = abs(x - center) + abs(y - center)
            score += (n * 2) - distance

            # 2. 连接奖励
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if board[ny][nx] == self.player:
                            score += 12
                        elif board[ny][nx] == self.opponent:
                            score += 2

            # 3. 防守价值
            defense_score = self._check_defense_value(board, x, y)
            score += defense_score

            # 4. 攻击价值
            attack_score = self._check_attack_value(board, x, y)
            score += attack_score

            # 随机波动
            score *= random.uniform(0.6, 1.4)
            scores.append(score)

        best_idx = np.argmax(scores)
        return valid_actions[best_idx]

    def _check_defense_value(self, board, x, y):
        """检查防守价值：这个位置是否阻止对手形成威胁"""
        n = self.board_size
        defense_score = 0

        # 模拟对手在这个位置下棋
        temp_board = board.copy()
        temp_board[y][x] = self.opponent

        # 检查四个方向
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            opponent_count = 1  # 刚下的这个子

            # 正向
            for i in range(1, 4):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.opponent:
                    opponent_count += 1
                else:
                    break

            # 反向
            for i in range(1, 4):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.opponent:
                    opponent_count += 1
                else:
                    break

            # 如果对手能形成3连，这个位置有防守价值
            if opponent_count >= 3:
                defense_score += 30
            elif opponent_count == 2:
                defense_score += 10

        return defense_score

    def _check_attack_value(self, board, x, y):
        """检查攻击价值：这个位置是否能形成攻击"""
        n = self.board_size
        attack_score = 0

        # 模拟自己在这个位置下棋
        temp_board = board.copy()
        temp_board[y][x] = self.player

        # 检查四个方向
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            player_count = 1  # 刚下的这个子

            # 正向
            for i in range(1, 4):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.player:
                    player_count += 1
                else:
                    break

            # 反向
            for i in range(1, 4):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.player:
                    player_count += 1
                else:
                    break

            # 如果能形成3连，有攻击价值
            if player_count >= 3:
                attack_score += 25
            elif player_count == 2:
                attack_score += 8

        return attack_score


# 为了保持完全兼容性，保留原类名
WeakRuleAI = LearningFriendlyAI


# 为了方便使用，创建不同等级的AI
class BabyAI(LearningFriendlyAI):
    """婴儿级AI - 最简单的对手"""

    def __init__(self, player=1, board_size=9, debug=False):
        super().__init__(player, board_size, skill_level=0.1, debug=debug)


class ToddlerAI(LearningFriendlyAI):
    """幼儿级AI - 学习连接概念"""

    def __init__(self, player=1, board_size=9, debug=False):
        super().__init__(player, board_size, skill_level=0.3, debug=debug)


class KidAI(LearningFriendlyAI):
    """儿童级AI - 学习防守概念"""

    def __init__(self, player=1, board_size=9, debug=False):
        super().__init__(player, board_size, skill_level=0.5, debug=debug)


class TeenAI(LearningFriendlyAI):
    """少年级AI - 学习攻击概念"""

    def __init__(self, player=1, board_size=9, debug=False):
        super().__init__(player, board_size, skill_level=0.8, debug=debug)