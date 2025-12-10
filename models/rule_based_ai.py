import numpy as np
from models.base_ai import BaseAI


class RuleBasedAI(BaseAI):
    """基于规则的AI，支持可变棋盘尺寸"""

    def __init__(self, player=1, board_size=9, aggression=0.7):
        super().__init__(player, "RuleBasedAI", board_size)
        self.aggression = aggression
        self.pattern_scores = self._init_pattern_scores()

    def _init_pattern_scores(self):
        """初始化棋型评分表"""
        return {
            'five': 100000,  # 五连
            'open_four': 10000,  # 活四
            'half_four': 5000,  # 冲四
            'open_three': 1000,  # 活三
            'half_three': 500,  # 眠三
            'open_two': 100,  # 活二
            'half_two': 10,  # 眠二
        }

    def evaluate_position(self, board, x, y, player):
        """评估位置价值"""
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            score += self._evaluate_direction(board, x, y, dx, dy, player)
        return score

    def _evaluate_direction(self, board, x, y, dx, dy, player):
        """评估单个方向 - 修正：移除了多余的self参数"""
        n = self.board_size
        count = 1
        open_ends = 0

        # 正向检查
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < n and 0 <= ny < n:
                if board[ny][nx] == player:
                    count += 1
                elif board[ny][nx] == 0:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break

        # 反向检查
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if 0 <= nx < n and 0 <= ny < n:
                if board[ny][nx] == player:
                    count += 1
                elif board[ny][nx] == 0:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break

        # 根据棋型评分
        if count >= 5:
            return self.pattern_scores['five']
        elif count == 4:
            if open_ends == 2:
                return self.pattern_scores['open_four']
            elif open_ends == 1:
                return self.pattern_scores['half_four']
        elif count == 3:
            if open_ends == 2:
                return self.pattern_scores['open_three']
            elif open_ends == 1:
                return self.pattern_scores['half_three']
        elif count == 2:
            if open_ends == 2:
                return self.pattern_scores['open_two']
            elif open_ends == 1:
                return self.pattern_scores['half_two']
        return 0

    def get_move(self, game_state, valid_moves):
        """获取最佳移动"""
        best_score = -float('inf')
        best_moves = []

        valid_indices = np.where(valid_moves == 1)[0]

        for action in valid_indices:
            x, y = action % self.board_size, action // self.board_size

            # 攻击和防守评分
            attack_score = self.evaluate_position(game_state, x, y, self.player)
            defense_score = self.evaluate_position(game_state, x, y, self.opponent)

            total_score = (attack_score * self.aggression +
                           defense_score * (1 - self.aggression))

            # 中心偏好
            center_bonus = self._get_center_bonus(x, y)
            total_score += center_bonus

            if total_score > best_score:
                best_score = total_score
                best_moves = [action]
            elif abs(total_score - best_score) < 1e-6:
                best_moves.append(action)

        return np.random.choice(best_moves) if best_moves else None

    def _get_center_bonus(self, x, y):
        """中心位置奖励"""
        center = self.board_size // 2
        distance = abs(x - center) + abs(y - center)
        return (self.board_size - distance) * 5