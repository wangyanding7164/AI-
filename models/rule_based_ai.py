# models/rule_based_ai.py
import numpy as np
from models.base_ai import BaseAI


class RuleBasedAI(BaseAI):
    """平衡攻防的规则AI"""

    def __init__(self, player=1, board_size=9, aggression=0.3, debug=False):
        """
        player: 玩家编号（1或2）
        board_size: 棋盘大小
        aggression: 攻击性参数（0.0-1.0）
                    0.0: 纯防守
                    0.5: 平衡
                    1.0: 纯攻击
        debug: 调试模式
        """
        super().__init__(player, "BalancedRuleAI", board_size)
        self.aggression = max(0.0, min(1.0, aggression))  # 限制在0-1之间
        self.debug = debug

        if self.debug:
            print(f"[AI初始化] 玩家: {self.player}, 攻击性: {self.aggression:.2f}")

    def _log(self, message):
        if self.debug:
            print(f"[AI] {message}")

    def get_move(self, game_state, valid_moves):
        """平衡攻防的决策逻辑"""
        n = self.board_size

        if self.debug:
            self._log(f"第{getattr(self, 'move_count', 0) + 1}步决策 (攻击性: {self.aggression:.2f})")
            self._log("当前棋盘:")
            for y in range(n):
                row = []
                for x in range(n):
                    if game_state[y][x] == 0:
                        row.append('.')
                    elif game_state[y][x] == 1:
                        row.append('X')
                    else:
                        row.append('O')
                self._log(f"  {''.join(row)}")

        # 1. 检查立即获胜（活四、冲四）
        win_move = self._find_immediate_win(game_state, valid_moves)
        if win_move is not None:
            self._log(f"立即获胜: {win_move}")
            return win_move

        # 2. 检查必须防守的威胁（根据攻击性调整）
        defense_urgency = 1.0 - self.aggression  # 攻击性越低，防守越紧急

        # 2.1 必须防守的威胁（无论攻击性如何）
        must_defend = self._find_must_defend_threats(game_state, valid_moves)
        if must_defend is not None:
            self._log(f"必须防守: {must_defend}")
            return must_defend

        # 2.2 根据攻击性决定是否防守普通威胁
        if defense_urgency > 0.3:  # 攻击性较低时才防守
            normal_defense = self._find_normal_threats(game_state, valid_moves)
            if normal_defense is not None:
                self._log(f"防守威胁: {normal_defense}")
                return normal_defense

        # 3. 寻找攻击机会（根据攻击性调整优先级）
        if self.aggression > 0.3:  # 攻击性较高时才积极攻击
            attack_move = self._find_attack_opportunity(game_state, valid_moves)
            if attack_move is not None:
                self._log(f"攻击机会: {attack_move}")
                return attack_move

        # 4. 选择战略位置
        strategic_move = self._find_strategic_position(game_state, valid_moves)
        if strategic_move is not None:
            self._log(f"战略位置: {strategic_move}")
            return strategic_move

        # 5. 随机选择
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) > 0:
            action = valid_indices[0]
            x, y = action % n, action // n
            self._log(f"随机选择: ({x}, {y})")
            return action

        return None

    def _find_immediate_win(self, board, valid_moves):
        """寻找立即获胜的位置（活四、冲四）"""
        n = self.board_size

        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            # 模拟自己落子
            temp_board = board.copy()
            temp_board[y][x] = self.player

            # 检查四个方向
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count = 1

                # 正向
                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.player:
                        count += 1
                    else:
                        break

                # 反向
                for i in range(1, 5):
                    nx, ny = x - dx * i, y - dy * i
                    if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.player:
                        count += 1
                    else:
                        break

                if count >= 5:  # 五连获胜
                    return action

                # 活四检查（两端开口的四连）
                if count == 4:
                    front_open = (0 <= x + dx * 4 < n and 0 <= y + dy * 4 < n and
                                  board[y + dy * 4][x + dx * 4] == 0)
                    back_open = (0 <= x - dx < n and 0 <= y - dy < n and
                                 board[y - dy][x - dx] == 0)
                    if front_open or back_open:  # 活四
                        return action

        return None

    def _find_must_defend_threats(self, board, valid_moves):
        """寻找必须防守的威胁（对手的活四、冲四）"""
        n = self.board_size

        # 检查对手立即获胜
        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            temp_board = board.copy()
            temp_board[y][x] = self.opponent

            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count = 1

                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.opponent:
                        count += 1
                    else:
                        break

                for i in range(1, 5):
                    nx, ny = x - dx * i, y - dy * i
                    if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.opponent:
                        count += 1
                    else:
                        break

                if count >= 5:  # 对手五连
                    return action

        # 检查对手的活四
        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            # 检查这个位置是否在对手的活四上
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                # 统计对手在这个方向上的连续棋子
                opponent_count = 0

                # 正向
                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < n and 0 <= ny < n and board[ny][nx] == self.opponent:
                        opponent_count += 1
                    else:
                        break

                # 反向
                for i in range(1, 5):
                    nx, ny = x - dx * i, y - dy * i
                    if 0 <= nx < n and 0 <= ny < n and board[ny][nx] == self.opponent:
                        opponent_count += 1
                    else:
                        break

                if opponent_count >= 4:  # 对手有四连
                    # 检查是否是活四
                    front_open = (0 <= x + dx * (opponent_count + 1) < n and
                                  0 <= y + dy * (opponent_count + 1) < n and
                                  board[y + dy * (opponent_count + 1)][x + dx * (opponent_count + 1)] == 0)
                    back_open = (0 <= x - dx < n and 0 <= y - dy < n and
                                 board[y - dy][x - dx] == 0)

                    if front_open or back_open:  # 活四威胁
                        return action

        return None

    def _find_normal_threats(self, board, valid_moves):
        """寻找普通威胁（对手的活三、活二等）"""
        n = self.board_size

        # 按威胁程度排序：活三 > 活二
        threat_levels = [
            (3, 1000),  # 活三
            (2, 100),  # 活二
        ]

        best_score = -1
        best_action = None

        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n
            threat_score = 0

            # 检查这个位置是否在对手的威胁线上
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                # 统计对手连续棋子
                opponent_count = 0

                # 正向
                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < n and 0 <= ny < n and board[ny][nx] == self.opponent:
                        opponent_count += 1
                    else:
                        break

                # 反向
                for i in range(1, 5):
                    nx, ny = x - dx * i, y - dy * i
                    if 0 <= nx < n and 0 <= ny < n and board[ny][nx] == self.opponent:
                        opponent_count += 1
                    else:
                        break

                # 根据连续棋子数评分
                for min_count, score in threat_levels:
                    if opponent_count >= min_count:
                        # 检查是否是活棋
                        front_open = (0 <= x + dx * (opponent_count + 1) < n and
                                      0 <= y + dy * (opponent_count + 1) < n and
                                      board[y + dy * (opponent_count + 1)][x + dx * (opponent_count + 1)] == 0)
                        back_open = (0 <= x - dx < n and 0 <= y - dy < n and
                                     board[y - dy][x - dx] == 0)

                        if front_open or back_open:  # 活棋
                            threat_score = max(threat_score, score)

            if threat_score > best_score:
                best_score = threat_score
                best_action = action

        return best_action if best_score > 0 else None

    def _find_attack_opportunity(self, board, valid_moves):
        """寻找攻击机会"""
        n = self.board_size
        best_score = -1
        best_action = None

        # 攻击优先级：活四 > 活三 > 活二
        attack_levels = [
            (4, 10000),  # 活四
            (3, 1000),  # 活三
            (2, 100),  # 活二
        ]

        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            # 模拟自己落子
            temp_board = board.copy()
            temp_board[y][x] = self.player

            attack_score = 0

            # 检查四个方向
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count = 1

                # 统计自己连续棋子
                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.player:
                        count += 1
                    else:
                        break

                for i in range(1, 5):
                    nx, ny = x - dx * i, y - dy * i
                    if 0 <= nx < n and 0 <= ny < n and temp_board[ny][nx] == self.player:
                        count += 1
                    else:
                        break

                # 根据形成的棋形评分
                for min_count, score in attack_levels:
                    if count >= min_count:
                        # 检查是否是活棋
                        front_open = (0 <= x + dx * count < n and 0 <= y + dy * count < n and
                                      board[y + dy * count][x + dx * count] == 0)
                        back_open = (0 <= x - dx < n and 0 <= y - dy < n and
                                     board[y - dy][x - dx] == 0)

                        if front_open or back_open:  # 活棋
                            attack_score += score
                            break

            if attack_score > best_score:
                best_score = attack_score
                best_action = action

        return best_action if best_score > 0 else None

    def _find_strategic_position(self, board, valid_moves):
        """寻找战略位置"""
        n = self.board_size
        center = n // 2
        best_score = -1
        best_action = None

        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            score = 0

            # 1. 中心偏好
            distance_to_center = abs(x - center) + abs(y - center)
            center_score = (n - distance_to_center) * 10
            score += center_score

            # 2. 靠近已有棋子（但不要太近）
            for dy in [-2, -1, 0, 1, 2]:
                for dx in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if board[ny][nx] == self.player:
                            # 自己棋子，适当靠近
                            score += 5
                        elif board[ny][nx] == self.opponent:
                            # 对手棋子，保持距离
                            score += 2

            # 3. 开局策略：优先中心
            total_pieces = np.sum(board != 0)
            if total_pieces < 4:  # 开局阶段
                score += center_score * 2

            if score > best_score:
                best_score = score
                best_action = action

        return best_action