# models/rule_based_ai_simple_working.py
import numpy as np
from models.base_ai import BaseAI


class RuleBasedAI(BaseAI):
    """修复开局策略的规则AI"""

    def __init__(self, player=1, board_size=9, aggression=0.3, debug=False):
        super().__init__(player, "FixedRuleAI", board_size)
        self.aggression = aggression
        self.debug = debug
        self.move_count = 0  # 添加步数计数器

        if self.debug:
            print(f"[AI初始化] 玩家: {self.player}, 攻击性: {self.aggression}")

    def _log(self, message):
        if self.debug:
            print(f"[AI] {message}")

    def get_move(self, game_state, valid_moves):
        """修复决策逻辑 - 确保开局合理"""
        n = self.board_size
        self.move_count += 1

        # 调试：显示当前棋盘
        if self.debug:
            self._log(f"第{self.move_count}步决策")
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

        # 0. 开局策略：前几步强制中心附近
        if self._is_opening_phase(game_state):
            opening_move = self._get_opening_move(game_state, valid_moves)
            if opening_move is not None:
                x, y = opening_move % n, opening_move // n
                self._log(f"开局策略: ({x}, {y})")
                return opening_move

        # 1. 检查立即获胜
        win_move = self._find_immediate_win(game_state, valid_moves)
        if win_move is not None:
            self._log(f"立即获胜: {win_move}")
            return win_move

        # 2. 检查必须防守的威胁
        defense_move = self._find_simple_defense(game_state, valid_moves)
        if defense_move is not None:
            self._log(f"防守威胁: {defense_move}")
            return defense_move

        # 3. 寻找攻击机会
        attack_move = self._find_simple_attack(game_state, valid_moves)
        if attack_move is not None:
            self._log(f"攻击机会: {attack_move}")
            return attack_move

        # 4. 选择最佳战略位置（修复随机选择问题）
        strategic_move = self._find_best_strategic_position(game_state, valid_moves)
        if strategic_move is not None:
            x, y = strategic_move % n, strategic_move // n
            self._log(f"战略位置: ({x}, {y})")
            return strategic_move

        # 5. 最后手段：选择第一个合法位置
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) > 0:
            action = valid_indices[0]
            x, y = action % n, action // n
            self._log(f"最后选择: ({x}, {y})")
            return action

        return None

    def _is_opening_phase(self, game_state):
        """判断是否处于开局阶段"""
        n = self.board_size
        total_pieces = np.sum(game_state != 0)
        return total_pieces <= 4  # 前4步为开局

    def _get_opening_move(self, game_state, valid_moves):
        """获取开局移动 - 绝对优先中心"""
        n = self.board_size
        center = n // 2

        if self.debug:
            self._log(f"开局阶段，棋盘棋子数: {np.sum(game_state != 0)}")

        # 优先级1: 正中心
        if game_state[center][center] == 0:
            action = center * n + center
            if valid_moves[action]:
                if self.debug:
                    self._log(f"选择正中心: ({center}, {center})")
                return action

        # 优先级2: 中心3x3区域，按距离排序
        center_positions = []
        for y in range(max(0, center - 1), min(n, center + 2)):
            for x in range(max(0, center - 1), min(n, center + 2)):
                if game_state[y][x] == 0:
                    action = y * n + x
                    if valid_moves[action]:
                        # 计算距离中心的距离
                        distance = abs(y - center) + abs(x - center)
                        center_positions.append((action, distance, y, x))

        if center_positions:
            # 按距离排序，选择最近的位置
            center_positions.sort(key=lambda x: x[1])
            best_action, best_distance, best_y, best_x = center_positions[0]

            if self.debug:
                self._log(f"选择中心附近: ({best_x}, {best_y}) 距离: {best_distance}")
            return best_action

        # 优先级3: 星位
        star_positions = [
            (center, center - 3), (center, center + 3),  # 上下
            (center - 3, center), (center + 3, center),  # 左右
            (center - 3, center - 3), (center - 3, center + 3),  # 四角
            (center + 3, center - 3), (center + 3, center + 3)
        ]

        for y, x in star_positions:
            if 0 <= y < n and 0 <= x < n and game_state[y][x] == 0:
                action = y * n + x
                if valid_moves[action]:
                    if self.debug:
                        self._log(f"选择星位: ({x}, {y})")
                    return action

        return None

    def _find_best_strategic_position(self, game_state, valid_moves):
        """寻找最佳战略位置 - 修复随机选择问题"""
        n = self.board_size
        center = n // 2

        best_score = -1
        best_action = None
        best_x, best_y = 0, 0

        # 评估所有合法位置
        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            score = 0

            # 1. 中心偏好（距离中心越近越好）
            distance_to_center = abs(x - center) + abs(y - center)
            center_score = (n - distance_to_center) * 10
            score += center_score

            # 2. 靠近已有棋子
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if game_state[ny][nx] == self.player:
                            score += 20
                        elif game_state[ny][nx] == self.opponent:
                            score += 10

            # 3. 边界惩罚
            if x == 0 or x == n - 1 or y == 0 or y == n - 1:
                score -= 5

            if score > best_score or (score == best_score and center_score > best_score):
                best_score = score
                best_action = action
                best_x, best_y = x, y

        if best_action is not None and self.debug:
            self._log(f"战略位置得分: ({best_x}, {best_y}) = {best_score}")

        return best_action

    # 以下是原有方法，保持不变
    def _find_immediate_win(self, board, valid_moves):
        """寻找立即获胜的位置"""
        n = self.board_size

        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            # 模拟落子
            temp_board = board.copy()
            temp_board[y][x] = self.player

            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count = 1

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

                if count >= 5:
                    return action

        return None

    def _find_simple_defense(self, board, valid_moves):
        """简化的防守检测"""
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

                if count >= 5:
                    return action

        # 检查威胁
        threat_positions = []

        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n
            threat_score = self._evaluate_simple_threat(board, x, y)

            if threat_score > 0:
                threat_positions.append((action, threat_score))

        if threat_positions:
            threat_positions.sort(key=lambda x: x[1], reverse=True)
            return threat_positions[0][0]

        return None

    def _evaluate_simple_threat(self, board, x, y):
        """威胁评估"""
        n = self.board_size
        max_threat = 0

        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            threat = self._check_direction_threat_simple(board, x, y, dx, dy)
            max_threat = max(max_threat, threat)

        return max_threat

    def _check_direction_threat_simple(self, board, x, y, dx, dy):
        """检查方向威胁"""
        n = self.board_size
        threat_score = 0

        forward_count = 0
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < n and 0 <= ny < n:
                if board[ny][nx] == self.opponent:
                    forward_count += 1
                else:
                    break
            else:
                break

        backward_count = 0
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if 0 <= nx < n and 0 <= ny < n:
                if board[ny][nx] == self.opponent:
                    backward_count += 1
                else:
                    break
            else:
                break

        total_count = forward_count + backward_count

        if total_count >= 4:
            threat_score = 10000
        elif total_count >= 3:
            threat_score = 5000
        elif total_count >= 2:
            threat_score = 1000

        return threat_score

    def _find_simple_attack(self, board, valid_moves):
        """攻击检测"""
        n = self.board_size
        best_score = -1
        best_action = None

        for action in np.where(valid_moves == 1)[0]:
            x, y = action % n, action // n

            temp_board = board.copy()
            temp_board[y][x] = self.player

            attack_score = 0

            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count = 1

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

                if count >= 5:
                    attack_score += 10000
                elif count >= 4:
                    attack_score += 5000
                elif count >= 3:
                    attack_score += 1000
                elif count >= 2:
                    attack_score += 100

            if attack_score > best_score:
                best_score = attack_score
                best_action = action

        return best_action