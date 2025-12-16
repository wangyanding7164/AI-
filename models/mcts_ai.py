# models/mcts_ai.py
import numpy as np
import math
import random
from models.base_ai import BaseAI
from collections import defaultdict
import time


class MCTSNode:
    """MCTS节点"""

    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = self._get_legal_actions(state)

    def _get_legal_actions(self, state):
        """获取合法动作"""
        n = state.shape[0]
        legal_actions = []
        for y in range(n):
            for x in range(n):
                if state[y][x] == 0:
                    legal_actions.append(y * n + x)
        return legal_actions

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """检查是否终局"""
        return self._check_winner() != 0 or len(self._get_legal_actions(self.state)) == 0

    def _check_winner(self):
        """检查获胜者"""
        n = self.state.shape[0]

        for y in range(n):
            for x in range(n):
                if self.state[y][x] != 0:
                    player = self.state[y][x]

                    # 检查四个方向
                    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                        count = 1
                        for i in range(1, 5):
                            nx, ny = x + dx * i, y + dy * i
                            if 0 <= nx < n and 0 <= ny < n and self.state[ny][nx] == player:
                                count += 1
                            else:
                                break

                        if count >= 5:
                            return player

        return 0  # 无获胜者

    def best_child(self, c_param=1.4):
        """选择最佳子节点（UCB公式）"""
        choices_weights = [
            (child.wins / child.visits) +
            c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]


class MCTSAI(BaseAI):
    """蒙特卡洛树搜索AI"""

    def __init__(self, board_size=9, player=1, iterations=1000, simulation_depth=50, debug=False):
        super().__init__(player, "MCTSAI", board_size)
        self.iterations = iterations
        self.simulation_depth = simulation_depth
        self.debug = debug

    def get_move(self, game_state, valid_moves):
        """MCTS选择动作"""
        start_time = time.time()
        root = MCTSNode(game_state.copy(), self.player)

        for i in range(self.iterations):
            node = self._tree_policy(root)
            winner = self._default_policy(node)
            self._backup(node, winner)

        # 选择访问次数最多的子节点
        best_child = max(root.children, key=lambda c: c.visits)

        if self.debug:
            print(f"[MCTS] 搜索{self.iterations}次, 耗时{time.time() - start_time:.2f}s")
            print(
                f"[MCTS] 最佳动作: {best_child.action}, 访问次数: {best_child.visits}, 胜率: {best_child.wins / best_child.visits:.2%}")

        return best_child.action

    def _tree_policy(self, node):
        """树策略：选择/扩展节点"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                node = node.best_child()
        return node

    def _expand(self, node):
        """扩展节点"""
        action = node.untried_actions.pop()
        next_state = self._make_move(node.state.copy(), action, node.player)
        next_player = 3 - node.player
        child_node = MCTSNode(next_state, next_player, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def _default_policy(self, node):
        """默认策略：随机模拟"""
        state = node.state.copy()
        player = node.player

        for _ in range(self.simulation_depth):
            # 检查是否结束
            winner = self._check_winner_sim(state)
            if winner != 0:
                return 1 if winner == player else -1

            legal_actions = self._get_legal_actions_sim(state)
            if not legal_actions:
                return 0  # 平局

            # 随机选择动作
            action = random.choice(legal_actions)
            state = self._make_move(state, action, player)
            player = 3 - player

        return 0  # 未分胜负

    def _backup(self, node, result):
        """反向传播"""
        while node is not None:
            node.visits += 1
            if result == 1:  # 获胜
                node.wins += 1
            elif result == 0:  # 平局
                node.wins += 0.5
            node = node.parent

    def _make_move(self, state, action, player):
        """执行动作"""
        n = int(math.sqrt(len(state.flatten())))
        x, y = action % n, action // n
        new_state = state.copy()
        new_state[y][x] = player
        return new_state

    def _get_legal_actions_sim(self, state):
        """模拟中的合法动作"""
        n = state.shape[0]
        legal_actions = []
        for y in range(n):
            for x in range(n):
                if state[y][x] == 0:
                    legal_actions.append(y * n + x)
        return legal_actions

    def _check_winner_sim(self, state):
        """模拟中的获胜检查"""
        n = state.shape[0]

        for y in range(n):
            for x in range(n):
                if state[y][x] != 0:
                    player = state[y][x]

                    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                        count = 1
                        for i in range(1, 5):
                            nx, ny = x + dx * i, y + dy * i
                            if 0 <= nx < n and 0 <= ny < n and state[ny][nx] == player:
                                count += 1
                            else:
                                break

                        if count >= 5:
                            return player

        return 0