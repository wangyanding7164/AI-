# train_dqn_fixed.py
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import random
from datetime import datetime
import argparse
import json
from collections import deque

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn_ai import DQNAgent
from models.rule_based_ai import RuleBasedAI
from models.random_ai import RandomAI


class TrainingEnvironment:
    """训练环境"""

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.reset()

    def reset(self):
        """重置环境"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.step_count = 0
        return self.board.copy()

    def get_valid_moves(self):
        """获取合法移动"""
        n = self.board_size
        valid_moves = np.zeros(n * n, dtype=int)
        for y in range(n):
            for x in range(n):
                if self.board[y][x] == 0:
                    action = y * n + x
                    valid_moves[action] = 1
        return valid_moves

    def is_valid_move(self, action):
        """检查移动是否合法"""
        if action is None:
            return False
        n = self.board_size
        x, y = action % n, action // n
        return 0 <= x < n and 0 <= y < n and self.board[y][x] == 0

    def step(self, action, player):
        """执行一步"""
        if not self.is_valid_move(action) or self.done:
            return self.board.copy(), 0, True, {}

        n = self.board_size
        x, y = action % n, action // n
        self.board[y][x] = player
        self.step_count += 1

        # 检查游戏是否结束
        if self.check_win(x, y, player):
            self.done = True
            self.winner = player
            reward = 1.0
        elif np.all(self.board != 0):  # 平局
            self.done = True
            self.winner = 0
            reward = 0.1
        else:
            self.current_player = 3 - self.current_player
            reward = 0.0

        return self.board.copy(), reward, self.done, {}

    def check_win(self, x, y, player):
        """检查是否获胜"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1

            # 正向
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] == player:
                    count += 1
                else:
                    break

            # 反向
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] == player:
                    count += 1
                else:
                    break

            if count >= 5:
                return True

        return False


class DQNTrainer:
    """DQN训练器"""

    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        # 确保所有必要的配置项都存在
        self.config = {**self.get_default_config(), **(config or {})}
        self.setup_training()

    def get_default_config(self):
        """默认配置"""
        return {
            'opponent_type': 'mixed',  # 'random', 'rule', 'mixed', 'self', 'previous'
            'rule_aggression': 0.3,
            'mixed_ratio': 0.5,
            'board_size': 9,
            'total_episodes': 2000,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.998,
            'target_update': 100,
            'memory_size': 20000,
            'batch_size': 64,
            'eval_interval': 50,
            'eval_games': 20,
            'save_interval': 100,
            'save_dir': 'saved_models',
            'log_dir': 'training_logs',
            'early_stop_patience': 10,
            'target_win_rate': 0.7,
            'previous_model_path': None,  # 旧版本模型路径
            'self_play_ratio': 0.3,  # 自我对弈比例
            'model_pool_size': 3,  # 模型池大小
            'pool_update_interval': 200  # 模型池更新间隔
        }

    def setup_training(self):
        """设置训练环境"""
        # 创建目录
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)

        # 创建环境
        self.env = TrainingEnvironment(self.config['board_size'])

        # 创建DQN智能体
        self.dqn_agent = DQNAgent(
            board_size=self.config['board_size'],
            player=1,
            lr=self.config['learning_rate'],
            gamma=self.config['gamma'],
            epsilon=self.config['epsilon'],
            epsilon_min=self.config['epsilon_min'],
            epsilon_decay=self.config['epsilon_decay'],
            target_update=self.config['target_update'],
            memory_size=self.config['memory_size']
        )

        # 加载旧版本模型（如果提供）
        if self.config['previous_model_path'] and os.path.exists(self.config['previous_model_path']):
            try:
                self.dqn_agent.load(self.config['previous_model_path'])
                self._log(f"✅ 成功加载旧版本模型: {self.config['previous_model_path']}")
            except Exception as e:
                self._log(f"⚠️ 加载旧版本模型失败: {e}")

        # 模型池（用于自我对弈）
        self.model_pool = []
        self.model_pool_update_counter = 0

        # 训练统计
        self.stats = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'win_rates': [],
            'steps': [],
            'memory_size': [],
            'opponent_types': []
        }

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.config['log_dir'], f'training_{timestamp}.log')
        self._log(f"训练配置: {json.dumps(self.config, indent=2)}")

    def get_opponent(self, episode):
        """根据配置获取对手"""
        opponent_type = self.config['opponent_type']

        if opponent_type == 'random':
            return RandomAI(player=2), 'random'

        elif opponent_type == 'rule':
            # 动态调整规则AI难度
            aggression = self.config['rule_aggression']
            if episode > self.config['total_episodes'] * 0.7:
                aggression = min(0.7, aggression + 0.2)  # 后期增加难度
            return RuleBasedAI(player=2, board_size=self.config['board_size'],
                               aggression=aggression, debug=False), 'rule'

        elif opponent_type == 'mixed':
            # 混合训练
            if episode < self.config['total_episodes'] * 0.3:
                # 前期主要用随机AI
                ratio = 0.2
            elif episode < self.config['total_episodes'] * 0.7:
                # 中期混合
                ratio = self.config['mixed_ratio']
            else:
                # 后期主要用规则AI
                ratio = 0.8

            if random.random() < ratio:
                aggression = random.uniform(0.2, 0.6)
                return RuleBasedAI(player=2, board_size=self.config['board_size'],
                                   aggression=aggression, debug=False), 'rule'
            else:
                return RandomAI(player=2), 'random'

        elif opponent_type == 'self':
            # 自我对弈
            return self.create_self_play_opponent(episode), 'self'

        elif opponent_type == 'previous':
            # 与旧版本对弈
            return self.create_previous_version_opponent(), 'previous'

        else:
            raise ValueError(f"未知的对手类型: {opponent_type}")

    def create_self_play_opponent(self, episode):
        """创建自我对弈对手"""
        # 从模型池中随机选择一个模型，或者使用当前模型
        if self.model_pool and random.random() < 0.7:
            # 70%的概率从模型池中选择
            opponent_model = random.choice(self.model_pool)
            opponent = DQNAgent(
                board_size=self.config['board_size'],
                player=2,
                lr=self.config['learning_rate'],
                gamma=self.config['gamma'],
                epsilon=0.01,  # 评估模式，探索率低
                epsilon_min=0.01,
                epsilon_decay=1.0
            )
            # 复制模型权重
            opponent.policy_net.load_state_dict(opponent_model['state_dict'])
            opponent.target_net.load_state_dict(opponent_model['state_dict'])
            return opponent
        else:
            # 30%的概率使用当前模型
            opponent = DQNAgent(
                board_size=self.config['board_size'],
                player=2,
                lr=self.config['learning_rate'],
                gamma=self.config['gamma'],
                epsilon=0.05,  # 稍微有点探索
                epsilon_min=0.01,
                epsilon_decay=1.0
            )
            opponent.policy_net.load_state_dict(self.dqn_agent.policy_net.state_dict())
            opponent.target_net.load_state_dict(self.dqn_agent.target_net.state_dict())
            return opponent

    def create_previous_version_opponent(self):
        """创建旧版本对手"""
        if self.config['previous_model_path'] and os.path.exists(self.config['previous_model_path']):
            opponent = DQNAgent(
                board_size=self.config['board_size'],
                player=2,
                lr=self.config['learning_rate'],
                gamma=self.config['gamma'],
                epsilon=0.02,
                epsilon_min=0.01,
                epsilon_decay=1.0
            )
            opponent.load(self.config['previous_model_path'])
            return opponent
        else:
            # 如果没有旧版本，使用规则AI
            self._log("⚠️ 未找到旧版本模型，使用规则AI代替")
            return RuleBasedAI(player=2, board_size=self.config['board_size'],
                               aggression=0.5, debug=False)

    def update_model_pool(self, episode):
        """更新模型池"""
        self.model_pool_update_counter += 1

        if self.model_pool_update_counter >= self.config['pool_update_interval']:
            self.model_pool_update_counter = 0

            # 保存当前模型到池中
            model_snapshot = {
                'episode': episode,
                'state_dict': self.dqn_agent.policy_net.state_dict().copy(),
                'timestamp': datetime.now()
            }

            self.model_pool.append(model_snapshot)

            # 保持模型池大小
            if len(self.model_pool) > self.config['model_pool_size']:
                # 移除最旧的模型
                self.model_pool.pop(0)

            self._log(f"📦 模型池更新: 当前大小 {len(self.model_pool)}")

    def play_episode(self, opponent, opponent_type, episode):
        """进行一局游戏"""
        state = self.env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        step_count = 0

        # 随机决定先手
        dqn_first = random.random() < 0.5
        dqn_player = 1 if dqn_first else 2
        opponent_player = 2 if dqn_first else 1

        self.dqn_agent.player = dqn_player
        opponent.player = opponent_player

        # 用于经验回放的玩家视角
        current_perspective = dqn_player

        while not done and step_count < self.config['board_size'] ** 2:
            valid_moves = self.env.get_valid_moves()

            if self.env.current_player == dqn_player:
                # DQN AI的回合
                action = self.dqn_agent.get_move(state, valid_moves, training=True)

                if action is None or not self.env.is_valid_move(action):
                    break

                next_state, reward, done, _ = self.env.step(action, dqn_player)

                # 计算最终奖励
                if done:
                    if self.env.winner == dqn_player:
                        final_reward = 1.0
                    elif self.env.winner == opponent_player:
                        final_reward = -1.0
                    else:
                        final_reward = 0.1
                else:
                    final_reward = 0.0

                # 保存经验
                self.dqn_agent.remember(
                    state=state.copy(),
                    action=action,
                    reward=final_reward,
                    next_state=next_state.copy() if not done else None,
                    done=done,
                    valid_moves=valid_moves.copy(),
                    player=current_perspective
                )

                # 训练
                if len(self.dqn_agent.memory) >= self.dqn_agent.batch_size:
                    loss = self.dqn_agent.replay()
                    if loss is not None:
                        total_loss += loss

                total_reward += final_reward
                state = next_state

            else:
                # 对手的回合
                action = opponent.get_move(state, valid_moves)
                if action is None or not self.env.is_valid_move(action):
                    break

                state, _, done, _ = self.env.step(action, opponent_player)

            step_count += 1

        # 更新模型池（如果是自我对弈）
        if opponent_type == 'self':
            self.update_model_pool(episode)

        return total_reward, total_loss, step_count, self.env.winner == dqn_player

    def evaluate_agent(self, num_games=20):
        """修复版评估函数"""
        wins = 0

        self._log(f"   开始评估{num_games}局游戏...")

        for game_idx in range(num_games):
            env = TrainingEnvironment(self.config['board_size'])
            state = env.reset()
            done = False

            # 随机决定先手
            dqn_first = random.random() < 0.5
            dqn_player = 1 if dqn_first else 2
            opponent_player = 2 if dqn_first else 1

            # 使用规则AI进行评估
            opponent = RuleBasedAI(player=opponent_player,
                                   board_size=self.config['board_size'],
                                   aggression=0.5, debug=False)

            self.dqn_agent.player = dqn_player

            step_count = 0
            while not done and step_count < 100:
                valid_moves = env.get_valid_moves()

                if env.current_player == dqn_player:
                    # DQN AI行动（关闭训练模式）
                    action = self.dqn_agent.get_move(state, valid_moves, training=False)
                else:
                    # 对手行动
                    action = opponent.get_move(state, valid_moves)

                if action is None or not env.is_valid_move(action):
                    break

                state, _, done, _ = env.step(action, env.current_player)
                step_count += 1

                if done:
                    # 正确判断胜负
                    if env.winner == dqn_player:
                        wins += 1
                        if game_idx < 3:  # 只显示前3局结果
                            self._log(f"    第{game_idx + 1}局: ✅ DQN获胜 (步数: {step_count})")
                    elif env.winner == opponent_player:
                        if game_idx < 3:
                            self._log(f"    第{game_idx + 1}局: ❌ DQN失败 (步数: {step_count})")
                    else:
                        if game_idx < 3:
                            self._log(f"    第{game_idx + 1}局: 🤝 平局 (步数: {step_count})")
                    break

        win_rate = wins / num_games
        self._log(f"  📊 最终评估: {wins}胜/{num_games}局 = {win_rate:.2%}")
        return win_rate

    def _log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def train(self):
        """主训练循环"""
        self._log("开始训练DQN AI")
        self._log(f"对手类型: {self.config['opponent_type']}")
        self._log(f"训练轮数: {self.config['total_episodes']}")

        best_win_rate = 0
        patience_counter = 0
        total_wins = 0

        for episode in tqdm(range(self.config['total_episodes']), desc="训练进度"):
            # 获取对手
            opponent, opponent_type = self.get_opponent(episode)

            # 进行一局游戏
            total_reward, total_loss, step_count, won = self.play_episode(opponent, opponent_type, episode)

            if won:
                total_wins += 1

            # 记录统计
            self.stats['episodes'].append(episode)
            self.stats['rewards'].append(total_reward)
            self.stats['losses'].append(total_loss / max(step_count, 1))
            self.stats['epsilon'].append(self.dqn_agent.epsilon)
            self.stats['steps'].append(step_count)
            self.stats['memory_size'].append(len(self.dqn_agent.memory))
            self.stats['opponent_types'].append(opponent_type)

            # 定期评估
            if (episode + 1) % self.config['eval_interval'] == 0 or episode == 0:
                # 添加调试信息
                self._log(f"🔍 开始评估第{episode + 1}轮...")
                # 先显示累计胜率
                cumulative_win_rate = total_wins / (episode + 1) if episode > 0 else 0
                self._log(f"  累计胜率: {cumulative_win_rate:.2%} ({total_wins}/{episode + 1})")

                win_rate = self.evaluate_agent(self.config['eval_games'])
                self.stats['win_rates'].append(win_rate)

                # 获取训练信息
                info = self.dqn_agent.get_training_info()

                self._log(f"\n轮数 {episode + 1}:")
                self._log(f"  评估胜率: {win_rate:.2%} (累计: {total_wins}/{episode + 1} = {cumulative_win_rate:.2%})")
                self._log(f"  总奖励: {total_reward:.3f}")
                self._log(f"  探索率: {self.dqn_agent.epsilon:.4f}")
                self._log(f"  记忆库: {len(self.dqn_agent.memory)}")
                self._log(f"  平均损失: {total_loss / max(step_count, 1):.6f}")
                self._log(f"  步数: {step_count}")
                self._log(f"  对手类型: {opponent_type}")

                # 保存最佳模型
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    patience_counter = 0

                    model_path = os.path.join(
                        self.config['save_dir'],
                        f"best_model_ep{episode + 1}_wr{win_rate:.3f}.pth"
                    )
                    self.dqn_agent.save(model_path)
                    self._log(f"  ✅ 保存最佳模型: {model_path}")

                    # 早停检查
                    if win_rate >= self.config['target_win_rate']:
                        self._log(f"🎉 达到目标胜率 {win_rate:.2%}，提前停止训练！")
                        break
                else:
                    patience_counter += 1

                # 早停：长时间没有提升
                if patience_counter >= self.config['early_stop_patience']:
                    self._log(f"⏹️  {self.config['early_stop_patience']}次评估没有提升，提前停止训练")
                    break

            # 定期保存检查点
            if (episode + 1) % self.config['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['save_dir'],
                    f"checkpoint_ep{episode + 1}.pth"
                )
                self.dqn_agent.save(checkpoint_path)
                self._log(f"  💾 保存检查点: {checkpoint_path}")

        # 保存最终模型
        final_path = os.path.join(self.config['save_dir'], "final_model.pth")
        self.dqn_agent.save(final_path)
        self._log(f"✅ 保存最终模型: {final_path}")

        # 绘制训练曲线
        self.plot_training_curves()

        return self.dqn_agent, self.stats

    def play_episode(self, opponent, opponent_type, episode):
        """进行一局游戏（修复版本）"""
        state = self.env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        step_count = 0

        # 随机决定先手
        dqn_first = random.random() < 0.5
        dqn_player = 1 if dqn_first else 2
        opponent_player = 2 if dqn_first else 1

        self.dqn_agent.player = dqn_player
        opponent.player = opponent_player

        while not done and step_count < self.config['board_size'] ** 2:
            valid_moves = self.env.get_valid_moves()

            if self.env.current_player == dqn_player:
                # DQN AI的回合
                action = self.dqn_agent.get_move(state, valid_moves, training=True)

                if action is None or not self.env.is_valid_move(action):
                    break

                next_state, reward, done, _ = self.env.step(action, dqn_player)

                # 计算最终奖励
                if done:
                    if self.env.winner == dqn_player:
                        final_reward = 1.0
                    elif self.env.winner == opponent_player:
                        final_reward = -1.0
                    else:
                        final_reward = 0.1
                else:
                    final_reward = 0.0

                # 保存经验
                self.dqn_agent.remember(
                    state=state.copy(),
                    action=action,
                    reward=final_reward,
                    next_state=next_state.copy() if not done else None,
                    done=done,
                    valid_moves=valid_moves.copy(),
                    player=dqn_player
                )

                # 训练
                if len(self.dqn_agent.memory) >= self.dqn_agent.batch_size:
                    loss = self.dqn_agent.replay()
                    if loss is not None:
                        total_loss += loss

                total_reward += final_reward
                state = next_state

            else:
                # 对手的回合
                action = opponent.get_move(state, valid_moves)
                if action is None or not self.env.is_valid_move(action):
                    break

                state, _, done, _ = self.env.step(action, opponent_player)

            step_count += 1

        return total_reward, total_loss, step_count, self.env.winner == dqn_player

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 12))

        # 1. 胜率曲线
        plt.subplot(3, 3, 1)
        if self.stats['win_rates']:
            eval_points = [self.config['eval_interval'] * (i + 1) for i in range(len(self.stats['win_rates']))]
            plt.plot(eval_points, self.stats['win_rates'], 'mo-', linewidth=2, markersize=5)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准')
            plt.axhline(y=self.config['target_win_rate'], color='g', linestyle='--', alpha=0.5, label='目标胜率')
            plt.xlabel('训练轮数')
            plt.ylabel('胜率')
            plt.title('胜率 vs 规则AI')
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend()

        # 2. 奖励曲线
        plt.subplot(3, 3, 2)
        window = 50
        if len(self.stats['rewards']) > window:
            rewards_smooth = np.convolve(self.stats['rewards'], np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(self.stats['rewards'])), rewards_smooth, 'b-', alpha=0.7, label='平滑')
        plt.plot(self.stats['rewards'], 'b-', alpha=0.3, label='原始')
        plt.xlabel('训练轮数')
        plt.ylabel('总奖励')
        plt.title('奖励曲线')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 3. 损失曲线
        plt.subplot(3, 3, 3)
        if len(self.stats['losses']) > window:
            loss_smooth = np.convolve(self.stats['losses'], np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(self.stats['losses'])), loss_smooth, 'r-', alpha=0.7, label='平滑')
        plt.plot(self.stats['losses'], 'r-', alpha=0.3, label='原始')
        plt.xlabel('训练轮数')
        plt.ylabel('平均损失')
        plt.title('损失曲线')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 4. 探索率曲线
        plt.subplot(3, 3, 4)
        plt.plot(self.stats['epsilon'], 'g-')
        plt.xlabel('训练轮数')
        plt.ylabel('探索率 (ε)')
        plt.title('探索率衰减')
        plt.grid(True, alpha=0.3)

        # 5. 步数曲线
        plt.subplot(3, 3, 5)
        if len(self.stats['steps']) > window:
            steps_smooth = np.convolve(self.stats['steps'], np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(self.stats['steps'])), steps_smooth, 'c-', alpha=0.7, label='平滑')
        plt.plot(self.stats['steps'], 'c-', alpha=0.3, label='原始')
        plt.xlabel('训练轮数')
        plt.ylabel('步数')
        plt.title('每局平均步数')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 6. 记忆库大小
        plt.subplot(3, 3, 6)
        plt.plot(self.stats['memory_size'], 'y-')
        plt.xlabel('训练轮数')
        plt.ylabel('记忆数量')
        plt.title('经验回放记忆库大小')
        plt.grid(True, alpha=0.3)

        # 7. 对手类型分布
        plt.subplot(3, 3, 7)
        if self.stats['opponent_types']:
            unique_types, counts = np.unique(self.stats['opponent_types'], return_counts=True)
            plt.pie(counts, labels=unique_types, autopct='%1.1f%%', startangle=90)
            plt.title('对手类型分布')

        plt.suptitle(f"DQN训练结果 - {self.config['board_size']}x{self.config['board_size']} 棋盘", fontsize=16)
        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(self.config['log_dir'], "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self._log(f"✅ 训练曲线已保存: {plot_path}")


def train_dqn(opponent_type='random', episodes=1000, **kwargs):
    """
    训练DQN的便捷函数

    参数:
        opponent_type: 对手类型 ('random', 'rule', 'mixed', 'self', 'previous')
        episodes: 训练轮数
        **kwargs: 其他配置参数
    """
    config = {
        'opponent_type': opponent_type,
        'total_episodes': episodes,
        **kwargs
    }

    trainer = DQNTrainer(config)
    return trainer.train()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练DQN五子棋AI')
    parser.add_argument('--opponent', type=str,
                        choices=['random', 'rule', 'mixed', 'self', 'previous'],
                        default='mixed', help='训练对手类型')
    parser.add_argument('--episodes', type=int, default=2000, help='训练轮数')
    parser.add_argument('--aggression', type=float, default=0.3,
                        help='规则AI的攻击性（0-1）')
    parser.add_argument('--mixed-ratio', type=float, default=0.5,
                        help='混合训练中使用规则AI的概率')
    parser.add_argument('--target-win-rate', type=float, default=0.7,
                        help='目标胜率（达到后提前停止）')
    parser.add_argument('--previous-model', type=str, default='',
                        help='旧版本模型路径（用于previous模式）')
    parser.add_argument('--self-play-ratio', type=float, default=0.3,
                        help='自我对弈比例')

    args = parser.parse_args()

    print("🚀 DQN训练配置:")
    print(f"  对手类型: {args.opponent}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练DQN五子棋AI')
    parser.add_argument('--opponent', type=str,
                        choices=['random', 'rule', 'mixed', 'self', 'previous'],
                        default='mixed', help='训练对手类型')
    parser.add_argument('--episodes', type=int, default=2000, help='训练轮数')
    parser.add_argument('--aggression', type=float, default=0.3,
                        help='规则AI的攻击性（0-1）')
    parser.add_argument('--mixed-ratio', type=float, default=0.5,
                        help='混合训练中使用规则AI的概率')
    parser.add_argument('--target-win-rate', type=float, default=0.7,
                        help='目标胜率（达到后提前停止）')
    parser.add_argument('--previous-model', type=str, default='',
                        help='旧版本模型路径（用于previous模式）')
    parser.add_argument('--self-play-ratio', type=float, default=0.3,
                        help='自我对弈比例')

    args = parser.parse_args()

    print("🚀 DQN训练配置:")
    print(f"  对手类型: {args.opponent}")
    print(f"  训练轮数: {args.episodes}")
    print(f"  规则AI攻击性: {args.aggression}")
    print(f"  混合比例: {args.mixed_ratio}")
    print(f"  目标胜率: {args.target_win_rate}")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'opponent_type': args.opponent,
            'total_episodes': args.episodes,
            'rule_aggression': args.aggression,
            'mixed_ratio': args.mixed_ratio,
            'target_win_rate': args.target_win_rate,
            'previous_model_path': args.previous_model if args.previous_model else None,
            'self_play_ratio': args.self_play_ratio
        }

        # 创建训练器并开始训练
        trainer = DQNTrainer(config)
        agent, stats = trainer.train()

        # 最终评估
        print("\n" + "=" * 60)
        print("📊 最终评估")
        print("=" * 60)

        final_win_rate = trainer.evaluate_agent(num_games=50)
        print(f"🎯 最终胜率: {final_win_rate:.2%}")
        print(f"✅ 训练完成！")

    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback

        traceback.print_exc()

# 与随机AI训练（最简单，适合初学者）
#python train_dqn.py --opponent random --episodes 1000

# 与规则AI训练（中等难度）
#python train_dqn.py --opponent rule --episodes 2000 --aggression 0.5

# 混合训练（推荐，效果最好）
#python train_dqn_fixed.py --opponent mixed --episodes 2000 --mixed-ratio 0.5

# 自我对弈（高级，需要已有基础模型）
#python train_dqn_fixed.py --opponent self --episodes 3000

# 与旧版本训练（持续改进）
#python train_dqn_fixed.py --opponent previous --episodes 2000 --previous-model "saved_models/best_model.pth"