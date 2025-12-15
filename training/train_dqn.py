# train_dqn_fixed.py
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入修复后的模型
from models.dqn_ai import DQNAgent
from models.rule_based_ai import RuleBasedAI  # 使用修复后的规则AI


class TrainingGomokuEnv:
    """训练环境"""

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_valid_moves(self):
        """获取所有合法移动"""
        n = self.board_size
        valid_moves = np.zeros(n * n, dtype=int)
        for y in range(n):
            for x in range(n):
                if self.board[y][x] == 0:
                    action = y * n + x
                    valid_moves[action] = 1
        return valid_moves

    def is_valid_move(self, action):
        if action is None:
            return False
        n = self.board_size
        x, y = action % n, action // n
        return 0 <= x < n and 0 <= y < n and self.board[y][x] == 0

    def step(self, action):
        """执行一步"""
        if not self.is_valid_move(action) or self.done:
            return self.board.copy(), 0, True, {}

        n = self.board_size
        x, y = action % n, action // n
        self.board[y][x] = self.current_player

        # 检查是否获胜
        if self.check_win(y, x):
            self.done = True
            self.winner = self.current_player
            reward = 1.0 if self.current_player == 1 else -1.0
        elif np.all(self.board != 0):  # 平局
            self.done = True
            self.winner = 0
            reward = 0.1
        else:
            self.current_player = 3 - self.current_player
            reward = 0.0

        return self.board.copy(), reward, self.done, {}

    def check_win(self, y, x):
        """检查是否获胜"""
        player = self.board[y][x]
        n = self.board_size

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dy, dx in directions:
            count = 1

            # 正向
            for i in range(1, 5):
                ny, nx = y + dy * i, x + dx * i
                if 0 <= ny < n and 0 <= nx < n and self.board[ny][nx] == player:
                    count += 1
                else:
                    break

            # 反向
            for i in range(1, 5):
                ny, nx = y - dy * i, x - dx * i
                if 0 <= ny < n and 0 <= nx < n and self.board[ny][nx] == player:
                    count += 1
                else:
                    break

            if count >= 5:
                return True

        return False


def train_dqn(config):
    """训练DQN模型"""
    print("=" * 60)
    print("开始训练DQN模型")
    print("=" * 60)

    # 创建环境
    env = TrainingGomokuEnv(config['board_size'])

    # 创建智能体
    print(f"创建DQN智能体 (棋盘大小: {config['board_size']}x{config['board_size']})")
    agent = DQNAgent(
        board_size=config['board_size'],
        player=1,
        lr=config['learning_rate'],
        gamma=config['gamma'],
        epsilon=config['epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
        target_update=config['target_update'],
        memory_size=config['memory_size']
    )

    # 创建对手
    print("创建规则AI对手...")
    opponent = RuleBasedAI(
        player=2,
        board_size=config['board_size'],
        aggression=config['opponent_aggression'],
        debug=False
    )

    # 训练统计
    stats = {
        'episode': [],
        'total_reward': [],
        'avg_loss': [],
        'epsilon': [],
        'win_rate': [],
        'steps': [],
        'memory_size': []
    }

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['save_dir'], f"train_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"保存目录: {save_dir}")
    print(f"训练轮数: {config['total_episodes']}")
    print(f"评估间隔: 每{config['eval_interval']}轮")
    print(f"评估局数: {config['eval_games']}局")
    print("-" * 40)

    best_win_rate = 0
    patience_counter = 0
    max_patience = config.get('patience', 10)

    # 训练循环
    for episode in tqdm(range(config['total_episodes']), desc="训练进度"):
        state = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        step_count = 0

        # 随机决定先手
        agent_first = np.random.random() < 0.5
        agent.player = 1 if agent_first else 2
        opponent.player = 2 if agent_first else 1

        # 保存当前玩家用于经验回放
        current_player_perspective = agent.player

        while not done:
            valid_moves = env.get_valid_moves()

            # 当前玩家行动
            if env.current_player == agent.player:
                # DQN Agent行动
                action = agent.get_move(state, valid_moves, training=True)
                if action is None or not env.is_valid_move(action):
                    # 没有合法移动，结束游戏
                    done = True
                    if not done:  # 如果还没结束，平局
                        env.done = True
                        env.winner = 0
                    break

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 计算最终奖励
                if done:
                    if env.winner == agent.player:
                        final_reward = 1.0
                    elif env.winner == opponent.player:
                        final_reward = -1.0
                    else:  # 平局
                        final_reward = 0.1
                else:
                    final_reward = 0.0

                # 保存经验
                agent.remember(
                    state=state.copy(),
                    action=action,
                    reward=final_reward,
                    next_state=next_state.copy() if not done else None,
                    done=done,
                    valid_moves=valid_moves.copy(),
                    player=current_player_perspective
                )

                # 训练
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.replay()
                    if loss is not None:
                        total_loss += loss

                total_reward += final_reward
                state = next_state

            else:
                # 对手行动
                action = opponent.get_move(state, valid_moves)
                if action is None or not env.is_valid_move(action):
                    # 没有合法移动，结束游戏
                    done = True
                    if not done:  # 如果还没结束，平局
                        env.done = True
                        env.winner = 0
                    break

                state, reward, done, _ = env.step(action)

            step_count += 1

            # 防止无限循环
            if step_count > config['board_size'] * config['board_size']:
                done = True
                env.done = True
                env.winner = 0  # 平局

        # 记录统计
        stats['episode'].append(episode)
        stats['total_reward'].append(total_reward)
        stats['avg_loss'].append(total_loss / max(step_count, 1))
        stats['epsilon'].append(agent.epsilon)
        stats['steps'].append(step_count)
        stats['memory_size'].append(len(agent.memory))

        # 定期评估
        if (episode + 1) % config['eval_interval'] == 0 or episode == 0:
            # 评估
            win_rate = evaluate_agent(agent, opponent, config)

            # 记录胜率
            stats['win_rate'].append(win_rate)

            # 获取训练信息
            info = agent.get_training_info()

            # 打印进度
            print(f"\n轮数 {episode + 1}/{config['total_episodes']}:")
            print(f"  胜率: {win_rate:.2%}")
            print(f"  平均奖励: {total_reward:.3f}")
            print(f"  探索率: {agent.epsilon:.4f}")
            print(f"  记忆库大小: {len(agent.memory)}")
            print(f"  平均损失: {total_loss / max(step_count, 1):.6f}")
            print(f"  平均步数: {step_count}")

            # 保存最佳模型
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                patience_counter = 0

                # 保存模型
                model_path = os.path.join(save_dir, f"best_model_ep{episode + 1}_wr{win_rate:.3f}.pth")
                agent.save(model_path)
                print(f"  ✅ 保存最佳模型: {model_path}")

                # 早停检查
                if win_rate >= config.get('target_win_rate', 0.7):
                    print(f"\n🎉 达到目标胜率 {win_rate:.2%}，提前停止训练！")
                    break
            else:
                patience_counter += 1

            # 早停：长时间没有提升
            if patience_counter >= max_patience:
                print(f"\n⚠️  {max_patience}次评估没有提升，提前停止训练")
                break

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_model.pth")
    agent.save(final_path)
    print(f"\n✅ 保存最终模型: {final_path}")

    # 绘制训练曲线
    plot_training_curves(stats, save_dir, config)

    return agent, stats, save_dir


def evaluate_agent(agent, opponent, config, verbose=False):
    """评估智能体性能"""
    env = TrainingGomokuEnv(config['board_size'])
    wins = 0
    total_games = config['eval_games']

    for game_idx in range(total_games):
        state = env.reset()
        done = False

        # 随机决定先手
        agent_first = np.random.random() < 0.5
        agent.player = 1 if agent_first else 2
        opponent.player = 2 if agent_first else 1

        while not done:
            valid_moves = env.get_valid_moves()

            if env.current_player == agent.player:
                # DQN Agent行动
                action = agent.get_move(state, valid_moves, training=False)
            else:
                # 对手行动
                action = opponent.get_move(state, valid_moves)

            if action is None or not env.is_valid_move(action):
                # 没有合法移动
                env.done = True
                env.winner = 0
                done = True
                break

            state, reward, done, _ = env.step(action)

            if env.done:
                if env.winner == agent.player:
                    wins += 1
                break

        if verbose and (game_idx + 1) % 5 == 0:
            print(f"  评估进度: {game_idx + 1}/{total_games}")

    win_rate = wins / total_games
    return win_rate


def plot_training_curves(stats, save_dir, config):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 10))

    # 1. 奖励曲线
    plt.subplot(2, 3, 1)
    window = 50
    if len(stats['total_reward']) > window:
        rewards_smooth = np.convolve(stats['total_reward'], np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(stats['total_reward'])), rewards_smooth, 'b-', alpha=0.7, label='平滑')
    plt.plot(stats['total_reward'], 'b-', alpha=0.3, label='原始')
    plt.xlabel('训练轮数')
    plt.ylabel('总奖励')
    plt.title('奖励曲线')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. 损失曲线
    plt.subplot(2, 3, 2)
    if len(stats['avg_loss']) > window:
        loss_smooth = np.convolve(stats['avg_loss'], np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(stats['avg_loss'])), loss_smooth, 'r-', alpha=0.7, label='平滑')
    plt.plot(stats['avg_loss'], 'r-', alpha=0.3, label='原始')
    plt.xlabel('训练轮数')
    plt.ylabel('平均损失')
    plt.title('损失曲线')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3. 探索率曲线
    plt.subplot(2, 3, 3)
    plt.plot(stats['epsilon'], 'g-')
    plt.xlabel('训练轮数')
    plt.ylabel('探索率 (ε)')
    plt.title('探索率衰减')
    plt.grid(True, alpha=0.3)

    # 4. 胜率曲线
    plt.subplot(2, 3, 4)
    if stats['win_rate']:
        eval_points = [config['eval_interval'] * (i + 1) for i in range(len(stats['win_rate']))]
        plt.plot(eval_points, stats['win_rate'], 'mo-', linewidth=2, markersize=5)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准')
        plt.xlabel('训练轮数')
        plt.ylabel('胜率')
        plt.title('胜率 vs 规则AI')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()

    # 5. 步数曲线
    plt.subplot(2, 3, 5)
    if len(stats['steps']) > window:
        steps_smooth = np.convolve(stats['steps'], np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(stats['steps'])), steps_smooth, 'c-', alpha=0.7, label='平滑')
    plt.plot(stats['steps'], 'c-', alpha=0.3, label='原始')
    plt.xlabel('训练轮数')
    plt.ylabel('步数')
    plt.title('每局平均步数')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 6. 记忆库大小
    plt.subplot(2, 3, 6)
    plt.plot(stats['memory_size'], 'y-')
    plt.xlabel('训练轮数')
    plt.ylabel('记忆数量')
    plt.title('经验回放记忆库大小')
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"DQN训练结果 - {config['board_size']}x{config['board_size']} 棋盘", fontsize=16)
    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 训练曲线已保存: {plot_path}")


def main():
    """主训练函数"""
    # 训练配置
    config = {
        'board_size': 9,  # 训练9x9棋盘
        'total_episodes': 2000,  # 总训练轮数
        'learning_rate': 0.001,
        'gamma': 0.95,  # 折扣因子
        'epsilon': 1.0,  # 初始探索率
        'epsilon_min': 0.1,  # 最小探索率
        'epsilon_decay': 0.998,  # 探索率衰减
        'target_update': 200,  # 目标网络更新频率
        'memory_size': 20000,  # 记忆库大小
        'eval_interval': 50,  # 评估间隔
        'eval_games': 20,  # 每次评估的局数
        'save_dir': 'saved_models',  # 保存目录
        'opponent_aggression': 0.3,  # 对手攻击性（0-1）
        'target_win_rate': 0.7,  # 目标胜率
        'patience': 10  # 早停耐心值
    }

    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 开始训练
    try:
        agent, stats, save_dir = train_dqn(config)

        # 最终评估
        print("\n" + "=" * 60)
        print("最终评估")
        print("=" * 60)

        # 创建对手
        opponent = RuleBasedAI(
            player=2,
            board_size=config['board_size'],
            aggression=0.3,
            debug=False
        )

        # 评估100局
        final_win_rate = evaluate_agent(agent, opponent, config, verbose=True)
        print(f"最终胜率: {final_win_rate:.2%}")

        # 保存最终结果
        result_path = os.path.join(save_dir, "training_results.txt")
        with open(result_path, 'w') as f:
            f.write("DQN训练结果总结\n")
            f.write("=" * 40 + "\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最终轮数: {len(stats['episode'])}\n")
            f.write(f"最终胜率: {final_win_rate:.2%}\n")
            f.write(f"最终探索率: {agent.epsilon:.4f}\n")
            f.write(f"记忆库大小: {len(agent.memory)}\n")
            f.write(f"模型保存目录: {save_dir}\n")

        print(f"\n✅ 训练完成！结果保存到: {save_dir}")

    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()