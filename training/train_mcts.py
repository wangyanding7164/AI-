# train_mcts.py
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mcts_ai import MCTSAI
from models.rule_based_ai import RuleBasedAI
from models.dqn_ai import DQNAgent


class MCTSTrainingEnv:
    """MCTS训练环境"""

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.history = []
        return self.board.copy()

    def get_valid_moves(self):
        n = self.board_size
        valid_moves = np.zeros(n * n, dtype=int)
        for y in range(n):
            for x in range(n):
                if self.board[y][x] == 0:
                    action = y * n + x
                    valid_moves[action] = 1
        return valid_moves

    def step(self, action, player):
        """执行一步"""
        n = self.board_size
        x, y = action % n, action // n

        if not (0 <= x < n and 0 <= y < n and self.board[y][x] == 0):
            return self.board.copy(), 0, True, {}

        self.board[y][x] = player
        self.history.append((action, player))

        # 检查获胜
        if self.check_win(x, y, player):
            self.done = True
            self.winner = player
            reward = 1.0
        elif np.all(self.board != 0):  # 平局
            self.done = True
            self.winner = 0
            reward = 0.1
        else:
            reward = 0.0
            self.current_player = 3 - player

        return self.board.copy(), reward, self.done, {}

    def check_win(self, x, y, player):
        """检查获胜"""
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

    def get_state(self):
        return self.board.copy()


def train_mcts_with_opponent(config):
    """用指定对手训练MCTS"""
    print("=" * 60)
    print(f"训练MCTS AI (对手: {config['opponent_type']})")
    print("=" * 60)

    # 创建MCTS AI
    mcts_ai = MCTSAI(
        board_size=config['board_size'],
        player=1,
        iterations=config['iterations_per_move'],
        debug=config.get('debug', False)
    )

    # 创建对手
    if config['opponent_type'] == 'rule_based':
        opponent = RuleBasedAI(
            player=2,
            board_size=config['board_size'],
            aggression=config.get('opponent_aggression', 0.5)
        )
    elif config['opponent_type'] == 'dqn':
        opponent = DQNAgent(
            board_size=config['board_size'],
            player=2,
            epsilon=0.01  # 推理模式
        )
        # 加载训练好的DQN
        dqn_path = config.get('dqn_model_path', 'saved_models/dqn_final.pth')
        if os.path.exists(dqn_path):
            opponent.load(dqn_path)
            print(f"✅ 加载DQN对手: {dqn_path}")
        else:
            print(f"❌ DQN模型不存在: {dqn_path}")
            return None
    else:
        print(f"❌ 未知对手类型: {config['opponent_type']}")
        return None

    # 训练统计
    stats = {
        'games': [],
        'mcts_wins': [],
        'opponent_wins': [],
        'draws': [],
        'avg_moves': [],
        'training_time': []
    }

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['save_dir'], f"mcts_vs_{config['opponent_type']}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"保存目录: {save_dir}")
    print(f"训练局数: {config['total_games']}")
    print(f"MCTS每步搜索次数: {config['iterations_per_move']}")
    print(f"对手: {config['opponent_type']}")

    mcts_wins = 0
    opponent_wins = 0
    draws = 0

    for game_idx in tqdm(range(config['total_games']), desc="训练进度"):
        start_time = datetime.now()

        env = MCTSTrainingEnv(config['board_size'])
        state = env.reset()
        done = False

        # 随机决定先手
        mcts_first = np.random.random() < 0.5
        mcts_player = 1 if mcts_first else 2
        opponent_player = 2 if mcts_first else 1

        moves = 0
        max_moves = config['board_size'] * config['board_size']

        while not done and moves < max_moves:
            valid_moves = env.get_valid_moves()

            if env.current_player == mcts_player:
                # MCTS行动
                action = mcts_ai.get_move(state, valid_moves)
            else:
                # 对手行动
                action = opponent.get_move(state, valid_moves)

            if action is None or valid_moves[action] == 0:
                # 没有合法移动，平局
                env.done = True
                env.winner = 0
                break

            state, reward, done, _ = env.step(action, env.current_player)
            moves += 1

            if done:
                if env.winner == mcts_player:
                    mcts_wins += 1
                elif env.winner == opponent_player:
                    opponent_wins += 1
                else:
                    draws += 1
                break

        # 记录统计
        game_time = (datetime.now() - start_time).total_seconds()

        stats['games'].append(game_idx + 1)
        stats['mcts_wins'].append(mcts_wins)
        stats['opponent_wins'].append(opponent_wins)
        stats['draws'].append(draws)
        stats['avg_moves'].append(moves)
        stats['training_time'].append(game_time)

        # 定期保存和显示
        if (game_idx + 1) % config['log_interval'] == 0:
            total_played = mcts_wins + opponent_wins + draws
            if total_played > 0:
                mcts_win_rate = mcts_wins / total_played
                opponent_win_rate = opponent_wins / total_played
                draw_rate = draws / total_played

                print(f"\n游戏 {game_idx + 1}/{config['total_games']}:")
                print(f"  MCTS胜率: {mcts_win_rate:.2%} ({mcts_wins}/{total_played})")
                print(f"  对手胜率: {opponent_win_rate:.2%} ({opponent_wins}/{total_played})")
                print(f"  平局率: {draw_rate:.2%} ({draws}/{total_played})")
                print(f"  平均步数: {np.mean(stats['avg_moves'][-config['log_interval']:]):.1f}")
                print(f"  平均时间/局: {np.mean(stats['training_time'][-config['log_interval']:]):.1f}s")

            # 保存检查点
            if (game_idx + 1) % config['save_interval'] == 0:
                checkpoint_path = os.path.join(save_dir, f"mcts_checkpoint_game{game_idx + 1}.pkl")
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'mcts_ai': mcts_ai,
                        'stats': stats,
                        'config': config
                    }, f)
                print(f"  💾 保存检查点: {checkpoint_path}")

    # 最终评估
    print("\n" + "=" * 60)
    print("训练完成！最终统计")
    print("=" * 60)

    total_played = mcts_wins + opponent_wins + draws
    if total_played > 0:
        print(f"MCTS总胜率: {mcts_wins / total_played:.2%}")
        print(f"对手总胜率: {opponent_wins / total_played:.2%}")
        print(f"平局率: {draws / total_played:.2%}")
        print(f"平均步数: {np.mean(stats['avg_moves']):.1f}")

    # 保存最终模型
    final_path = os.path.join(save_dir, "mcts_final.pkl")
    with open(final_path, 'wb') as f:
        pickle.dump({
            'mcts_ai': mcts_ai,
            'stats': stats,
            'config': config
        }, f)

    print(f"\n✅ 最终模型保存到: {final_path}")

    # 绘制训练曲线
    plot_training_curves(stats, save_dir, config)

    return mcts_ai, stats, save_dir


def plot_training_curves(stats, save_dir, config):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 10))

    # 胜率曲线
    plt.subplot(2, 3, 1)
    games = stats['games']

    if len(games) > 0:
        mcts_win_rates = [m / t if t > 0 else 0 for m, t in zip(stats['mcts_wins'], games)]
        opponent_win_rates = [o / t if t > 0 else 0 for o, t in zip(stats['opponent_wins'], games)]
        draw_rates = [d / t if t > 0 else 0 for d, t in zip(stats['draws'], games)]

        plt.plot(games, mcts_win_rates, 'g-', label='MCTS胜率', linewidth=2)
        plt.plot(games, opponent_win_rates, 'r-', label='对手胜率', linewidth=2)
        plt.plot(games, draw_rates, 'b-', label='平局率', linewidth=2)

        plt.xlabel('游戏局数')
        plt.ylabel('胜率')
        plt.title('胜率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

    # 累计胜率
    plt.subplot(2, 3, 2)
    if len(games) > 0:
        window = 50
        if len(mcts_win_rates) > window:
            smoothed = np.convolve(mcts_win_rates, np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(games)), smoothed, 'g-', linewidth=2, label='MCTS胜率(平滑)')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准')
        plt.xlabel('游戏局数')
        plt.ylabel('胜率(平滑)')
        plt.title('MCTS胜率(平滑)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 平均步数
    plt.subplot(2, 3, 3)
    if len(stats['avg_moves']) > 0:
        plt.plot(games, stats['avg_moves'], 'c-', alpha=0.5)

        # 平滑
        window = 20
        if len(stats['avg_moves']) > window:
            smoothed_moves = np.convolve(stats['avg_moves'], np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(games)), smoothed_moves, 'b-', linewidth=2, label='平滑')

        plt.xlabel('游戏局数')
        plt.ylabel('步数')
        plt.title('平均步数')
        plt.grid(True, alpha=0.3)

    # 时间统计
    plt.subplot(2, 3, 4)
    if len(stats['training_time']) > 0:
        plt.plot(games, stats['training_time'], 'y-', alpha=0.3)

        window = 20
        if len(stats['training_time']) > window:
            smoothed_time = np.convolve(stats['training_time'], np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(games)), smoothed_time, 'orange', linewidth=2, label='平滑')

        plt.xlabel('游戏局数')
        plt.ylabel('时间(秒)')
        plt.title('每局训练时间')
        plt.grid(True, alpha=0.3)

    # 胜负统计柱状图
    plt.subplot(2, 3, 5)
    if len(games) > 0:
        labels = ['MCTS胜利', '对手胜利', '平局']
        values = [stats['mcts_wins'][-1], stats['opponent_wins'][-1], stats['draws'][-1]]
        colors = ['green', 'red', 'blue']

        bars = plt.bar(labels, values, color=colors)
        plt.ylabel('局数')
        plt.title('胜负统计')

        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{value}', ha='center', va='bottom')

    # 训练信息
    plt.subplot(2, 3, 6)
    plt.axis('off')

    info_text = f"训练配置:\n"
    info_text += f"棋盘大小: {config['board_size']}x{config['board_size']}\n"
    info_text += f"对手: {config['opponent_type']}\n"
    info_text += f"训练局数: {config['total_games']}\n"
    info_text += f"搜索次数/步: {config['iterations_per_move']}\n"

    if len(games) > 0:
        info_text += f"\n最终统计:\n"
        info_text += f"MCTS胜率: {stats['mcts_wins'][-1] / games[-1]:.2%}\n"
        info_text += f"对手胜率: {stats['opponent_wins'][-1] / games[-1]:.2%}\n"
        info_text += f"平局率: {stats['draws'][-1] / games[-1]:.2%}\n"
        info_text += f"平均步数: {np.mean(stats['avg_moves']):.1f}\n"

    plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')

    plt.suptitle(f"MCTS训练结果 - 对手: {config['opponent_type']}", fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 训练曲线保存到: {plot_path}")


def main():
    """主训练函数"""
    # 根据您的DQN表现选择训练策略

    # 方案A：如果DQN表现一般（<60%胜率），先用规则AI训练
    if True:  # 替换为您的判断条件
        print("使用方案A：MCTS vs 规则AI（快速启动）")
        config = {
            'board_size': 9,
            'opponent_type': 'rule_based',  # 使用规则AI
            'opponent_aggression': 0.5,  # 中等难度
            'iterations_per_move': 500,  # 每步搜索次数
            'total_games': 1000,  # 训练局数
            'log_interval': 50,  # 日志间隔
            'save_interval': 200,  # 保存间隔
            'save_dir': 'saved_models/mcts',
            'debug': False
        }

    # 方案B：如果DQN表现优秀（>60%胜率），用DQN训练
    else:
        print("使用方案B：MCTS vs DQN（强化训练）")
        config = {
            'board_size': 9,
            'opponent_type': 'dqn',  # 使用DQN
            'dqn_model_path': 'saved_models/dqn_final.pth',  # DQN模型路径
            'iterations_per_move': 800,  # 增加搜索深度
            'total_games': 2000,  # 更多训练局数
            'log_interval': 100,
            'save_interval': 500,
            'save_dir': 'saved_models/mcts_vs_dqn',
            'debug': False
        }

    # 开始训练
    try:
        mcts_ai, stats, save_dir = train_mcts_with_opponent(config)

        # 最终评估报告
        report_path = os.path.join(save_dir, "training_report.txt")
        with open(report_path, 'w') as f:
            f.write("MCTS训练报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"对手类型: {config['opponent_type']}\n")
            f.write(f"训练局数: {len(stats['games'])}\n")

            if len(stats['games']) > 0:
                total_games = stats['games'][-1]
                mcts_wins = stats['mcts_wins'][-1]
                opponent_wins = stats['opponent_wins'][-1]
                draws = stats['draws'][-1]

                f.write(f"MCTS胜率: {mcts_wins / total_games:.2%}\n")
                f.write(f"对手胜率: {opponent_wins / total_games:.2%}\n")
                f.write(f"平局率: {draws / total_games:.2%}\n")
                f.write(f"平均步数: {np.mean(stats['avg_moves']):.1f}\n")
                f.write(f"模型保存目录: {save_dir}\n")

        print(f"\n📄 训练报告保存到: {report_path}")

    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()