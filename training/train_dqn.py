import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training_env import TrainingGomokuEnv
from models.dqn_ai import DQNAgent
from models.rule_based_ai import RuleBasedAI


def train_dqn_multisize(config):
    """训练多尺寸DQN模型"""
    results = {}

    for board_size in config['board_sizes']:
        print(f"\n=== 训练 {board_size}x{board_size} 棋盘 DQN ===")

        # 更新配置
        current_config = config.copy()
        current_config['board_size'] = board_size
        current_config['save_dir'] = f"{config['save_dir']}/size_{board_size}"

        # 训练单个尺寸
        agent, stats = train_single_size(current_config)
        results[board_size] = {'agent': agent, 'stats': stats}

    return results


def train_single_size(config):
    """训练单个棋盘尺寸的DQN"""
    # 创建环境
    env = TrainingGomokuEnv(config['board_size'])

    # 创建智能体
    agent = DQNAgent(
        board_size=config['board_size'],
        player=1,
        lr=config['learning_rate'],
        gamma=config['gamma'],
        epsilon=config['epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay']
    )

    # 创建对手
    opponent = RuleBasedAI(player=2, board_size=config['board_size'])

    # 训练统计
    stats = {
        'episodes': [], 'rewards': [], 'losses': [],
        'epsilons': [], 'win_rates': [], 'avg_steps': []
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    best_win_rate = 0
    convergence_count = 0

    for episode in tqdm(range(config['total_episodes'])):
        state = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0

        # 随机决定先手
        agent_turn = np.random.random() < 0.5
        agent.player = 1 if agent_turn else 2
        opponent.player = 2 if agent_turn else 1

        while not done:
            valid_moves = env.get_valid_moves()

            if agent_turn:
                action = agent.get_move(state, valid_moves)
                if action is None:
                    break

                next_state, reward, done, _ = env.step(action)

                # 计算最终奖励
                if done:
                    if env.winner == agent.player:
                        final_reward = 1.0
                    elif env.winner == opponent.player:
                        final_reward = -1.0
                    else:
                        final_reward = 0.1
                else:
                    final_reward = 0.0

                agent.remember(state, action, final_reward, next_state, done, valid_moves)
                loss = agent.replay()

                total_reward += final_reward
                if loss is not None:
                    total_loss += loss
                state = next_state
            else:
                action = opponent.get_move(state, valid_moves)
                if action is None:
                    break
                state, reward, done, _ = env.step(action)

            agent_turn = not agent_turn
            steps += 1

        # 记录统计
        stats['episodes'].append(episode)
        stats['rewards'].append(total_reward)
        stats['losses'].append(total_loss / max(steps, 1))
        stats['epsilons'].append(agent.epsilon)
        stats['avg_steps'].append(steps)

        # 定期评估
        if (episode + 1) % config['eval_interval'] == 0:
            win_rate = evaluate_agent(agent, opponent, config['board_size'],
                                      num_games=config['eval_games'])
            stats['win_rates'].append(win_rate)

            print(f"\nEpisode {episode + 1}: Win Rate = {win_rate:.2%}, "
                  f"Epsilon = {agent.epsilon:.3f}, Avg Reward = {total_reward:.3f}")

            # 保存最佳模型
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                model_path = os.path.join(config['save_dir'], "best_model.pth")
                agent.save(model_path)
                print(f"保存最佳模型: {model_path} (胜率: {win_rate:.2%})")

            # 早停检查
            if win_rate > config['convergence_threshold']:
                convergence_count += 1
                if convergence_count >= 3:
                    print(f"模型已收敛，提前停止训练")
                    break
            else:
                convergence_count = 0

    # 保存最终模型
    final_path = os.path.join(config['save_dir'], "final_model.pth")
    agent.save(final_path)

    # 绘制训练曲线
    plot_training_curve(stats, config)

    return agent, stats


def evaluate_agent(agent, opponent, board_size, num_games=20):
    """评估智能体性能"""
    env = TrainingGomokuEnv(board_size)
    wins = 0

    for game_idx in range(num_games):
        state = env.reset()
        done = False

        # 随机决定先手
        agent_turn = np.random.random() < 0.5
        agent.player = 1 if agent_turn else 2
        opponent.player = 2 if agent_turn else 1

        while not done:
            valid_moves = env.get_valid_moves()

            if agent_turn:
                action = agent.get_move(state, valid_moves)
            else:
                action = opponent.get_move(state, valid_moves)

            if action is None:
                break

            state, reward, done, _ = env.step(action)
            agent_turn = not agent_turn

            if env.done:
                if env.winner == agent.player:
                    wins += 1
                break

    return wins / num_games


def plot_training_curve(stats, config):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(stats['rewards'])
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(stats['losses'])
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(stats['epsilons'])
    plt.title('Exploration Rate')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)

    if stats['win_rates']:
        plt.subplot(2, 3, 4)
        eval_episodes = [config['eval_interval'] * (i + 1)
                         for i in range(len(stats['win_rates']))]
        plt.plot(eval_episodes, stats['win_rates'])
        plt.title('Win Rate vs Rule-Based AI')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.plot(stats['avg_steps'])
    plt.title('Average Steps per Game')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"DQN Training - Board Size {config['board_size']}x{config['board_size']}")
    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(config['save_dir'], "training_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主训练函数"""
    config = {
        'board_sizes': [9, 13, 15],  # 要训练的棋盘尺寸
        'total_episodes': 2000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'eval_interval': 100,
        'eval_games': 20,
        'convergence_threshold': 0.8,  # 收敛阈值
        'save_dir': 'saved_models/dqn'
    }

    print("开始训练多尺寸DQN模型...")
    results = train_dqn_multisize(config)

    # 输出最终结果
    print("\n=== 训练结果总结 ===")
    for board_size, result in results.items():
        win_rates = result['stats']['win_rates']
        final_win_rate = win_rates[-1] if win_rates else 0
        print(f"棋盘尺寸 {board_size}x{board_size}: 最终胜率 = {final_win_rate:.2%}")


if __name__ == "__main__":
    main()