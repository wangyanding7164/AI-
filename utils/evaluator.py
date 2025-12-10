# utils/evaluator.py
import numpy as np
from training.training_env import TrainingGomokuEnv


def evaluate_ai(ai1, ai2, board_size=9, num_games=100, verbose=False):
    """评估两个AI的对战胜率"""
    env = TrainingGomokuEnv(board_size)

    wins_ai1 = 0
    wins_ai2 = 0
    draws = 0

    for game_idx in range(num_games):
        state = env.reset()
        done = False

        # 随机决定先手
        ai1_turn = np.random.random() < 0.5
        if ai1_turn:
            ai1.player = 1
            ai2.player = 2
        else:
            ai1.player = 2
            ai2.player = 1

        while not done:
            valid_moves = env.get_valid_moves()

            if ai1_turn:
                action = ai1.get_move(state, valid_moves)
            else:
                action = ai2.get_move(state, valid_moves)

            if action is None:
                break

            state, reward, done, info = env.step(action)
            ai1_turn = not ai1_turn

            if env.done:
                if env.winner == ai1.player:
                    wins_ai1 += 1
                elif env.winner == ai2.player:
                    wins_ai2 += 1
                else:
                    draws += 1
                break

        if verbose and (game_idx + 1) % 20 == 0:
            print(f"已完成 {game_idx + 1}/{num_games} 局")

    win_rate_ai1 = wins_ai1 / num_games
    win_rate_ai2 = wins_ai2 / num_games
    draw_rate = draws / num_games

    if verbose:
        print(f"\n评估结果 (共{num_games}局):")
        print(f"  {ai1.name} 胜率: {win_rate_ai1:.2%} ({wins_ai1}胜)")
        print(f"  {ai2.name} 胜率: {win_rate_ai2:.2%} ({wins_ai2}胜)")
        print(f"  平局率: {draw_rate:.2%} ({draws}平)")

    return win_rate_ai1