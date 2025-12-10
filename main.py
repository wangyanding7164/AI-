# main.py
import pygame
import sys
import numpy as np
import os
import torch
import torch.nn as nn

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from models.dqn_ai import DQNAgent, DQN
    from models.rule_based_ai import RuleBasedAI
    from models.base_ai import BaseAI
except ImportError as e:
    print(f"导入错误: {e}")


    # 如果导入失败，创建简化版本
    class BaseAI:
        def __init__(self, player=1, name="BaseAI"):
            self.player = player
            self.name = name

        def get_move(self, game_state, valid_moves):
            valid_indices = np.where(valid_moves == 1)[0]
            return np.random.choice(valid_indices) if len(valid_indices) > 0 else None


class GomokuGame:
    def __init__(self, board_size=9):
        self.n = board_size
        self.board = None
        self.current_player = None
        self.done = False
        self.winner = None
        self.ai_player = None
        self.human_player = None
        self.reset()

    def set_players(self, human_is_black=True):
        """设置人类和AI的角色"""
        if human_is_black:
            self.human_player = 1
            self.ai_player = 2
        else:
            self.human_player = 2
            self.ai_player = 1
        print(f"人类执{'黑' if human_is_black else '白'}棋, AI执{'白' if human_is_black else '黑'}棋")

    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def is_valid_move(self, action):
        """判断落子是否合法"""
        if action is None:
            return False
        x, y = action % self.n, action // self.n
        return (0 <= x < self.n and 0 <= y < self.n and self.board[y][x] == 0)

    def make_move(self, action):
        """执行落子"""
        if not self.is_valid_move(action) or self.done:
            return self.get_state(), 0, True, {}

        x, y = action % self.n, action // self.n
        self.board[y][x] = self.current_player

        if self.check_win(y, x):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
            reward = 0.1
        else:
            self.current_player = 3 - self.current_player
            reward = 0.0

        return self.get_state(), reward, self.done, {}

    def check_win(self, y, x):
        """检查是否获胜"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player = self.board[y][x]

        for dy, dx in directions:
            count = 1
            for i in range(1, 5):
                ny, nx = y + dy * i, x + dx * i
                if 0 <= ny < self.n and 0 <= nx < self.n and self.board[ny][nx] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                ny, nx = y - dy * i, x - dx * i
                if 0 <= ny < self.n and 0 <= nx < self.n and self.board[ny][nx] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def get_state(self):
        """获取当前棋盘状态"""
        return self.board.copy()

    def get_reward(self):
        """定义奖励函数"""
        if not self.done:
            return 0
        if self.winner == 1:
            return 1
        elif self.winner == 2:
            return -1
        else:
            return 0.1

    def get_valid_moves(self):
        """获取所有合法移动的掩码"""
        return np.array([1 if self.is_valid_move(i) else 0 for i in range(self.n * self.n)])

    def get_current_player_type(self):
        """获取当前玩家类型"""
        if self.ai_player is None or self.human_player is None:
            return 'human'
        return 'human' if self.current_player == self.human_player else 'ai'

    def render_text(self):
        """文本渲染棋盘（用于调试）"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        result = []
        for row in self.board:
            result.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(result)


class AIManager:
    """AI管理器，负责加载和使用不同的AI模型"""

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.ai_models = {}
        self.current_ai = None
        self.load_all_models()

    def load_all_models(self):
        """加载所有可用的AI模型"""
        try:
            # 加载规则AI
            self.ai_models['rule_based'] = RuleBasedAI(player=2, board_size=self.board_size)
            print("✅ 规则AI加载成功")
        except Exception as e:
            print(f"❌ 规则AI加载失败: {e}")
            # 创建备用规则AI
            self.ai_models['rule_based'] = BaseAI(player=2, name="FallbackRuleAI")

        try:
            # 尝试加载DQN模型
            dqn_model_path = self.find_latest_model()
            if dqn_model_path:
                self.ai_models['dqn'] = self.load_dqn_model(dqn_model_path)
                print(f"✅ DQN AI加载成功: {dqn_model_path}")
                self.current_ai = 'dqn'  # 默认使用DQN
            else:
                print("⚠️ 未找到DQN模型，使用规则AI")
                self.current_ai = 'rule_based'
        except Exception as e:
            print(f"❌ DQN AI加载失败: {e}")
            self.current_ai = 'rule_based'

    def find_latest_model(self):
        """查找最新的模型文件"""
        possible_paths = [
            'saved_models/dqn_final.pth',
            'saved_models/dqn/best_model.pth',
            'saved_models/size_9/dqn_final.pth',
            'saved_models/size_9/best_model.pth'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 搜索整个saved_models目录
        for root, dirs, files in os.walk('saved_models'):
            for file in files:
                if file.endswith(('.pth', '.pt')) and 'dqn' in file.lower():
                    return os.path.join(root, file)
        return None

    def load_dqn_model(self, model_path):
        """加载DQN模型[6,8](@ref)"""
        # 创建DQN智能体
        agent = DQNAgent(
            board_size=self.board_size,
            player=2,
            lr=0.001,
            epsilon=0.01  # 推理时使用很小的探索率
        )

        # 加载模型权重[6](@ref)
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')

        # 加载模型状态[8](@ref)
        if 'policy_net_state_dict' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        else:
            # 如果保存的是整个模型
            agent.policy_net = checkpoint

        agent.policy_net.eval()  # 设置为评估模式
        return agent

    def set_ai(self, ai_type):
        """设置当前使用的AI"""
        if ai_type in self.ai_models:
            self.current_ai = ai_type
            print(f"切换到AI: {ai_type}")
        else:
            print(f"未知的AI类型: {ai_type}")

    def get_move(self, game_state, valid_moves, current_player):
        """获取AI的移动"""
        if self.current_ai not in self.ai_models:
            # 回退到随机
            valid_indices = np.where(valid_moves == 1)[0]
            return np.random.choice(valid_indices) if len(valid_indices) > 0 else None

        ai = self.ai_models[self.current_ai]
        ai.player = current_player  # 设置当前玩家
        return ai.get_move(game_state, valid_moves)


class GomokuGUI:
    """五子棋人机对弈界面 - 集成训练好的AI"""

    def __init__(self, board_size=9):
        self.game = GomokuGame(board_size)
        self.board_size = board_size
        self.cell_size = 60
        self.margin = 50
        self.width = self.cell_size * board_size + 2 * self.margin
        self.height = self.cell_size * board_size + 2 * self.margin + 150

        # 初始化AI管理器
        self.ai_manager = AIManager(board_size)

        # 颜色
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BROWN = (210, 180, 140)
        self.BOARD_COLOR = (220, 179, 92)
        self.GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 120, 255)
        self.YELLOW = (255, 200, 0)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku AI Game - 训练模型集成版")

        # 字体
        self.font = self.get_chinese_font(24)
        self.large_font = self.get_chinese_font(36)
        self.small_font = self.get_chinese_font(18)

        # AI选择界面状态
        self.showing_ai_selection = False
        self.ai_buttons = []

    def get_chinese_font(self, size):
        """获取支持中文的字体"""
        try:
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "C:/Windows/Fonts/msyh.ttc",
                None
            ]

            for font_path in font_paths:
                try:
                    if font_path and os.path.exists(font_path):
                        font = pygame.font.Font(font_path, size)
                        test_surface = font.render("测试", True, (0, 0, 0))
                        if test_surface.get_width() > 0:
                            return font
                    else:
                        font = pygame.font.SysFont(None, size)
                        return font
                except:
                    continue
            return pygame.font.Font(None, size)
        except:
            return pygame.font.Font(None, size)

    def render_text(self, text, font, color):
        """渲染文本"""
        try:
            return font.render(text, True, color)
        except:
            # 如果中文渲染失败，使用英文回退
            english_mapping = {
                "选择AI对手": "Select AI Opponent",
                "规则AI": "Rule-based AI",
                "DQN AI": "DQN AI",
                "开始游戏": "Start Game",
                "人类获胜!": "Human Wins!",
                "AI获胜!": "AI Wins!",
                "平局!": "Draw!",
                "当前AI": "Current AI",
                "切换AI": "Switch AI"
            }
            english_text = english_mapping.get(text, text)
            return font.render(english_text, True, color)

    def choose_role_and_ai(self):
        """角色和AI选择界面"""
        choosing = True
        human_is_black = True
        selected_ai = self.ai_manager.current_ai

        while choosing:
            self.screen.fill(self.WHITE)

            # 显示选择提示
            title_text = self.render_text("选择您的棋子颜色和AI对手:", self.large_font, self.BLACK)
            self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, 30))

            # 棋子颜色选择按钮
            color_y = 100
            black_rect = pygame.Rect(self.width // 2 - 250, color_y, 200, 50)
            white_rect = pygame.Rect(self.width // 2 + 50, color_y, 200, 50)

            pygame.draw.rect(self.screen, self.BROWN, black_rect, 2)
            pygame.draw.rect(self.screen, self.BROWN, white_rect, 2)

            if human_is_black:
                pygame.draw.rect(self.screen, self.YELLOW, black_rect, 3)
            else:
                pygame.draw.rect(self.screen, self.YELLOW, white_rect, 3)

            black_text = self.render_text("黑棋（先手）", self.font, self.BLACK)
            white_text = self.render_text("白棋（后手）", self.font, self.BLACK)

            self.screen.blit(black_text, (black_rect.centerx - black_text.get_width() // 2,
                                          black_rect.centery - black_text.get_height() // 2))
            self.screen.blit(white_text, (white_rect.centerx - white_text.get_width() // 2,
                                          white_rect.centery - white_text.get_height() // 2))

            # AI选择按钮
            ai_y = 180
            ai_title = self.render_text("选择AI对手:", self.font, self.BLACK)
            self.screen.blit(ai_title, (self.width // 2 - ai_title.get_width() // 2, ai_y))

            ai_buttons = []
            ai_types = ['rule_based', 'dqn'] if 'dqn' in self.ai_manager.ai_models else ['rule_based']
            ai_names = {'rule_based': '规则AI', 'dqn': 'DQN AI'}

            for i, ai_type in enumerate(ai_types):
                ai_rect = pygame.Rect(self.width // 2 - 100 + i * 220, ai_y + 40, 180, 40)
                ai_buttons.append((ai_rect, ai_type))

                # 绘制按钮
                if selected_ai == ai_type:
                    pygame.draw.rect(self.screen, self.BLUE, ai_rect)
                    text_color = self.WHITE
                else:
                    pygame.draw.rect(self.screen, self.BROWN, ai_rect, 2)
                    text_color = self.BLACK

                ai_text = self.render_text(ai_names.get(ai_type, ai_type), self.font, text_color)
                self.screen.blit(ai_text, (ai_rect.centerx - ai_text.get_width() // 2,
                                           ai_rect.centery - ai_text.get_height() // 2))

            # 开始游戏按钮
            start_rect = pygame.Rect(self.width // 2 - 100, ai_y + 120, 200, 50)
            pygame.draw.rect(self.screen, self.GREEN, start_rect)
            start_text = self.render_text("开始游戏", self.font, self.WHITE)
            self.screen.blit(start_text, (start_rect.centerx - start_text.get_width() // 2,
                                          start_rect.centery - start_text.get_height() // 2))

            # 显示当前选择的AI信息
            info_y = ai_y + 190
            info_text = self.render_text(f"当前AI: {ai_names.get(selected_ai, selected_ai)}", self.small_font,
                                         self.BLACK)
            self.screen.blit(info_text, (self.width // 2 - info_text.get_width() // 2, info_y))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    # 检查棋子颜色选择
                    if black_rect.collidepoint(pos):
                        human_is_black = True
                    elif white_rect.collidepoint(pos):
                        human_is_black = False

                    # 检查AI选择
                    for rect, ai_type in ai_buttons:
                        if rect.collidepoint(pos):
                            selected_ai = ai_type

                    # 开始游戏
                    if start_rect.collidepoint(pos):
                        self.game.set_players(human_is_black)
                        self.ai_manager.set_ai(selected_ai)
                        choosing = False
                        print(f"游戏开始！人类执{'黑' if human_is_black else '白'}棋，AI对手: {selected_ai}")

            pygame.display.flip()

    def draw_board(self):
        """绘制棋盘和游戏信息"""
        self.screen.fill(self.BROWN)

        # 绘制棋盘区域
        board_rect = pygame.Rect(self.margin - 10, self.margin - 10,
                                 self.cell_size * self.board_size + 20,
                                 self.cell_size * self.board_size + 20)
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)

        # 绘制网格线
        for i in range(self.board_size):
            # 横线
            start_x, end_x = self.margin, self.margin + self.cell_size * (self.board_size - 1)
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.BLACK, (start_x, y), (end_x, y), 2)

            # 竖线
            start_y, end_y = self.margin, self.margin + self.cell_size * (self.board_size - 1)
            x = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.BLACK, (x, start_y), (x, end_y), 2)

        # 绘制棋子
        for y in range(self.board_size):
            for x in range(self.board_size):
                center_x = self.margin + x * self.cell_size
                center_y = self.margin + y * self.cell_size

                if self.game.board[y][x] == 1:
                    pygame.draw.circle(self.screen, self.BLACK, (center_x, center_y), self.cell_size // 2 - 4)
                elif self.game.board[y][x] == 2:
                    pygame.draw.circle(self.screen, self.WHITE, (center_x, center_y), self.cell_size // 2 - 4)
                    pygame.draw.circle(self.screen, self.BLACK, (center_x, center_y), self.cell_size // 2 - 4, 1)

        # 显示游戏信息
        info_y = self.margin + self.cell_size * self.board_size + 20

        # 当前玩家信息
        current_player_text = "当前回合: " + ("人类" if self.game.get_current_player_type() == 'human' else "AI")
        status_text = self.render_text(current_player_text, self.font, self.BLACK)
        self.screen.blit(status_text, (self.margin, info_y))

        # AI信息
        ai_info = self.render_text(f"AI: {self.ai_manager.current_ai}", self.small_font, self.BLUE)
        self.screen.blit(ai_info, (self.width - self.margin - ai_info.get_width(), info_y))

        # 游戏结果
        if self.game.done:
            result_y = info_y + 30
            if self.game.winner == 0:
                result_text = self.render_text("平局!", self.large_font, self.RED)
            else:
                winner = "人类" if self.game.winner == self.game.human_player else "AI"
                result_text = self.render_text(f"{winner}获胜!", self.large_font, self.RED)
            self.screen.blit(result_text, (self.width // 2 - result_text.get_width() // 2, result_y))

            # 重新开始按钮
            restart_y = result_y + 50
            restart_rect = pygame.Rect(self.width // 2 - 80, restart_y, 160, 40)
            pygame.draw.rect(self.screen, self.GREEN, restart_rect)
            restart_text = self.render_text("重新开始", self.font, self.WHITE)
            self.screen.blit(restart_text, (restart_rect.centerx - restart_text.get_width() // 2,
                                            restart_rect.centery - restart_text.get_height() // 2))

        # 切换AI按钮（游戏未结束时）
        if not self.game.done:
            switch_y = info_y + 30
            switch_rect = pygame.Rect(self.width - 120, switch_y, 100, 30)
            pygame.draw.rect(self.screen, self.BLUE, switch_rect)
            switch_text = self.render_text("切换AI", self.small_font, self.WHITE)
            self.screen.blit(switch_text, (switch_rect.centerx - switch_text.get_width() // 2,
                                           switch_rect.centery - switch_text.get_height() // 2))

    def handle_ai_move(self):
        """处理AI的移动"""
        if self.game.get_current_player_type() == 'ai' and not self.game.done:
            # 添加短暂延迟，使AI思考可见
            pygame.time.delay(500)

            state = self.game.get_state()
            valid_moves = self.game.get_valid_moves()

            # 获取AI移动
            action = self.ai_manager.get_move(state, valid_moves, self.game.current_player)

            if action is not None and self.game.is_valid_move(action):
                self.game.make_move(action)
                return True
        return False

    def run(self):
        """运行游戏主循环"""
        self.choose_role_and_ai()
        clock = pygame.time.Clock()

        # AI自动移动计时器
        ai_move_delay = 1000  # 1秒延迟
        last_ai_time = 0

        running = True
        while running:
            current_time = pygame.time.get_ticks()
            ai_should_move = (self.game.get_current_player_type() == 'ai' and
                              not self.game.done and
                              current_time - last_ai_time > ai_move_delay)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game.done:
                    pos = pygame.mouse.get_pos()

                    # 处理人类玩家移动
                    if self.game.get_current_player_type() == 'human':
                        board_x = round((pos[0] - self.margin) / self.cell_size)
                        board_y = round((pos[1] - self.margin) / self.cell_size)

                        if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                            action = board_y * self.board_size + board_x
                            if self.game.is_valid_move(action):
                                self.game.make_move(action)

                    # 检查切换AI按钮
                    switch_rect = pygame.Rect(self.width - 120, self.margin + self.cell_size * self.board_size + 50,
                                              100, 30)
                    if switch_rect.collidepoint(pos):
                        # 切换AI类型
                        current_ai = self.ai_manager.current_ai
                        if current_ai == 'rule_based' and 'dqn' in self.ai_manager.ai_models:
                            self.ai_manager.set_ai('dqn')
                        else:
                            self.ai_manager.set_ai('rule_based')
                        print(f"已切换到{self.ai_manager.current_ai}")

                elif event.type == pygame.MOUSEBUTTONDOWN and self.game.done:
                    # 检查重新开始按钮
                    restart_rect = pygame.Rect(self.width // 2 - 80,
                                               self.margin + self.cell_size * self.board_size + 100,
                                               160, 40)
                    if restart_rect.collidepoint(pos):
                        self.game.reset()
                        self.choose_role_and_ai()

            # 处理AI移动
            if ai_should_move:
                if self.handle_ai_move():
                    last_ai_time = current_time

            self.draw_board()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    try:
        gui = GomokuGUI(board_size=9)
        gui.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()