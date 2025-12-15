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

    def __init__(self, board_size=9, debug=False):
        self.board_size = board_size
        self.ai_models = {}
        self.current_ai = None
        self.debug = debug
        self.load_all_models()

    def load_all_models(self):
        """加载所有可用的AI模型 - 添加详细调试"""
        print(f"\n{'=' * 60}")
        print("AI加载调试信息")
        print(f"{'=' * 60}")

        try:
            # 1. 检查模块内容
            import models.rule_based_ai
            print(f"1. AI模块文件: {models.rule_based_ai.__file__}")
            print(f"2. 模块中的类: {[name for name in dir(models.rule_based_ai) if 'AI' in name]}")

            # 2. 检查RuleBasedAI类的属性
            if hasattr(models.rule_based_ai, 'RuleBasedAI'):
                ai_class = models.rule_based_ai.RuleBasedAI
                print(f"3. RuleBasedAI类存在")
                print(f"4. 类的方法: {[m for m in dir(ai_class) if not m.startswith('_')]}")

                # 3. 尝试创建实例
                try:
                    ai_instance = ai_class(
                        player=2,
                        board_size=self.board_size,
                        debug=self.debug,
                    )
                    print(f"5. AI实例创建成功: {ai_instance}")
                    print(f"6. AI名称: {getattr(ai_instance, 'name', '无name属性')}")
                    self.ai_models['rule_based'] = ai_instance
                    print("✅ 规则AI加载成功")

                except TypeError as e:
                    print(f"❌ 创建AI实例参数错误: {e}")
                    # 尝试简化版本
                    try:
                        ai_instance = ai_class(player=2, board_size=self.board_size)
                        self.ai_models['rule_based'] = ai_instance
                        print("✅ 使用简化参数创建成功")
                    except Exception as e2:
                        print(f"❌ 简化参数也失败: {e2}")
                        self._create_fallback_ai()

                except Exception as e:
                    print(f"❌ 创建AI实例失败: {e}")
                    self._create_fallback_ai()

            else:
                print("❌ RuleBasedAI类不存在")
                self._create_fallback_ai()

        except Exception as e:
            print(f"❌ 导入AI模块失败: {e}")
            self._create_fallback_ai()

        print(f"{'=' * 60}\n")

    def _create_fallback_ai(self):
        """创建备用AI"""
        print("创建备用AI...")
        self.ai_models['rule_based'] = BaseAI(player=2, name="FallbackAI")

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
        """加载DQN模型"""
        # 创建DQN智能体
        agent = DQNAgent(
            board_size=self.board_size,
            player=2,
            lr=0.001,
            epsilon=0.01  # 推理时使用很小的探索率
        )

        # 加载模型权重
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')

        # 加载模型状态
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
        """获取AI的移动 - 超级调试版"""
        if self.current_ai not in self.ai_models:
            valid_indices = np.where(valid_moves == 1)[0]
            return np.random.choice(valid_indices) if len(valid_indices) > 0 else None

        ai = self.ai_models[self.current_ai]
        ai.player = current_player

        # 强制开启调试
        ai.debug = True

        print(f"\n{'=' * 60}")
        print(f"AI决策开始 - 玩家{current_player}")
        print(f"AI类型: {self.current_ai}")

        # 显示人类棋子的位置
        human_positions = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if game_state[y][x] == 1:  # 人类是黑棋
                    human_positions.append((x, y))

        print(f"人类棋子位置: {human_positions}")

        # 检查是否有连续棋子
        for x, y in human_positions:
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count = 1
                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and game_state[ny][nx] == 1:
                        count += 1
                    else:
                        break
                if count >= 2:
                    print(f"发现人类连续棋子: ({x},{y})方向({dx},{dy}) 长度{count}")

        action = ai.get_move(game_state, valid_moves)

        if action is not None:
            x, y = action % self.board_size, action // self.board_size
            print(f"AI最终选择: ({x}, {y})")
        else:
            print("AI返回了None!")

        print(f"{'=' * 60}\n")
        return action


class GomokuGUI:
    """五子棋人机对弈界面 - 集成训练好的AI"""

    def __init__(self, board_size=9, debug=False):
        self.game = GomokuGame(board_size)
        self.board_size = board_size
        self.cell_size = 60
        self.margin = 50
        self.width = self.cell_size * board_size + 2 * self.margin
        self.height = self.cell_size * board_size + 2 * self.margin + 200  # 增加高度以容纳更多按钮

        # 调试模式
        self.debug = debug

        # 初始化AI管理器
        self.ai_manager = AIManager(board_size, debug=debug)

        # 颜色
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BROWN = (210, 180, 140)
        self.BOARD_COLOR = (220, 179, 92)
        self.GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 120, 255)
        self.YELLOW = (255, 200, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku AI Game - 增强版")

        # 字体
        self.font = self.get_chinese_font(24)
        self.large_font = self.get_chinese_font(36)
        self.small_font = self.get_chinese_font(18)
        self.tiny_font = self.get_chinese_font(14)

        # 游戏状态
        self.showing_menu = True
        self.game_started = False
        self.human_is_black = True
        self.selected_ai = 'rule_based'

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
                "切换AI": "Switch AI",
                "重新开始": "Restart Game",
                "返回主菜单": "Back to Menu",
                "继续游戏": "Continue",
                "五子棋AI对战": "Gomoku AI Game",
                "选择您的棋子颜色和AI对手:": "Select your piece color and AI opponent:",
                "黑棋（先手）": "Black (First)",
                "白棋（后手）": "White (Second)",
                "游戏结束": "Game Over"
            }
            english_text = english_mapping.get(text, text)
            return font.render(english_text, True, color)

    def draw_button(self, rect, text, bg_color, text_color, border_color=None, border_width=2):
        """绘制按钮"""
        if border_color:
            pygame.draw.rect(self.screen, border_color, rect, border_width)

        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, bg_color, inner_rect)

        text_surface = self.render_text(text, self.font, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

        return rect

    def show_main_menu(self):
        """显示主菜单"""
        self.screen.fill(self.WHITE)

        # 标题
        title_text = self.render_text("五子棋AI对战", self.large_font, self.BLACK)
        self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, 30))

        # 棋子颜色选择
        color_y = 100
        title = self.render_text("选择您的棋子颜色和AI对手:", self.font, self.BLACK)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, color_y))

        # 棋子颜色选择按钮
        color_y += 50
        black_rect = pygame.Rect(self.width // 2 - 250, color_y, 200, 50)
        white_rect = pygame.Rect(self.width // 2 + 50, color_y, 200, 50)

        # 绘制颜色选择按钮
        black_button = self.draw_button(
            black_rect,
            "黑棋（先手）",
            self.BLACK if self.human_is_black else self.WHITE,
            self.WHITE if self.human_is_black else self.BLACK,
            self.YELLOW if self.human_is_black else self.BROWN
        )

        white_button = self.draw_button(
            white_rect,
            "白棋（后手）",
            self.WHITE if not self.human_is_black else self.BLACK,
            self.BLACK if not self.human_is_black else self.WHITE,
            self.YELLOW if not self.human_is_black else self.BROWN
        )

        # AI选择
        ai_y = color_y + 80
        ai_title = self.render_text("选择AI对手:", self.font, self.BLACK)
        self.screen.blit(ai_title, (self.width // 2 - ai_title.get_width() // 2, ai_y))

        ai_y += 50
        ai_buttons = []
        ai_types = ['rule_based', 'dqn'] if 'dqn' in self.ai_manager.ai_models else ['rule_based']
        ai_names = {'rule_based': '规则AI', 'dqn': 'DQN AI'}

        for i, ai_type in enumerate(ai_types):
            ai_rect = pygame.Rect(self.width // 2 - 100 + (i - 0.5) * 220, ai_y, 180, 40)

            # 高亮选中的AI
            bg_color = self.BLUE if self.selected_ai == ai_type else self.WHITE
            text_color = self.WHITE if self.selected_ai == ai_type else self.BLACK
            border_color = self.YELLOW if self.selected_ai == ai_type else self.BROWN

            ai_button = self.draw_button(ai_rect, ai_names.get(ai_type, ai_type),
                                         bg_color, text_color, border_color)
            ai_buttons.append((ai_button, ai_type))

        # 开始游戏按钮
        start_y = ai_y + 60
        start_rect = pygame.Rect(self.width // 2 - 100, start_y, 200, 50)
        start_button = self.draw_button(start_rect, "开始游戏", self.GREEN, self.WHITE)

        # 显示当前选择
        info_y = start_y + 70
        info_text = self.render_text(f"当前选择: {ai_names.get(self.selected_ai, self.selected_ai)}",
                                     self.small_font, self.BLACK)
        self.screen.blit(info_text, (self.width // 2 - info_text.get_width() // 2, info_y))

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                # 检查棋子颜色选择
                if black_button.collidepoint(pos):
                    self.human_is_black = True
                elif white_button.collidepoint(pos):
                    self.human_is_black = False

                # 检查AI选择
                for button_rect, ai_type in ai_buttons:
                    if button_rect.collidepoint(pos):
                        self.selected_ai = ai_type

                # 开始游戏
                if start_button.collidepoint(pos):
                    self.game.set_players(self.human_is_black)
                    self.ai_manager.set_ai(self.selected_ai)
                    self.showing_menu = False
                    self.game_started = True
                    print(f"游戏开始！人类执{'黑' if self.human_is_black else '白'}棋，AI对手: {self.selected_ai}")

        pygame.display.flip()
        return True

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

        # 绘制坐标（调试模式）
        if self.debug:
            for y in range(self.board_size):
                for x in range(self.board_size):
                    center_x = self.margin + x * self.cell_size
                    center_y = self.margin + y * self.cell_size
                    coord_text = f"{x},{y}"
                    coord_surface = self.tiny_font.render(coord_text, True, (100, 100, 100))
                    self.screen.blit(coord_surface, (center_x - 12, center_y - 8))

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
        ai_names = {'rule_based': '规则AI', 'dqn': 'DQN AI'}
        current_ai_name = ai_names.get(self.ai_manager.current_ai, self.ai_manager.current_ai)
        ai_info = self.render_text(f"AI: {current_ai_name}", self.small_font, self.BLUE)
        self.screen.blit(ai_info, (self.width - self.margin - ai_info.get_width(), info_y))

        # 游戏结果
        if self.game.done:
            result_y = info_y + 30
            if self.game.winner == 0:
                result_text = self.render_text("平局!", self.large_font, self.ORANGE)
            else:
                winner = "人类" if self.game.winner == self.game.human_player else "AI"
                result_text = self.render_text(f"{winner}获胜!", self.large_font, self.RED)
            self.screen.blit(result_text, (self.width // 2 - result_text.get_width() // 2, result_y))

            # 按钮区域
            button_y = result_y + 50
            button_width = 120
            button_height = 40
            button_spacing = 20

            # 重新开始按钮
            restart_x = self.width // 2 - button_width - button_spacing // 2
            restart_rect = pygame.Rect(restart_x, button_y, button_width, button_height)

            # 返回主菜单按钮
            menu_x = self.width // 2 + button_spacing // 2
            menu_rect = pygame.Rect(menu_x, button_y, button_width, button_height)

            # 绘制按钮
            restart_button = self.draw_button(restart_rect, "重新开始", self.GREEN, self.WHITE)
            menu_button = self.draw_button(menu_rect, "返回主菜单", self.PURPLE, self.WHITE)

            return restart_button, menu_button
        else:
            # 游戏进行中，显示切换AI按钮
            switch_y = info_y + 30
            switch_rect = pygame.Rect(self.width - 120, switch_y, 100, 30)
            switch_button = self.draw_button(switch_rect, "切换AI", self.BLUE, self.WHITE)

            return switch_button, None

    def handle_ai_move(self):
        """处理AI的移动"""
        if self.game.get_current_player_type() == 'ai' and not self.game.done:
            # 添加短暂延迟，使AI思考可见
            pygame.time.delay(500)

            state = self.game.get_state()
            valid_moves = self.game.get_valid_moves()

            # 获取AI移动
            action = self.ai_manager.get_move(state, valid_moves, self.game.current_player)

            if action is not None:
                if self.debug:
                    n = self.board_size
                    x, y = action % n, action // n
                    print(f"[UI调试] AI返回action: {action}")
                    print(f"        转换坐标: (x={x}, y={y})")
                    print(f"        棋盘状态: board[{y}][{x}] = {state[y][x]}")

                if self.game.is_valid_move(action):
                    self.game.make_move(action)
                    return True
                elif self.debug:
                    print(f"[UI调试] 移动不合法!")

        return False

    def run(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        last_ai_time = 0
        ai_move_delay = 1000  # 1秒延迟

        # 存储按钮引用
        current_buttons = (None, None)

        running = True
        while running:
            current_time = pygame.time.get_ticks()

            if self.showing_menu:
                self.show_main_menu()
            else:
                # 检查是否需要AI移动
                ai_should_move = (self.game.get_current_player_type() == 'ai' and
                                  not self.game.done and
                                  current_time - last_ai_time > ai_move_delay)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()

                        if not self.game.done:
                            # 处理人类玩家移动
                            if self.game.get_current_player_type() == 'human':
                                # 计算棋盘坐标
                                board_x = round((pos[0] - self.margin) / self.cell_size)
                                board_y = round((pos[1] - self.margin) / self.cell_size)

                                if self.debug:
                                    print(f"[UI调试] 鼠标点击: 屏幕({pos[0]},{pos[1]}) -> 棋盘({board_x},{board_y})")

                                if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                                    # 注意：这里使用 board_y * n + board_x
                                    # 因为 board_y 是行，board_x 是列
                                    action = board_y * self.board_size + board_x
                                    if self.game.is_valid_move(action):
                                        if self.debug:
                                            x, y = action % self.board_size, action // self.board_size
                                            print(f"[UI调试] 人类落子: action={action}, 坐标(x={x}, y={y})")
                                        self.game.make_move(action)

                            # 检查切换AI按钮
                            switch_button, _ = current_buttons
                            if switch_button and switch_button.collidepoint(pos):
                                # 切换AI类型
                                current_ai = self.ai_manager.current_ai
                                if current_ai == 'rule_based' and 'dqn' in self.ai_manager.ai_models:
                                    self.ai_manager.set_ai('dqn')
                                else:
                                    self.ai_manager.set_ai('rule_based')
                                print(f"已切换到{self.ai_manager.current_ai}")

                        elif self.game.done:
                            # 游戏结束，检查按钮
                            restart_button, menu_button = current_buttons
                            if restart_button and restart_button.collidepoint(pos):
                                # 重新开始游戏
                                self.game.reset()
                                self.game.set_players(self.human_is_black)
                                print(f"重新开始游戏")

                            elif menu_button and menu_button.collidepoint(pos):
                                # 返回主菜单
                                self.showing_menu = True
                                self.game_started = False
                                print(f"返回主菜单")

                # 处理AI移动
                if ai_should_move:
                    if self.handle_ai_move():
                        last_ai_time = current_time

                # 绘制棋盘和获取按钮
                current_buttons = self.draw_board()

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    try:
        # 开启调试模式以查看坐标转换
        gui = GomokuGUI(board_size=9, debug=True)
        gui.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()