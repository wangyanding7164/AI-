import pygame
import sys
import numpy as np
import os
from gomokuGame import GomokuGame
class GomokuGUI:
    def __init__(self, board_size=9):
        self.game = GomokuGame(board_size)
        self.board_size = board_size
        self.cell_size = 60
        self.margin = 50
        self.width = self.cell_size * board_size + 2 * self.margin
        self.height = self.cell_size * board_size + 2 * self.margin + 100

        # 颜色
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BROWN = (210, 180, 140)
        self.BOARD_COLOR = (220, 179, 92)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku AI Game")  # 英文标题避免乱码

        # 修复中文显示：使用系统字体
        self.font = self.get_chinese_font(24)
        self.large_font = self.get_chinese_font(36)

    def get_chinese_font(self, size):
        """获取支持中文的字体"""
        # 尝试多种字体方案
        font_paths = [
            # Windows 系统字体
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            # macOS 系统字体
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            # Linux 系统字体
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            # 如果系统字体都不可用，使用默认字体
            None
        ]

        for font_path in font_paths:
            try:
                if font_path and os.path.exists(font_path):
                    font = pygame.font.Font(font_path, size)
                    # 测试字体是否能显示中文
                    test_surface = font.render("测试", True, (0, 0, 0))
                    if test_surface.get_width() > 0:
                        print(f"成功加载字体: {font_path}")
                        return font
                else:
                    # 使用pygame默认字体
                    font = pygame.font.SysFont(None, size)
                    test_surface = font.render("Test", True, (0, 0, 0))
                    if test_surface.get_width() > 0:
                        print("使用系统默认字体")
                        return font
            except:
                continue

        # 最后尝试使用None（pygame默认）
        print("使用pygame默认字体（可能不支持中文）")
        return pygame.font.Font(None, size)

    def render_text(self, text, font, color):
        """渲染文本，如果中文显示有问题则使用英文回退"""
        try:
            # 先尝试渲染中文
            surface = font.render(text, True, color)
            if surface.get_width() > 0:
                return surface
        except:
            pass

        # 如果中文渲染失败，使用英文回退
        english_text = self.chinese_to_english(text)
        return font.render(english_text, True, color)

    def chinese_to_english(self, text):
        """中文到英文的映射"""
        mapping = {
            "选择您的棋子颜色:": "Choose your color:",
            "黑棋（先手）": "Black (First)",
            "白棋（后手）": "White (Second)",
            "开始游戏": "Start Game",
            "人类获胜!": "Human Wins!",
            "AI获胜!": "AI Wins!",
            "平局!": "Draw!",
            "当前回合: 人类": "Current: Human",
            "当前回合: AI": "Current: AI",
            "人类执黑棋, AI执白棋": "Human: Black, AI: White",
            "人类执白棋, AI执黑棋": "Human: White, AI: Black"
        }
        return mapping.get(text, text)

    def choose_role(self):
        """角色选择界面"""
        choosing = True
        human_is_black = True

        while choosing:
            self.screen.fill(self.WHITE)

            # 显示选择提示
            text_surface = self.render_text("选择您的棋子颜色:", self.large_font, self.BLACK)
            self.screen.blit(text_surface, (self.width // 2 - text_surface.get_width() // 2, 50))

            # 绘制选择按钮
            button_width, button_height = 200, 50
            black_rect = pygame.Rect(self.width // 2 - button_width // 2, 120, button_width, button_height)
            white_rect = pygame.Rect(self.width // 2 - button_width // 2, 190, button_width, button_height)
            start_rect = pygame.Rect(self.width // 2 - button_width // 2, 260, button_width, button_height)

            # 绘制按钮
            pygame.draw.rect(self.screen, self.BROWN, black_rect, 2)
            pygame.draw.rect(self.screen, self.BROWN, white_rect, 2)
            pygame.draw.rect(self.screen, (0, 200, 0), start_rect)

            # 高亮当前选择
            if human_is_black:
                pygame.draw.rect(self.screen, (255, 200, 0), black_rect, 3)
            else:
                pygame.draw.rect(self.screen, (255, 200, 0), white_rect, 3)

            # 按钮文字
            black_text = self.render_text("黑棋（先手）", self.font, self.BLACK)
            white_text = self.render_text("白棋（后手）", self.font, self.BLACK)
            start_text = self.render_text("开始游戏", self.font, self.WHITE)

            self.screen.blit(black_text, (black_rect.centerx - black_text.get_width() // 2,
                                          black_rect.centery - black_text.get_height() // 2))
            self.screen.blit(white_text, (white_rect.centerx - white_text.get_width() // 2,
                                          white_rect.centery - white_text.get_height() // 2))
            self.screen.blit(start_text, (start_rect.centerx - start_text.get_width() // 2,
                                          start_rect.centery - start_text.get_height() // 2))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if black_rect.collidepoint(pos):
                        human_is_black = True
                    elif white_rect.collidepoint(pos):
                        human_is_black = False
                    elif start_rect.collidepoint(pos):
                        self.game.set_players(human_is_black)
                        choosing = False

            pygame.display.flip()

    def run(self):
        """运行游戏主循环"""
        self.choose_role()
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game.done:
                    if self.game.get_current_player_type() == 'human':
                        pos = pygame.mouse.get_pos()
                        board_x = (pos[0] - self.margin) // self.cell_size
                        board_y = (pos[1] - self.margin) // self.cell_size

                        if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                            action = board_y * self.board_size + board_x
                            if self.game.is_valid_move(action):
                                self.game.make_move(action)

                                if not self.game.done and self.game.get_current_player_type() == 'ai':
                                    ai_action = self.get_ai_move()
                                    if ai_action is not None:
                                        self.game.make_move(ai_action)

            self.draw_board()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def get_ai_move(self):
        """AI落子决策"""
        valid_moves = self.game.get_valid_moves()
        valid_indices = [i for i, valid in enumerate(valid_moves) if valid]
        return np.random.choice(valid_indices) if valid_indices else None

    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(self.BROWN)

        # 绘制棋盘区域
        board_rect = pygame.Rect(self.margin - 10, self.margin - 10,
                                 self.cell_size * self.board_size + 20,
                                 self.cell_size * self.board_size + 20)
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)

        # 绘制网格线
        for i in range(self.board_size):
            start_x = self.margin
            end_x = self.margin + self.cell_size * (self.board_size - 1)
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.BLACK, (start_x, y), (end_x, y), 2)

            start_y = self.margin
            end_y = self.margin + self.cell_size * (self.board_size - 1)
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

        if self.game.done:
            if self.game.winner == 0:
                status = "平局!"
            else:
                winner = "人类" if self.game.winner == self.game.human_player else "AI"
                status = f"{winner}获胜!"
        else:
            current = "人类" if self.game.get_current_player_type() == 'human' else "AI"
            status = f"当前回合: {current}"

        status_text = self.render_text(status, self.font, self.BLACK)
        self.screen.blit(status_text, (self.margin, info_y))