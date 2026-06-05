from __future__ import annotations

import cv2
import numpy as np

from ar_shuffleboard.models.shuffleboard import *
from ar_shuffleboard.utils.constants import GameConfig

GAME_BOARD_SCALE: int = 2


class GameView:
    def __init__(
        self,
        game_model: Shuffleboard,
        window_title: str,
        board_size: tuple[int, int],
        callback,
    ):
        self.game_model = game_model
        self.window_title = window_title
        self.board_size = board_size

        cv2.namedWindow(self.window_title)
        if callback is not None:
            cv2.setMouseCallback(self.window_title, callback)

        self.last_mouse_pos = None

        self.game_canvas: np.ndarray
        self.clearGameCanvas()
        self.game_overlay: np.ndarray
        self.clearGameOverlay()

    def showGameWindow(self):
        """게임 윈도우 출력"""
        self.refreshGameView()
        image = self.merge4ChannelImage(self.game_canvas, self.game_overlay)
        cv2.imshow(self.window_title, image)

    def refreshGameView(self):
        """게임 뷰 새로고침"""
        self.clearGameCanvas()
        self.clearGameOverlay()
        self.drawGameCanvas()
        self.drawGameOverlay()

    def merge4ChannelImage(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        # 1. 각 이미지의 채널 분리 (B, G, R, A)
        b1, g1, r1, a1 = cv2.split(img1)
        b2, g2, r2, a2 = cv2.split(img2)

        # 2. 알파 값을 0~1 범위로 정규화
        alpha_fg = a1 / 255.0
        alpha_bg = a2 / 255.0

        # 3. 알파 블렌딩 연산
        # 수식: Result = (Foreground * alpha_fg) + (Background * alpha_bg * (1 - alpha_fg))
        out_b = (b1 * alpha_fg) + (b2 * alpha_bg * (1.0 - alpha_fg))
        out_g = (g1 * alpha_fg) + (g2 * alpha_bg * (1.0 - alpha_fg))
        out_r = (r1 * alpha_fg) + (r2 * alpha_bg * (1.0 - alpha_fg))

        # 4. 결과 이미지의 알파 채널 계산 (전경 알파 + 배경 알파)
        out_a = a1 + (a2 * (255 - a1) / 255.0)

        # 5. 0~255 범위의 8비트 정수로 변환하여 채널 병합
        merged_img = cv2.merge(
            [
                np.clip(out_b, 0, 255).astype(np.uint8),
                np.clip(out_g, 0, 255).astype(np.uint8),
                np.clip(out_r, 0, 255).astype(np.uint8),
                np.clip(out_a, 0, 255).astype(np.uint8),
            ]
        )

        return merged_img

    # === Game Canvas ===

    def clearGameCanvas(self):
        w, h = self.board_size
        self.game_canvas = np.zeros((GAME_BOARD_SCALE * h, GAME_BOARD_SCALE * w, 4), dtype=np.uint8)

    def drawScoreArea(self):
        """점수 영역 그리기"""
        x, y = self.getGameOrigin()
        w, h = self.board_size

        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            x1 = int(x + (prev_area - score_area.ratio) * w)
            x2 = int(x + prev_area * w)
            color = score_area.color
            cv2.rectangle(self.game_canvas, (x1, y), (x2, y + h), color, -1)

    def drawGameBoarderLines(self):
        """게임 보드 경계선 그리기"""
        x, y = self.getGameOrigin()
        w, h = self.board_size

        line_color = (127, 127, 127, 255)  # black
        # 모서리 경계선
        cv2.line(self.game_canvas, (x, y), (x + w, y), line_color, 2)  # 위
        cv2.line(self.game_canvas, (x, y + h), (x + w, y + h), line_color, 2)  # 아래
        cv2.line(self.game_canvas, (x, y), (x, y + h), line_color, 2)  # 좌
        cv2.line(self.game_canvas, (x + w, y), (x + w, y + h), line_color, 2)  # 우
        # 점수 경계선
        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            _x = int(x + prev_area * w)
            cv2.line(self.game_canvas, (_x, y), (_x, y + h), line_color, 2)

    def drawPucks(self):
        """게임 퍽 그리기"""
        x, y = self.getGameOrigin()
        for player in self.game_model.getPlayers():
            for puck in player.pucks:
                cx, cy = puck.body.position
                center = (int(cx + x), int(cy + y))
                cv2.circle(self.game_canvas, center, GameConfig.PUCK_RADIUS, player.color, -1)

    def drawGameCanvas(self):
        """게임 캔버스 그리기"""
        self.drawScoreArea()
        self.drawGameBoarderLines()
        self.drawPucks()

    def getGameCanvas(self):
        """게임 캔버스 반환"""
        return self.game_canvas

    def getProjectBoardCanvas(self) -> np.ndarray:
        """
        게임 보드(원본 크기, 보드 바깥 여백 없음)를 RGBA 포맷으로 반환합니다.
        이는 카메라 프레임에 투영할 때 사용됩니다.
        """
        w, h = self.board_size
        canvas = np.zeros((h, w, 4), dtype=np.uint8)

        # draw score areas
        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            x1 = int((prev_area - score_area.ratio) * w)
            x2 = int(prev_area * w)
            color = score_area.color
            cv2.rectangle(canvas, (x1, 0), (x2, h), color, -1)

        # border lines
        line_color = (127, 127, 127, 255)
        cv2.line(canvas, (0, 0), (w, 0), line_color, 2)
        cv2.line(canvas, (0, h), (w, h), line_color, 2)
        cv2.line(canvas, (0, 0), (0, h), line_color, 2)
        cv2.line(canvas, (w, 0), (w, h), line_color, 2)

        # score boundaries
        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            _x = int(prev_area * w)
            cv2.line(canvas, (_x, 0), (_x, h), line_color, 2)

        # pucks
        for player in self.game_model.getPlayers():
            for puck in player.pucks:
                cx, cy = puck.body.position
                center = (int(cx), int(cy))
                cv2.circle(canvas, center, GameConfig.PUCK_RADIUS, player.color, -1)

        return canvas

    # === Game Overlay ===

    def clearGameOverlay(self):
        w, h = self.board_size
        self.game_overlay = np.zeros((GAME_BOARD_SCALE * h, GAME_BOARD_SCALE * w, 4), dtype=np.uint8)

    def drawPuckArrow(self):
        if self.last_mouse_pos is None:
            return
        origin_x, origin_y = self.getGameOrigin()
        puck = self.game_model.getReadyPuck()
        if puck is not None:
            puck_x, puck_y = puck.body.position
            start_pos = (int(puck_x + origin_x), int(puck_y + origin_y))
            cv2.arrowedLine(self.game_overlay, start_pos, self.last_mouse_pos, (20, 20, 20), 2, tipLength=0.2)

    def drawPlayerInfo(self):
        """각 플레이어 별 남은 퍽의 개수, 현재 플레이어 오버레이 그리기"""
        if not self.game_model.shuffleboard_fsm.is_end():  # 게임 진행 중
            origin_y = 30
            for player in self.game_model.getPlayers():
                text = f"{player.name} has {player.left_pucks} pucks"
                if player == self.game_model.getCurrentPlayer():
                    text = f"{text} <- Turn"
                cv2.putText(self.game_overlay, text, (5, origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.color, 2)
                origin_y += 20

    def drawPlayerScores(self):
        origin_y = 30
        for player in self.game_model.getPlayers():
            text = f"{player.name} Score: {player.score}"
            cv2.putText(self.game_overlay, text, (400, origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.color, 2)
            origin_y += 20

    def drawGameOverlay(self):
        """게임 오버레이 그리기"""
        self.drawPlayerInfo()
        self.drawPlayerScores()

    def getGameOverlay(self):
        """게임 오버레이 반환"""
        return self.game_overlay

    # === etc. ===

    def getGameOrigin(self) -> tuple[int, int]:
        """게임 이미지 내 게임 보드의 오프셋을 반환한다."""
        w, h = self.board_size
        s = (GAME_BOARD_SCALE - 1) / 2
        return (int(w * s), int(h * s))

    def setLastMousePos(self, pos):
        self.last_mouse_pos = pos
