from __future__ import annotations

import cv2
import numpy as np

from ar_shuffleboard.models.shuffleboard import PlayerData, Puck, SCORE_AREA_LIST
from ar_shuffleboard.utils.constants import GameConfig

GAME_BOARD_SCALE: int = 2


class GameView:
    def __init__(
        self,
        board_size: tuple[int, int],
    ):
        self.board_size = board_size

    # === Game Canvas ===

    def drawGameBoarderLines(self, image: np.ndarray):
        """게임 보드 경계선 그리기"""
        x, y = self.getGameOrigin()
        w, h = self.board_size

        line_color = (238, 238, 238, 255)
        # 모서리 경계선
        cv2.line(image, (x, y), (x + w, y), line_color, 2)  # 위
        cv2.line(image, (x, y + h), (x + w, y + h), line_color, 2)  # 아래
        cv2.line(image, (x, y), (x, y + h), line_color, 2)  # 좌
        cv2.line(image, (x + w, y), (x + w, y + h), line_color, 2)  # 우
        # 점수 경계선
        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            _x = int(x + prev_area * w)
            cv2.line(image, (_x, y), (_x, y + h), line_color, 2)

    def drawScoreArea(self, img: np.ndarray):
        """점수 영역 그리기"""
        x, y = self.getGameOrigin()
        w, h = self.board_size

        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            x1 = int(x + (prev_area - score_area.ratio) * w)
            x2 = int(x + prev_area * w)
            color = score_area.color
            cv2.rectangle(img, (x1, y), (x2, y + h), color, -1)

    def drawPucks(self, image: np.ndarray, players: list[PlayerData]):
        """게임 퍽 그리기"""
        x, y = self.getGameOrigin()
        for player in players:
            for puck in player.pucks:
                cx, cy = puck.body.position
                center = (int(cx + x), int(cy + y))
                cv2.circle(image, center, GameConfig.PUCK_RADIUS, player.color, -1)

    def getGameCanvas(self, players: list[PlayerData]):
        """게임 캔버스 반환"""
        w, h = self.board_size
        canvas: np.ndarray = np.zeros((GAME_BOARD_SCALE * h, GAME_BOARD_SCALE * w, 4), dtype=np.uint8)
        self.drawGameBoarderLines(canvas)
        self.drawScoreArea(canvas)
        self.drawPucks(canvas, players)
        return canvas

    # === Game Overlay ===

    def drawGameOverlay(self, image: np.ndarray):
        """게임 오버레이 그리기"""
        pass

    # === etc. ===

    def getGameOrigin(self) -> tuple[int, int]:
        """게임 이미지 내 게임 보드의 오프셋을 반환한다."""
        w, h = self.board_size
        s = (GAME_BOARD_SCALE - 1) / 2
        return (int(w * s), int(h * s))
