from __future__ import annotations

import cv2
import numpy as np

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ar_shuffleboard.controllers.main_controller import MainController


class MainView:
    def __init__(
        self,
        controller: MainController,
        window_title: str,  # 윈도우 제목
        resolution: tuple[int, int],  # 비디오 해상도
        fps: float,  # 비디오 fps
        frame_interval_ms: int,  # 프레임 간격 (ms)
        mirror: bool,  # 비디오 좌우반전 여부
        chessboard_pattern_size: tuple[int, int],  # (w, h)
        chessboard_square_size: float,  # mm 단위
        **kwargs,
    ):
        self.controller = controller

        self.window_title = window_title
        self.resolution = resolution
        self.fps = fps
        self.frame_interval_ms = frame_interval_ms
        self.mirror = mirror

        self.chessboard_pattern_size = chessboard_pattern_size
        self.chessboard_cell_size = chessboard_square_size

        # 캔버스
        self.canvas = self.get_canvas()

        # 화살표
        self.arrow_origin: Optional[tuple[int, int]] = None  # 마우스 우클릭 시작 위치
        self.max_arrow_length: float = 80  # 화살표 길이 (px)

    def show_frame(self, frame: np.ndarray, apply_canvas: bool = False):
        """프레임을 출력한다."""
        if apply_canvas:
            self.add_canvas(frame, self.canvas)
        cv2.imshow(self.window_title, frame)

    def close_all(self):
        """모든 윈도우를 닫는다."""
        cv2.destroyAllWindows()

    def put_text(self, frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]):
        """프레임에 텍스트 출력한다."""
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def get_canvas(self):
        return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

    def add_canvas(self, frame: np.ndarray, add: np.ndarray):
        """캔버스 추가"""
        mask = np.any(add != 0, axis=2)
        frame[mask] = add[mask]

    def clear_canvas(self):
        self.canvas.fill(0)

    def draw_arrow(self, pt1, pt2, color):
        cv2.arrowedLine(self.canvas, pt1, pt2, color, 2, tipLength=0.2)
