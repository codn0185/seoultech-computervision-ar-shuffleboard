from __future__ import annotations

import cv2
import numpy as np

from typing import TYPE_CHECKING

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

    def show_frame(self, frame: np.ndarray):
        """프레임을 출력한다."""
        cv2.imshow(self.window_title, frame)

    def close_all(self):
        """모든 윈도우를 닫는다."""
        cv2.destroyAllWindows()

    def put_text(self, frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]):
        """프레임에 텍스트 출력한다."""
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
