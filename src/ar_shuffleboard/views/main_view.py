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
        mirror: bool,  # 비디오 좌우반전 여부
        chessboard_pattern_size: tuple[int, int],  # (w, h)
        chessboard_square_size: float,  # mm 단위
    ):
        self.controller = controller

        self.window_title = window_title
        self.resolution = resolution
        self.fps = fps
        self.frame_interval_ms = 1000.0 / self.fps
        self.mirror = mirror

        self.chessboard_pattern_size = chessboard_pattern_size
        self.chessboard_cell_size = chessboard_square_size
