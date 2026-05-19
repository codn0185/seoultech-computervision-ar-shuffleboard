from ar_shuffleboard.models.main_model import MainModel
from ar_shuffleboard.views.main_view import MainView


class MainController:
    def __init__(self, config: ConfigParser):
        self.model = MainModel(self)

        # config에서 값 추출
        window_title = config.get("window", "title")
        resolution = (config.getint("video", "width"), config.getint("video", "height"))
        fps = config.getfloat("video", "fps")
        mirror = config.getboolean("video", "mirror")

        chessboard_pattern_size = (config.getint("chessboard", "width"), config.getint("chessboard", "height"))
        chessboard_square_size = config.getint("chessboard", "cell_size")

        self.view = MainView(
            controller=self,
            window_title=window_title,
            resolution=resolution,
            fps=fps,
            mirror=mirror,
            chessboard_pattern_size=chessboard_pattern_size,
            chessboard_square_size=chessboard_square_size,
        )

    def run(self):
        pass
