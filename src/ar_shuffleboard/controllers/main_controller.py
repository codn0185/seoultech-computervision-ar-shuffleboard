from configparser import ConfigParser
import cv2
import numpy as np
from typing import Optional


from ar_shuffleboard.models.main_model import MainModel
from ar_shuffleboard.models.video_model import VideoModel
from ar_shuffleboard.models.gesture_detector import GestureDetector
from ar_shuffleboard.models.camera_calibrator import CameraCalibrator
from ar_shuffleboard.views.main_view import MainView
from ar_shuffleboard.utils.constants import Keycode


class MainController:
    def __init__(self, config: ConfigParser):
        # config에서 값 추출 및 가공
        self.config_data = {
            "window_title": config.get("window", "title"),
            "resolution": (config.getint("video", "width"), config.getint("video", "height")),
            "fps": config.getfloat("video", "fps"),
            "frame_interval_ms": int(1000.0 / config.getfloat("video", "fps")),
            "mirror": config.getboolean("video", "mirror"),
            "chessboard_pattern_size": (config.getint("chessboard", "width"), config.getint("chessboard", "height")),
            "chessboard_square_size": config.getint("chessboard", "cell_size"),
        }

        self.main_model = MainModel(self, **self.config_data)
        self.main_view = MainView(self, **self.config_data)

        self.video_model = VideoModel(**self.config_data)
        self.camera_calibrator = CameraCalibrator(**self.config_data)
        self.gesture_detector = GestureDetector()

        self.frame_index = 0  # 현재 프레임 인덱스
        self.timestamp_ms = 0  # 현재 타임스탬프 (ms)

        self.calibration_state: int = -1  # -1: 캘리브레이션 수행 전 / 0: 캘리브레이션에 필요한 프레임 수집 중 / 1: 캘리브레이션 완료
        # 플래그 딕셔너리
        self.flags = {
            "terminated": False,  # 앱 종료 플래그
            "show_hands": False,  # 손 랜드마크 출력 플래그
        }

        # 임시
        self.video_model.set_source(0)

    def run(self):
        while not self.flags["terminated"]:
            # 프레임 읽기
            frame = self.video_model.read_frame()
            if frame is None:
                break

            if self.config_data["mirror"]:
                frame = np.ascontiguousarray(np.flip(frame, axis=1))

            # 캘리브레이션
            if self.calibration_state == 0:
                self.camera_calibrator.extract_corners(frame, save=True)
                if self.camera_calibrator.is_ready_for_calibration():  # 캘리브레이션 준비
                    self.calibration_state = 1
                    self.camera_calibrator.calibrate()
            if self.calibration_state == 1:
                # self.camera_calibrator.project()
                pass

            # self.camera_calibrator.draw_on_chessboard(frame)

            # 제스처 감지
            self.gesture_detector.detect(frame, self.timestamp_ms)
            if self.flags["show_hands"]:
                self.gesture_detector.draw_hands(frame)

            # 프레임 가공
            processed_frame = self.main_model.get_processed_frame(frame)

            # 오버레이 적용
            self.drawOverlay(frame)

            # 화면에 출력
            self.main_view.show_frame(processed_frame)

            # 키 입력 감지 및 이벤트 핸들러 호출
            keycode = cv2.waitKeyEx(self.config_data["frame_interval_ms"])
            self.keyEventHandler(keycode)

            # 기타 이벤트 핸들러 호출
            self.windowCloseEventHandler()

            # 기타
            self.timestamp_ms += self.config_data["frame_interval_ms"]
            self.frame_index += 1

        self.video_model.release()
        self.main_view.close_all()

    def drawOverlay(self, frame: np.ndarray):
        """프레임에 오버레이를 적용한다."""
        if self.calibration_state == -1:  # 캘리브레이션 전
            self.main_view.put_text(
                frame,
                "Calibration required (press ENTER to calibrate)",
                (20, 20),
                (0, 80, 255),
            )
        elif self.calibration_state == 0:  # 캘리브레이션 중
            collected = self.camera_calibrator.get_calibration_views_count()
            required = self.camera_calibrator.MIN_CALIBRATION_VIEWS
            self.main_view.put_text(
                frame,
                f"Calibrating... ({collected}/{required}) - Please align the chessboard",
                (20, 20),
                (0, 80, 255),
            )
        else:  # 캘리브레이션 완료
            self.main_view.put_text(
                frame,
                "Calibration completed",
                (20, 20),
                (0, 80, 255),
            )

    # === Flag Handlers ===

    def setFlag(self, flag: str, value: bool):
        """플래그 설정"""
        if flag in self.flags:
            self.flags[flag] = value

    def toggleFlag(self, flag: str):
        """플래그 토글"""
        if flag in self.flags:
            self.flags[flag] = not self.flags[flag]

    # === Event Handlers ===

    def keyEventHandler(self, event):
        match event:
            case Keycode.ESC:
                self.setFlag("terminated", True)
            case Keycode.SPACE:
                self.toggleFlag("show_hands")
            case Keycode.ENTER:
                if self.calibration_state == -1:
                    self.calibration_state = 0

    def mouseClickEventHandler(self, event):
        pass

    def mouseReleaseEventHandler(self, event):
        pass

    def mouseMoveEventHandler(self, event):
        pass

    def windowCloseEventHandler(self):
        """윈도우 닫힘을 확인하여 앱 종료를 설정한다."""
        try:
            visible = cv2.getWindowProperty(self.config_data["window_title"], cv2.WND_PROP_VISIBLE)
            if visible < 1:
                self.setFlag("terminated", True)
        except:
            self.setFlag("terminated", True)
