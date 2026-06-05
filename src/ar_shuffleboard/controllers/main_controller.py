from configparser import ConfigParser
import cv2
import numpy as np
from typing import Optional
from enum import Enum


from ar_shuffleboard.models.main_model import MainModel
from ar_shuffleboard.models.video_model import VideoModel
from ar_shuffleboard.models.shuffleboard import Shuffleboard
from ar_shuffleboard.models.gesture_detector import GestureDetector
from ar_shuffleboard.models.camera_calibrator import CameraCalibrator
from ar_shuffleboard.views.main_view import MainView
from ar_shuffleboard.views.game_view import GameView
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

        # 플래그 딕셔너리
        self.flags = {
            "terminated": False,  # 앱 종료 플래그
            "show_hands": False,  # 손 랜드마크 출력 플래그
        }

        # 메인 뷰
        cv2.namedWindow(self.config_data["window_title"])
        cv2.setMouseCallback(self.config_data["window_title"], self.mouseCallBackForMainWindow)

        # 임시
        self.video_model.set_source(0)

        # 게임 모델
        w, h = self.config_data["chessboard_pattern_size"]
        cell_size = self.config_data["chessboard_square_size"]
        board_size = ((w + 1) * cell_size, (h + 1) * cell_size)
        self.game_model = Shuffleboard(board_size)

        # 게임 뷰
        self.game_view = GameView(
            game_model=self.game_model,
            window_title="Game Window",
            board_size=board_size,
            callback=self.mouseCallBackForGameWindow,
        )

    def run(self):
        while not self.flags["terminated"]:
            # 프레임 읽기
            frame = self.video_model.read_frame()
            if frame is None:
                break

            # 거울 모드 설정
            if self.config_data["mirror"]:
                frame = np.ascontiguousarray(np.flip(frame, axis=1))

            # 현재 프레임 전달
            self.camera_calibrator.set_frame(frame)

            # 캘리브레이션
            if self.camera_calibrator.calibration_fsm.is_collecting():
                self.camera_calibrator.extract_corners(scale=0.7, save=True)  # 체스보드 코너 추출
                if self.camera_calibrator.is_ready_for_calibration():  # 캘리브레이션 준비
                    self.camera_calibrator.calibration_fsm.to_complete()
                    self.camera_calibrator.calibrate()
            if self.camera_calibrator.calibration_fsm.is_complete():
                # self.camera_calibrator.project()
                pass

            self.camera_calibrator.draw_on_chessboard(scale=0.3, return_canvas=False)

            # 제스처 감지
            self.gesture_detector.detect(frame, self.timestamp_ms)
            if self.flags["show_hands"]:
                self.gesture_detector.draw_hands(frame)

            # 프레임 가공
            processed_frame = self.main_model.get_processed_frame(frame)

            # 게임 모델
            self.game_model.stepSpace(self.config_data["frame_interval_ms"])
            if self.camera_calibrator.calibration_fsm.is_complete():
                # 프로젝트용 보드 캔버스(원본 크기, 보드만)를 생성하여 실제 체스보드에 맞춰 투영
                board_img = self.game_view.getProjectBoardCanvas()
                pixel_to_unit = 1.0
                plane_pts = self.camera_calibrator.image_to_plane_points(board_img, ignore_background=False, pixel_to_unit=pixel_to_unit)
                # 한 칸(cell_size) 만큼 이동 — 사용자가 요청한 반대 방향(기존과 반대로)
                cell_size = float(self.config_data["chessboard_square_size"])  # in mm (and pixels when pixel_to_unit==1)
                shift_px = cell_size / float(pixel_to_unit)
                plane_pts.origin = (int(shift_px), int(shift_px))
                projected_game = self.camera_calibrator.project(plane_pts)
                if projected_game is not None:  # 화면에 체스보드가 보일 때
                    self.video_model.add_image_on_frame(
                        processed_frame,
                        projected_game,
                    )

            # 오버레이 적용
            self.main_view.apply_overlay(processed_frame)

            # 화면에 출력
            self.main_view.show_frame(processed_frame, apply_canvas=True)
            self.game_view.showGameWindow()

            # 키 입력 감지 및 이벤트 핸들러 호출
            keycode = cv2.waitKeyEx(self.config_data["frame_interval_ms"])
            self.keyEventHandler(keycode)

            # 기타 이벤트 핸들러 호출
            self.windowCloseEventHandler()

            # 기타
            self.timestamp_ms += self.config_data["frame_interval_ms"]
            self.frame_index += 1

            self.game_model.check()

        self.video_model.release()
        self.main_view.close_all()

    # === Flag Handlers ===

    def enableFlag(self, flag: str):
        """플래그 활성화"""
        if flag in self.flags:
            self.flags[flag] = True

    def disableFlag(self, flag: str):
        """플래그 비활성화"""
        if flag in self.flags:
            self.flags[flag] = False

    def toggleFlag(self, flag: str):
        """플래그 토글"""
        if flag in self.flags:
            self.flags[flag] = not self.flags[flag]

    # === Event Handlers ===

    def keyEventHandler(self, keycode: int, **kwargs):
        """키 입력에 별 이벤트를 처리한다."""
        if keycode == -1:
            return
        print(f"[MainController] Key Pressed: {Keycode.to_str(keycode)}")
        match keycode:
            case Keycode.ESC:  # 프로그램 종료
                self.enableFlag("terminated")
            case Keycode.SPACE:  # 손 랜드마크 표시 on/off
                self.toggleFlag("show_hands")
            case Keycode.ENTER:  # 카메라 캘리브레이션 수행 (외부 파라미터 구하기)
                if self.camera_calibrator.calibration_fsm.is_waiting():
                    self.camera_calibrator.calibration_fsm.to_collecting()
            case Keycode.L | Keycode.l:  # 카메라 내부 파라미터 고정 on/off
                if self.camera_calibrator.is_lock():
                    self.camera_calibrator.lock_extrinsic_parameters(lock=False)
                else:
                    self.camera_calibrator.lock_extrinsic_parameters(lock=True)
            case Keycode.R | Keycode.r:  # 게임 초기화
                self.game_model.resetGame()

    def windowCloseEventHandler(self):
        """윈도우 닫힘을 확인하여 앱 종료를 설정한다."""
        try:
            visible = cv2.getWindowProperty(self.config_data["window_title"], cv2.WND_PROP_VISIBLE)
            if visible < 1:
                self.enableFlag("terminated")
        except:
            self.enableFlag("terminated")

    # === Callbacks ===

    def mouseCallBackForMainWindow(self, event, x, y, flags, param):
        """마우스 이벤트 콜백 메서드"""
        # # 당기기 화살표 그리기
        # if event == cv2.EVENT_RBUTTONDOWN:  # RMB click
        #     # TODO: 조건 불만족 시 None 할당 (ex: 기물 바깥 클릭)
        #     self.arrow_origin = (x, y)
        # elif event == cv2.EVENT_RBUTTONUP:  # RMB release
        #     self.main_view.clear_canvas()
        #     if self.arrow_origin is not None:
        #         x0, y0 = self.arrow_origin
        #         dx, dy = x - x0, y - y0
        #         length = np.hypot(dx, dy)
        #         scale = min(1.0, self.max_arrow_length / length)
        #         # TODO scale[0, 1]을 전달하여 이벤트 호출
        # elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):  # RMB move w/ click
        #     if self.arrow_origin is not None:
        #         self.drawArrow(self.arrow_origin, (x, y))
        pass

    def mouseCallBackForGameWindow(self, event, x, y, flags, param):
        """
        LMB Click: 퍽 배치
        RMB Click / Move / Release: 퍽 힘 조절 및 놓기
        """
        if self.game_model.shuffleboard_fsm.is_end():
            return

        origin_x, origin_y = self.game_view.getGameOrigin()
        board_x, board_y = self.game_view.board_size
        if event == cv2.EVENT_RBUTTONDOWN:  # RMB click - 퍽 배치
            puck_x = board_x
            puck_y = np.clip(y - origin_y, 0, board_y)
            self.game_model.placePuck((puck_x, puck_y))
        if event == cv2.EVENT_LBUTTONDOWN:  # LMB click - 발사
            puck_x, puck_y = self.game_model.getReadyPuck().body.position
            puck_x, puck_y = origin_x + puck_x, origin_y + puck_y
            scale = 3.0
            velocity = (int(scale * (puck_x - x)), int(scale * (puck_y - y)))
            self.game_model.hitPuckForReady(velocity)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # LMB move w/ click - 마우스 좌클릭 드래그
            end_pos = (x, y)
            self.game_view.setLastMousePos(end_pos)
            self.game_view.drawPuckArrow()
        elif event == cv2.EVENT_LBUTTONUP:  # LMB release
            end_pos = (x, y)

        self.game_view.setLastMousePos(None)

    # === UI Methods ===

    def drawArrow(self, puck_pos: tuple[int, int], mouse_pos: tuple[int, int]):
        """퍽의 위치와 마우스 위치를 바탕으로 화면에 화살표를 그린다."""
        # temp_canvas 초기화
        self.main_view.clear_canvas()
        x0, y0 = puck_pos
        x, y = mouse_pos
        dx, dy = x - x0, y - y0
        length = np.hypot(dx, dy)
        # 최대 길이 제한
        scale = min(1.0, self.max_arrow_length / length)
        x1 = int(x0 - dx * scale)
        y1 = int(y0 - dy * scale)
        # 화살표 그리기
        self.main_view.draw_arrow((x0, y0), (x1, y1), (255, 0, 0))
