import cv2
import numpy as np
from typing import TYPE_CHECKING, Optional
from enum import Enum


class CameraCalibrator:
    class CalibrationFiniteStateMachine:
        """카메라 캘리브레이션 유한상태머신"""

        class CalibrationState(Enum):
            waiting = 0  # 캘리브레이션 대기
            collecting = 1  # 캘리브레이션에 필요한 데이터 수집 중
            complete = 2  # 캘리브레이션 완료

        _current_state: CalibrationState = CalibrationState.waiting

        def current_state(self):
            return self._current_state

        def _switch_state(self, new_state: CalibrationState):
            print(f"[CameraCalibrator.CalibrationFiniteStateMachine] State Switched: {self._current_state} -> {new_state}")
            self._current_state = new_state

        def to_waiting(self):
            self._current_state = self.CalibrationState.waiting

        def to_collecting(self):
            self._current_state = self.CalibrationState.collecting

        def to_complete(self):
            self._current_state = self.CalibrationState.complete

        def _is_state(self, target_state: CalibrationState) -> bool:
            return self._current_state == target_state

        def is_waiting(self):
            return self._is_state(self.CalibrationState.waiting)

        def is_collecting(self):
            return self._is_state(self.CalibrationState.collecting)

        def is_complete(self):
            return self._is_state(self.CalibrationState.complete)

    MIN_CALIBRATION_VIEWS = 15  # 캘리브레이션에 필요한 프레임 수

    def __init__(
        self,
        resolution: tuple[int, int],  # 프레임 해상도 (w, h)
        chessboard_pattern_size: tuple[int, int],  # 체스보드 코너 (w, h)
        chessboard_square_size: float,  # 체스보드 칸 크기 (mm 단위)
        **kwargs,
    ):
        self.resolution = resolution

        self.chessboard_pattern_size = chessboard_pattern_size
        self.chessboard_square_size = chessboard_square_size

        # 카메라 캘리브레이션에 사용되는 좌표들 (3d -> 2d)
        self.object_points = []
        self.image_points = []

        # 3D 공간의 실제 기준점 좌표 생성 (z=0)
        objp = np.zeros(
            (chessboard_pattern_size[0] * chessboard_pattern_size[1], 3),
            dtype=np.float32,
        )
        objp[:, :2] = np.mgrid[0 : chessboard_pattern_size[0], 0 : chessboard_pattern_size[1]].T.reshape(-1, 2)
        self.base_object_points = objp * chessboard_square_size

        # 현재 프레임
        self.frame: Optional[np.ndarray] = None

        # 캘리브레이션 결과
        self.calibration_result = {}

        # 플래그
        self._is_lock = False  # 외부 파라미터(rvec, tvec) 값 고정 여부

        # 외부 파라미터
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None

        # 캘리브레이션 FSM
        self.calibration_fsm = self.CalibrationFiniteStateMachine()

    # === Core Methods ===

    def set_frame(self, frame: np.ndarray):
        """현재 프레임을 설정한다."""
        self.frame = frame

    def extract_corners(self, scale: float = 1.0, save: bool = False) -> Optional[np.ndarray]:
        """
        이미지에서 체스보드의 코너를 추출한다.

        - scale은 우선 탐색용 속도 힌트로만 사용한다.
        - 축소 이미지에서 먼저 찾고, 실패하면 원본 해상도에서 다시 찾는다.
        - 최종 코너 정밀화는 항상 원본 해상도에서 수행한다.
        - 추출 실패 시 None을 반환한다.
        - save=True이면 추출된 코너를 캘리브레이션 데이터로 저장한다.

        Args:
            scale (float): 체스보드 검출에 사용할 축소 비율 (0 < scale <= 1, 1.0=원본)
            save (bool): True면 추출 결과를 캘리브레이션 데이터에 저장
        Returns:
            np.ndarray or None: 검출된 코너 좌표 (N,1,2) 또는 None
        """
        if not (0 < scale <= 1.0):  # 스케일링 범위 확인
            scale = 1.0

        # 이미지 흑백 전환
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        corners = None
        search_images = []
        if scale != 1.0:
            search_images.append((cv2.resize(gray, (0, 0), fx=scale, fy=scale), scale))
        search_images.append((gray, 1.0))

        for search_image, current_scale in search_images:
            found, found_corners = cv2.findChessboardCorners(search_image, self.chessboard_pattern_size)
            if not found:
                continue

            if current_scale != 1.0:
                found_corners = found_corners / current_scale  # 원본 해상도로 복원

            corners = found_corners
            break

        if corners is None:
            return None

        # 코너 정밀화
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 저장 (카메라 캘리브레이션에 사용)
        if save:
            self.object_points.append(self.base_object_points.copy())
            self.image_points.append(corners)

        return corners

    def calibrate(self):
        """카메라 캘리브레이션을 수행 후 결과를 저장한다."""
        if not self.object_points or not self.image_points:
            return

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, self.resolution, None, None)
        self.calibration_result = {
            "ret": ret,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvecs": rvecs,
            "tvecs": tvecs,
        }

    def get_intrinsic_parameters(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """내부 파라미터(camera_matrix, dist_coeffs)를 반환한다. 구할 수 없다면 None을 반환한다."""
        camera_matrix = self.calibration_result.get("camera_matrix")
        dist_coeffs = self.calibration_result.get("dist_coeffs")
        if camera_matrix is None or dist_coeffs is None:
            return None

        return (camera_matrix, dist_coeffs)

    def extract_extrinsic_parameters(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """외부 파라미터 rvec, tvec를 추출 및 반환한다. 구할 수 없다면 None을 반환한다."""
        # 프레임 내 체스보드 코너 확인
        corners = self.extract_corners(scale=0.3, save=False)
        if corners is None:
            print("[CameraCalibrator.extract_extrinsic_parameters] Failed to extract chessboard corners")
            return None

        # 내부 파라미터 가져오기
        intrinsic_parameters = self.get_intrinsic_parameters()
        if intrinsic_parameters is None:
            print("[CameraCalibrator.extract_extrinsic_parameters] Failed to get intrinsic parameters")
            return None
        camera_matrix, dist_coeffs = intrinsic_parameters

        # solvePnP() 수행
        obj_pts = np.asarray(self.base_object_points, dtype=np.float64).reshape(-1, 1, 3)
        img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            print("[CameraCalibrator.extract_extrinsic_parameters] Failed to get extrinsic parameters")
            return None

        # rvec, tvec 반환
        return (rvec, tvec)

    def project(self, object_points) -> Optional[np.ndarray]:
        """
        현재 프레임에서 체스보드를 찾고, 찾으면 3D 점들을 2D로 투영한 결과를 반환한다.

        투명 실패 시 None을 반환한다.
        """
        if self._is_lock:
            # 고정된 외부 파라미터(rvec, tvec) 사용
            rvec, tvec = self.rvec, self.tvec
        else:
            # 외부 파라미터(rvec, tvec) 추출
            extrinsic_parameters = self.extract_extrinsic_parameters()
            if extrinsic_parameters is None:
                return None
            rvec, tvec = extrinsic_parameters

        # 내부 파라미터(camera_matrix, dist_coeffs) 가져오기
        intrinsic_parameters = self.get_intrinsic_parameters()
        if intrinsic_parameters is None:
            return None
        camera_matrix, dist_coeffs = intrinsic_parameters

        # 프레임에 투영
        obj_for_proj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
        image_points, _ = cv2.projectPoints(obj_for_proj, rvec, tvec, camera_matrix, dist_coeffs)
        return image_points.reshape(-1, 2)

    def draw_on_chessboard(self, scale: float = 1.0, return_canvas: bool = False) -> Optional[np.ndarray]:
        """
        체스보드 코너를 프레임에 그린다.
        - 외부/내부 파라미터가 고정되어 있으면, 3D 기준점들을 2D로 투영하여 그린다.
        - 그렇지 않으면 기존 방식대로 코너를 추출해서 그린다.
        - return_canvas=True면 frame에 적용하는 대신 빈 캔버스에 그려 반환한다.

        Args:
            scale: 체스보드 검출용 스케일(기본 1.0)
            return_canvas: True면 빈 캔버스에 그림
        Returns:
            None 또는 새 캔버스(np.ndarray)
        """
        target = None
        if return_canvas:
            # frame과 동일한 크기의 검정 배경 캔버스 생성
            target = np.zeros_like(self.frame)
        else:
            target = self.frame

        # 내부/외부 파라미터가 고정되어 있으면 투영 방식 사용
        if self._is_lock and self.rvec is not None and self.tvec is not None:
            intrinsic = self.get_intrinsic_parameters()
            if intrinsic is not None:
                camera_matrix, dist_coeffs = intrinsic
                obj_points = np.asarray(self.base_object_points, dtype=np.float64).reshape(-1, 1, 3)
                image_points, _ = cv2.projectPoints(obj_points, self.rvec, self.tvec, camera_matrix, dist_coeffs)
                # 2D 투영점 그리기
                for pt in image_points.reshape(-1, 2):
                    cv2.circle(target, tuple(np.round(pt).astype(int)), 5, (0, 255, 0), -1)
                if return_canvas:
                    return target
                return None
        # 그렇지 않으면 기존 방식
        corners = self.extract_corners(scale=scale, save=False)
        if corners is not None:
            cv2.drawChessboardCorners(target, self.chessboard_pattern_size, corners, True)
            if return_canvas:
                return target
        return None

    def clear(self):
        """체스보드 코너 추출 및 캘리브레이션 결과 제거"""
        self.object_points = []
        self.image_points = []
        self.calibration_result.clear()

    def is_ready_for_calibration(self) -> bool:
        """캘리브레이션 준비 완료 여부를 반환한다."""
        return len(self.image_points) >= self.MIN_CALIBRATION_VIEWS

    def get_calibration_views_count(self) -> int:
        """캘리브레이션에 수집된 뷰(프레임) 개수를 반환한다."""
        return len(self.image_points)

    def lock_extrinsic_parameters(self, lock: bool):
        """
        현재 외부 파라미터(tvec, rvec) 고정 여부를 설정한다.
        - lock=True이면, 현재 프레임에서 외부 파라미터(rvec, tvec)를 추출하고 저장한다.
        - lock=False이면, 저장된 외부 파라미터(rvec, tvec)를 제거한다.
        """
        if not lock:
            self._is_lock = False
            self.rvec, self.tvec = None, None
            print("[CameraCalibrator.lock_extrinsic_parameters] Extrinsic Parameters Unlocked")
            return

        extrinsic_parameters = self.extract_extrinsic_parameters()
        if extrinsic_parameters is None:
            self._is_lock = False
            print("[CameraCalibrator.lock_extrinsic_parameters] Extrinsic Parameters Unlocked (failed to extract extrinsic parameters)")
            return

        self._is_lock = True
        self.rvec, self.tvec = extrinsic_parameters
        print("[CameraCalibrator.lock_extrinsic_parameters] Extrinsic Parameters Locked")
        return

    def is_lock(self) -> bool:
        """외부 파라미터(tvec, rvec) 고정 여부를 반환한다."""
        return self._is_lock
