import cv2
import numpy as np
from typing import Optional


class CameraCalibrator:
    MIN_CALIBRATION_VIEWS = 15

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

        # 캘리브레이션 결과
        self.calibration_result = {}

    def extract_corners(self, frame: np.ndarray, scale: float = 1.0, save: bool = False) -> Optional[np.ndarray]:
        """
        이미지에서 체스보드의 코너를 추출한다.

        - 속도 향상을 위해 scale < 1.0으로 축소된 이미지에서 먼저 코너를 찾고 원본 해상도로 복원한다.
        - 추출 실패 시 None을 반환한다.
        - save=True이면 추출된 코너를 캘리브레이션 데이터로 저장한다.

        Args:
            frame (np.ndarray): 입력 프레임
            scale (float): 체스보드 검출에 사용할 축소 비율 (0 < scale <= 1, 1.0=원본)
            save (bool): True면 추출 결과를 캘리브레이션 데이터에 저장
        Returns:
            np.ndarray or None: 검출된 코너 좌표 (N,1,2) 또는 None
        """
        if not (0 < scale <= 1.0):  # 스케일링 범위 확인
            scale = 1.0

        # 이미지 흑백 전환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if scale != 1.0:  # 스케일링 (속도 향상)
            small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            found, corners = cv2.findChessboardCorners(small, self.chessboard_pattern_size)
            if not found:
                return None
            corners = corners / scale  # 원본 해상도로 복원
        else:  # 원본
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_pattern_size)
            if not found:
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

    def project(self, frame: np.ndarray, object_points) -> Optional[np.ndarray]:
        """
        현재 프레임에서 체스보드를 찾고, 찾으면 3D 점들을 2D로 투영한 결과를 반환한다.

        투명 실패 시 None을 반환한다.
        """
        # 프레임 내 체스보드 코너 확인
        corners = self.extract_corners(frame, save=False)
        if corners is None:
            return None

        # 카메라 캘리브레이션을 수행한 결과 확인
        camera_matrix = self.calibration_result.get("camera_matrix")
        dist_coeffs = self.calibration_result.get("dist_coeffs")
        if camera_matrix is None or dist_coeffs is None:
            return None

        # solvePnP() 수행
        obj_pts = np.asarray(self.base_object_points, dtype=np.float64).reshape(-1, 1, 2)
        img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        # 프레임에 투영
        obj_for_proj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
        image_points, _ = cv2.projectPoints(obj_for_proj, rvec, tvec, camera_matrix, dist_coeffs)
        return image_points.reshape(-1, 2)

    def draw_on_chessboard(self, frame: np.ndarray, scale: float = 1.0):
        """체스보드 코너를 프레임에 그린다."""
        corners = self.extract_corners(frame, scale=scale, save=False)
        if corners is not None:
            cv2.drawChessboardCorners(frame, self.chessboard_pattern_size, corners, True)

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
