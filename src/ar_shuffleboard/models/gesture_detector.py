import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# 손 랜드마크 연결 정보
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


class GestureDetector:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.hand_data = {}  # 손 랜드마크 감지 데이터
        self.timestamp_ms = 0

    def detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> bool:
        """프레임과 타임스탬프를 통해 손 랜드마크를 감지하고 결과를 저장 후 성공 여부를 반환한다."""
        # frame 가공
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # timestamp 설정
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms < self.timestamp_ms:
            timestamp_ms = self.timestamp_ms + 1
        self.timestamp_ms = timestamp_ms

        # hand landmark 감지 결과
        results = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        self.hand_data.clear()

        # 감지 결과 없는 경우
        if not results.hand_landmarks:
            return False

        # 결과 가공
        for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            # hand_landmarks: 한 손의 21개 랜드마크
            # handedness: 해당 손의 좌/우 판정 후보
            self.hand_data[handedness[0].category_name] = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=float
            )

        return True

    def draw_hands(self, frame: np.ndarray):
        """프레임에 미리 저장된 감지 결과를 통해 손 랜드마크를 그린다."""
        if not self.hand_data:
            return

        for handedness in self.hand_data:
            self.draw_one_hand(frame, self.hand_data[handedness])

    def draw_one_hand(self, frame: np.ndarray, hand_points: np.ndarray[tuple[float, float, float]]):
        """프레임에 하나의 손 랜드마크를 그린다"""
        height, width = frame.shape[:2]

        # 손 랜드마크 사이의 연결선 그리기
        for idx1, idx2 in HAND_CONNECTIONS:
            pt1 = self.to_pixel(hand_points[idx1], height, width)
            pt2 = self.to_pixel(hand_points[idx2], height, width)
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # green

        # 손 랜드마크 점 그리기
        for x, y, _ in hand_points:
            point = self.to_pixel((x, y), height, width)
            cv2.circle(frame, point, 3, (0, 0, 255), -1)  # red

    def to_pixel(self, normalized_position_2d: tuple[float, float], height, width) -> tuple[int, int]:
        """2차원 노멀라이즈[0, 1] 좌표(x, y)를 픽셀 좌표로 변환"""
        return (int(normalized_position_2d[0] * width), int(normalized_position_2d[1] * height))
