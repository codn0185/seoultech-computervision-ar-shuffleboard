import cv2
import numpy as np
from typing import Optional


class VideoModel:
    video_capture: Optional[cv2.VideoCapture]

    def __init__(
        self,
        resolution: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        self.video_capture = None
        self.resolution = resolution

    def set_source(self, source: int):
        self.release()
        self.video_capture = cv2.VideoCapture(source)

        if self.resolution is not None:
            w, h = self.resolution
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def release(self):
        if self.video_capture is None:
            return
        self.video_capture.release()
        self.video_capture = None

    def read_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.video_capture.read()
        return frame if ret else None

    def add_image_on_frame(
        self,
        frame: np.ndarray,
        img: np.ndarray,
        origin: tuple[int, int] = (0, 0),
    ):
        """
        프레임에 이미지를 병합합니다.

        Args:
            frame (np.ndarray): 병합에 사용할 원본 프레임
            img (np.ndarray): 프레임에 병합할 이미지
            offset (tuple[int, int]): img를 frame에 병합할 때 오프셋
        """
        if frame is None or img is None:
            return

        if hasattr(img, "image") and hasattr(img, "destination_corners"):
            source_image = img.image
            destination_corners = np.asarray(img.destination_corners, dtype=np.float32)

            if source_image is None or source_image.size == 0 or destination_corners.shape[0] < 4:
                return

            source_h, source_w = source_image.shape[:2]
            source_corners = np.array(
                [
                    [0.0, 0.0],
                    [float(source_w), 0.0],
                    [float(source_w), float(source_h)],
                    [0.0, float(source_h)],
                ],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(source_corners, destination_corners[:4])
            img = cv2.warpPerspective(source_image, transform, (frame.shape[1], frame.shape[0]))
            origin = (0, 0)

        x, y = origin
        frame_h, frame_w = frame.shape[:2]
        img_h, img_w = img.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + img_w)
        y2 = min(frame_h, y + img_h)

        if x1 >= x2 or y1 >= y2:
            return

        img_x1 = x1 - x
        img_y1 = y1 - y
        img_x2 = img_x1 + (x2 - x1)
        img_y2 = img_y1 + (y2 - y1)

        frame_roi = frame[y1:y2, x1:x2]
        img_roi = img[img_y1:img_y2, img_x1:img_x2]

        if img_roi.ndim == 2:
            img_roi = np.repeat(img_roi[:, :, np.newaxis], 3, axis=2)
        elif img_roi.ndim != 3 or img_roi.shape[2] not in (3, 4):
            return

        if img_roi.ndim == 3 and img_roi.shape[2] == 4:
            alpha_mask = img_roi[:, :, 3].astype(np.float32) / 255.0
            valid_mask = alpha_mask > 0.0

            if not np.any(valid_mask):
                return

            src_rgb = img_roi[:, :, :3].astype(np.float32)
            dst_rgb = frame_roi.astype(np.float32)
            alpha_3 = alpha_mask[:, :, np.newaxis]
            blended = src_rgb * alpha_3 + dst_rgb * (1.0 - alpha_3)
            frame_roi[valid_mask] = blended.astype(np.uint8)[valid_mask]
            return

        if img_roi.ndim == 3 and img_roi.shape[2] == 3:
            frame_roi[:] = img_roi
        else:
            return
