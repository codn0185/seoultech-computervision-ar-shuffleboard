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
