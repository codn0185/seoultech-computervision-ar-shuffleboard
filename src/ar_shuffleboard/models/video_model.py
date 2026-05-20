import cv2
import numpy as np
from typing import Optional


class VideoModel:
    video_capture: Optional[cv2.VideoCapture]

    def __init__(self):
        self.video_capture = None

    def set_source(self, source: int):
        self.release()
        self.video_capture = cv2.VideoCapture(source)

    def release(self):
        if self.video_capture is None:
            return
        self.video_capture.release()
        self.video_capture = None

    def read_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.video_capture.read()
        return frame if ret else None
