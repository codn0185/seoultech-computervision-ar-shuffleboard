from __future__ import annotations

import cv2
import numpy as np
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ar_shuffleboard.controllers.main_controller import MainController


class MainModel:
    def __init__(self, controller: MainController, **kwargs):
        pass

    def get_processed_frame(self, frame):
        """프레임 가공 후 반환한다."""
        return frame
