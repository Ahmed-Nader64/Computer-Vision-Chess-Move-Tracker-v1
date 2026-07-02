"""
Phase 1: Board Localization & Perspective Warping.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


class BoardLocalizer:
    """
    Locates the chessboard in a video frame and warps it to a standardized
    top-down view, using a fine-tuned YOLO11-Pose model that predicts 4
    keypoints: a1, h1, a8, h8.

    Once a good detection is found, the perspective-transform matrix is
    "locked" and reused for every subsequent frame (until `reset()` is
    called), which avoids re-running the pose model on every frame and
    keeps the warp stable even if a hand temporarily occludes a corner.
    """

    def __init__(
        self,
        model_path: str,
        target_size: int = 640,
        conf_thresh: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.target_size = target_size
        self.conf_thresh = conf_thresh

        # Memory
        self.M: np.ndarray | None = None  # 3x3 perspective transform matrix
        self.is_locked = False
        self.locked_points: np.ndarray | None = None

        # Destination points for the perspective transform:
        # top-left=a8, top-right=h8, bottom-left=a1, bottom-right=h1
        self.dst_points = np.array(
            [
                [0, target_size],           # a1
                [target_size, target_size], # h1
                [0, 0],                     # a8
                [target_size, 0],           # h8
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        """Clear the locked calibration (call between videos or if the
        camera moved)."""
        self.M = None
        self.is_locked = False
        self.locked_points = None

    def is_calibrated(self) -> bool:
        return self.is_locked

    def get_locked_points(self) -> np.ndarray | None:
        return self.locked_points

    def _attempt_calibration(self, frame: np.ndarray) -> bool:
        """Try to detect the 4 board corners in `frame` and lock the
        perspective transform. Returns True on success."""
        results = self.model(frame, conf=self.conf_thresh, verbose=False)

        if len(results) == 0 or len(results[0].keypoints) == 0:
            return False

        kpts = results[0].keypoints.xy[0].cpu().numpy()   # (4, 2)
        confs = results[0].keypoints.conf[0].cpu().numpy()

        # All 4 keypoints must have valid (non-zero/negative) coordinates
        if np.any(kpts <= 0):
            return False

        # Average confidence must clear the threshold
        if np.mean(confs) < self.conf_thresh:
            return False

        self.M = cv2.getPerspectiveTransform(kpts.astype(np.float32), self.dst_points)
        self.is_locked = True
        self.locked_points = kpts.astype(np.float32)
        return True

    def get_warped_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Return a (target_size, target_size) top-down warp of the board, or
        None if the board hasn't been (or couldn't be) calibrated yet.
        """
        if not self.is_locked:
            if not self._attempt_calibration(frame):
                return None

        return cv2.warpPerspective(
            frame, M=self.M, dsize=(self.target_size, self.target_size)
        )
