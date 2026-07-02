"""
Default configuration for the chess move tracking pipeline.

The two models below are the fine-tuned YOLO11 weights from the original
notebook, hosted on Hugging Face. `ultralytics.YOLO(...)` can load a model
directly from an https:// URL (it downloads and caches it locally), so no
manual download step is required.
"""

# Fine-tuned YOLO11s-Pose model: detects the 4 board corners (a1, h1, a8, h8)
DEFAULT_POSE_MODEL_URL = (
    "https://huggingface.co/surawut/chess-move-tracking-yolo11/"
    "resolve/main/models/yolo11s_pose_chessboard.pt"
)

# Fine-tuned YOLO11m model: detects the 12 piece classes + "Hand"
DEFAULT_PIECE_MODEL_URL = (
    "https://huggingface.co/surawut/chess-move-tracking-yolo11/"
    "resolve/main/models/yolo11m_pieces.pt"
)

# Warped top-down board size in pixels (square)
DEFAULT_BOARD_SIZE = 640

# Confidence thresholds
DEFAULT_POSE_CONF = 0.5
DEFAULT_PIECE_CONF = 0.25

# How many seconds a board state must stay unchanged before a move is
# confirmed. Multiply by the video's FPS to get the frame-count threshold.
DEFAULT_STABILITY_SECONDS = 1.5
