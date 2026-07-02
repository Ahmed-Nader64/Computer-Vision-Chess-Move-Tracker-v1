"""
Full pipeline: video in -> per-move FEN log + PGN out.

This ties together all four phases from the original notebook:
    1. BoardLocalizer   - find & warp the board
    2. YOLO piece model - detect pieces on the warped board
    3. ChessStateAnalyzer - temporal filtering -> confirmed moves
    4. ChessPGNGenerator  - rule validation -> SAN/FEN/PGN
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import torch
from ultralytics import YOLO

from .board_localizer import BoardLocalizer
from .config import (
    DEFAULT_BOARD_SIZE,
    DEFAULT_PIECE_CONF,
    DEFAULT_PIECE_MODEL_URL,
    DEFAULT_POSE_CONF,
    DEFAULT_POSE_MODEL_URL,
    DEFAULT_STABILITY_SECONDS,
)
from .pgn_generator import ChessPGNGenerator, generate_fen_from_dict
from .state_analyzer import ChessStateAnalyzer

# Called as progress_callback(frame_idx, total_frames)
ProgressCallback = Callable[[int, int], None]

# Called as frame_callback(annotated_bgr_frame_or_None, frame_idx, total_frames)
FrameCallback = Callable[[object, int, int], None]

# Called as move_callback(MoveRecord) every time a new move is confirmed
MoveCallback = Callable[["MoveRecord"], None]


@dataclass
class MoveRecord:
    ply: int
    color: str          # 'w' or 'b'
    from_square: str
    to_square: str
    san: str
    fen: str             # board FEN *after* this move


class ChessVideoTracker:
    """
    High-level entry point. Instantiate once (this loads both YOLO models),
    then call `process_video()` for one or more videos.
    """

    def __init__(
        self,
        pose_model_path: str = DEFAULT_POSE_MODEL_URL,
        piece_model_path: str = DEFAULT_PIECE_MODEL_URL,
        board_size: int = DEFAULT_BOARD_SIZE,
        pose_conf: float = DEFAULT_POSE_CONF,
        piece_conf: float = DEFAULT_PIECE_CONF,
        stability_seconds: float = DEFAULT_STABILITY_SECONDS,
        device: Optional[str] = None,
    ):
        self.board_size = board_size
        self.piece_conf = piece_conf
        self.stability_seconds = stability_seconds

        self.device = device or ("0" if torch.cuda.is_available() else "cpu")

        self.localizer = BoardLocalizer(
            model_path=pose_model_path, target_size=board_size, conf_thresh=pose_conf
        )
        self.piece_model = YOLO(piece_model_path)

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[ProgressCallback] = None,
        frame_callback: Optional[FrameCallback] = None,
        move_callback: Optional[MoveCallback] = None,
        frame_stride: int = 1,
    ) -> dict:
        """
        Run the full pipeline on a single video.

        Args:
            video_path: Path to the input .mp4/.mov/etc video.
            progress_callback: optional fn(frame_idx, total_frames) called every frame.
            frame_callback: optional fn(annotated_frame_bgr_or_None, frame_idx, total_frames)
                called every frame; useful for a live preview (e.g. in Streamlit).
            move_callback: optional fn(MoveRecord) called every time a move is confirmed.
            frame_stride: process every Nth frame (>=1). Speeds up long videos at the
                cost of potentially missing very quick moves; the stability threshold
                is automatically adjusted for the stride.

        Returns:
            dict with keys:
                'moves': List[MoveRecord]
                'pgn': str (full PGN with headers)
                'final_fen': str (FEN of the final position, or starting FEN if no moves)
        """
        frame_stride = max(1, frame_stride)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        effective_fps = fps / frame_stride
        stability_thresh = max(1, int(self.stability_seconds * effective_fps))

        self.localizer.reset()
        state_analyzer = ChessStateAnalyzer(
            board_size=self.board_size, stability_thresh=stability_thresh
        )
        pgn_engine = ChessPGNGenerator()

        moves: List[MoveRecord] = []
        ply = 0
        frame_idx = 0
        video_start_fen: Optional[str] = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_stride > 1 and (frame_idx - 1) % frame_stride != 0:
                    if progress_callback:
                        progress_callback(frame_idx, total_frames)
                    continue

                annotated = None
                warped = self.localizer.get_warped_frame(frame)

                if warped is not None:
                    results = self.piece_model.predict(
                        warped,
                        imgsz=self.board_size,
                        conf=self.piece_conf,
                        device=self.device,
                        verbose=False,
                    )

                    detections = []
                    if results[0].boxes:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                            conf = box.conf.cpu().item()
                            cls_id = int(box.cls.cpu().item())
                            cls_name = self.piece_model.names[cls_id]
                            detections.append([x1, y1, x2, y2, conf, cls_id, cls_name])

                    if frame_callback is not None:
                        annotated = results[0].plot()

                    event = state_analyzer.process_frame(detections)

                    if event:
                        if event["type"] == "first_move":
                            initial_board_state = event["initial_board_state"]
                            active_color = event["active_color"]
                            from_sq, to_sq = event["move"]

                            start_fen = generate_fen_from_dict(
                                board_state=initial_board_state, active_color=active_color
                            )
                            video_start_fen = start_fen
                            pgn_engine.set_board_from_fen(start_fen)
                            san = pgn_engine.push_move(from_sq, to_sq, ignore_rules=True)

                            if san:
                                ply += 1
                                record = MoveRecord(
                                    ply=ply,
                                    color=active_color,
                                    from_square=from_sq,
                                    to_square=to_sq,
                                    san=san,
                                    fen=pgn_engine.current_fen(),
                                )
                                moves.append(record)
                                if move_callback:
                                    move_callback(record)

                        elif event["type"] == "move":
                            from_sq, to_sq = event["move"]
                            mover_color = "w" if pgn_engine.board.turn else "b"
                            san = pgn_engine.push_move(from_sq, to_sq)

                            if san:
                                ply += 1
                                record = MoveRecord(
                                    ply=ply,
                                    color=mover_color,
                                    from_square=from_sq,
                                    to_square=to_sq,
                                    san=san,
                                    fen=pgn_engine.current_fen(),
                                )
                                moves.append(record)
                                if move_callback:
                                    move_callback(record)

                if frame_callback is not None:
                    frame_callback(annotated, frame_idx, total_frames)
                if progress_callback:
                    progress_callback(frame_idx, total_frames)
        finally:
            cap.release()
            self.localizer.reset()

        return {
            "moves": moves,
            "pgn": pgn_engine.get_pgn_string(headers=True, clean_format=False),
            "final_fen": pgn_engine.current_fen(),
            "start_fen": video_start_fen,
        }


def save_moves_csv(moves: List[MoveRecord], path: str) -> None:
    """Write the per-move FEN log to a CSV file."""
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ply", "color", "from_square", "to_square", "san", "fen"]
        )
        writer.writeheader()
        for m in moves:
            writer.writerow(asdict(m))


def save_moves_json(moves: List[MoveRecord], path: str) -> None:
    """Write the per-move FEN log to a JSON file."""
    with open(str(path), "w") as f:
        json.dump([asdict(m) for m in moves], f, indent=2)


def save_pgn(pgn_text: str, path: str) -> None:
    Path(path).write_text(pgn_text)
