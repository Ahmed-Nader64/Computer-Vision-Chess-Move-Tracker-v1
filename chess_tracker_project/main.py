"""
Command-line entry point (no Streamlit needed).

Usage:
    python main.py path/to/video.mp4 --out-dir ./output
    python main.py path/to/video.mp4 --frame-stride 2 --stability 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chess_tracker.config import (
    DEFAULT_PIECE_CONF,
    DEFAULT_PIECE_MODEL_URL,
    DEFAULT_POSE_CONF,
    DEFAULT_POSE_MODEL_URL,
    DEFAULT_STABILITY_SECONDS,
)
from chess_tracker.pipeline import ChessVideoTracker, save_moves_csv, save_moves_json, save_pgn


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track a chess game video into FEN/PGN.")
    parser.add_argument("video", type=str, help="Path to the input video file.")
    parser.add_argument("--out-dir", type=str, default="./output", help="Directory to write results to.")
    parser.add_argument("--pose-model", type=str, default=DEFAULT_POSE_MODEL_URL)
    parser.add_argument("--piece-model", type=str, default=DEFAULT_PIECE_MODEL_URL)
    parser.add_argument("--pose-conf", type=float, default=DEFAULT_POSE_CONF)
    parser.add_argument("--piece-conf", type=float, default=DEFAULT_PIECE_CONF)
    parser.add_argument("--stability", type=float, default=DEFAULT_STABILITY_SECONDS,
                         help="Seconds a board state must hold before a move is confirmed.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = ChessVideoTracker(
        pose_model_path=args.pose_model,
        piece_model_path=args.piece_model,
        pose_conf=args.pose_conf,
        piece_conf=args.piece_conf,
        stability_seconds=args.stability,
    )

    def on_progress(frame_idx: int, total_frames: int) -> None:
        if total_frames:
            print(f"\rFrame {frame_idx}/{total_frames}", end="", flush=True)

    def on_move(record) -> None:
        print(f"\n\u265f\ufe0f  {record.ply:>3} {record.color} {record.san:<8} FEN: {record.fen}")

    print(f"Processing {args.video} ...")
    result = tracker.process_video(
        args.video,
        progress_callback=on_progress,
        move_callback=on_move,
        frame_stride=args.frame_stride,
    )
    print()

    video_stem = Path(args.video).stem
    save_moves_csv(result["moves"], out_dir / f"{video_stem}_moves.csv")
    save_moves_json(result["moves"], out_dir / f"{video_stem}_moves.json")
    save_pgn(result["pgn"], out_dir / f"{video_stem}.pgn")

    print(f"\n{len(result['moves'])} move(s) detected.")
    print(f"Final FEN: {result['final_fen']}")
    print(f"Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
