"""
chess_tracker
=============

A computer-vision pipeline that watches a chess game on video, tracks the
pieces with two fine-tuned YOLO11 models, and reconstructs the game as a
sequence of FEN positions / a PGN file.

Modules:
    config               - default model URLs & pipeline constants
    board_localizer      - locates the board and produces a top-down warp
    state_analyzer       - turns per-frame detections into confirmed moves
    pgn_generator        - validates moves and builds FEN/PGN output
    pipeline             - orchestrates the full video -> moves pipeline
    stockfish_analyzer   - chess engine integration for move analysis
    opening_recognition  - ECO classification and opening names
    game_summary         - game statistics and player analytics
    board_viewer         - interactive chessboard visualization
"""

from .board_viewer import BoardViewer
from .game_summary import GameAnalyzer, GameSummary
from .opening_recognition import OpeningRecognizer
from .pipeline import ChessVideoTracker, MoveRecord
from .stockfish_analyzer import StockfishAnalyzer

__all__ = [
    "ChessVideoTracker",
    "MoveRecord",
    "StockfishAnalyzer",
    "OpeningRecognizer",
    "GameAnalyzer",
    "GameSummary",
    "BoardViewer",
]
