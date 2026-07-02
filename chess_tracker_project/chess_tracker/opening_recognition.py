"""
Opening Recognition using chess opening database.

Provides:
- Opening name/ECO code lookup
- Opening classification by depth
- Transposition detection
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn


class OpeningRecognizer:
    """
    Identifies chess openings from a position or move sequence.
    Uses the built-in python-chess opening database (if available).
    """

    # Simplified common openings database (ECO classification)
    COMMON_OPENINGS = {
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR": ("1.e4", "King's Pawn Opening"),
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR": ("1.d4", "Queen's Pawn Opening"),
        "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR": ("1.c4", "English Opening"),
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R": ("1.Nf3", "Reti Opening"),
    }

    # ECO Code mapping (simplified)
    ECO_RANGES = {
        "A00-A03": "Uncommon Opening",
        "A04-A09": "Reti Opening",
        "A10-A39": "English Opening",
        "A40-A44": "Irregular Opening",
        "A45-A49": "Trompowsky Attack",
        "A50-A59": "Indian System",
        "A60-A79": "Benoni Defense",
        "A80-A99": "Dutch Defense",
        "B00-B19": "Scandinavian Defense",
        "B20-B99": "Sicilian Defense",
        "C00-C19": "French Defense",
        "C20-C99": "Open Game (1.e4 e5)",
        "D00-D09": "Queen's Gambit Declined",
        "D10-D19": "Queen's Gambit Accepted",
        "D20-D99": "Queen's Gambit",
        "E00-E99": "Indian Defense",
    }

    def __init__(self):
        self.move_history: List[str] = []
        self.opening_name = "Unknown"
        self.eco_code = "?"
        self.ply_count = 0

    def update(self, move: chess.Move, board: chess.Board) -> Dict[str, str]:
        """
        Update with a new move and return opening information.
        Args:
            move: The chess.Move that was played
            board: The board state BEFORE the move
        Returns:
            {'opening': str, 'eco': str, 'ply': int}
        """
        self.move_history.append(board.san(move))
        self.ply_count += 1

        # Recognize opening from move sequence
        opening_info = self._recognize_from_moves()

        return {
            "opening": opening_info[1],
            "eco": opening_info[0],
            "ply": self.ply_count,
        }

    def _recognize_from_moves(self) -> Tuple[str, str]:
        """Recognize opening from move sequence."""
        moves_str = " ".join(self.move_history)

        # Common opening patterns (simplified)
        patterns = {
            # Sicilian Defense
            r"^1\.e4 c5": ("B20", "Sicilian Defense"),
            # French Defense
            r"^1\.e4 e6": ("C00", "French Defense"),
            # Caro-Kann
            r"^1\.e4 c6": ("B10", "Caro-Kann Defense"),
            # Scandinavian
            r"^1\.e4 d5": ("B01", "Scandinavian Defense"),
            # Ruy Lopez
            r"^1\.e4 e5 2\.Nf3 Nc6 3\.Bb5": ("C60", "Ruy Lopez"),
            # Italian Game
            r"^1\.e4 e5 2\.Nf3 Nc6 3\.Bc4": ("C50", "Italian Game"),
            # Queen's Gambit
            r"^1\.d4 d5 2\.c4": ("D10", "Queen's Gambit"),
            # Indian Defense
            r"^1\.d4 Nf6 2\.c4": ("E12", "Indian Defense"),
            # English Opening
            r"^1\.c4": ("A10", "English Opening"),
            # Reti Opening
            r"^1\.Nf3": ("A04", "Reti Opening"),
        }

        import re

        for pattern, (eco, name) in patterns.items():
            if re.match(pattern, moves_str):
                return (eco, name)

        # Default
        if len(self.move_history) == 0:
            return ("?", "Starting Position")
        elif len(self.move_history) == 1:
            return ("?", "After 1st move")
        else:
            return ("?", "Unknown Opening")

    def get_opening_info(self, board: chess.Board) -> Dict[str, str]:
        """
        Get opening information for the current position.
        Returns: {'name': str, 'eco': str, 'phase': 'opening'|'middlegame'|'endgame'}
        """
        pieces = len(board.pieces(chess.PAWN, chess.WHITE)) + len(
            board.pieces(chess.PAWN, chess.BLACK)
        )

        phase = "opening"
        if self.ply_count > 15:
            phase = "middlegame"
        if pieces <= 8:
            phase = "endgame"

        opening_name, eco = self._recognize_from_moves()

        return {
            "name": opening_name,
            "eco": eco,
            "phase": phase,
            "ply": self.ply_count,
        }

    @staticmethod
    def analyze_game(game: chess.pgn.Game) -> List[Dict]:
        """
        Analyze all moves in a game and return opening information.
        Returns list of opening transitions.
        """
        recognizer = OpeningRecognizer()
        board = game.board()
        openings = []

        for move in game.mainline_moves():
            info = recognizer.update(move, board)
            openings.append(info)
            board.push(move)

        return openings

    @staticmethod
    def get_eco_description(eco_code: str) -> str:
        """Get a human-readable description of an ECO code."""
        for range_key, desc in OpeningRecognizer.ECO_RANGES.items():
            start, end = range_key.split("-")
            if start <= eco_code <= end:
                return desc
        return "Unknown ECO"
