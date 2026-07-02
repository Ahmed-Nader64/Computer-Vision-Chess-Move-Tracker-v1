"""
Phase 4: Rules Validation & FEN/PGN Generation.

Uses `python-chess` as the referee: every visually-detected move is checked
for legality (except the very first move, whose legality can't be verified
since we don't know the true game history), the internal board state is
kept in sync, and both FEN snapshots and a full PGN can be exported.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Optional

import chess
import chess.pgn

STANDARD_START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def generate_fen_from_dict(board_state: Dict[str, str], active_color: str) -> str:
    """
    Convert a `{square: piece_code}` dict (e.g. `{'e4': 'wP'}`) into a FEN
    string. Detects the standard starting position automatically; otherwise
    derives castling rights from whether kings/rooks are still on their
    home squares.
    """
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]

    piece_rows = []
    for rank in ranks:
        empty_count = 0
        row_str = ""
        for file in files:
            square = file + rank
            if square in board_state:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                piece_code = board_state[square]
                color, role = piece_code[0], piece_code[1]
                row_str += role.upper() if color == "w" else role.lower()
            else:
                empty_count += 1
        if empty_count > 0:
            row_str += str(empty_count)
        piece_rows.append(row_str)

    piece_placement = "/".join(piece_rows)

    if piece_placement == STANDARD_START_PLACEMENT and active_color == "w":
        return f"{STANDARD_START_PLACEMENT} w KQkq - 0 1"

    castling_rights = ""
    if board_state.get("e1") == "wK":
        if board_state.get("h1") == "wR":
            castling_rights += "K"
        if board_state.get("a1") == "wR":
            castling_rights += "Q"
    if board_state.get("e8") == "bK":
        if board_state.get("h8") == "bR":
            castling_rights += "k"
        if board_state.get("a8") == "bR":
            castling_rights += "q"
    if castling_rights == "":
        castling_rights = "-"

    return f"{piece_placement} {active_color} {castling_rights} - 0 1"


class ChessPGNGenerator:
    """
    Bridges the CV pipeline and the official chess rules. Validates raw
    (from_square, to_square) moves, keeps the board state in sync, builds
    the PGN game tree, and exposes FEN snapshots at any point.
    """

    def __init__(self, event_name: str = "Chess Move Tracking"):
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = event_name
        self.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self.node = self.game

    def set_board_from_fen(self, fen_str: str) -> bool:
        """Initialize the board from a FEN string (for mid-game starts)."""
        try:
            self.board = chess.Board(fen_str)
            self.game.setup(self.board)
            self.node = self.game
            return True
        except ValueError:
            return False

    def _is_promotion(self, move: chess.Move) -> bool:
        piece = self.board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(move.to_square)
            if rank in (0, 7):
                return True
        return False

    def push_move(
        self, from_sq: str, to_sq: str, ignore_rules: bool = False
    ) -> Optional[str]:
        """
        Validate and apply a move given as square strings (e.g. 'e2', 'e4').
        Returns the SAN string (e.g. 'Nf3') on success, or None if the move
        is illegal (and `ignore_rules=False`) or otherwise invalid.

        `ignore_rules=True` is used for the very first detected move, since
        legality can't be checked against an unknown prior game history.
        """
        try:
            move = chess.Move.from_uci(f"{from_sq}{to_sq}")

            if self._is_promotion(move):
                move.promotion = chess.QUEEN

            if not ignore_rules and move not in self.board.legal_moves:
                return None

            san_move = self.board.san(move)
            self.node = self.node.add_variation(move)
            self.board.push(move)
            return san_move

        except Exception:
            return None

    def current_fen(self) -> str:
        """FEN snapshot of the board *right now*."""
        return self.board.fen()

    def get_pgn_string(self, headers: bool = True, clean_format: bool = False) -> str:
        exporter = chess.pgn.StringExporter(headers=headers, variations=True, comments=False)
        pgn_raw = self.game.accept(exporter)

        if clean_format:
            pgn_clean = re.sub(r"\s+(1-0|0-1|1\/2-1\/2|\*)$", "", pgn_raw)
            pgn_clean = pgn_clean.replace("...", ".")
            pgn_clean = pgn_clean.replace("\n", " ")
            pgn_clean = re.sub(r"\s+", " ", pgn_clean)
            return pgn_clean.strip()

        return pgn_raw

    def save_pgn_file(self, filename: str = "game.pgn") -> None:
        with open(filename, "w") as f:
            f.write(self.get_pgn_string())
