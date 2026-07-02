"""
Game Summary & Statistics Dashboard.

Provides:
- Move counts and game statistics
- Accuracy metrics per player
- Time spent in each phase (opening, middlegame, endgame)
- Critical decision analysis
- Overall game narrative
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess
import chess.pgn

from .opening_recognition import OpeningRecognizer


@dataclass
class PlayerStats:
    """Statistics for one player."""

    color: str  # 'w' or 'b'
    name: str = "Player"
    total_moves: int = 0
    accurate_moves: int = 0
    inaccuracies: int = 0
    mistakes: int = 0
    blunders: int = 0
    brilliant_moves: int = 0
    average_accuracy: float = 0.0
    best_move_percentage: float = 0.0
    material_disadvantage_moves: int = 0
    pieces_captured: List[str] = field(default_factory=list)
    pieces_lost: List[str] = field(default_factory=list)


@dataclass
class GameSummary:
    """Complete game analysis summary."""

    white: PlayerStats
    black: PlayerStats
    total_moves: int = 0
    result: str = "*"  # "*", "1-0", "0-1", "1/2-1/2"
    opening_name: str = "Unknown"
    eco_code: str = "?"
    game_length_phase: str = "unknown"  # opening, middlegame, endgame

    # Critical moments
    critical_moves: List[Dict] = field(default_factory=list)
    turning_points: List[Dict] = field(default_factory=list)

    # Phase analysis
    opening_moves: int = 0
    middlegame_moves: int = 0
    endgame_moves: int = 0

    # Material balance
    final_material_white: int = 0
    final_material_black: int = 0
    biggest_swing: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "white": {
                "name": self.white.name,
                "moves": self.white.total_moves,
                "accuracy": self.white.average_accuracy,
                "best_move_pct": self.white.best_move_percentage,
                "brilliant": self.white.brilliant_moves,
                "inaccuracies": self.white.inaccuracies,
                "mistakes": self.white.mistakes,
                "blunders": self.white.blunders,
            },
            "black": {
                "name": self.black.name,
                "moves": self.black.total_moves,
                "accuracy": self.black.average_accuracy,
                "best_move_pct": self.black.best_move_percentage,
                "brilliant": self.black.brilliant_moves,
                "inaccuracies": self.black.inaccuracies,
                "mistakes": self.black.mistakes,
                "blunders": self.black.blunders,
            },
            "game": {
                "result": self.result,
                "total_moves": self.total_moves,
                "opening": self.opening_name,
                "eco": self.eco_code,
                "opening_moves": self.opening_moves,
                "middlegame_moves": self.middlegame_moves,
                "endgame_moves": self.endgame_moves,
            },
        }


class GameAnalyzer:
    """Analyze a complete chess game and generate summary statistics."""

    def __init__(self):
        self.white_stats = PlayerStats(color="w", name="White")
        self.black_stats = PlayerStats(color="b", name="Black")

    def analyze_game(
        self, game: chess.pgn.Game, move_analyses: Optional[List[Dict]] = None
    ) -> GameSummary:
        """
        Analyze a complete game.
        Args:
            game: chess.pgn.Game object
            move_analyses: Optional list of move quality analyses from StockfishAnalyzer
        Returns:
            GameSummary object with detailed statistics
        """
        board = game.board()

        summary = GameSummary(
            white=PlayerStats(color="w", name=game.headers.get("White", "White")),
            black=PlayerStats(color="b", name=game.headers.get("Black", "Black")),
        )

        # Determine result
        result = game.headers.get("Result", "*")
        summary.result = result

        recognizer = OpeningRecognizer()

        # Analyze each move
        move_idx = 0
        for move in game.mainline_moves():
            move_idx += 1
            current_player = summary.white if board.turn else summary.black

            # Recognize opening moves progressively
            recognizer.update(move, board)

            # Determine game phase by move count
            if move_idx <= 20:
                phase = "opening"
                summary.opening_moves += 1
            elif move_idx <= 40:
                phase = "middlegame"
                summary.middlegame_moves += 1
            else:
                phase = "endgame"
                summary.endgame_moves += 1

            # Update move count
            current_player.total_moves += 1
            summary.total_moves += 1

            # Apply move quality analysis if available
            if move_analyses and move_idx - 1 < len(move_analyses):
                analysis = move_analyses[move_idx - 1]
                self._apply_move_analysis(current_player, analysis, summary)

            board.push(move)

        opening_info = recognizer.get_opening_info(board)
        summary.opening_name = opening_info["name"]
        summary.eco_code = opening_info["eco"]

        # Post-game analysis
        self._calculate_final_stats(summary, game)

        return summary

    @staticmethod
    def _apply_move_analysis(
        player: PlayerStats, analysis: Dict, summary: GameSummary
    ) -> None:
        """Apply a move analysis to player stats."""
        quality = analysis.get("quality", "unknown")

        if quality == "brilliant":
            player.brilliant_moves += 1
            player.accurate_moves += 1
        elif quality == "excellent":
            player.accurate_moves += 1
        elif quality == "good":
            player.accurate_moves += 1
        elif quality == "inaccuracy":
            player.inaccuracies += 1
        elif quality == "mistake":
            player.mistakes += 1
        elif quality == "blunder":
            player.blunders += 1

    @staticmethod
    def _calculate_final_stats(summary: GameSummary, game: chess.pgn.Game) -> None:
        """Calculate final summary statistics."""
        board = game.board()

        # Set game length phase
        if summary.opening_moves > 0 and summary.middlegame_moves == 0:
            summary.game_length_phase = "opening"
        elif summary.middlegame_moves > 0 and summary.endgame_moves == 0:
            summary.game_length_phase = "middlegame"
        else:
            summary.game_length_phase = "endgame"

        # Calculate accuracy percentages
        if summary.white.total_moves > 0:
            summary.white.average_accuracy = (
                summary.white.accurate_moves / summary.white.total_moves * 100
            )
            summary.white.best_move_percentage = (
                (summary.white.accurate_moves + summary.white.brilliant_moves)
                / summary.white.total_moves
                * 100
            )

        if summary.black.total_moves > 0:
            summary.black.average_accuracy = (
                summary.black.accurate_moves / summary.black.total_moves * 100
            )
            summary.black.best_move_percentage = (
                (summary.black.accurate_moves + summary.black.brilliant_moves)
                / summary.black.total_moves
                * 100
            )

        # Determine winner narrative
        if summary.result == "1-0":
            if summary.white.blunders > summary.black.blunders:
                summary.white.name += " (won despite errors)"
            elif summary.black.blunders > summary.white.blunders:
                summary.black.name += " (capitulated)"
        elif summary.result == "0-1":
            if summary.black.blunders > summary.white.blunders:
                summary.black.name += " (won despite errors)"
            elif summary.white.blunders > summary.black.blunders:
                summary.white.name += " (capitulated)"

    @staticmethod
    def get_narrative(summary: GameSummary) -> str:
        """Generate a human-readable game narrative."""
        lines = [
            f"Opening: {summary.opening_name} ({summary.eco_code})",
            f"Result: {summary.result} after {summary.total_moves} moves",
            "",
            "White Statistics:",
            f"  • Moves: {summary.white.total_moves}",
            f"  • Accuracy: {summary.white.average_accuracy:.1f}%",
            f"  • Brilliant: {summary.white.brilliant_moves} | Inaccuracies: {summary.white.inaccuracies} | "
            f"Mistakes: {summary.white.mistakes} | Blunders: {summary.white.blunders}",
            "",
            "Black Statistics:",
            f"  • Moves: {summary.black.total_moves}",
            f"  • Accuracy: {summary.black.average_accuracy:.1f}%",
            f"  • Brilliant: {summary.black.brilliant_moves} | Inaccuracies: {summary.black.inaccuracies} | "
            f"Mistakes: {summary.black.mistakes} | Blunders: {summary.black.blunders}",
            "",
            "Game Breakdown:",
            f"  • Opening phase: {summary.opening_moves} moves",
            f"  • Middlegame phase: {summary.middlegame_moves} moves",
            f"  • Endgame phase: {summary.endgame_moves} moves",
        ]

        return "\n".join(lines)
