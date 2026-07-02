"""
Stockfish Integration for move analysis and quality scoring.

Provides:
- Best move suggestions
- Move quality scoring (brilliant, excellent, good, inaccuracy, etc.)
- Evaluation deltas
- Principal variation (best line) suggestions
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn


class StockfishAnalyzer:
    """
    Wraps Stockfish for:
    1. Position evaluation
    2. Best move computation
    3. Move quality assessment (compared to best move)
    4. Opening classification
    """

    # Move quality thresholds (in centipawns)
    BLUNDER_THRESHOLD = 200  # > 200 cp loss = blunder
    MISTAKE_THRESHOLD = 100  # > 100 cp loss = mistake
    INACCURACY_THRESHOLD = 50  # > 50 cp loss = inaccuracy
    EXCELLENT_THRESHOLD = -50  # < -50 cp gain = excellent/brilliant

    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 20):
        """
        Args:
            stockfish_path: Path to stockfish binary. If None, tries to find it in PATH.
            depth: Search depth for analysis (higher = slower but more accurate).
        """
        self.depth = depth
        self.stockfish_path = stockfish_path or self._find_stockfish()

    @staticmethod
    def _find_stockfish() -> Optional[str]:
        """Try to locate stockfish in common places or PATH."""
        common_paths = [
            r"C:\Users\anahm\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe",
            r"C:\Users\anahm\Downloads\stockfish-windows-x86-64-avx2\stockfish.exe",
            r"C:\Users\anahm\Downloads\stockfish-windows-x86-64-avx2\stockfish",
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "stockfish",
            "stockfish.exe",
        ]
        for path in common_paths:
            try:
                subprocess.run([path, "--version"], capture_output=True, timeout=5, check=True)
                return path
            except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError, OSError):
                continue
        return None

    def analyze_position(self, board: chess.Board) -> Optional[Dict]:
        """
        Analyze a single position.
        Returns: {
            'best_move': 'e2e4',
            'eval': 0.35 (in pawns),
            'mate': None or mate-in-N,
            'pv': ['e2e4', 'e7e5', ...],  # principal variation
        }
        """
        if not self.stockfish_path:
            return None

        try:
            cmd = [self.stockfish_path]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Set up the engine
            process.stdin.write("setoption name Threads value 1\n")
            process.stdin.write("setoption name Hash value 128\n")
            process.stdin.write("isready\n")
            process.stdin.flush()

            # Wait for readiness
            for _ in range(1000):
                line = process.stdout.readline()
                if "readyok" in line:
                    break

            # Send position
            process.stdin.write(f"position fen {board.fen()}\n")
            process.stdin.write(f"go depth {self.depth}\n")
            process.stdin.flush()

            best_move = None
            eval_cp = None
            mate_in = None
            pv = []

            # Parse output
            for _ in range(10000):
                line = process.stdout.readline()
                if not line:
                    break

                if "bestmove" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move = parts[1]
                    break

                if "info" in line:
                    parts = line.split()
                    try:
                        # Extract evaluation
                        if "cp" in parts:
                            idx = parts.index("cp")
                            eval_cp = int(parts[idx + 1])
                        elif "mate" in parts:
                            idx = parts.index("mate")
                            mate_in = int(parts[idx + 1])

                        # Extract principal variation
                        if "pv" in parts:
                            idx = parts.index("pv")
                            pv = parts[idx + 1 :]
                    except (ValueError, IndexError):
                        pass

            process.stdin.write("quit\n")
            process.stdin.flush()
            process.terminate()

            if best_move:
                eval_pawns = eval_cp / 100.0 if eval_cp is not None else 0
                return {
                    "best_move": best_move,
                    "eval": eval_pawns,
                    "eval_cp": eval_cp,
                    "mate": mate_in,
                    "pv": pv[:5],  # First 5 moves
                }

        except Exception as e:
            print(f"Stockfish error: {e}")

        return None

    def score_move(
        self, board: chess.Board, move_uci: str
    ) -> Dict[str, any]:
        """
        Evaluate a specific move by comparing it to the best move.
        Returns: {
            'quality': 'brilliant' | 'excellent' | 'good' | 'ok' | 'inaccuracy' | 'mistake' | 'blunder',
            'best_move': 'e2e4',
            'eval_before': 0.5,
            'eval_after': 0.3,
            'delta': -0.2,  # How much the position changed (negative = got worse)
            'mate_threat': None or mate-in-N,
        }
        """
        if not self.stockfish_path:
            return {"quality": "unknown", "reason": "Stockfish not found"}

        try:
            # Evaluate before move
            before = self.analyze_position(board)
            if not before:
                return {"quality": "unknown", "reason": "Failed to analyze before"}

            # Make the move and evaluate after
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return {"quality": "illegal", "reason": "Illegal move"}

            board.push(move)
            after = self.analyze_position(board)
            board.pop()

            if not after:
                return {"quality": "unknown", "reason": "Failed to analyze after"}

            before_eval = before.get("eval", 0)
            after_eval = after.get("eval", 0)

            # From the mover's perspective
            # If we're black (eval is negative from white's view), flip the sign
            if not board.turn:  # Black just moved
                delta = -(after_eval - before_eval)
            else:  # White just moved
                delta = after_eval - before_eval

            best_move = before.get("best_move", "?")
            quality = self._classify_move_quality(delta)

            return {
                "quality": quality,
                "best_move": best_move,
                "eval_before": before_eval,
                "eval_after": after_eval,
                "delta": delta,
                "mate_threat": after.get("mate"),
            }

        except Exception as e:
            return {"quality": "error", "reason": str(e)}

    @classmethod
    def _classify_move_quality(cls, delta: float) -> str:
        """Classify move quality based on evaluation delta."""
        delta_cp = delta * 100
        if delta_cp <= -cls.MISTAKE_THRESHOLD:
            return "brilliant"
        elif delta_cp < -cls.INACCURACY_THRESHOLD:
            return "excellent"
        elif delta_cp < cls.INACCURACY_THRESHOLD:
            return "good"
        elif delta_cp < cls.MISTAKE_THRESHOLD:
            return "inaccuracy"
        elif delta_cp < cls.BLUNDER_THRESHOLD:
            return "mistake"
        else:
            return "blunder"

    def analyze_game(
        self, game: chess.pgn.Game
    ) -> List[Dict]:
        """
        Analyze all moves in a game.
        Returns list of move analyses with quality scores.
        """
        board = game.board()
        analyses = []

        for move in game.mainline_moves():
            move_uci = move.uci()
            analysis = self.score_move(board, move_uci)
            analysis["move"] = move.uci()
            analysis["san"] = board.san(move)
            analyses.append(analysis)
            board.push(move)

        return analyses
