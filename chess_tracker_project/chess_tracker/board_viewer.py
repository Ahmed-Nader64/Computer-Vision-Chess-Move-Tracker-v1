"""
Interactive Chess Board Viewer for Streamlit.

Provides:
- Interactive chessboard display
- Move highlighting and animation
- FEN visualization
- PV (Principal Variation) display
"""
from __future__ import annotations

import chess


class BoardViewer:
    """
    Generates HTML/SVG representation of a chess position for
    interactive display in Streamlit.
    """

    PIECE_UNICODE = {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
        "P": "♙",
        "k": "♚",
        "q": "♛",
        "r": "♜",
        "b": "♝",
        "n": "♞",
        "p": "♟",
    }

    COLORS = {
        "light": "#f0d9b5",
        "dark": "#baca44",
        "highlight_move": "#baca44",
        "highlight_square": "#f6d66f",
        "text_light": "#111111",
        "text_dark": "#f0d9b5",
    }

    SQUARE_SIZE = 60  # pixels

    @staticmethod
    def get_square_color(square: int) -> str:
        """Get the color of a square."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if (file + rank) % 2 == 0:
            return BoardViewer.COLORS["light"]
        else:
            return BoardViewer.COLORS["dark"]

    @staticmethod
    def get_text_color(square: int) -> str:
        """Get the text color for a square."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if (file + rank) % 2 == 0:
            return BoardViewer.COLORS["text_light"]
        else:
            return BoardViewer.COLORS["text_dark"]

    @classmethod
    def board_to_svg(
        cls,
        board: chess.Board,
        last_move: chess.Move = None,
        orientation: str = "white",
    ) -> str:
        """
        Convert a board to an interactive SVG.
        Args:
            board: chess.Board object
            last_move: Optional last move to highlight
            orientation: 'white' or 'black' for board flip
        Returns:
            SVG string
        """
        size = cls.SQUARE_SIZE * 8
        margin = 30
        width = size + 2 * margin
        height = size + 2 * margin
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#eef2f7"/>',
            f'<rect x="{margin - 8}" y="{margin - 8}" width="{size + 16}" height="{size + 16}" rx="28" fill="#2f3746"/>',
            f'<rect x="{margin}" y="{margin}" width="{size}" height="{size}" rx="18" fill="#f7f4ec" stroke="#2f3746" stroke-width="2"/>',
        ]

        # Determine board orientation
        flip = orientation == "black"

        # Draw squares and pieces
        for rank in range(8):
            for file in range(8):
                if flip:
                    x = margin + (7 - file) * cls.SQUARE_SIZE
                    y = margin + rank * cls.SQUARE_SIZE
                    square = chess.square(7 - file, 7 - rank)
                else:
                    x = margin + file * cls.SQUARE_SIZE
                    y = margin + (7 - rank) * cls.SQUARE_SIZE
                    square = chess.square(file, rank)

                color = cls.get_square_color(square)

                # Highlight last move
                if last_move and square in (last_move.from_square, last_move.to_square):
                    color = cls.COLORS["highlight_square"]

                # Draw square
                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cls.SQUARE_SIZE}" '
                    f'height="{cls.SQUARE_SIZE}" fill="{color}"/>'
                )

                # Draw piece
                piece = board.piece_at(square)
                if piece:
                    text_color = cls.get_text_color(square)
                    piece_char = cls.PIECE_UNICODE[piece.symbol()]
                    cx = x + cls.SQUARE_SIZE / 2
                    cy = y + cls.SQUARE_SIZE / 2

                    svg_parts.append(
                        f'<text x="{cx}" y="{cy}" font-size="48" font-weight="bold" '
                        f'fill="{text_color}" text-anchor="middle" dominant-baseline="central">'
                        f"{piece_char}</text>"
                    )

                # Draw coordinates
                if file == 0:
                    rank_label = 8 - rank if not flip else rank + 1
                    svg_parts.append(
                        f'<text x="{x - 6}" y="{y + cls.SQUARE_SIZE / 2 + 5}" font-size="12" '
                        f'fill="#2f2f2f" text-anchor="end">{rank_label}</text>'
                    )
                if rank == 7:
                    file_label = chr(ord("a") + file if not flip else ord("h") - file)
                    svg_parts.append(
                        f'<text x="{x + cls.SQUARE_SIZE / 2}" y="{y + cls.SQUARE_SIZE + 18}" '
                        f'font-size="12" fill="#2f2f2f" text-anchor="middle">{file_label}</text>'
                    )
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def board_to_text(cls, board: chess.Board) -> str:
        """
        Convert a board to ASCII representation.
        Returns:
            ASCII board string
        """
        lines = ["  a b c d e f g h"]
        for rank in range(8, 0, -1):
            row = f"{rank} "
            for file in range(8):
                square = chess.square(file, rank - 1)
                piece = board.piece_at(square)
                row += (piece.symbol() if piece else ".") + " "
            row += str(rank)
            lines.append(row)
        lines.append("  a b c d e f g h")
        return "\n".join(lines)

    @staticmethod
    def get_html_viewer(
        board: chess.Board,
        move_history: list,
        last_move: chess.Move = None,
        title: str = "Chessboard",
    ) -> str:
        """
        Generate a complete HTML page for board viewing with move history.
        """
        board_svg = BoardViewer.board_to_svg(board, last_move)

        move_list = ""
        for i, move_san in enumerate(move_history):
            if i % 2 == 0:
                move_list += f"<div class='move-pair'>"
            move_list += f"<span class='move'>{i + 1}.{move_san if i % 2 == 0 else ''} {move_san if i % 2 == 1 else ''}</span>"
            if i % 2 == 1:
                move_list += "</div>"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Inter', Arial, sans-serif; background: #f4f6fb; color: #1f2937; margin: 0; }}
                .wrapper {{ max-width: 1200px; margin: 0 auto; padding: 18px; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }}
                .header h2 {{ margin: 0; font-size: 20px; color: #111827; }}
                .container {{ display: flex; gap: 24px; flex-wrap: wrap; }}
                .board {{ flex: 0 0 auto; }}
                .info {{ flex: 1; min-width: 300px; }}
                .move-list {{ background: #ffffff; padding: 16px; border-radius: 16px; box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08); max-height: 520px; overflow-y: auto; }}
                .move-pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 8px; }}
                .move {{ padding: 8px 10px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; color: #111827; font-size: 14px; }}
                .fen-card {{ background: #ffffff; padding: 16px; border-radius: 16px; border: 1px solid #e5e7eb; margin-top: 16px; }}
                .fen {{ font-family: monospace; font-size: 13px; color: #111827; word-break: break-all; }}
            </style>
        </head>
        <body>
            <div class="wrapper">
                <div class="header">
                    <h2>{title}</h2>
                    <span style="color:#6b7280;font-size:14px">{board.turn and 'White to move' or 'Black to move'}</span>
                </div>
                <div class="container">
                    <div class="board">{board_svg}</div>
                    <div class="info">
                        <h3>Move History</h3>
                        <div class="move-list">{move_list or '<em>No moves available</em>'}</div>
                        <div class="fen-card">
                            <h3 style="margin-top:0;">FEN</h3>
                            <div class="fen">{board.fen()}</div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html
