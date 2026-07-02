"""
Chess Board Viewer – uses python-chess's built-in SVG renderer for
a professional Lichess-style board with proper piece graphics.

Provides:
- board_to_svg()       — single-position SVG (Lichess-style)
- board_to_text()      — ASCII fallback
- get_html_viewer()    — standalone HTML page with move list
- wrap_svg()           — convenience wrapper for Streamlit st.components.v1.html
"""
from __future__ import annotations

import chess
import chess.svg


# ── Colour themes ──────────────────────────────────────────────────────────
THEMES = {
    "lichess": {
        "square light":           "#f0d9b5",
        "square dark":            "#b58863",
        "square light lastmove":  "#cdd16e",
        "square dark lastmove":   "#aaa23a",
    },
    "chess.com": {
        "square light":           "#eeeed2",
        "square dark":            "#769656",
        "square light lastmove":  "#f6f669",
        "square dark lastmove":   "#baca44",
    },
    "blue": {
        "square light":           "#dee3e6",
        "square dark":            "#8ca2ad",
        "square light lastmove":  "#c4d4da",
        "square dark lastmove":   "#6f8fa0",
    },
}

DEFAULT_THEME = "lichess"


class BoardViewer:
    """
    Wraps chess.svg to render professional, Lichess-style chess boards
    for display in Streamlit via st.components.v1.html().
    """

    @staticmethod
    def board_to_svg(
        board: chess.Board,
        last_move: chess.Move | None = None,
        orientation: str = "white",
        size: int = 400,
        theme: str = DEFAULT_THEME,
        arrows: list | None = None,
    ) -> str:
        """
        Render the board as an SVG string using python-chess's built-in renderer.

        Args:
            board:       chess.Board to render.
            last_move:   Move to highlight (from/to squares highlighted).
            orientation: 'white' or 'black' – which side faces the viewer.
            size:        Pixel size of the square board (default 400).
            theme:       One of 'lichess', 'chess.com', 'blue'.
            arrows:      Optional list of chess.svg.Arrow objects.

        Returns:
            SVG string ready to embed in HTML.
        """
        colors = THEMES.get(theme, THEMES[DEFAULT_THEME])
        orient = chess.WHITE if orientation == "white" else chess.BLACK

        svg = chess.svg.board(
            board,
            orientation=orient,
            lastmove=last_move,
            size=size,
            colors=colors,
            arrows=arrows or [],
            coordinates=True,
        )
        return svg

    @staticmethod
    def board_to_text(board: chess.Board) -> str:
        """Return a plain ASCII representation of the board."""
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
    def wrap_svg(
        svg: str,
        width: int = 420,
        height: int = 440,
        bg: str = "#f8fafc",
    ) -> str:
        """
        Wrap an SVG string in a minimal HTML page suitable for
        st.components.v1.html().

        Args:
            svg:    The SVG string from board_to_svg().
            width:  iframe width in pixels.
            height: iframe height in pixels.
            bg:     Page background colour (should match app theme).
        """
        return (
            f"<!DOCTYPE html><html><head>"
            f"<style>body{{margin:0;background:{bg};"
            f"display:flex;justify-content:center;align-items:center;height:100vh;}}</style>"
            f"</head><body>{svg}</body></html>"
        )

    @classmethod
    def get_html_viewer(
        cls,
        board: chess.Board,
        move_history: list[str],
        last_move: chess.Move | None = None,
        title: str = "Chessboard",
        orientation: str = "white",
        theme: str = DEFAULT_THEME,
    ) -> str:
        """
        Generate a complete, self-contained HTML page showing the board
        on the left and the move list on the right.

        Args:
            board:        Current chess.Board.
            move_history: List of SAN move strings (displayed as a move list).
            last_move:    Move to highlight.
            title:        Page / section title.
            orientation:  'white' or 'black'.
            theme:        Board colour theme.

        Returns:
            Full HTML string for st.components.v1.html().
        """
        board_svg = cls.board_to_svg(board, last_move, orientation=orientation,
                                     size=380, theme=theme)

        # Build move list HTML
        move_rows = ""
        for i in range(0, len(move_history), 2):
            move_num = i // 2 + 1
            white_san = move_history[i] if i < len(move_history) else ""
            black_san = move_history[i + 1] if i + 1 < len(move_history) else ""
            move_rows += (
                f"<tr>"
                f"<td class='num'>{move_num}.</td>"
                f"<td class='mv'>{white_san}</td>"
                f"<td class='mv'>{black_san}</td>"
                f"</tr>"
            )

        turn_label = "White to move" if board.turn == chess.WHITE else "Black to move"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f8fafc;
    color: #1e293b;
    padding: 12px;
  }}
  .layout {{
    display: flex;
    gap: 20px;
    align-items: flex-start;
    flex-wrap: wrap;
  }}
  .board-wrap {{ flex: 0 0 auto; }}
  .board-title {{
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 6px;
  }}
  .turn-badge {{
    display: inline-block;
    margin-top: 8px;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: #e2e8f0;
    color: #334155;
  }}
  .panel {{
    flex: 1;
    min-width: 180px;
  }}
  .panel h3 {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #64748b;
    margin-bottom: 8px;
  }}
  .movelist {{
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    max-height: 380px;
    overflow-y: auto;
    padding: 4px 0;
  }}
  .movelist table {{ width: 100%; border-collapse: collapse; }}
  .movelist td {{ padding: 5px 10px; font-size: 13px; }}
  .movelist tr:nth-child(even) {{ background: #f8fafc; }}
  .movelist td.num {{ color: #94a3b8; width: 36px; font-weight: 600; }}
  .movelist td.mv  {{ font-weight: 500; color: #1e293b; }}
  .fen-box {{
    margin-top: 12px;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 12px;
  }}
  .fen-box h3 {{ font-size: 11px; text-transform: uppercase; letter-spacing:.05em; color:#64748b; margin-bottom:4px; }}
  .fen-val {{ font-family: monospace; font-size: 11px; color: #1e293b; word-break: break-all; }}
</style>
</head>
<body>
  <div class="layout">
    <div class="board-wrap">
      <div class="board-title">{title}</div>
      {board_svg}
      <div class="turn-badge">{turn_label}</div>
    </div>
    <div class="panel">
      <h3>Move History</h3>
      <div class="movelist">
        <table>{move_rows or '<tr><td colspan="3" style="color:#94a3b8;padding:12px;">No moves yet</td></tr>'}</table>
      </div>
      <div class="fen-box">
        <h3>FEN</h3>
        <div class="fen-val">{board.fen()}</div>
      </div>
    </div>
  </div>
</body>
</html>"""
        return html
