"""
Enhanced Streamlit app - Chess Move Tracker with Analysis Features.

Features:
- Interactive board viewer with move navigator (step through every move)
- Board flip (White / Black perspective)
- ASCII board fallback view
- Stockfish eval bar & best-move / PV display for each position
- Eval-over-time chart
- Opening recognition with phase detection and ECO info
- Per-player accuracy comparison (radar / bar charts)
- Move quality distribution pie chart
- Full game narrative
- Download CSV / JSON / PGN / Summary JSON

Run with:
    streamlit run app_enhanced.py
"""
from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import chess
import chess.pgn

from chess_tracker.config import (
    DEFAULT_BOARD_SIZE,
    DEFAULT_PIECE_CONF,
    DEFAULT_PIECE_MODEL_URL,
    DEFAULT_POSE_CONF,
    DEFAULT_POSE_MODEL_URL,
    DEFAULT_STABILITY_SECONDS,
)
from chess_tracker.pipeline import ChessVideoTracker, MoveRecord
from chess_tracker.board_viewer import BoardViewer
from chess_tracker.game_summary import GameAnalyzer, GameSummary
from chess_tracker.opening_recognition import OpeningRecognizer
from chess_tracker.stockfish_analyzer import StockfishAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chess Move Tracker Pro",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Quality badge colours */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 13px;
        color: #fff;
    }
    .badge-brilliant  { background: #059669; }
    .badge-excellent  { background: #2563eb; }
    .badge-good       { background: #d97706; }
    .badge-inaccuracy { background: #dc2626; }
    .badge-mistake    { background: #991b1b; }
    .badge-blunder    { background: #450a0a; }
    .badge-unknown    { background: #6b7280; }

    /* Eval bar */
    .eval-bar-wrap { display:flex; align-items:center; gap:10px; margin:8px 0; }
    .eval-bar { flex:1; height:22px; border-radius:6px; overflow:hidden; border:1px solid #d1d5db; }
    .eval-white { background:#f0d9b5; display:inline-block; height:100%; }
    .eval-black { background:#312e2b; display:inline-block; height:100%; }
    .eval-label { font-weight:700; font-size:14px; min-width:60px; text-align:center; }

    /* Narrative card */
    .narrative-card {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    /* Phase pill */
    .phase-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    .phase-opening    { background:#dbeafe; color:#1e40af; }
    .phase-middlegame { background:#fef3c7; color:#92400e; }
    .phase-endgame    { background:#f3e8ff; color:#6b21a8; }

    /* Player stat rows */
    .stat-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #f1f5f9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
QUALITY_BADGE = {
    "brilliant":  '<span class="badge badge-brilliant">⭐ Brilliant</span>',
    "excellent":  '<span class="badge badge-excellent">✨ Excellent</span>',
    "good":       '<span class="badge badge-good">👍 Good</span>',
    "inaccuracy": '<span class="badge badge-inaccuracy">⚠️ Inaccuracy</span>',
    "mistake":    '<span class="badge badge-mistake">❌ Mistake</span>',
    "blunder":    '<span class="badge badge-blunder">💣 Blunder</span>',
}

QUALITY_COLORS = {
    "Brilliant":  "#059669",
    "Excellent":  "#2563eb",
    "Good":       "#d97706",
    "Inaccuracy": "#dc2626",
    "Mistake":    "#991b1b",
    "Blunder":    "#450a0a",
}

PHASE_HTML = {
    "opening":    '<span class="phase-pill phase-opening">Opening</span>',
    "middlegame": '<span class="phase-pill phase-middlegame">Middlegame</span>',
    "endgame":    '<span class="phase-pill phase-endgame">Endgame</span>',
}


def eval_bar_html(eval_pawns: float | None, mate: int | None) -> str:
    """Render a visual evaluation bar."""
    if eval_pawns is None and mate is None:
        return ""
    if mate is not None:
        if mate > 0:
            label = f"M{mate}"
            white_pct = 95
        else:
            label = f"-M{abs(mate)}"
            white_pct = 5
    else:
        clamped = max(-5.0, min(5.0, eval_pawns))
        white_pct = int((clamped + 5) / 10 * 100)
        sign = "+" if eval_pawns > 0 else ""
        label = f"{sign}{eval_pawns:.2f}"

    return (
        f'<div class="eval-bar-wrap">'
        f'<span class="eval-label">{label}</span>'
        f'<div class="eval-bar">'
        f'<span class="eval-white" style="width:{white_pct}%"></span>'
        f'<span class="eval-black" style="width:{100-white_pct}%"></span>'
        f'</div></div>'
    )


def pv_to_san(board_before: chess.Board, pv_uci: list[str]) -> list[str]:
    """Convert a list of UCI PV moves to SAN notation."""
    b = board_before.copy()
    san_list = []
    for uci in pv_uci:
        try:
            move = chess.Move.from_uci(uci)
            if move in b.legal_moves:
                san_list.append(b.san(move))
                b.push(move)
            else:
                break
        except Exception:
            break
    return san_list


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("♟️ Chess Move Tracker")
st.sidebar.caption("YOLO11 + Stockfish + Interactive Analysis")

with st.sidebar.expander("Model sources", expanded=False):
    pose_model_path = st.text_input(
        "Board pose model (local path)",
        value=DEFAULT_POSE_MODEL_URL,
    )
    piece_model_path = st.text_input(
        "Piece detection model (local path)",
        value=DEFAULT_PIECE_MODEL_URL,
    )

with st.sidebar.expander("Detection settings", expanded=False):
    pose_conf = st.slider("Board corner confidence", 0.1, 0.95, DEFAULT_POSE_CONF, 0.05)
    piece_conf = st.slider("Piece detection confidence", 0.1, 0.95, DEFAULT_PIECE_CONF, 0.05)
    stability_seconds = st.slider(
        "Move stability window (seconds)",
        0.3, 4.0, DEFAULT_STABILITY_SECONDS, 0.1,
        help="How long a board state must stay unchanged before a move is confirmed.",
    )
    frame_stride = st.slider(
        "Process every Nth frame", 1, 10, 1,
        help="Increase to speed up processing on long videos.",
    )
    show_live_preview = st.checkbox("Show live detection preview", value=True)

with st.sidebar.expander("Analysis settings", expanded=False):
    enable_stockfish = st.checkbox(
        "Enable Stockfish analysis", value=False,
        help="Requires Stockfish binary (already installed on this server)",
    )
    stockfish_depth = st.slider(
        "Stockfish search depth", 5, 25, 15,
        help="Higher = slower but more accurate. Depth 10–15 is fast for most games.",
    )
    board_orientation = st.radio("Board orientation", ["White", "Black"], index=0)
    board_theme = st.selectbox(
        "Board colour theme",
        ["lichess", "chess.com", "blue"],
        index=0,
        help="Lichess = brown/cream  |  Chess.com = green/ivory  |  Blue = steel blue",
    )
    show_ascii_board = st.checkbox("Also show ASCII board", value=False)

st.sidebar.divider()
uploaded_video = st.sidebar.file_uploader(
    "Upload a chess game video", type=["mp4", "mov", "avi", "mkv"]
)
run_button = st.sidebar.button(
    "▶️ Run tracking", type="primary", disabled=uploaded_video is None
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO11 models…")
def load_tracker(pose_path, piece_path, board_size, pose_c, piece_c, stability):
    return ChessVideoTracker(
        pose_model_path=pose_path,
        piece_model_path=piece_path,
        board_size=board_size,
        pose_conf=pose_c,
        piece_conf=piece_c,
        stability_seconds=stability,
    )


@st.cache_resource(show_spinner="Initialising Stockfish…")
def load_stockfish(depth: int):
    try:
        sf = StockfishAnalyzer(depth=depth)
        if sf.stockfish_path:
            return sf
        st.sidebar.warning("⚠️ Stockfish binary not found. Analysis disabled.")
        return None
    except Exception as e:
        st.sidebar.warning(f"⚠️ Stockfish error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
for key in ("result", "analysis", "navigator_idx"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "navigator_idx" else 0

# ─────────────────────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────────────────────
st.title("♟️ Chess Move Tracker Pro")
st.write(
    "Upload a chess game video for AI-powered analysis: move detection, "
    "Stockfish evaluation, opening recognition, and comprehensive statistics."
)

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_video is not None and run_button:
    st.session_state.navigator_idx = 0
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.read())
        tmp.flush()
        video_path = tmp.name

    # ── Step 1: Load models from local disk ─────────────────────────────────
    model_status = st.status("⚙️ Loading AI models…", expanded=False)
    with model_status:
        try:
            tracker = load_tracker(
                pose_model_path, piece_model_path,
                DEFAULT_BOARD_SIZE, pose_conf, piece_conf, stability_seconds,
            )
            model_status.update(label="✅ Models loaded", state="complete", expanded=False)
        except Exception as model_err:
            model_status.update(label="❌ Model loading failed", state="error")
            st.error(
                f"**Could not load YOLO models.**\n\n"
                f"```\n{model_err}\n```\n\n"
                f"Expected model files:\n"
                f"- `{pose_model_path}`\n"
                f"- `{piece_model_path}`\n\n"
                "Make sure these files exist in the `models/` directory."
            )
            Path(video_path).unlink(missing_ok=True)
            st.stop()

    # ── Step 2: Live processing UI ───────────────────────────────────────────
    progress_bar = st.progress(0.0, text="Starting…")

    # Three-column live layout: camera feed | live chess board | move list
    col_preview, col_live_board, col_moves = st.columns([5, 4, 3])
    with col_preview:
        st.caption("📹 Camera detection")
        preview_slot = st.empty()
    with col_live_board:
        st.caption("♟️ Live board")
        live_board_slot = st.empty()
        live_move_label_slot = st.empty()
    with col_moves:
        st.caption("📋 Moves (live)")
        moves_slot = st.empty()

    live_moves: list[MoveRecord] = []
    live_board_state = {"board": chess.Board(), "last_move": None}
    ui_state = {"last_update": 0.0, "board_update": 0.0}

    def _render_live_board(board: chess.Board, last_move, label: str = ""):
        """Render board as inline SVG into the live_board_slot via markdown."""
        svg = BoardViewer.board_to_svg(
            board, last_move,
            orientation=board_orientation.lower(),
            size=320,
            theme=board_theme,
        )
        # Inline SVG in a centred div — works reliably in Streamlit callbacks
        live_board_slot.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:center;">'
            f'{svg}'
            f'<div style="font-size:12px;color:#64748b;margin-top:6px;">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Show starting position immediately
    _render_live_board(live_board_state["board"], None, "Starting position")

    def on_progress(frame_idx, total_frames):
        if total_frames > 0:
            progress_bar.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Frame {frame_idx} / {total_frames}",
            )

    def on_frame(annotated_bgr, frame_idx, total_frames):
        if not show_live_preview or annotated_bgr is None:
            return
        if time.time() - ui_state["last_update"] < 0.15:
            return
        ui_state["last_update"] = time.time()
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        preview_slot.image(
            rgb, caption=f"Frame {frame_idx}/{total_frames}", use_container_width=True
        )

    def on_move(record: MoveRecord):
        live_moves.append(record)

        # Update live chess board position
        try:
            uci = f"{record.from_square}{record.to_square}"
            m = chess.Move.from_uci(uci)
            b = live_board_state["board"]
            if m in b.legal_moves:
                b.push(m)
                live_board_state["last_move"] = m

                now = time.time()
                # Throttle renders to ~4 fps — keeps UI responsive
                if now - ui_state["board_update"] >= 0.25:
                    ui_state["board_update"] = now
                    move_num = (record.ply + 1) // 2
                    color_str = "White" if record.color == "w" else "Black"
                    label = f"Move {move_num} — {color_str}: {record.san}"
                    _render_live_board(b, m, label)
        except Exception:
            pass

        # Update moves table
        df = pd.DataFrame([
            {"#": mv.ply, "Color": "⬜" if mv.color == "w" else "⬛", "Move": mv.san}
            for mv in live_moves
        ])
        moves_slot.dataframe(df, use_container_width=True, hide_index=True, height=400)

    # ── Step 3: Run tracking ─────────────────────────────────────────────────
    try:
        with st.spinner("🎥 Analysing video…"):
            result = tracker.process_video(
                video_path,
                progress_callback=on_progress,
                frame_callback=on_frame if show_live_preview else None,
                move_callback=on_move,
                frame_stride=frame_stride,
            )
    except Exception as proc_err:
        st.error(
            f"**Video processing failed.**\n\n"
            f"```\n{proc_err}\n```\n\n"
            "Make sure the video shows a clearly lit, unobstructed chess board from above. "
            "Try increasing the frame stride or lowering the confidence thresholds in Detection settings."
        )
        Path(video_path).unlink(missing_ok=True)
        st.stop()
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except (PermissionError, FileNotFoundError):
            pass

    progress_bar.progress(1.0, text="✅ Done!")
    st.session_state.result = result

    # --- Build game object ---
    game = chess.pgn.read_game(io.StringIO(result["pgn"])) or chess.pgn.Game()
    game.headers.setdefault("Event", "Analyzed Game")
    game.headers.setdefault("White", "White Player")
    game.headers.setdefault("Black", "Black Player")

    # Always build a base summary (no Stockfish)
    summary = GameAnalyzer().analyze_game(game)
    opening_transitions = OpeningRecognizer.analyze_game(game)
    eco_description = OpeningRecognizer.get_eco_description(summary.eco_code)

    st.session_state.analysis = {
        "summary": summary,
        "move_analyses": [],
        "opening_transitions": opening_transitions,
        "eco_description": eco_description,
    }

    st.success(f"✅ Tracking complete — **{len(result['moves'])}** move(s) detected.")

    # --- Stockfish analysis ---
    if enable_stockfish and result["moves"]:
        stockfish = load_stockfish(stockfish_depth)
        if stockfish and stockfish.stockfish_path:
            with st.status("🔍 Running Stockfish analysis…", expanded=True) as sf_status:
                # Replay the game move by move and call analyze_game()
                sf_game = chess.pgn.Game()
                sf_game.headers["Event"] = "Analyzed Game"
                board = chess.Board()
                node = sf_game
                for move_rec in result["moves"]:
                    try:
                        uci = f"{move_rec.from_square}{move_rec.to_square}"
                        m = chess.Move.from_uci(uci)
                        if m in board.legal_moves:
                            node = node.add_variation(m)
                            board.push(m)
                    except Exception:
                        pass

                # Use StockfishAnalyzer.analyze_game() directly
                move_analyses = stockfish.analyze_game(sf_game)
                sf_status.update(label="✅ Stockfish analysis complete!", state="complete")

            # Rebuild summary with Stockfish quality data
            summary = GameAnalyzer().analyze_game(sf_game, move_analyses)
            opening_transitions = OpeningRecognizer.analyze_game(sf_game)
            eco_description = OpeningRecognizer.get_eco_description(summary.eco_code)

            st.session_state.analysis = {
                "summary": summary,
                "move_analyses": move_analyses,
                "opening_transitions": opening_transitions,
                "eco_description": eco_description,
            }


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.result
analysis = st.session_state.analysis

if result is None:
    st.info("Upload a video and click **Run tracking** in the sidebar to get started.")
    st.stop()

moves = result["moves"]
orientation = board_orientation.lower()
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab_board, tab_analysis, tab_stockfish, tab_opening, tab_log, tab_export = st.tabs([
    "♟️ Board Navigator",
    "📊 Game Analysis",
    "🤖 Stockfish",
    "📖 Opening & Phases",
    "📋 Move Log",
    "📥 Export",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — INTERACTIVE BOARD NAVIGATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_board:
    st.subheader("♟️ Interactive Board Navigator")
    st.caption("Step through every move of the game. Board, position info, and Stockfish eval update automatically.")

    # Replay all legal moves and store board snapshots
    snapshots: list[dict] = []
    board = chess.Board()
    snapshots.append({"board": board.copy(), "move": None, "san": "Start", "ply": 0, "fen": board.fen()})
    for move_rec in moves:
        try:
            uci = f"{move_rec.from_square}{move_rec.to_square}"
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                snap_move = m
                san = board.san(m)
                board.push(m)
                snapshots.append({"board": board.copy(), "move": snap_move, "san": san,
                                   "ply": move_rec.ply, "fen": board.fen()})
        except Exception:
            pass

    total_snaps = len(snapshots)

    if total_snaps <= 1:
        st.warning("No moves were detected in this video to navigate.")
    else:
        # Navigator controls
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 3, 1, 1])
        with nav_col1:
            if st.button("⏮ Start", use_container_width=True):
                st.session_state.navigator_idx = 0
        with nav_col2:
            if st.button("◀ Prev", use_container_width=True) and st.session_state.navigator_idx > 0:
                st.session_state.navigator_idx -= 1
        with nav_col3:
            new_idx = st.slider(
                "Position", 0, total_snaps - 1,
                st.session_state.navigator_idx,
                label_visibility="collapsed",
            )
            if new_idx != st.session_state.navigator_idx:
                st.session_state.navigator_idx = new_idx
        with nav_col4:
            if st.button("Next ▶", use_container_width=True) and st.session_state.navigator_idx < total_snaps - 1:
                st.session_state.navigator_idx += 1
        with nav_col5:
            if st.button("End ⏭", use_container_width=True):
                st.session_state.navigator_idx = total_snaps - 1

        idx = st.session_state.navigator_idx
        snap = snapshots[idx]
        curr_board = snap["board"]
        curr_move = snap["move"]

        # Get opening info for current position
        recognizer = OpeningRecognizer()
        temp_board = chess.Board()
        for s in snapshots[1: idx + 1]:
            if s["move"] and s["move"] in temp_board.legal_moves:
                recognizer.update(s["move"], temp_board)
                temp_board.push(s["move"])
        opening_info = recognizer.get_opening_info(curr_board)

        # Phase pill
        phase = opening_info.get("phase", "opening")
        phase_badge = PHASE_HTML.get(phase, "")

        board_col, info_col = st.columns([3, 2])

        with board_col:
            # Move label
            if idx == 0:
                move_label = "Starting position"
            else:
                move_num = (snap["ply"] + 1) // 2
                color_str = "White" if snap["ply"] % 2 == 1 else "Black"
                move_label = f"Move {move_num} — {color_str}: **{snap['san']}**"

            st.markdown(
                f"{move_label} &nbsp; {phase_badge} &nbsp; "
                f"<span style='color:#6b7280;font-size:13px'>{idx}/{total_snaps-1}</span>",
                unsafe_allow_html=True,
            )

            svg = BoardViewer.board_to_svg(
                curr_board, curr_move,
                orientation=orientation,
                size=420,
                theme=board_theme,
            )
            st.components.v1.html(
                BoardViewer.wrap_svg(svg, width=460, height=460, bg="#f8fafc"),
                height=460,
            )

            if show_ascii_board:
                with st.expander("ASCII board"):
                    st.code(BoardViewer.board_to_text(curr_board), language=None)

        with info_col:
            # Opening info
            opening_name = opening_info.get("name", "Unknown")
            eco = opening_info.get("eco", "?")
            st.markdown(f"**Opening:** {opening_name}")
            st.markdown(f"**ECO:** `{eco}` — {OpeningRecognizer.get_eco_description(eco)}")
            st.markdown(f"**FEN:**")
            st.code(snap["fen"], language=None)

            # Stockfish eval for this position (if analysis available)
            move_analyses = analysis["move_analyses"] if analysis else []
            if move_analyses and idx > 0 and idx - 1 < len(move_analyses):
                ma = move_analyses[idx - 1]
                eval_val = ma.get("eval_after")
                mate_val = ma.get("mate_threat")
                quality = ma.get("quality", "unknown")
                best_move_uci = ma.get("best_move", "")

                st.markdown("#### 🤖 Stockfish Eval")
                st.markdown(eval_bar_html(eval_val, mate_val), unsafe_allow_html=True)

                badge = QUALITY_BADGE.get(quality, f"<span class='badge badge-unknown'>{quality}</span>")
                st.markdown(f"**Move quality:** {badge}", unsafe_allow_html=True)
                st.markdown(f"**Eval before:** {ma.get('eval_before', 0):+.2f} &nbsp; "
                            f"**After:** {ma.get('eval_after', 0):+.2f} &nbsp; "
                            f"**Δ:** {ma.get('delta', 0):+.2f}",
                            unsafe_allow_html=True)
            elif move_analyses and idx == 0:
                st.info("Stockfish eval appears after the first move.")
            elif not move_analyses:
                st.info("Enable Stockfish in Analysis settings and re-run for eval data.")

            # Turn indicator
            turn_str = "⬜ White to move" if curr_board.turn == chess.WHITE else "⬛ Black to move"
            st.markdown(f"**{turn_str}**")

            # Check / Checkmate / Stalemate status
            if curr_board.is_checkmate():
                winner = "Black" if curr_board.turn == chess.WHITE else "White"
                st.error(f"♟️ Checkmate! {winner} wins.")
            elif curr_board.is_stalemate():
                st.warning("⚖️ Stalemate.")
            elif curr_board.is_check():
                st.warning("⚡ Check!")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GAME ANALYSIS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_analysis:
    st.subheader("📊 Game Analysis Dashboard")

    if analysis is None:
        st.info("Run tracking first to see analysis.")
        st.stop()

    summary: GameSummary = analysis["summary"]
    move_analyses = analysis["move_analyses"]

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Moves", summary.total_moves)
    c2.metric("Result", summary.result)
    c3.metric("Opening", f"{summary.eco_code}", summary.opening_name[:20])
    c4.metric("Phase reached", summary.game_length_phase.title())
    c5.metric("Opening moves", summary.opening_moves)

    c1, c2, c3 = st.columns(3)
    c1.metric("Opening phase", f"{summary.opening_moves} moves")
    c2.metric("Middlegame phase", f"{summary.middlegame_moves} moves")
    c3.metric("Endgame phase", f"{summary.endgame_moves} moves")

    st.divider()

    # ── Phase breakdown bar ───────────────────────────────────────────────────
    fig_phase = px.bar(
        pd.DataFrame({
            "Phase": ["Opening", "Middlegame", "Endgame"],
            "Moves": [summary.opening_moves, summary.middlegame_moves, summary.endgame_moves],
            "Color": ["#3b82f6", "#f59e0b", "#8b5cf6"],
        }),
        x="Phase", y="Moves", color="Phase",
        color_discrete_map={"Opening": "#3b82f6", "Middlegame": "#f59e0b", "Endgame": "#8b5cf6"},
        title="Game Breakdown by Phase",
        text_auto=True,
    )
    fig_phase.update_layout(showlegend=False)
    st.plotly_chart(fig_phase, use_container_width=True)

    st.divider()

    # ── Player comparison ────────────────────────────────────────────────────
    st.subheader("👥 Player Comparison")

    pw = summary.white
    pb = summary.black

    col_w, col_b = st.columns(2)

    def render_player_card(player, color_icon):
        rows = [
            ("Moves played", player.total_moves),
            ("Accuracy", f"{player.average_accuracy:.1f}%"),
            ("Best-move %", f"{player.best_move_percentage:.1f}%"),
            ("Accurate moves", player.accurate_moves),
            ("⭐ Brilliant", player.brilliant_moves),
            ("⚠️ Inaccuracies", player.inaccuracies),
            ("❌ Mistakes", player.mistakes),
            ("💣 Blunders", player.blunders),
        ]
        html = ""
        for label, val in rows:
            html += (
                f'<div class="stat-row">'
                f'<span style="color:#6b7280">{label}</span>'
                f'<strong>{val}</strong>'
                f'</div>'
            )
        return html

    with col_w:
        st.markdown(f"### ⬜ White — {pw.name}")
        st.markdown(render_player_card(pw, "⬜"), unsafe_allow_html=True)

    with col_b:
        st.markdown(f"### ⬛ Black — {pb.name}")
        st.markdown(render_player_card(pb, "⬛"), unsafe_allow_html=True)

    st.divider()

    # ── Side-by-side comparison bar chart ────────────────────────────────────
    compare_df = pd.DataFrame([
        {"Category": "Accurate", "White": pw.accurate_moves, "Black": pb.accurate_moves},
        {"Category": "Brilliant", "White": pw.brilliant_moves, "Black": pb.brilliant_moves},
        {"Category": "Inaccuracies", "White": pw.inaccuracies, "Black": pb.inaccuracies},
        {"Category": "Mistakes", "White": pw.mistakes, "Black": pb.mistakes},
        {"Category": "Blunders", "White": pw.blunders, "Black": pb.blunders},
    ])
    fig_compare = go.Figure(data=[
        go.Bar(name="White", x=compare_df["Category"], y=compare_df["White"],
               marker_color="#f0d9b5", marker_line_color="#888", marker_line_width=1),
        go.Bar(name="Black", x=compare_df["Category"], y=compare_df["Black"],
               marker_color="#312e2b"),
    ])
    fig_compare.update_layout(
        barmode="group", title="White vs Black — Move Quality Counts",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Accuracy radar ─────────────────────────────────────────────────────
    categories = ["Accuracy %", "Best Move %", "Brilliant", "Low Inaccuracies", "Low Blunders"]
    white_vals = [
        min(pw.average_accuracy, 100),
        min(pw.best_move_percentage, 100),
        min(pw.brilliant_moves * 10, 100),
        max(0, 100 - pw.inaccuracies * 10),
        max(0, 100 - pw.blunders * 20),
    ]
    black_vals = [
        min(pb.average_accuracy, 100),
        min(pb.best_move_percentage, 100),
        min(pb.brilliant_moves * 10, 100),
        max(0, 100 - pb.inaccuracies * 10),
        max(0, 100 - pb.blunders * 20),
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=white_vals + [white_vals[0]], theta=categories + [categories[0]],
        fill="toself", name="White", line_color="#d97706",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=black_vals + [black_vals[0]], theta=categories + [categories[0]],
        fill="toself", name="Black", line_color="#2563eb", opacity=0.7,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Player Strength Radar",
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Move quality pie ───────────────────────────────────────────────────
    if move_analyses:
        st.divider()
        st.subheader("🎯 Move Quality Distribution")
        quality_counts = {"brilliant": 0, "excellent": 0, "good": 0,
                          "inaccuracy": 0, "mistake": 0, "blunder": 0}
        for ma in move_analyses:
            q = ma.get("quality", "unknown")
            if q in quality_counts:
                quality_counts[q] += 1

        q_df = pd.DataFrame([
            {"Quality": k.title(), "Count": v}
            for k, v in quality_counts.items() if v > 0
        ])
        if not q_df.empty:
            fig_pie = px.pie(
                q_df, values="Count", names="Quality",
                color="Quality", color_discrete_map=QUALITY_COLORS,
                title="All Moves — Quality Distribution",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Narrative ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Game Narrative")
    narrative = GameAnalyzer.get_narrative(summary)
    st.markdown(f'<div class="narrative-card">{narrative}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — STOCKFISH DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab_stockfish:
    st.subheader("🤖 Stockfish Analysis")

    move_analyses = analysis["move_analyses"] if analysis else []

    if not move_analyses:
        st.info(
            "No Stockfish data yet. Enable **Stockfish analysis** in the sidebar "
            "settings and click **Run tracking** again."
        )
    else:
        # ── Eval over time chart ───────────────────────────────────────────
        st.markdown("#### 📈 Evaluation Over Time")
        eval_data = []
        board = chess.Board()
        for i, (move_rec, ma) in enumerate(zip(moves, move_analyses)):
            eval_val = ma.get("eval_after", 0) or 0
            mate = ma.get("mate_threat")
            if mate is not None:
                eval_val = 10.0 if mate > 0 else -10.0
            eval_data.append({
                "Ply": i + 1,
                "Move": ma.get("san", move_rec.san),
                "Color": "White" if (i % 2 == 0) else "Black",
                "Eval": round(float(eval_val), 3),
                "Quality": ma.get("quality", "?"),
            })

        eval_df = pd.DataFrame(eval_data)
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=eval_df["Ply"], y=eval_df["Eval"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=2),
            marker=dict(
                color=[
                    {"brilliant": "#059669", "excellent": "#2563eb",
                     "good": "#d97706", "inaccuracy": "#dc2626",
                     "mistake": "#991b1b", "blunder": "#450a0a"}.get(q, "#9ca3af")
                    for q in eval_df["Quality"]
                ],
                size=8,
            ),
            hovertemplate="<b>%{customdata}</b><br>Eval: %{y:.2f}<extra></extra>",
            customdata=eval_df["Move"],
        ))
        fig_eval.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
        fig_eval.update_layout(
            title="Evaluation After Each Move (positive = White advantage)",
            xaxis_title="Ply",
            yaxis_title="Eval (pawns)",
            yaxis=dict(range=[-6, 6]),
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        st.caption("Marker color = move quality  |  Green=Brilliant  Blue=Excellent  Orange=Good  Red=Inaccuracy/Mistake  Dark=Blunder")

        # ── Move-by-move quality table with PV ────────────────────────────
        st.markdown("#### 🔍 Move-by-Move Breakdown")

        board = chess.Board()
        board_snapshots_for_pv = [board.copy()]
        for move_rec in moves:
            try:
                uci = f"{move_rec.from_square}{move_rec.to_square}"
                m = chess.Move.from_uci(uci)
                if m in board.legal_moves:
                    board.push(m)
                    board_snapshots_for_pv.append(board.copy())
            except Exception:
                board_snapshots_for_pv.append(board.copy())

        for i, ma in enumerate(move_analyses):
            quality = ma.get("quality", "unknown")
            badge = QUALITY_BADGE.get(quality, f"<b>{quality}</b>")
            san = ma.get("san", f"Move {i+1}")
            move_num = i // 2 + 1
            color_str = "White" if i % 2 == 0 else "Black"
            eval_b = ma.get("eval_before", 0) or 0
            eval_a = ma.get("eval_after", 0) or 0
            delta = ma.get("delta", 0) or 0
            best_uci = ma.get("best_move", "")

            # PV in SAN
            pv_uci_list = []  # PV not stored in score_move result; shown via best_move only
            best_san = ""
            if best_uci and i < len(board_snapshots_for_pv):
                try:
                    bm = chess.Move.from_uci(best_uci)
                    snap_before = board_snapshots_for_pv[i]
                    if bm in snap_before.legal_moves:
                        best_san = snap_before.san(bm)
                except Exception:
                    best_san = best_uci

            with st.expander(
                f"{'⬜' if i%2==0 else '⬛'} {move_num}{'.' if i%2==0 else '…'} {san}  —  {quality.title()}",
                expanded=False,
            ):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"**Quality:** {badge}", unsafe_allow_html=True)
                    st.markdown(eval_bar_html(eval_a, ma.get("mate_threat")), unsafe_allow_html=True)
                    st.markdown(
                        f"Before: **{eval_b:+.2f}** &nbsp;→&nbsp; After: **{eval_a:+.2f}** &nbsp; (Δ {delta:+.2f})",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    if best_san:
                        st.markdown(f"**Stockfish best move:** `{best_san}` (UCI: `{best_uci}`)")
                    if ma.get("mate_threat") is not None:
                        st.warning(f"⚠️ Mate in {abs(ma['mate_threat'])} {'for the side to move' if ma['mate_threat'] > 0 else 'against'}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — OPENING & PHASES
# ─────────────────────────────────────────────────────────────────────────────
with tab_opening:
    st.subheader("📖 Opening Recognition & Phase Analysis")

    if analysis is None:
        st.info("Run tracking first.")
        st.stop()

    summary = analysis["summary"]
    opening_transitions = analysis.get("opening_transitions", [])
    eco_description = analysis.get("eco_description", "Unknown ECO")

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Opening", summary.opening_name)
    col2.metric("ECO Code", summary.eco_code)
    col3.metric("ECO Family", eco_description)

    # Phase metrics
    st.divider()
    st.markdown("#### 🔄 Game Phase Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("Opening Phase", f"{summary.opening_moves} moves", "Moves 1–20")
    c2.metric("Middlegame Phase", f"{summary.middlegame_moves} moves", "Moves 21–40")
    c3.metric("Endgame Phase", f"{summary.endgame_moves} moves", "Moves 41+")

    fig_donut = go.Figure(go.Pie(
        labels=["Opening", "Middlegame", "Endgame"],
        values=[summary.opening_moves, summary.middlegame_moves, summary.endgame_moves],
        hole=0.5,
        marker_colors=["#3b82f6", "#f59e0b", "#8b5cf6"],
    ))
    fig_donut.update_layout(title="Phase Distribution", showlegend=True)
    st.plotly_chart(fig_donut, use_container_width=True)

    # Opening transitions table
    st.divider()
    st.markdown("#### 🧭 Opening Recognition Trace")
    st.caption("Shows how opening classification evolved as moves were played.")

    if opening_transitions:
        distinct = []
        last_name = None
        for info in opening_transitions:
            name = info.get("opening", "?")
            if name != last_name:
                distinct.append({
                    "Ply": info["ply"],
                    "Move Number": (info["ply"] + 1) // 2,
                    "Side": "White" if info["ply"] % 2 == 1 else "Black",
                    "Opening": name,
                    "ECO": info.get("eco", "?"),
                })
                last_name = name

        if distinct:
            st.dataframe(pd.DataFrame(distinct), use_container_width=True, hide_index=True)
        else:
            st.info("No opening transitions detected.")
    else:
        st.info("Opening recognition did not find a match for this game.")

    # ECO range reference
    st.divider()
    with st.expander("📚 ECO Code Reference Chart", expanded=False):
        st.caption("Standard ECO (Encyclopaedia of Chess Openings) code ranges and their families.")
        eco_df = pd.DataFrame([
            {"ECO Range": k, "Opening Family": v}
            for k, v in OpeningRecognizer.ECO_RANGES.items()
        ])
        st.dataframe(eco_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — DETAILED MOVE LOG
# ─────────────────────────────────────────────────────────────────────────────
with tab_log:
    st.subheader("📋 Detailed Move Log")

    if not moves:
        st.info("No moves were detected in this video.")
    else:
        move_analyses = analysis["move_analyses"] if analysis else []

        move_data = []
        for i, m in enumerate(moves):
            row = {
                "#": m.ply,
                "Color": "⬜ White" if m.color == "w" else "⬛ Black",
                "Move": m.san,
                "From": m.from_square.upper(),
                "To": m.to_square.upper(),
                "FEN (short)": m.fen[:40] + "…" if len(m.fen) > 40 else m.fen,
            }
            if i < len(move_analyses):
                ma = move_analyses[i]
                row["Quality"] = ma.get("quality", "?").title()
                row["Eval Before"] = f"{ma.get('eval_before', 0):+.2f}"
                row["Eval After"] = f"{ma.get('eval_after', 0):+.2f}"
                row["Delta"] = f"{ma.get('delta', 0):+.2f}"
                row["Best Move"] = ma.get("best_move", "")
            move_data.append(row)

        df = pd.DataFrame(move_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=500)

        # PGN viewer
        st.divider()
        st.markdown("#### 📄 PGN Notation")
        st.code(result["pgn"], language=None)

        st.markdown("#### 🎯 Final Position FEN")
        st.code(result["final_fen"], language=None)
        st.link_button(
            "🔗 Analyse on Lichess",
            f"https://lichess.org/editor/{result['final_fen'].replace(' ', '_')}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.subheader("📥 Export Game Data")

    summary = analysis["summary"] if analysis else None
    move_analyses = analysis["move_analyses"] if analysis else []

    st.markdown("#### Downloads")
    col1, col2 = st.columns(2)

    with col1:
        # PGN
        st.download_button(
            "♟️ Download PGN",
            result["pgn"].encode("utf-8"),
            "game.pgn",
            "application/x-chess-pgn",
            use_container_width=True,
        )

    with col2:
        # Summary JSON
        if summary:
            st.download_button(
                "📊 Download Summary JSON",
                json.dumps(summary.to_dict(), indent=2),
                "game_summary.json",
                "application/json",
                use_container_width=True,
            )

    if moves:
        move_data = []
        for i, m in enumerate(moves):
            row = {
                "ply": m.ply,
                "color": "White" if m.color == "w" else "Black",
                "san": m.san,
                "from": m.from_square,
                "to": m.to_square,
                "fen": m.fen,
            }
            if i < len(move_analyses):
                ma = move_analyses[i]
                row["quality"] = ma.get("quality", "")
                row["eval_before"] = ma.get("eval_before", "")
                row["eval_after"] = ma.get("eval_after", "")
                row["delta"] = ma.get("delta", "")
                row["best_move"] = ma.get("best_move", "")
            move_data.append(row)

        df_export = pd.DataFrame(move_data)
        col3, col4 = st.columns(2)

        with col3:
            st.download_button(
                "📄 Download Moves CSV",
                df_export.to_csv(index=False).encode("utf-8"),
                "moves.csv",
                "text/csv",
                use_container_width=True,
            )
        with col4:
            st.download_button(
                "🗂 Download Moves JSON",
                df_export.to_json(orient="records", indent=2).encode("utf-8"),
                "moves.json",
                "application/json",
                use_container_width=True,
            )

    # FEN link
    st.divider()
    st.markdown("#### 🔗 External Links")
    st.link_button(
        "📝 Edit final position on Lichess",
        f"https://lichess.org/editor/{result['final_fen'].replace(' ', '_')}",
    )
    if result.get("start_fen"):
        st.link_button(
            "🎬 View start position on Lichess",
            f"https://lichess.org/editor/{result['start_fen'].replace(' ', '_')}",
        )
