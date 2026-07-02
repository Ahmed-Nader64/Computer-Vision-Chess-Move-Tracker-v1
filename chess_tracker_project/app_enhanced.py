"""
Enhanced Streamlit app - Chess Move Tracker with Analysis Features.

New Features:
- Interactive board viewer
- Stockfish-powered move analysis
- Move quality scoring (brilliant, good, inaccuracy, blunder)
- Opening recognition
- Game summary dashboard
- Move statistics and player accuracy metrics

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
import chess
import chess.pgn

st.set_page_config(
    page_title="Chess Move Tracker", page_icon="♟️", layout="wide"
)

# CSS for enhanced styling
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .quality-brilliant { color: #2ecc71; font-weight: bold; }
    .quality-excellent { color: #3498db; font-weight: bold; }
    .quality-good { color: #f39c12; font-weight: bold; }
    .quality-inaccuracy { color: #e74c3c; font-weight: bold; }
    .quality-mistake { color: #c0392b; font-weight: bold; }
    .quality-blunder { color: #8b0000; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar - configuration
# --------------------------------------------------------------------------
st.sidebar.title("♟️ Chess Move Tracker")
st.sidebar.caption("YOLO11 + Stockfish + Interactive Analysis")

with st.sidebar.expander("Model sources", expanded=False):
    pose_model_path = st.text_input(
        "Board pose model (URL or local path)",
        value=DEFAULT_POSE_MODEL_URL,
    )
    piece_model_path = st.text_input(
        "Piece detection model (URL or local path)",
        value=DEFAULT_PIECE_MODEL_URL,
    )

with st.sidebar.expander("Detection settings", expanded=False):
    pose_conf = st.slider("Board corner confidence", 0.1, 0.95, DEFAULT_POSE_CONF, 0.05)
    piece_conf = st.slider(
        "Piece detection confidence", 0.1, 0.95, DEFAULT_PIECE_CONF, 0.05
    )
    stability_seconds = st.slider(
        "Move stability window (seconds)",
        0.3,
        4.0,
        DEFAULT_STABILITY_SECONDS,
        0.1,
        help="How long a board state must stay unchanged before a move is confirmed. "
        "Higher = fewer false positives from hands/flicker, but slower to react.",
    )
    frame_stride = st.slider(
        "Process every Nth frame",
        1,
        10,
        1,
        help="Increase to speed up processing on long videos or slower (CPU-only) machines.",
    )
    show_live_preview = st.checkbox("Show live detection preview", value=True)

with st.sidebar.expander("Analysis settings", expanded=False):
    enable_stockfish = st.checkbox(
        "Enable Stockfish analysis", value=False, help="Requires Stockfish binary"
    )
    stockfish_depth = st.slider(
        "Stockfish search depth", 10, 25, 20, help="Higher = slower but more accurate"
    )

uploaded_video = st.sidebar.file_uploader(
    "Upload a chess game video", type=["mp4", "mov", "avi", "mkv"]
)
run_button = st.sidebar.button(
    "▶️ Run tracking", type="primary", disabled=uploaded_video is None
)

# --------------------------------------------------------------------------
# Cached model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO11 models...")
def load_tracker(
    pose_path: str,
    piece_path: str,
    board_size: int,
    pose_c: float,
    piece_c: float,
    stability: float,
):
    return ChessVideoTracker(
        pose_model_path=pose_path,
        piece_model_path=piece_path,
        board_size=board_size,
        pose_conf=pose_c,
        piece_conf=piece_c,
        stability_seconds=stability,
    )


@st.cache_resource(show_spinner="Initializing Stockfish...")
def load_stockfish(depth: int):
    try:
        return StockfishAnalyzer(depth=depth)
    except Exception as e:
        st.warning(f"Stockfish not available: {e}")
        return None


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("♟️ Chess Move Tracker")
st.write(
    "Upload a chess game video for AI-powered analysis: move detection, "
    "Stockfish evaluation, opening recognition, and comprehensive statistics."
)

if "result" not in st.session_state:
    st.session_state.result = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if uploaded_video is not None and run_button:
    # Persist the upload to a temp file. Close it immediately to avoid Windows file locks.
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_video.read())
        tmp_file.flush()
        video_path = tmp_file.name

    tracker = load_tracker(
        pose_model_path,
        piece_model_path,
        DEFAULT_BOARD_SIZE,
        pose_conf,
        piece_conf,
        stability_seconds,
    )

    progress_bar = st.progress(0.0, text="Starting...")
    col_preview, col_moves = st.columns([2, 1])
    with col_preview:
        preview_slot = st.empty()
    with col_moves:
        st.subheader("Moves (live)")
        moves_slot = st.empty()

    live_moves: list[MoveRecord] = []
    ui_state = {"last_update": 0.0}

    def on_progress(frame_idx: int, total_frames: int) -> None:
        if total_frames > 0:
            progress_bar.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Frame {frame_idx}/{total_frames}",
            )

    def on_frame(annotated_bgr, frame_idx: int, total_frames: int) -> None:
        if not show_live_preview or annotated_bgr is None:
            return
        now = time.time()
        if now - ui_state["last_update"] < 0.15:
            return
        ui_state["last_update"] = now
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        preview_slot.image(rgb, caption=f"Frame {frame_idx}/{total_frames}", use_container_width=True)

    def on_move(record: MoveRecord) -> None:
        live_moves.append(record)
        df = pd.DataFrame(
            [{"#": m.ply, "Color": m.color, "Move": m.san, "FEN": m.fen} for m in live_moves]
        )
        moves_slot.dataframe(df, use_container_width=True, hide_index=True, height=420)

    try:
        with st.spinner("Processing video..."):
            result = tracker.process_video(
                video_path,
                progress_callback=on_progress,
                frame_callback=on_frame if show_live_preview else None,
                move_callback=on_move,
                frame_stride=frame_stride,
            )
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except PermissionError:
            st.warning("Could not delete the temporary video file immediately; it will be removed when the OS releases it.")

    progress_bar.progress(1.0, text="Done!")
    st.session_state.result = result

    # Reconstruct the game and build a summary even when Stockfish is disabled.
    game = chess.pgn.read_game(io.StringIO(result["pgn"]))
    if game is None:
        game = chess.pgn.Game()
    game.headers["Event"] = "Analyzed Game"
    game.headers["White"] = "White Player"
    game.headers["Black"] = "Black Player"

    summary = GameAnalyzer().analyze_game(game)
    opening_transitions = OpeningRecognizer.analyze_game(game)
    eco_description = OpeningRecognizer.get_eco_description(summary.eco_code)
    st.session_state.analysis = {
        "summary": summary,
        "move_analyses": [],
        "opening_transitions": opening_transitions,
        "eco_description": eco_description,
    }

    # ---- ANALYSIS PHASE ----
    st.success(f"✓ Tracking complete — {len(result['moves'])} move(s) detected.")

    if enable_stockfish and len(result['moves']) > 0:
        with st.spinner("Running Stockfish analysis..."):
            stockfish = load_stockfish(stockfish_depth)
            if stockfish:
                # Reconstruct the game
                game = chess.pgn.Game()
                game.headers["Event"] = "Analyzed Game"
                game.headers["White"] = "White Player"
                game.headers["Black"] = "Black Player"

                board = chess.Board()
                for move_rec in result['moves']:
                    move = chess.Move.from_uci(f"{move_rec.from_square}{move_rec.to_square}")
                    if move in board.legal_moves:
                        board.push(move)

                # Analyze all moves
                move_analyses = []
                board = chess.Board()
                with st.progress(0, text="Analyzing moves...") as progress:
                    for idx, move_rec in enumerate(result['moves']):
                        move_uci = f"{move_rec.from_square}{move_rec.to_square}"
                        analysis = stockfish.score_move(board, move_uci)
                        move_analyses.append(analysis)
                        board.push(chess.Move.from_uci(move_uci))
                        progress.progress((idx + 1) / len(result['moves']))

                # Generate summary
                game_analyzer = GameAnalyzer()
                game = chess.pgn.Game()
                game.headers["Event"] = "Analyzed Game"

                board = chess.Board()
                for move_rec in result['moves']:
                    move = chess.Move.from_uci(f"{move_rec.from_square}{move_rec.to_square}")
                    if move in board.legal_moves:
                        board.push(move)

                summary = game_analyzer.analyze_game(game, move_analyses)
                opening_transitions = OpeningRecognizer.analyze_game(game)
                eco_description = OpeningRecognizer.get_eco_description(summary.eco_code)
                st.session_state.analysis = {
                    "summary": summary,
                    "move_analyses": move_analyses,
                    "opening_transitions": opening_transitions,
                    "eco_description": eco_description,
                }

# --------------------------------------------------------------------------
# Results section
# --------------------------------------------------------------------------
result = st.session_state.result
analysis = st.session_state.analysis

if result is not None:
    moves = result["moves"]
    st.divider()

    if result.get("start_fen"):
        start_board = chess.Board(result["start_fen"])
        st.subheader("🎬 Start Position from Video")
        st.components.v1.html(
            BoardViewer.get_html_viewer(start_board, [], None, title="Video Start"),
            height=520,
        )
        st.markdown(f"**Start FEN:** `{result['start_fen']}`")
        st.divider()

    if moves:
        board = chess.Board()
        move_sans = []
        last_move = None
        for move_record in moves:
            move = chess.Move.from_uci(f"{move_record.from_square}{move_record.to_square}")
            if move in board.legal_moves:
                board.push(move)
                move_sans.append(move_record.san)
                last_move = move

        st.subheader("♟️ Final Board & Move Log")
        col_board, col_info = st.columns([2, 1])
        with col_board:
            html = BoardViewer.get_html_viewer(board, move_sans, last_move, title="Final Position")
            st.components.v1.html(html, height=520)
        with col_info:
            st.write("### Live Moves")
            move_df = pd.DataFrame(
                [
                    {
                        "#": m.ply,
                        "Color": "White" if m.color == "w" else "Black",
                        "Move": m.san,
                        "From": m.from_square,
                        "To": m.to_square,
                    }
                    for m in moves
                ]
            )
            st.dataframe(move_df, use_container_width=True, hide_index=True)

    if analysis is not None:
        summary = analysis["summary"]
        move_analyses = analysis["move_analyses"]
        opening_transitions = analysis.get("opening_transitions", [])
        eco_description = analysis.get("eco_description", "Unknown ECO")

        # ---- GAME SUMMARY DASHBOARD ----
        st.subheader("📊 Game Summary Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Moves", summary.total_moves)
        with col2:
            st.metric("Result", summary.result)
        with col3:
            st.metric("Opening", summary.opening_name, f"{summary.eco_code}")
        with col4:
            st.metric("Game Length", f"{summary.game_length_phase.title()}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Opening Moves", summary.opening_moves)
        with col2:
            st.metric("Middlegame Moves", summary.middlegame_moves)
        with col3:
            st.metric("Endgame Moves", summary.endgame_moves)

        st.markdown(f"**Detected Opening:** {summary.opening_name} ({summary.eco_code})")
        st.markdown(f"**ECO Description:** {eco_description}")

        if opening_transitions:
            distinct_transitions = []
            last_opening = None
            for info in opening_transitions:
                if info["opening"] != last_opening:
                    distinct_transitions.append(
                        {
                            "Ply": info["ply"],
                            "Opening": info["opening"],
                            "ECO": info["eco"],
                        }
                    )
                    last_opening = info["opening"]

            if distinct_transitions:
                with st.expander("Opening recognition trace", expanded=False):
                    st.write(
                        "This shows how the opening classification evolves as the moves are applied."
                    )
                    st.dataframe(distinct_transitions, use_container_width=True)
        else:
            st.info("Opening recognition did not match a known pattern for this game.")

        st.subheader("🧠 Game Narrative")
        st.write(GameAnalyzer.get_narrative(summary))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("White Player")
            st.write(f"**Moves:** {summary.white.total_moves}")
            st.write(f"**Accuracy:** {summary.white.average_accuracy:.1f}%")
            st.write(f"**Brilliant:** {summary.white.brilliant_moves}")
            st.write(f"**Mistakes:** {summary.white.mistakes} | "
                    f"**Blunders:** {summary.white.blunders}")

        with col2:
            st.subheader("Black Player")
            st.write(f"**Moves:** {summary.black.total_moves}")
            st.write(f"**Accuracy:** {summary.black.average_accuracy:.1f}%")
            st.write(f"**Brilliant:** {summary.black.brilliant_moves}")
            st.write(f"**Mistakes:** {summary.black.mistakes} | "
                    f"**Blunders:** {summary.black.blunders}")

        summary_json = json.dumps(summary.to_dict(), indent=2)
        st.download_button(
            "📥 Download Summary JSON",
            summary_json,
            "game_summary.json",
            "application/json",
        )

        # ---- ACCURACY CHART ----
        st.subheader("📈 Accuracy by Phase")

        phase_data = {
            "Phase": ["Opening", "Middlegame", "Endgame"],
            "Moves": [summary.opening_moves, summary.middlegame_moves, summary.endgame_moves],
        }
        fig = px.bar(
            phase_data,
            x="Phase",
            y="Moves",
            color="Moves",
            title="Game Breakdown by Phase",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- MOVE QUALITY DISTRIBUTION ----
        st.subheader("🎯 Move Quality Distribution")

        quality_counts = {
            "brilliant": 0,
            "excellent": 0,
            "good": 0,
            "inaccuracy": 0,
            "mistake": 0,
            "blunder": 0,
        }

        for analysis in move_analyses:
            quality = analysis.get("quality", "unknown")
            if quality in quality_counts:
                quality_counts[quality] += 1

        quality_df = pd.DataFrame(
            [
                {"Quality": "Brilliant", "Count": quality_counts["brilliant"]},
                {"Quality": "Excellent", "Count": quality_counts["excellent"]},
                {"Quality": "Good", "Count": quality_counts["good"]},
                {"Quality": "Inaccuracy", "Count": quality_counts["inaccuracy"]},
                {"Quality": "Mistake", "Count": quality_counts["mistake"]},
                {"Quality": "Blunder", "Count": quality_counts["blunder"]},
            ]
        )

        fig = px.pie(
            quality_df,
            values="Count",
            names="Quality",
            color="Quality",
            color_discrete_map={
                "Brilliant": "#2ecc71",
                "Excellent": "#3498db",
                "Good": "#f39c12",
                "Inaccuracy": "#e74c3c",
                "Mistake": "#c0392b",
                "Blunder": "#8b0000",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- MOVE LOG ----
    st.subheader("📋 Detailed Move Log")

    if moves:
        if analysis:
            # Enhanced move log with quality scores
            move_data = []
            for i, m in enumerate(moves):
                row = {
                    "#": m.ply,
                    "Color": "White" if m.color == "w" else "Black",
                    "From": m.from_square.upper(),
                    "To": m.to_square.upper(),
                    "Move": m.san,
                    "FEN": m.fen[:30] + "...",
                }
                if i < len(move_analyses):
                    quality = move_analyses[i].get("quality", "?")
                    row["Quality"] = quality
                move_data.append(row)

            df = pd.DataFrame(move_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            # Basic move log without analysis
            df = pd.DataFrame(
                [
                    {
                        "#": m.ply,
                        "Color": "White" if m.color == "w" else "Black",
                        "From": m.from_square.upper(),
                        "To": m.to_square.upper(),
                        "Move (SAN)": m.san,
                        "FEN": m.fen[:30] + "...",
                    }
                    for m in moves
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Download options
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Moves (CSV)", csv_bytes, "moves.csv", "text/csv")
        with col2:
            st.download_button(
                "📥 Download Moves (JSON)", json_bytes, "moves.json", "application/json"
            )
    else:
        st.info("No moves were confirmed in this video.")

    # ---- PGN ----
    st.subheader("📄 PGN Notation")
    st.code(result["pgn"], language=None)
    st.download_button(
        "📥 Download PGN", result["pgn"].encode("utf-8"), "game.pgn", "application/x-chess-pgn"
    )

    # ---- FEN ----
    st.subheader("🎯 Final Position (FEN)")
    st.code(result["final_fen"], language=None)
    st.link_button(
        "🔗 View on Lichess",
        f"https://lichess.org/editor/{result['final_fen'].replace(' ', '_')}",
    )

else:
    st.info("Upload a video and click **Run tracking** in the sidebar to get started.")
