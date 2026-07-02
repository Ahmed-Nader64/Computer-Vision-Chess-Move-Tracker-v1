"""
Phase 3: State Analysis - the temporal state machine that turns raw
per-frame piece detections into confirmed "a move happened" events.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict


class MoveEvent(TypedDict, total=False):
    type: str                       # "first_move" or "move"
    initial_board_state: Dict[str, str]
    active_color: str
    move: Tuple[str, str]


class ChessStateAnalyzer:
    """
    A filter + state machine. It takes raw YOLO detections for a single
    frame, converts them into a logical board dict, and only "confirms" a
    board state once it has stayed identical for `stability_thresh`
    consecutive frames. This filters out hand occlusions and detection
    flicker. When a confirmed state differs from the previous one, it
    infers which squares changed and reports a move.
    """

    def __init__(self, board_size: int = 640, stability_thresh: int = 10):
        self.board_size = board_size
        self.square_size = board_size / 8
        self.stability_thresh = stability_thresh

        # image y=0 is the top of the board => rank 8
        self.files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]

        # State-machine memory
        self.prev_stable_board_state: Optional[Dict[str, str]] = None
        self.candidate_board_str: Optional[str] = None
        self.stability_counter = 0

        # Game-start memory
        self.initial_board_state: Optional[Dict[str, str]] = None
        self.game_initialized = False

    def _get_square_from_xy(self, x: float, y: float) -> str:
        col = int(x // self.square_size)
        row = int(y // self.square_size)
        col = max(0, min(7, col))
        row = max(0, min(7, row))
        return self.files[col] + self.ranks[row]

    def detections_to_board(self, detections: List) -> Optional[Dict[str, str]]:
        """
        Convert a list of detections `[x1, y1, x2, y2, conf, cls_id, cls_name]`
        into a `{square: piece_code}` dict, using each box's top-center point
        (more robust to perspective distortion than the box center).

        Returns None if a "Hand" is among the detections (frame considered
        unstable/occluded).
        """
        board_state: Dict[str, str] = {}
        conf_map: Dict[str, float] = {}

        for box in detections:
            x1, y1, x2, y2 = box[:4]
            conf = box[4]
            cls_name = box[6]

            if cls_name == "Hand":
                return None

            ref_x = (x1 + x2) / 2
            ref_y = y1 + 0.25 * (y2 - y1)
            square = self._get_square_from_xy(ref_x, ref_y)

            if square in board_state:
                if conf > conf_map[square]:
                    board_state[square] = cls_name
                    conf_map[square] = conf
            else:
                board_state[square] = cls_name
                conf_map[square] = conf

        return board_state

    def _detect_move_diff(
        self, old_board: Dict[str, str], new_board: Dict[str, str]
    ) -> Optional[Tuple[str, str]]:
        """Infer (from_square, to_square) between two stable board states."""
        all_squares = set(old_board.keys()) | set(new_board.keys())
        diff_squares = {
            sq for sq in all_squares if old_board.get(sq) != new_board.get(sq)
        }

        from_sq, to_sq = None, None
        for sq in diff_squares:
            piece_old = old_board.get(sq)
            piece_new = new_board.get(sq)
            if piece_old and not piece_new:
                from_sq = sq
            elif piece_new:
                to_sq = sq

        if from_sq and to_sq:
            return (from_sq, to_sq)
        return None

    def process_frame(self, detections: List) -> Optional[MoveEvent]:
        """
        Main entry point: call once per video frame with the raw piece
        detections. Returns a move-event dict when a move has just been
        confirmed, otherwise None.
        """
        current_board_state = self.detections_to_board(detections)

        if current_board_state is None or len(current_board_state) < 2:
            self.stability_counter = 0
            self.candidate_board_str = None
            return None

        current_board_str = str(sorted(current_board_state.items()))

        if current_board_str == self.candidate_board_str:
            self.stability_counter += 1
        else:
            self.candidate_board_str = current_board_str
            self.stability_counter = 0

        if self.stability_counter < self.stability_thresh:
            return None

        # ---- Initialization phase: waiting for the first move ----
        if not self.game_initialized:
            if self.initial_board_state is None:
                self.initial_board_state = current_board_state
                return None

            prev_board_str = str(sorted(self.initial_board_state.items()))
            if current_board_str == prev_board_str:
                return None

            move = self._detect_move_diff(self.initial_board_state, current_board_state)
            if not move:
                return None

            from_sq, _to_sq = move
            moving_piece = self.initial_board_state.get(from_sq)
            if not moving_piece:
                return None

            active_color = moving_piece[0]  # 'wP' -> 'w'
            self.game_initialized = True
            self.prev_stable_board_state = current_board_state
            self.stability_counter = 0

            return {
                "type": "first_move",
                "initial_board_state": self.initial_board_state,
                "active_color": active_color,
                "move": move,
            }

        # ---- Game loop phase: subsequent moves ----
        prev_board_str = str(sorted(self.prev_stable_board_state.items()))
        if current_board_str == prev_board_str:
            return None

        move = self._detect_move_diff(self.prev_stable_board_state, current_board_state)
        if not move:
            return None

        self.prev_stable_board_state = current_board_state
        self.stability_counter = 0

        return {"type": "move", "move": move}
