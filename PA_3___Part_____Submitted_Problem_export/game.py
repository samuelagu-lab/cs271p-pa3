"""
CS 271P — PA 3D-s : the game (King-of-the-Hill Los Alamos chess) — PROVIDED, do not edit
======================================================================================
This is the two-player, constant-sum game your MCTS agent plays. You DON'T need to
read this file to do the assignment — you only need the interface your agent talks to:

    g = LosAlamos()                 # the 6x6 game
    g.initial()                     # the starting state
    g.to_move(state)                # 'w' or 'b' — whose turn
    g.actions(state)                # the legal moves in `state`
    g.result(state, move)           # the state after playing `move`
    g.is_terminal(state)            # is the game over?
    g.winner(state)                 # 'w' / 'b' / None (draw / not over)

  WIN CONDITION (this variant — 'King of the Hill'):
    A side wins EITHER by capturing the enemy king, OR by moving its own king onto one
    of the four central squares (the 'hill': (2,2), (2,3), (3,2), (3,3)). The game ends
    the instant either happens.

A move is (from_square, to_square, promo); a square is (col, row). Helpers you may find
useful for a game-specific idea: `king_sq(state, color)`, `HILL` (the winning squares),
`material(state, color)`, `PIECE_VALUE`, `is_capture(state, move)` — pick whatever fits
how THIS game is actually won.

Design notes (King-capture model, stdlib-only):
  * King-capture model: a move that captures a king is legal; the game is over the
    instant a king leaves the board.  We generate PSEUDO-LEGAL moves (we do NOT
    filter moves that leave your own king attacked).  This is what the 1956 MANIAC
    program effectively did and it removes all the check/checkmate/stalemate/pin
    machinery while being behaviorally near-identical at shallow search depth.
  * Parameterized: board size + piece set are config.  6x6 (R N Q K N R, no bishops)
    for PA 3A/3B/3D; 8x8 standard piece set for PA 3C's node-count benchmark.
  * Simplified pawns: single-step only (no double-step, no en passant); promotion to
    Q / R / N on the last rank.  No castling.
  * A ply cap guarantees every game (and every MCTS rollout) terminates.

Coordinates: square = (col, row), col in [0,W), row in [0,H), ROW 0 IS THE TOP.
White starts on the bottom ranks and moves UP (row decreases); Black starts on the
top ranks and moves DOWN (row increases).  Each board cell is a 2-char string like
'wP' / 'bK', or '.' for empty.  State is an immutable (hashable) namedtuple.
"""

from collections import namedtuple

# ---------------------------------------------------------------- piece values
# King has no material value (capturing it ends the game).
PIECE_VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

# A terminal score that dwarfs any evaluation-function magnitude. Used by utility() for the
# minimax-based PAs; the MCTS grader reads winner() directly and never touches this. Subtracting
# the ply number makes a minimax search prefer FASTER wins and SLOWER losses.
WIN = 10 ** 7

# ------------------------------------------------------------ move-offset tables
KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING   = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
ROOK   = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
QUEEN  = ROOK + BISHOP

# State: board is a tuple of W*H cell-strings; stm = side to move ('w'/'b');
# ply = number of plies played so far (drives the move cap).
State = namedtuple("State", ["board", "stm", "ply"])

# A move is a 3-tuple (from_sq, to_sq, promo) with promo in {None, 'Q', 'R', 'N'}.

BACK_RANKS = {
    "losalamos": ["R", "N", "Q", "K", "N", "R"],          # 6 wide
    "standard8": ["R", "N", "B", "Q", "K", "B", "N", "R"],  # 8 wide
}


def other(color):
    return "b" if color == "w" else "w"


def state_from_pieces(W, H, placements, stm="w", ply=0):
    """Build a State from a {(col, row): 'wK', ...} placement dict (for test positions)."""
    board = ["."] * (W * H)
    for (col, row), pc in placements.items():
        board[row * W + col] = pc
    return State(tuple(board), stm, ply)


class LosAlamos:
    """Parameterized Los Alamos chess (king-capture model)."""

    def __init__(self, size=6, pieces="losalamos", move_cap=120):
        self.W = self.H = size
        self.pieces = pieces
        self.move_cap = move_cap
        self.back = BACK_RANKS[pieces]
        assert len(self.back) == self.W, "back rank must span the board width"
        # The 'hill': the central 2x2 block — a king reaching one of these wins immediately.
        # Computed from the board size (6 -> {(2,2),(2,3),(3,2),(3,3)}, 8 -> {(3,3)..(4,4)}) so the
        # win squares stay centred on any even board, not pinned to the 6x6 coordinates.
        m = size // 2
        self.HILL = frozenset({(m - 1, m - 1), (m - 1, m), (m, m - 1), (m, m)})

    # ------------------------------------------------------------- core helpers
    def _idx(self, col, row):
        return row * self.W + col

    def initial(self):
        W, H = self.W, self.H
        board = ["."] * (W * H)
        # Black on the top two ranks (back rank row 0, pawns row 1).
        for c, p in enumerate(self.back):
            board[self._idx(c, 0)] = "b" + p
        for c in range(W):
            board[self._idx(c, 1)] = "bP"
        # White on the bottom two ranks (back rank row H-1, pawns row H-2).
        for c, p in enumerate(self.back):
            board[self._idx(c, H - 1)] = "w" + p
        for c in range(W):
            board[self._idx(c, H - 2)] = "wP"
        return State(tuple(board), "w", 0)

    def to_move(self, s):
        return s.stm

    # ------------------------------------------------------- move generation
    def _add_pawn(self, moves, frm, to, color):
        last = 0 if color == "w" else self.H - 1
        if to[1] == last:
            for pr in ("Q", "R", "N"):
                moves.append((frm, to, pr))
        else:
            moves.append((frm, to, None))

    def moves_for(self, s, color):
        """All PSEUDO-LEGAL moves for `color` in state s (regardless of whose turn)."""
        moves = []
        b, W, H = s.board, self.W, self.H
        for idx, cell in enumerate(b):
            if cell == "." or cell[0] != color:
                continue
            col, row = idx % W, idx // W
            p = cell[1]
            if p == "P":
                d = -1 if color == "w" else 1
                nr = row + d
                if 0 <= nr < H and b[self._idx(col, nr)] == ".":
                    self._add_pawn(moves, (col, row), (col, nr), color)
                for dc in (-1, 1):
                    nc = col + dc
                    if 0 <= nc < W and 0 <= nr < H:
                        t = b[self._idx(nc, nr)]
                        if t != "." and t[0] != color:
                            self._add_pawn(moves, (col, row), (nc, nr), color)
            elif p == "N":
                for dc, dr in KNIGHT:
                    nc, nr = col + dc, row + dr
                    if 0 <= nc < W and 0 <= nr < H:
                        t = b[self._idx(nc, nr)]
                        if t == "." or t[0] != color:
                            moves.append(((col, row), (nc, nr), None))
            elif p == "K":
                for dc, dr in KING:
                    nc, nr = col + dc, row + dr
                    if 0 <= nc < W and 0 <= nr < H:
                        t = b[self._idx(nc, nr)]
                        if t == "." or t[0] != color:
                            moves.append(((col, row), (nc, nr), None))
            else:  # sliding piece: R, B, or Q
                dirs = ROOK if p == "R" else BISHOP if p == "B" else QUEEN
                for dc, dr in dirs:
                    nc, nr = col + dc, row + dr
                    while 0 <= nc < W and 0 <= nr < H:
                        t = b[self._idx(nc, nr)]
                        if t == ".":
                            moves.append(((col, row), (nc, nr), None))
                        else:
                            if t[0] != color:
                                moves.append(((col, row), (nc, nr), None))
                            break
                        nc += dc
                        nr += dr
        return moves

    def actions(self, s):
        return self.moves_for(s, s.stm)

    def attacks_for(self, s, color):
        """Count of squares attacked (capture-style) by `color` -> {square: n_attackers}.

        Differs from move generation for pawns: a pawn attacks its two forward DIAGONALS
        (occupied or not) but does NOT 'attack' the square straight ahead.  Sliders attack
        up to and including the first blocker.  Used for king-safety evaluation.
        """
        atk = {}
        b, W, H = s.board, self.W, self.H

        def bump(c, r):
            atk[(c, r)] = atk.get((c, r), 0) + 1

        for idx, cell in enumerate(b):
            if cell == "." or cell[0] != color:
                continue
            col, row = idx % W, idx // W
            p = cell[1]
            if p == "P":
                d = -1 if color == "w" else 1
                for dc in (-1, 1):
                    nc, nr = col + dc, row + d
                    if 0 <= nc < W and 0 <= nr < H:
                        bump(nc, nr)
            elif p in ("N", "K"):
                for dc, dr in (KNIGHT if p == "N" else KING):
                    nc, nr = col + dc, row + dr
                    if 0 <= nc < W and 0 <= nr < H:
                        bump(nc, nr)
            else:
                dirs = ROOK if p == "R" else BISHOP if p == "B" else QUEEN
                for dc, dr in dirs:
                    nc, nr = col + dc, row + dr
                    while 0 <= nc < W and 0 <= nr < H:
                        bump(nc, nr)
                        if b[self._idx(nc, nr)] != ".":
                            break
                        nc += dc
                        nr += dr
        return atk

    def result(self, s, move):
        frm, to, promo = move
        b = list(s.board)
        fi, ti = self._idx(*frm), self._idx(*to)
        piece = b[fi]
        b[fi] = "."
        b[ti] = piece[0] + promo if promo else piece
        return State(tuple(b), other(s.stm), s.ply + 1)

    # ----------------------------------------------------------- terminal logic
    def king_sq(self, s, color):
        """(col, row) of `color`'s king, or None if it has been captured."""
        tgt = color + "K"
        for i, cell in enumerate(s.board):
            if cell == tgt:
                return (i % self.W, i // self.W)
        return None

    def is_terminal(self, s):
        return self.winner(s) is not None or s.ply >= self.move_cap

    def winner(self, s):
        """'w' / 'b' if a side has won, else None (draw / ongoing).

        A side wins EITHER by capturing the enemy king OR by moving its own king onto
        one of the four central HILL squares.
        """
        wk, bk = self.king_sq(s, "w"), self.king_sq(s, "b")
        if wk is not None and bk is None:
            return "w"
        if bk is not None and wk is None:
            return "b"
        if wk in self.HILL:
            return "w"
        if bk in self.HILL:
            return "b"
        return None

    def utility(self, s, player="w"):
        """Terminal value from `player`'s perspective: +(WIN-ply) win, -(WIN-ply) loss, 0 draw.

        The search uses the fixed White perspective (White = MAX, Black = MIN), so
        evaluation functions and this utility both read '+ = good for White'.
        """
        w = self.winner(s)
        if w == "w":
            val = WIN - s.ply
        elif w == "b":
            val = -(WIN - s.ply)
        else:
            val = 0
        return val if player == "w" else -val

    def is_capture(self, s, move):
        return s.board[self._idx(*move[1])] != "."

    # ------------------------------------------------------------------ display
    def display(self, s):
        W, H = self.W, self.H
        rows = []
        for r in range(H):
            cells = []
            for c in range(W):
                cell = s.board[self._idx(c, r)]
                cells.append(".." if cell == "." else cell)
            rows.append(" ".join(cells))
        side = "White" if s.stm == "w" else "Black"
        return "\n".join(rows) + f"\n[{side} to move, ply {s.ply}]"


class Hexapawn:
    """Hexapawn (Martin Gardner, 1962): 3x3, pawns only — the PA 3A warm-up.

    Same interface as LosAlamos so the search functions run unchanged.  A side wins
    by advancing a pawn to the opponent's home row, by capturing all enemy pawns, or
    when the opponent has no legal move.  White moves up (row 0 is the top); Black down.
    """

    def __init__(self, size=3, move_cap=30):
        self.W = self.H = size
        self.pieces = "hexapawn"
        self.move_cap = move_cap

    def _idx(self, col, row):
        return row * self.W + col

    def initial(self):
        W, H = self.W, self.H
        board = ["."] * (W * H)
        for c in range(W):
            board[self._idx(c, 0)] = "bP"       # Black on the top row
            board[self._idx(c, H - 1)] = "wP"   # White on the bottom row
        return State(tuple(board), "w", 0)

    def to_move(self, s):
        return s.stm

    def moves_for(self, s, color):
        moves = []
        b, W, H = s.board, self.W, self.H
        d = -1 if color == "w" else 1
        for idx, cell in enumerate(b):
            if cell == "." or cell[0] != color:
                continue
            col, row = idx % W, idx // W
            nr = row + d
            if 0 <= nr < H:
                if b[self._idx(col, nr)] == ".":
                    moves.append(((col, row), (col, nr), None))
                for dc in (-1, 1):
                    nc = col + dc
                    if 0 <= nc < W:
                        t = b[self._idx(nc, nr)]
                        if t != "." and t[0] != color:
                            moves.append(((col, row), (nc, nr), None))
        return moves

    def actions(self, s):
        return self.moves_for(s, s.stm)

    def result(self, s, move):
        frm, to, _ = move
        b = list(s.board)
        b[self._idx(*to)] = b[self._idx(*frm)]
        b[self._idx(*frm)] = "."
        return State(tuple(b), other(s.stm), s.ply + 1)

    def winner(self, s):
        W, H = self.W, self.H
        # A pawn on the opponent's home row wins.
        if any(s.board[self._idx(c, 0)] == "wP" for c in range(W)):
            return "w"
        if any(s.board[self._idx(c, H - 1)] == "bP" for c in range(W)):
            return "b"
        # No pawns left for a side -> that side has lost.
        wp = any(cell == "wP" for cell in s.board)
        bp = any(cell == "bP" for cell in s.board)
        if not wp:
            return "b"
        if not bp:
            return "w"
        # Side to move with no legal move loses.
        if not self.moves_for(s, s.stm):
            return other(s.stm)
        return None

    def is_terminal(self, s):
        return self.winner(s) is not None or s.ply >= self.move_cap

    def utility(self, s, player="w"):
        w = self.winner(s)
        if w == "w":
            val = WIN - s.ply
        elif w == "b":
            val = -(WIN - s.ply)
        else:
            val = 0
        return val if player == "w" else -val

    def is_capture(self, s, move):
        return s.board[self._idx(*move[1])] != "."

    def display(self, s):
        W, H = self.W, self.H
        rows = []
        for r in range(H):
            rows.append(" ".join(".." if s.board[self._idx(c, r)] == "." else s.board[self._idx(c, r)]
                                 for c in range(W)))
        side = "White" if s.stm == "w" else "Black"
        return "\n".join(rows) + f"\n[{side} to move, ply {s.ply}]"


# ----------------------------------------------------- a handy game-specific helper
def material(state, color):
    """Total material value of `color`'s pieces. King excluded (capturing it ends the game)."""
    return sum(PIECE_VALUE[c[1]] for c in state.board if c != "." and c[0] == color and c[1] != "K")
