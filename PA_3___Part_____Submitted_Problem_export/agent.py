"""
CS 271P — PA 3D-s : YOUR Monte-Carlo agent   ✍️  (this is the file you submit)
===========================================================================
You are given a complete but DELIBERATELY NAIVE MCTS agent: selection is random, expansion adds all
children and returns a random one, the play-out is uniformly random, and back-propagation is the
standard one. It plays legally — but weakly.

Your job: make it stronger by rewriting any of the four phase methods below (and adding any helpers
or `__init__` state you like). How you improve each phase is up to you — draw on what you know about
MCTS and about this game. Your lecture notes and the shipped survey
(`A_Survey_of_Monte_Carlo_Tree_Search_Methods.pdf`, Browne et al. 2012) are good places to start.

You are graded by how often your agent BEATS our hidden agents in a tournament (see the README). Only
this file is submitted; do not edit `game.py` or `mcts_framework.py`.


PUCT + MCTS-Solver agent for King of the Hill: Los Alamos (CS 271P PA 3D).

The game
--------
6x6, no bishops, no castling, no double pawn step, no en passant; pawns promote to
queen, rook or knight. A side wins by capturing the enemy king or by moving its own
king onto one of the four central squares, (2,2), (2,3), (3,2) and (3,3). Moves into
attack are legal: the engine uses king capture, not check.

That second condition dominates the design. Every square on a 6x6 board lies within
two king moves of those four, so a hill win is always two moves away for both sides,
and a piece parked on a hill square defends nothing because the enemy king captures
it and lands there anyway. The only answers to a king one step from the hill are to
take that king first or to win first, which is what `_forced_win_next` and
`_safe_after` test. Material matters much less here than in ordinary chess.

Design notes
------------
The CPU-time budget per move is small, so the number of iterations is important.
Every heuristic here is written to avoid `game.result()` and `game.actions()`
wherever a board lookup can answer the same question:

  * win detection is a board test (capture the enemy king, or step our king onto
    a hill square), confirmed with the real rules only for the few moves it flags;
  * the safety penalty on priors is applied to the leading moves only, and stops
    as soon as the leading move is sound;
  * `_attacked` scans outward from one square instead of generating every reply.

Only `game.W`, `game.H`, `game.HILL`, `game.move_cap`, `game.to_move`,
`game.actions`, `game.result`, `game.winner`, `game.is_terminal`,
`game.is_capture` and `game.king_sq` are used, plus `other` and `PIECE_VALUE`.
Direction tables are defined here rather than imported, so a missing name in
game.py can never stop this module importing.
"""

import heapq
import math

from mcts_framework import Node
from game import other, PIECE_VALUE

_ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_KING_DIRS = _ROOK_DIRS + _BISHOP_DIRS
_KNIGHT_DIRS = ((1, 2), (2, 1), (2, -1), (1, -2),
                (-1, -2), (-2, -1), (-2, 1), (-1, 2))


class Agent:
    """A Monte-Carlo Tree Search agent using PUCT and MCTS-Solver."""

    name = "my-agent"

    # --- tuning constants (override per instance to A/B them) -------------
    C_PUCT = 1.0             # exploration weight in the PUCT term
    SOFTMAX_TEMP = 1.0       # >1 flattens priors, <1 sharpens them
    CAPTURE_SCALE = 2.0      # prior-score bonus for capturing the best non-king piece
    HILL_STEP = 0.5          # prior-score bonus per king step closer to the hill
    PROMO_BONUS = 0.5
    PLAYOUT_DEPTH = 40
    RANDOM_MOVE_PROB = 0.25  # share of playout plies played at random
    SAFETY_CHECKS = 2        # playout candidates safety-checked per ply (0 disables)
    PRIOR_SAFETY_TOP = 4     # leading moves safety-checked when building priors (0 disables)
    UNSAFE_PENALTY = 3.0     # score subtracted from a move that hangs the king
    FORCED_WIN_SCORE = 10.0  # score for a move that forces a hill win next turn
    PLAYOUT_FORCED_WIN = True
    EVAL_MARGIN_FRAC = 0.3   # material edge (as a fraction of the top piece value) to call a cutoff
    RACE_MARGIN = 1          # tempi to spare before a cut-off race counts as decided
    RACE_WEIGHT = 0.5        # value of one tempo of race lead, in top-piece units
    RACE_LIMIT = 5           # every square on a 6x6 board is within 2 king moves of the
                             # central hill, so only blockers push the walk past that
    RACE_ATTACK_AWARE = True # treat squares the opponent attacks as impassable en route
    REWARD_DECAY = 0.98      # per ply; 1.0 restores plain win/draw/loss scoring
    AB_PLAYOUT = False       # choose each playout move with a selective two-ply search
                             # (measured no gain on stand-in rules; flip on to test here)
    AB_WIDTH = 3             # own moves examined at each playout ply
    AB_REPLY_WIDTH = 6       # opponent replies examined for each of those
    AB_PLAYOUT_DEPTH = 12    # playout length when AB_PLAYOUT is on (see simulate)

    def __init__(self):
        self._game = None
        self._root = None
        self._proven = {}
        self._prior_cache = {}
        self._hills = None
        self._hill_set = None
        self._last_plies = 0     # length of the most recent playout, for the reward decay
        # Largest non-king piece value, used to normalise capture scores.
        vals = [v for k, v in PIECE_VALUE.items() if str(k)[-1:] != 'K']
        self._max_val = max(vals) if vals else 1.0
        self._pawn_val = PIECE_VALUE.get('P', 1.0)

    # ------------------------------------------------------------------
    # Per-search bookkeeping
    # ------------------------------------------------------------------
    def _sync_root(self, node):
        """Reset the per-search caches only when a genuinely new tree starts.

        Holding a reference to the root also stops CPython reusing its id for a
        later Node, which would make this identity test lie.
        """
        root = node
        while root.parent is not None:
            root = root.parent
        # `is not` compares object identity: every new search builds a brand-new root,
        # so a different object means the old caches belong to the previous move.
        if root is not self._root:
            self._root = root
            self._proven.clear()
            self._prior_cache.clear()

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------
    def _hill_squares(self, game):
        if self._hills is None:
            self._hills = tuple((int(c), int(r)) for c, r in game.HILL)
            self._hill_set = frozenset(self._hills)
        return self._hills

    def _hill_dist(self, game, sq):
        """Distance in king moves (Chebyshev) to the nearest hill square.

        The king moves diagonally, so Manhattan distance would overstate how far
        away the hill is and mis-rank the race to it.
        """
        col, row = sq
        return min(max(abs(col - hc), abs(row - hr))
                   for hc, hr in self._hill_squares(game))

    def _on_hill(self, game, sq):
        col, row = sq
        return any(col == hc and row == hr for hc, hr in self._hill_squares(game))

    def _captured_value(self, game, state, to):
        """Value of the piece standing on `to`.

        These rules have no en passant, so the empty-square branch should never be
        reached; it is kept because an exception here would forfeit the game.
        """
        piece = state.board[to[1] * game.W + to[0]]
        if len(piece) >= 2:
            return PIECE_VALUE.get(piece[1], self._pawn_val)
        return self._pawn_val

    # ------------------------------------------------------------------
    # Win and safety tests
    # ------------------------------------------------------------------
    def _wins_now(self, game, state, move, me):
        """Does `move` win on the spot for `me`?

        The two winning conditions are visible on the board, so this filters with
        lookups and pays for the rules check only on the moves it flags.
        """
        frm, to, _ = move
        W = game.W
        # There are only two ways to win and both are visible on the board, so check
        # them with lookups and call the rules only for a move that looks like a win.
        takes_king = state.board[to[1] * W + to[0]] == other(me) + 'K'
        king_to_hill = (state.board[frm[1] * W + frm[0]] == me + 'K'
                        and self._on_hill(game, to))
        if not (takes_king or king_to_hill):
            return False
        return game.winner(game.result(state, move)) == me

    def _attacked(self, game, board, col, row, by):
        """Is square (col, row) attacked by a piece of colour `by`?

        Scans outward from the square rather than generating every reply, which
        keeps the safety test affordable inside playouts.
        """
        W, H = game.W, game.H
        # Pawns are the one asymmetric case: an attacker sits one row "behind" the
        # square, and which way that is depends on colour. White moves toward row 0.
        pawn_row = row + 1 if by == 'w' else row - 1
        if 0 <= pawn_row < H:
            pawn = by + 'P'
            for dc in (-1, 1):
                c = col + dc
                if 0 <= c < W and board[pawn_row * W + c] == pawn:
                    return True
        knight = by + 'N'
        for dc, dr in _KNIGHT_DIRS:
            c, r = col + dc, row + dr
            if 0 <= c < W and 0 <= r < H and board[r * W + c] == knight:
                return True
        king = by + 'K'
        for dc, dr in _KING_DIRS:
            c, r = col + dc, row + dr
            if 0 <= c < W and 0 <= r < H and board[r * W + c] == king:
                return True
        queen = by + 'Q'
        for dirs, slider in ((_ROOK_DIRS, by + 'R'), (_BISHOP_DIRS, by + 'B')):
            for dc, dr in dirs:
                c, r = col + dc, row + dr
                while 0 <= c < W and 0 <= r < H:
                    cell = board[r * W + c]
                    if cell != '.':
                        if cell == slider or cell == queen:
                            return True
                        break
                    c += dc
                    r += dr
        return False

    def _hill_step_free(self, game, board, ksq, colour):
        """Can the king on `ksq` step straight onto a hill square?

        A square held by the king's own side blocks it; anything else can simply
        be captured, because landing on the hill ends the game at once.
        """
        kc, kr = ksq
        for hc, hr in self._hill_squares(game):
            if abs(hc - kc) <= 1 and abs(hr - kr) <= 1 and not (hc == kc and hr == kr):
                cell = board[hr * game.W + hc]
                if cell == '.' or cell[0] != colour:
                    return True
        return False

    def _safe_after(self, game, s2, me):
        """Fast test on the state after our move: can the opponent win at once,
        by taking our king or by walking onto the hill?"""
        if game.is_terminal(s2):
            w = game.winner(s2)
            return w is None or w == me
        opp = other(me)
        mk = game.king_sq(s2, me)
        if mk is None:
            return False
        if self._attacked(game, s2.board, mk[0], mk[1], opp):
            return False
        ok = game.king_sq(s2, opp)
        if ok is not None and self._hill_step_free(game, s2.board, ok, opp):
            return False
        return True

    def _safe_after_exact(self, game, s2, me):
        """Same question, answered from the real move list instead of `_attacked`.

        Used only where a wrong answer would be expensive, since it costs one
        `game.actions()` call.
        """
        if game.is_terminal(s2):
            w = game.winner(s2)
            return w is None or w == me
        opp = other(me)
        return not any(self._wins_now(game, s2, m, opp) for m in game.actions(s2))

    def _is_safe(self, game, state, move, me):
        return self._safe_after(game, game.result(state, move), me)

    def _forced_win_next(self, game, s2, me):
        """In s2 (opponent to move), can our king step onto a free hill square next
        turn whatever the opponent does?

        A piece dropped on that square is no defence, because our king captures it
        and still lands on the hill. So the only answers are taking our king or
        winning first, which is exactly what the safety test asks.
        """
        if game.is_terminal(s2) or s2.ply + 1 >= game.move_cap:
            return False
        ksq = game.king_sq(s2, me)
        if ksq is None or not self._hill_step_free(game, s2.board, ksq, me):
            return False
        return self._safe_after_exact(game, s2, me)

    # ------------------------------------------------------------------
    # Move heuristic (board lookups only: no result() or actions() calls)
    # ------------------------------------------------------------------
    def _move_score(self, game, state, move, me):
        frm, to, promo = move
        score = 0.0

        if game.is_capture(state, move):
            v = min(self._captured_value(game, state, to), self._max_val)
            score += self.CAPTURE_SCALE * v / self._max_val

        if state.board[frm[1] * game.W + frm[0]] == me + 'K':
            improvement = self._hill_dist(game, frm) - self._hill_dist(game, to)
            if improvement > 0:
                score += improvement * self.HILL_STEP

        if promo:
            # Pawns promote to queen, rook or knight, so the bonus follows the piece
            # actually chosen rather than assuming a queen every time.
            got = min(PIECE_VALUE.get(promo, self._max_val), self._max_val)
            score += self.PROMO_BONUS * got / self._max_val

        return score

    # ------------------------------------------------------------------
    # Priors, computed once per node and cached
    # ------------------------------------------------------------------
    def _compute_priors(self, game, state, moves):
        if not moves:
            return {}
        me = game.to_move(state)

        wins = [m for m in moves if self._wins_now(game, state, m, me)]
        if wins:
            share = 1.0 / len(wins)
            winning = set(wins)
            return {m: (share if m in winning else 0.0) for m in moves}

        scores = {m: self._move_score(game, state, m, me) for m in moves}

        # A king move that lands next to the hill may force a win. Few moves can,
        # so the board filter keeps this to a couple of rule calls per node.
        king = me + 'K'
        for m in moves:
            frm, to, _ = m
            if (state.board[frm[1] * game.W + frm[0]] == king
                    and self._hill_dist(game, to) == 1
                    and self._forced_win_next(game, game.result(state, m), me)):
                scores[m] = self.FORCED_WIN_SCORE

        # Penalise moves that hang the king, but only around the top of the order:
        # those are the moves that actually get expanded, and the cost then stays
        # roughly constant instead of scaling with the number of legal moves.
        # Work down from the current best move and stop as soon as one is sound: the
        # top of the order is what actually gets expanded, so the rest rarely matter.
        if self.PRIOR_SAFETY_TOP > 0:
            checked = set()
            for _ in range(self.PRIOR_SAFETY_TOP):
                rest = [m for m in moves if m not in checked]
                if not rest:
                    break
                best = max(rest, key=lambda m: scores[m])
                if scores[best] >= self.FORCED_WIN_SCORE:
                    break
                checked.add(best)
                if self._is_safe(game, state, best, me):
                    break            # the leading move is sound; stop paying
                scores[best] -= self.UNSAFE_PENALTY

        # Turn scores into probabilities. Subtracting the best score first cannot change
        # the result but stops exp() overflowing on large values.
        top = max(scores.values())
        exps = {m: math.exp((scores[m] - top) / self.SOFTMAX_TEMP) for m in moves}
        total = sum(exps.values())
        return {m: e / total for m, e in exps.items()}

    def _node_priors(self, node, game):
        cached = self._prior_cache.get(node)
        if cached is None:
            # At first expansion every legal move is still untried, but include the
            # children too so this is correct however the node is first reached.
            moves = [c.move for c in node.children] + list(node.untried or ())
            cached = self._compute_priors(game, node.state, moves)
            self._prior_cache[node] = cached
        return cached

    # ------------------------------------------------------------------
    # MCTS-Solver
    # ------------------------------------------------------------------
    def _mark(self, node, status):
        """Record a proven status, demoting the node's statistics if it is a loss.

        The framework picks the move with the most visits. A child can bank visits
        before its refutation turns up, and `select` then stops visiting it, so a
        move the solver has *proven* lost could still finish with the highest count
        and be played. Dropping its visits to one keeps it out of that contest while
        any unrefuted sibling remains. The root is left alone: nothing chooses
        between roots, and its visit count feeds the exploration term.
        """
        self._proven[node] = status
        if status == 'loss' and node.parent is not None:
            node.visits = 1
            node.value = 0.0

    def _update_proven(self, node, winner, game, mover):
        """Mark nodes whose value is settled. A child's status is from the point of
        view of this node's side to move, so a child win makes this node a loss."""
        if node in self._proven:
            return

        if node.children:
            statuses = [self._proven.get(c) for c in node.children]
            # A child's 'win' belongs to our OPPONENT, so one of them refutes us.
            # One refutation is enough; "all children lose" needs every child checked.
            if any(s == 'win' for s in statuses):
                self._mark(node, 'loss')
            elif node.untried:
                return               # untested moves remain: nothing else is settled
            elif all(s == 'loss' for s in statuses):
                self._mark(node, 'win')
            elif all(s in ('loss', 'draw') for s in statuses):
                self._mark(node, 'draw')
            return

        if node.untried:
            return                   # an unexpanded leaf settles nothing
        if game.is_terminal(node.state):
            if winner == mover:
                self._mark(node, 'win')
            elif winner is None:
                self._mark(node, 'draw')
            else:
                self._mark(node, 'loss')

    # ------------------------------------------------------------------
    # MCTS hooks
    # ------------------------------------------------------------------
    def select(self, node, rng):
        game = self._game
        if not node.children:
            # Defensive: the framework only descends into a node that has children,
            # so this cannot normally fire. Kept because an exception here forfeits
            # the game, which is a far worse outcome than two unused lines.
            return node

        live = []
        for child in node.children:
            p = self._proven.get(child)
            if p == 'win':
                return child         # a forced win: pile every remaining visit on it
            if p != 'loss':
                live.append(child)
        if not live:
            live = node.children

        priors = self._node_priors(node, game) if game is not None else {}
        parent_visits = node.visits or 1
        root_term = math.sqrt(parent_visits)

        def puct(child):
            # q = how well this move has actually done. u = how much it still deserves
            # a look: starts high for moves the heuristic likes, fades as it is visited.
            q = child.value / child.visits if child.visits else 0.0
            u = (self.C_PUCT * priors.get(child.move, 0.0)
                 * root_term / (1 + child.visits))
            return q + u

        return max(live, key=puct)

    def expand(self, node, game, rng):
        self._game = game
        self._sync_root(node)

        if not node.untried:
            # Defensive: the framework only expands a node that has untried moves.
            # Return the node itself rather than a child, so a stray call cannot
            # award one child a visit it never earned.
            return node

        priors = self._node_priors(node, game)
        move = max(node.untried, key=lambda m: priors.get(m, 0.0))
        node.untried.remove(move)
        child = Node(game.result(node.state, move), node, move)
        node.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Playout
    # ------------------------------------------------------------------
    def _race_dist(self, game, s, colour):
        """Fewest king moves for `colour` to stand on a hill square.

        A breadth-first walk, not a straight-line distance: the king's own pieces
        block a square, while an enemy piece does not, because landing on the hill
        ends the game and a capture on the way is just a move. Returns None when no
        route is found within RACE_LIMIT moves.

        With RACE_ATTACK_AWARE on, a square the opponent attacks is impassable too,
        because a king that pauses there is simply taken. The hill squares are exempt
        from that test: arriving on one ends the game before any reply, so it does
        not matter how well defended it is. Since blocking a hill square is no
        defence either — the king captures the blocker and still lands there — the
        attacked squares *en route* are the only thing that can slow a race down,
        which is why ignoring them left the old walk systematically optimistic.

        The attack map is read from the current board, so it misses discoveries from
        the king's own departure and rays its arrival would block. That is a
        heuristic, not a proof.
        """
        start = game.king_sq(s, colour)
        if start is None:
            return None
        self._hill_squares(game)
        hills = self._hill_set
        if start in hills:
            return 0

        W, H, board = game.W, game.H, s.board
        enemy = other(colour)
        guarded = self.RACE_ATTACK_AWARE
        seen = {start}
        frontier = [start]
        for dist in range(1, self.RACE_LIMIT + 1):
            nxt = []
            for col, row in frontier:
                for dc, dr in _KING_DIRS:
                    c, r = col + dc, row + dr
                    if not (0 <= c < W and 0 <= r < H) or (c, r) in seen:
                        continue
                    cell = board[r * W + c]
                    if cell != '.' and cell[0] == colour:
                        continue             # our own piece is in the way
                    seen.add((c, r))
                    if (c, r) in hills:
                        return dist          # arriving wins, however guarded it is
                    if guarded and self._attacked(game, board, c, r, enemy):
                        continue             # the king would be captured standing here
                    nxt.append((c, r))
            if not nxt:
                return None
            frontier = nxt
        return None

    def _static_eval(self, game, s, me):
        """Score a cut-off position as a win, loss or draw for `me`.

        Without this, every playout that hits the depth cap would come back as a
        draw and tell the search almost nothing.

        The hill race is settled on tempo rather than distance. If the side to move
        needs k moves and the other side needs j, the mover arrives on ply 2k-1 and
        the opponent on ply 2j, so the mover wins the race exactly when k <= j. One
        move of slack decides the position outright; a smaller edge is only worth
        something alongside material, since the race ignores the pieces that can
        interfere with it.
        """
        opp = other(me)
        my_king, opp_king = game.king_sq(s, me), game.king_sq(s, opp)
        if my_king is None:
            return None if opp_king is None else opp
        if opp_king is None:
            return me

        stm = game.to_move(s)
        far = self.RACE_LIMIT + 1
        dist = {c: self._race_dist(game, s, c) for c in (me, opp)}
        d_stm = dist[stm] if dist[stm] is not None else far
        d_wait = dist[other(stm)] if dist[other(stm)] is not None else far

        # Moving first is worth half a tempo, so the mover needs no slack to win the
        # race but the waiting side needs a full move more than the margin.
        # The mover arrives on ply 2k-1 and the waiting side on ply 2j, so the mover
        # wins the race whenever k <= j. `arrival` is when the winner of it gets there.
        lead_stm = d_wait - d_stm
        arrival = (2 * d_stm - 1) if lead_stm >= 0 else (2 * d_wait)
        if arrival <= game.move_cap - s.ply and (d_stm < far or d_wait < far):
            if lead_stm >= self.RACE_MARGIN:
                return stm
            if -lead_stm >= self.RACE_MARGIN + 1:
                return other(stm)

        material = {me: 0.0, opp: 0.0}
        for piece in s.board:
            if len(piece) < 2:
                continue
            colour, kind = piece[0], piece[1]
            if kind != 'K' and colour in material:
                material[colour] += PIECE_VALUE.get(kind, 0.0)

        diff = material[me] - material[opp]
        lead_me = lead_stm if stm == me else -lead_stm
        diff += lead_me * self.RACE_WEIGHT * self._max_val

        margin = self.EVAL_MARGIN_FRAC * self._max_val
        if diff > margin:
            return me
        if diff < -margin:
            return opp
        return None

    def _fast_eval(self, game, s, me):
        """Material plus Chebyshev hill distance, in top-piece units.

        Deliberately cheaper than `_static_eval`: no breadth-first race walk, because
        this runs at every leaf of the playout's two-ply search rather than once per
        playout.
        """
        W = game.W
        score = 0.0
        my_king = opp_king = None
        for i, piece in enumerate(s.board):
            if len(piece) < 2:
                continue
            colour, kind = piece[0], piece[1]
            if kind == 'K':
                if colour == me:
                    my_king = (i % W, i // W)
                else:
                    opp_king = (i % W, i // W)
            elif colour == me:
                score += PIECE_VALUE.get(kind, 0.0)
            else:
                score -= PIECE_VALUE.get(kind, 0.0)

        if my_king is None:
            return -1e6
        if opp_king is None:
            return 1e6
        score += ((self._hill_dist(game, opp_king) - self._hill_dist(game, my_king))
                  * self.RACE_WEIGHT * self._max_val)
        return score

    def _ab_playout_move(self, game, s, moves, rng):
        """Pick a playout move with a selective two-ply alpha-beta search.

        This is the MC-alpha-beta idea of Winands and Bjornsson: the playout keeps
        running, but each of its moves is chosen by a shallow search instead of a
        greedy rule. Only the best `AB_WIDTH` moves and `AB_REPLY_WIDTH` replies are
        considered, and the alpha cutoff below abandons a move as soon as the
        opponent has a reply making it no better than the best move already found.
        """
        me = game.to_move(s)
        opp = other(me)
        candidates = heapq.nlargest(
            self.AB_WIDTH, moves,
            key=lambda mv: self._move_score(game, s, mv, me) + rng.random() * 1e-6)

        best, best_val = candidates[0], None
        for m in candidates:
            s2 = game.result(s, m)
            if game.is_terminal(s2):
                w = game.winner(s2)
                val = 1e6 if w == me else (-1e6 if w is not None else 0.0)
            else:
                replies = game.actions(s2)
                if not replies:
                    val = self._fast_eval(game, s2, me)
                else:
                    worst = None
                    for r in heapq.nlargest(
                            self.AB_REPLY_WIDTH, replies,
                            key=lambda rv: self._move_score(game, s2, rv, opp)):
                        if self._wins_now(game, s2, r, opp):
                            worst = -1e6
                            break
                        v = self._fast_eval(game, game.result(s2, r), me)
                        if worst is None or v < worst:
                            worst = v
                            if best_val is not None and worst <= best_val:
                                break          # alpha cutoff: this move cannot win
                    val = worst if worst is not None else self._fast_eval(game, s2, me)
            if best_val is None or val > best_val:
                best, best_val = m, val
        return best

    def _playout_move(self, game, s, moves, rng):
        me = game.to_move(s)
        king = me + 'K'
        enemy_king = other(me) + 'K'

        for m in moves:
            frm, to, _ = m
            if s.board[to[1] * game.W + to[0]] == enemy_king:
                return m
            if s.board[frm[1] * game.W + frm[0]] == king:
                if self._on_hill(game, to):
                    return m
                if (self.PLAYOUT_FORCED_WIN and self._hill_dist(game, to) == 1
                        and self._forced_win_next(game, game.result(s, m), me)):
                    return m

        if self.AB_PLAYOUT:
            return self._ab_playout_move(game, s, moves, rng)

        n = max(1, self.SAFETY_CHECKS)
        if rng.random() < self.RANDOM_MOVE_PROB:
            candidates = rng.sample(moves, min(n, len(moves)))
        else:
            # Random tie-breaking keeps playouts from funnelling into one line.
            candidates = heapq.nlargest(
                n, moves,
                key=lambda mv: self._move_score(game, s, mv, me) + rng.random() * 1e-6)

        if self.SAFETY_CHECKS <= 0:
            return candidates[0]

        for m in candidates:
            if self._is_safe(game, s, m, me):
                return m
        return candidates[0]         # every option loses; play on regardless

    def simulate(self, game, state, rng):
        self._game = game
        s, plies = state, 0
        root_player = game.to_move(state)
        # A two-ply search per move costs far more per ply, so the playout is cut
        # shorter and `_static_eval` is left to judge the position it stops at.
        limit = self.AB_PLAYOUT_DEPTH if self.AB_PLAYOUT else self.PLAYOUT_DEPTH

        while plies < limit and not game.is_terminal(s):
            moves = game.actions(s)
            if not moves:
                break
            s = game.result(s, self._playout_move(game, s, moves, rng))
            plies += 1

        # Handed to backpropagate, which consumes and clears it. The reward decay
        # needs the playout's length and the framework's signature cannot carry it.
        self._last_plies = plies
        winner = game.winner(s)
        if winner is not None or game.is_terminal(s):
            return winner
        return self._static_eval(game, s, root_player)

    def backpropagate(self, node, winner, game):
        """Walk from the new leaf to the root, updating proofs and statistics.

        With `REWARD_DECAY` below 1.0 the reward shrinks with the number of plies
        between a node and the end of the playout, so a quick win outscores a slow
        one and a slow loss outscores a quick one. Draws stay at 0.5 and values stay
        inside [0, 1], but the spread narrows, so `C_PUCT` wants retuning alongside
        this.
        """
        self._game = game
        decay = self.REWARD_DECAY
        # Consume the playout length and clear it, so a backpropagate that somehow
        # arrives without a preceding simulate decays from zero instead of reusing a
        # stale count from the previous iteration.
        steps, self._last_plies = self._last_plies, 0
        while node is not None:
            # state.stm is who moves NEXT from here, so the player who made the move
            # that led here is the other one. All scoring is from their point of view.
            mover = other(node.state.stm)
            self._update_proven(node, winner, game, mover)

            node.visits += 1
            if winner is None:
                node.value += 0.5    # half credit keeps draws above losses
            elif decay >= 1.0:
                if winner == mover:
                    node.value += 1.0
            else:
                edge = 0.5 * (decay ** steps)
                node.value += 0.5 + edge if winner == mover else 0.5 - edge
            steps += 1
            node = node.parent
            