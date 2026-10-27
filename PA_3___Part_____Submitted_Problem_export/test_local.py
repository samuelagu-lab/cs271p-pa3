"""
CS 271P — PA 3D-s : local self-test  (run this to check your agent before submitting)
====================================================================================
Plays YOUR agent (in agent.py) against the **naive baseline** you started from — the same opponent
as the 45%-of-the-grade tier — using the same 25%-more-time handicap the real references get. It does
NOT include our three stronger hidden agents, so a great score here is necessary but not sufficient for
full marks; beating those earns the remaining 60%.

    python3 test_local.py            # ~1 minute at the default time budget
    python3 test_local.py 0.05       # faster / weaker, for quick iteration
    python3 test_local.py --final    # larger public-baseline practice sample; fresh seed
    python3 test_local.py --final --seed 12345  # repeat the printed opening inputs
    python3 test_local.py --list     # numbered openings and illustrative boards
    python3 test_local.py --board 5  # one board: legal moves, your move, resulting board
    python3 test_local.py --board "near hill, White"

The one-board route uses only public game rules. It shows all legal moves and any
immediate winning moves, rather than claiming a unique expected MCTS choice.
Seeds repeat inputs and random streams; CPU-timed choices can still vary.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import secrets
import subprocess
import sys
import time
import types

from mcts_framework import Node, choose_move, play_match, opening_positions, random_playout
from game import LosAlamos, State, other, state_from_pieces

# Student-safe public practice settings. manifest.py checks these against the
# FINAL naive group's count/depth mix, without shipping its seed or hidden tiers.
FINAL_PRACTICE_OPENINGS = 36
FINAL_PRACTICE_PLIES = (4, 8, 12)


class _Naive:
    """The naive baseline you started from (local practice opponent only)."""
    def select(self, node, rng):
        return rng.choice(node.children)

    def expand(self, node, game, rng):
        for m in node.untried:
            node.children.append(Node(game.result(node.state, m), node, m))
        node.untried = []
        return rng.choice(node.children)

    def simulate(self, game, state, rng):
        return random_playout(game, state, rng)

    def backpropagate(self, node, winner, game):
        while node is not None:
            node.visits += 1
            mover = other(node.state.stm)
            node.value += 1.0 if winner == mover else 0.5 if winner is None else 0.0
            node = node.parent


def _board_catalog(game, openings):
    """Public practice openings followed by fixed, illustrative rule cases."""
    position = lambda pieces, side="w", ply=0: state_from_pieces(
        game.W, game.H, pieces, side, ply)
    return [(f"opening #{i}", state) for i, state in enumerate(openings, 1)] + [
        ("initial board", game.initial()),
        ("near hill, White", position({(1, 4): "wK", (5, 0): "bK"})),
        ("near hill, Black", position({(4, 1): "bK", (0, 5): "wK"}, "b")),
        ("king capture, White", position({(0, 5): "wK", (5, 5): "wR", (5, 0): "bK"})),
        ("king capture, Black", position({(5, 0): "bK", (0, 0): "bR", (0, 5): "wK"}, "b")),
        ("promotion, White", position({(0, 1): "wP", (5, 5): "wK", (5, 0): "bK"})),
        ("White already on hill", position({(2, 3): "wK", (5, 0): "bK"}, "b", 1)),
        ("Black already on hill", position({(3, 2): "bK", (0, 5): "wK"}, "w", 1)),
        ("move-cap draw", position({(0, 5): "wK", (5, 0): "bK"}, "w", game.move_cap)),
    ]


def _find_board(catalog, wanted):
    if wanted.isdigit() and 1 <= int(wanted) <= len(catalog):
        return catalog[int(wanted) - 1]
    return next(((name, state) for name, state in catalog
                 if name.casefold() == wanted.casefold()), None)


def _move_label(game, move):
    square = lambda sq: f"{chr(ord('a') + sq[0])}{game.H - sq[1]}"
    return f"{square(move[0])} -> {square(move[1])}" + (f"={move[2]}" if move[2] else "")


def _print_board(game, state):
    print("      " + " ".join(f"{chr(ord('a') + c):^4}" for c in range(game.W)))
    for row in range(game.H):
        cells = []
        for col in range(game.W):
            cell = state.board[row * game.W + col]
            cell = ".." if cell == "." else cell
            cells.append(f"[{cell}]" if (col, row) in game.HILL else f" {cell} ")
        print(f"   {game.H - row}  " + " ".join(cells) + f"   row {row}")
    print(f"  {'White' if state.stm == 'w' else 'Black'} to move; ply {state.ply}.")


def _position_outcome(game, state):
    winner = game.winner(state)
    if winner is not None:
        side = "White" if winner == "w" else "Black"
        reason = "king capture" if game.king_sq(state, other(winner)) is None else "reaching the hill"
        return f"{side} wins by {reason}."
    if state.ply >= game.move_cap:
        return f"Draw: the game's {game.move_cap}-ply cap has been reached."
    if not game.actions(state):
        return "Draw: the side to move has no legal move."
    return "Game continues; neither side has won."


def _board_worker():
    """One fresh, bounded interpreter; stdout carries a single JSON response."""
    request = json.load(sys.stdin)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            # Read current source so a quick same-size edit cannot reuse stale bytecode.
            path = Path(__file__).resolve().with_name("agent.py")
            module = types.ModuleType("agent")
            module.__file__ = str(path)
            sys.modules["agent"] = module
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
            agent = module.Agent()
            game = LosAlamos()
            state = State(tuple(request["board"]), request["stm"], request["ply"])
            started_cpu, started_wall = time.process_time(), time.monotonic()
            move, iterations = choose_move(agent, game, state, request["time"],
                                           random.Random(request["seed"]))
            result = dict(ok=move in game.actions(state), move=move, iterations=iterations,
                          cpu_seconds=time.process_time() - started_cpu,
                          wall_seconds=time.monotonic() - started_wall)
            if not result["ok"]:
                result["error"] = f"Illegal selected move: {move!r}"
                result["move"] = None
    except Exception as error:
        result = dict(ok=False, error=f"{type(error).__name__}: {error}")
    result["agent_output"] = captured.getvalue()[-2000:]
    print(json.dumps(result))


def _show_one_board(game, name, state, time_budget, seed):
    print("=" * 64)
    print(f" Board: {name}   |   local seed {seed}")
    print("=" * 64)
    print("  Coordinates: (col, row); row 0 is the top. Files a-f, ranks 6-1.")
    print("  w/b = White/Black; K/Q/R/N/P = pieces; .. = empty; [ ] = hill.")
    print("  King capture or a king on the hill wins immediately; moves into attack are legal.")
    _print_board(game, state)
    # choose_move does not itself guard terminal positions. Do not load student code here.
    if game.is_terminal(state) or not game.actions(state):
        print("\n  Existing outcome:", _position_outcome(game, state))
        print("  This position is finished; no agent search is needed.")
        return 0
    legal = game.actions(state)
    wins = [move for move in legal if game.winner(game.result(state, move)) == state.stm]
    print(f"\n  Legal moves ({len(legal)}); any one is an allowed answer:")
    for index, move in enumerate(legal, 1):
        print(f"    {index:2}. {_move_label(game, move):14} {move!r}"
              + ("  [wins immediately]" if move in wins else ""))
    print("\n  Immediate winning moves:", ", ".join(_move_label(game, move) for move in wins) or "none")
    print("  These are one-move outcomes, not a solved value for every other move.")
    limit = max(5.0, time_budget * 20 + 2)
    print(f"\n  Asking your agent for one move ({time_budget:g}s CPU budget; {limit:g}s wall safety limit)...")
    request = dict(board=state.board, stm=state.stm, ply=state.ply, time=time_budget, seed=seed)
    try:
        process = subprocess.run([sys.executable, "-c", "import test_local; test_local._board_worker()"],
                                 input=json.dumps(request), capture_output=True, text=True,
                                 cwd=Path(__file__).resolve().parent, timeout=limit)
    except subprocess.TimeoutExpired:
        print("  Agent check failed: the safety deadline stopped the worker. Check that importing,\n"
              "  constructing, and running your agent all finish.")
        return 1
    try:
        result = json.loads(process.stdout)
    except (json.JSONDecodeError, ValueError):
        print(f"  Agent check failed: worker exited with code {process.returncode} or unreadable output.")
        if process.stderr:
            print(process.stderr[-2000:])
        return 1
    if result.get("agent_output"):
        print("  Agent output:", result["agent_output"])
    if not result["ok"]:
        print("  Agent check failed:", result["error"])
        return 1
    raw = result["move"]
    move = (tuple(raw[0]), tuple(raw[1]), raw[2])
    if move not in legal:
        print(f"  Legality: FAIL -- your move {move!r} is outside the legal set above.")
        return 1
    print(f"  Your move: {_move_label(game, move)}   {move!r}")
    print(f"  Legality: PASS; {result['iterations']} iterations; "
          f"search CPU {result['cpu_seconds']:.3f}s, wall {result['wall_seconds']:.3f}s.")
    if wins and move not in wins:
        print("  Your agent chose another legal move instead of an available immediate win.\n"
              "  This is a search-strength observation, not a failed phase/API check.")
    after = game.result(state, move)
    print("\n  Board after your move:")
    _print_board(game, after)
    print("  Outcome:", _position_outcome(game, after))
    print("  A legal move on one board does not establish tournament strength. Timed choices may vary.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Practice against the public naive baseline, not the full graded tournament.")
    parser.add_argument("time", nargs="?", type=float, default=0.10, help="CPU-time budget per move (default 0.10s)")
    parser.add_argument("--final", action="store_true", help="larger mixed-depth public-baseline practice sample")
    parser.add_argument("--seed", type=int, default=None, help="repeat the opening inputs from a printed local seed")
    one_board = parser.add_mutually_exclusive_group()
    one_board.add_argument("--list", action="store_true", help="list numbered openings and illustrative boards")
    one_board.add_argument("--board", metavar="NAME_OR_NUMBER", help="inspect ONE board and your agent's chosen move")
    args = parser.parse_args()
    if args.time <= 0 or not args.time < float("inf"):
        parser.error("time must be a finite positive number")
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be nonnegative")
    t = args.time
    game = LosAlamos()
    # Preserve the original small local test and its positional-time argument.
    n_open = FINAL_PRACTICE_OPENINGS if args.final else 6
    depths = FINAL_PRACTICE_PLIES if args.final else 4
    seed = args.seed if args.seed is not None else secrets.randbits(64) if args.final else 1000
    match_seed = seed + 1 if args.final or args.seed is not None else 7
    openings = opening_positions(game, n=n_open, plies=depths, seed=seed)
    if len(openings) != n_open:
        raise RuntimeError("Opening acceptance screen did not fill the practice sample; try another seed.")
    if args.list or args.board is not None:
        catalog = _board_catalog(game, openings)
        if args.list:
            print(f"Available boards: local seed {seed}; {n_open} sampled openings, depths {depths}.")
            print('Run one with python3 test_local.py --board NUMBER (or its name in quotes).')
            if args.final or args.seed is not None:
                print(f"Repeat opening inputs with {'--final ' if args.final else ''}--seed {seed}; timed choices can vary.")
            for index, (name, state) in enumerate(catalog, 1):
                kind = "sampled opening" if index <= n_open else "illustrative rule case"
                print(f"  {index:2}. {name:25} {'White' if state.stm == 'w' else 'Black':5} to move  ({kind})")
            return 0
        found = _find_board(catalog, args.board)
        if found is None:
            parser.error(f"there is no board {args.board!r}; run python3 test_local.py --list")
        return _show_one_board(game, *found, t, seed)
    from agent import Agent  # Listing and finished boards work even with a broken student file.
    print(f"{'Larger final practice' if args.final else 'Quick local practice'}: seed {seed}; "
          f"{len(openings)} openings, depths {depths}. Seeded inputs repeat; timed search results may vary.")
    print(f"Your agent ({t:.3f}s/move)  vs  the naive baseline ({t * 1.25:.3f}s/move, +25%)  "
          f"over {len(openings) * 2} games ...")
    started = time.monotonic()
    incidents = []
    w, d, l = play_match(game, Agent(), t, _Naive(), t * 1.25, openings,
                         seed_base=match_seed, incidents=incidents)
    n = w + d + l
    print(f"  {w} wins, {d} draws, {l} losses   ->   beat-rate {(w + 0.5 * d) / n:.0%}")
    print("  The graded 'beat the naive agent' tier is worth 45%. Our three stronger hidden agents "
          "earn the other 60%.")
    print(f"  Elapsed {time.monotonic() - started:.1f}s. This public-baseline practice is not a predicted grade.")
    if incidents:
        print("  Forfeits: " + "; ".join(incidents))


if __name__ == "__main__":
    sys.exit(main())
