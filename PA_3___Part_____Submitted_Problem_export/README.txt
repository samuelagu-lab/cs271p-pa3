CS 271P — PA 3D-s : Monte-Carlo Tree Search Agent  (SUBMISSION)
=============================================================

WHAT TO SUBMIT
  agent.py   —  your Monte-Carlo agent, this ONE file.
  (Do not edit or submit game.py or mcts_framework.py — we grade with our own copies.)

THE GAME:  "King of the Hill" chess on a 6x6 board (no bishops).
  Simplified piece moves. A side WINS in EITHER of two ways:
    * capturing the enemy king, OR
    * moving its own king onto one of the four CENTRAL squares -- (2,2), (2,3), (3,2), (3,3).
  The game ends the instant either happens.  A long game with neither is a draw.
  Moves into attack are allowed: the engine uses king capture, not check/checkmate.
  There is no castling, double pawn step, or en passant. Pawns can promote to queen, rook, or knight.

  HEADS UP: this is NOT the game from the PA 3D notebook, which plays ordinary Los Alamos chess.
  The second win condition changes what matters here, so do not assume an idea that worked in the
  notebook still works.  Think about how THIS game is won.

WHAT YOU'RE GIVEN
  agent.py            <-  YOU EDIT THIS.  A complete but deliberately NAIVE MCTS agent: random
                          selection, all-children expansion, random play-out, standard backprop.
  mcts_framework.py   <-  the MCTS loop + game-playing code (provided).  It calls the four phase
                          methods your agent defines.  Do not edit.
  game.py             <-  the game your agents play: King-of-the-Hill Los Alamos chess, 6x6
                          (provided).  Do not edit.
  test_local.py       <-  public baseline games, or one board and your agent's chosen move.
  PA 3 Agent Lab.ipynb <-  optional board, move, and replay support for your agent.py.
  requirements-visual.txt <- optional packages for the Agent Lab; the command-line test and
                             submitted agent still use only Python's standard library.
  A_Survey_of_Monte_Carlo_Tree_Search_Methods.pdf   <-  reference reading (Browne et al., 2012).

THE TASK
  Make your agent as strong as you can by rewriting any of the four phase methods in agent.py
  (select / expand / simulate / backpropagate) and adding any helpers or state you like.  How you
  improve each phase is up to you -- draw on what you know about MCTS and about this game.  Your
  lecture notes and the shipped survey (Browne et al., 2012) are good places to start.

HOW TO RUN
  # Only Python's standard library is required (no packages to install).
  python3 test_local.py          # plays your agent vs the naive baseline (~1 minute)
  python3 test_local.py 0.05     # smaller time budget = faster, for quick iteration
  python3 test_local.py --final  # more public-baseline games, with a fresh printed seed
  python3 test_local.py --final --seed 12345  # repeat those seeded opening inputs
  python3 test_local.py --list   # numbered openings and named illustrative positions
  python3 test_local.py --board 5  # inspect one move on the fifth listed board
  python3 test_local.py --board "near hill, White"  # select a position by name

  The plain command keeps the original quick test: six openings at depth 4, colors swapped.
  --final uses 36 mixed-depth openings, colors swapped (72 games), against the PUBLIC NAIVE
  baseline only. It does not contain our three stronger opponents and cannot predict your
  tournament grade. Its seed is drawn locally, not taken from the final grader. --seed N repeats
  a printed local sample's inputs and random streams; timed searches may still choose differently.

  --list shows the six quick openings (36 with --final), followed by nine named public positions.
  It works even if agent.py is missing or broken. --board accepts a listed number or a quoted,
  case-insensitive name. It shows coordinates, legal moves, any immediate winning moves, your
  chosen move, its legality, and the resulting board and outcome. --list and --board are alternatives.
  To revisit a sampled final opening, use --final --seed N --board K with its printed seed and number.
  A terminal position reports its existing outcome without loading or running your agent.
  Each single-move search has a wall-time limit of at least five seconds to catch a stuck agent.
  PASS means the chosen move is legal; it is not a best-move or grade guarantee. Missing an
  available immediate win is an observation, not a failure of MCTS phase correctness. A timed,
  stochastic search has no single required move on every position.
  The optional Agent Lab visualizes the same local agent.py; edit and submit only agent.py.

HOW YOU'RE GRADED
  Your agent plays a tournament against four of our hidden agents; your score is how often you
  BEAT each one (beat% counts a draw as half):
     45%    in proportion to how often you beat our NAIVE RANDOM agent
    +20%    in proportion to how often you beat a STRONGER agent
    +25%    in proportion to how often you beat a MUCH STRONGER agent
    +15%    in proportion to how often you beat our STRONGEST agent       (up to 105% total)
  IMPORTANT: our agents get 25% MORE thinking time per move than yours.  So just copying an
  equal agent tends to lose — to win you must be smarter, or more efficient, per sample.

TWO GRADING RUNS
  Before the deadline, the regular grader gives feedback on a fresh random sample each run.
  Once all submissions and extensions are in, we regrade everyone with the FINAL grader:
  more positions from the same opening-depth distributions, shared across students using one
  fixed seed. The game, thinking-time calibration, handicap, opponents, and weights stay the same.
  The FINAL run's score is your assignment grade. The assignment is out of 50 points, with a
  maximum of 52.5 (105%). Every run identifies its profile, seed, and sample sizes in the results.
  A fixed seed repeats the opening inputs and random streams. CPU-time searches can complete
  different iteration counts, so it does not guarantee identical scores on every repeat run.
