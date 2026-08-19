"""Interactive Ultimate Tic-Tac-Toe — Play against the AI.

Usage:
    python play.py [--difficulty easy|medium|hard] [--checkpoint PATH]
"""

import sys
import os
import glob
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
from importlib import import_module

board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
viz_mod = import_module('01_game.visualizer')
model_factory_mod = import_module('02_network.model_factory')
search_mod = import_module('03_mcts.search')
trainer_mod = import_module('04_training.trainer')
gradcam_mod = import_module('05_explainability.gradcam')

GameState = board_mod.GameState
create_initial_state = board_mod.create_initial_state
decode_move = board_mod.decode_move
encode_move = board_mod.encode_move
get_legal_moves = rules_mod.get_legal_moves
apply_move = rules_mod.apply_move
render_board_ascii = viz_mod.render_board_ascii
run_mcts = search_mod.run_mcts
select_move = search_mod.select_move
compute_gradcam = gradcam_mod.compute_gradcam

# ── ANSI Colors ──────────────────────────────────────────────────────────────

BOLD = '\033[1m'
DIM = '\033[2m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BG_BLUE = '\033[44m'
BG_GREEN = '\033[42m'

DIFFICULTY_SIMS = {
    'easy': 500,
    'medium': 1000,
    'hard': 2000,
}

# ── Numpad mapping ───────────────────────────────────────────────────────────
# Numpad layout:        Internal index:
#   7 8 9                 0 1 2
#   4 5 6                 3 4 5
#   1 2 3                 6 7 8

NUMPAD_TO_IDX = {7: 0, 8: 1, 9: 2, 4: 3, 5: 4, 6: 5, 1: 6, 2: 7, 3: 8}
IDX_TO_NUMPAD = {v: k for k, v in NUMPAD_TO_IDX.items()}


def idx_to_np(idx):
    """Convert internal index (0-8) to numpad number."""
    return IDX_TO_NUMPAD[idx]


def np_to_idx(np_num):
    """Convert numpad number (1-9) to internal index (0-8)."""
    if np_num not in NUMPAD_TO_IDX:
        raise ValueError(f"{np_num} is not a valid numpad key (use 1-9)")
    return NUMPAD_TO_IDX[np_num]


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗
║         ULTIMATE  TIC-TAC-TOE  vs  AI                ║
║              AlphaZero-style MCTS                     ║
╚══════════════════════════════════════════════════════╝{RESET}
""")


def print_board_guide():
    print(f"""{BOLD}Sub-board layout:{RESET}          {BOLD}Cell layout (within each sub-board):{RESET}
┌───┬───┬───┐              ┌───┬───┬───┐
│ 7 │ 8 │ 9 │              │ 7 │ 8 │ 9 │
├───┼───┼───┤              ├───┼───┼───┤
│ 4 │ 5 │ 6 │              │ 4 │ 5 │ 6 │
├───┼───┼───┤              ├───┼───┼───┤
│ 1 │ 2 │ 3 │              │ 1 │ 2 │ 3 │
└───┴───┴───┘              └───┴───┴───┘
{DIM}(Numbers match your numpad!){RESET}

{DIM}Enter moves as: board cell  (e.g. "5 7" = center board, top-left)
Type "help" for this guide, "moves" to see legal moves, "quit" to exit.{RESET}
""")


def load_network(checkpoint_path=None):
    """Load network from checkpoint or create fresh one."""
    # Try to find a checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        cp_path = checkpoint_path
    else:
        # Look for best checkpoint in checkpoints/
        pts = sorted(glob.glob('checkpoints/*.pt'))
        cp_path = pts[-1] if pts else None

    if cp_path:
        checkpoint = torch.load(cp_path, weights_only=False, map_location='cpu')
        net, model_config = model_factory_mod.create_network_from_checkpoint(checkpoint)
        iteration = checkpoint.get('iteration', '?')
        elo = checkpoint.get('elo', '?')
        depth = 'layers' if model_config.architecture == 'transformer' else 'blocks'
        print(f"{GREEN}Loaded checkpoint: {cp_path}{RESET}")
        print(
            f"{DIM}  Iteration: {iteration}, Elo: {elo}, Network: "
            f"{model_config.architecture} {model_config.channels}ch x "
            f"{model_config.num_blocks}{depth}{RESET}"
        )
        return net, True
    else:
        # Even the untrained fallback uses the same JSON architecture source.
        config = trainer_mod.load_training_config(
            os.path.join(
                os.path.dirname(__file__),
                'configs',
                'training',
                'lite_transformer.json',
            )
        )
        net = model_factory_mod.create_network(config.model_config())
        print(f"{YELLOW}No checkpoint found — using untrained network (random play){RESET}")
        return net, False


def format_legal_moves(state):
    """Format legal moves for display using numpad numbers."""
    legal = get_legal_moves(state)
    if state.active_sub_board == -1:
        # Group by sub-board
        by_sb = {}
        for m in legal:
            sb, cell = decode_move(m)
            by_sb.setdefault(sb, []).append(cell)
        parts = []
        for sb in sorted(by_sb):
            np_sb = idx_to_np(sb)
            cells = ','.join(str(idx_to_np(c)) for c in sorted(by_sb[sb]))
            parts.append(f"board {np_sb}:[{cells}]")
        return ' '.join(parts)
    else:
        np_sb = idx_to_np(state.active_sub_board)
        cells = sorted(idx_to_np(decode_move(m)[1]) for m in legal)
        return f"board {np_sb}, cells: {cells}"


def get_human_move(state):
    """Get and validate human move input."""
    legal = get_legal_moves(state)

    while True:
        try:
            raw = input(f"{BOLD}{GREEN}Your move > {RESET}").strip().lower()

            if raw in ('quit', 'exit', 'q'):
                print(f"\n{DIM}Thanks for playing!{RESET}")
                sys.exit(0)

            if raw == 'help':
                print_board_guide()
                continue

            if raw == 'moves':
                print(f"{DIM}Legal: {format_legal_moves(state)}{RESET}")
                continue

            parts = raw.replace(',', ' ').split()

            if len(parts) == 2:
                # numpad board + numpad cell
                np_sb, np_cell = int(parts[0]), int(parts[1])
                sb = np_to_idx(np_sb)
                cell = np_to_idx(np_cell)
                move = encode_move(sb, cell)
            else:
                print(f"{RED}Enter: board cell (e.g. '5 7' = center board, top-left){RESET}")
                continue

            if move not in legal:
                # Give a helpful error explaining WHY it's illegal
                entered_sb, entered_cell = sb, cell
                np_entered_sb = idx_to_np(entered_sb)
                np_entered_cell = idx_to_np(entered_cell)

                if state.active_sub_board != -1 and entered_sb != state.active_sub_board:
                    # Wrong sub-board
                    np_required = idx_to_np(state.active_sub_board)
                    print(f"{RED}You must play in board {np_required} "
                          f"(you entered board {np_entered_sb}). "
                          f"Try: {np_required} <cell>{RESET}")
                elif state.sub_board_results[entered_sb] != 0:
                    # Sub-board already decided
                    print(f"{RED}Board {np_entered_sb} is already decided. "
                          f"Pick a different board.{RESET}")
                elif state.cells[entered_sb, entered_cell] != 0:
                    # Cell occupied
                    print(f"{RED}Cell {np_entered_cell} in board {np_entered_sb} "
                          f"is already taken. Pick another cell.{RESET}")
                else:
                    print(f"{RED}Illegal move (board {np_entered_sb}, cell {np_entered_cell}).{RESET}")
                continue

            return move

        except ValueError as e:
            print(f"{RED}{e}. Type 'help' for instructions.{RESET}")
        except EOFError:
            print(f"\n{DIM}Goodbye!{RESET}")
            sys.exit(0)


def ai_move(state, network, num_sims, human_player=1):
    """Run MCTS and return AI's chosen move with analysis."""
    print(f"{DIM}AI thinking ({num_sims} simulations)...{RESET}", end='', flush=True)
    t0 = time.time()

    root = run_mcts(
        state, network,
        num_simulations=num_sims,
        dirichlet_epsilon=0.0,  # no exploration noise during play
        device='cpu',
    )

    move = select_move(root, temperature=0.0)
    elapsed = time.time() - t0

    # Analysis
    visits = root.get_visit_counts()
    total_visits = sum(visits.values())
    top_moves = sorted(visits.items(), key=lambda x: -x[1])[:5]

    # Get value estimate from YOUR (human) perspective
    # The network evaluates from the current player's perspective (AI here)
    # Positive = good for AI, so we negate to show human's advantage
    net_output = network.predict(state, device='cpu')
    ai_value = net_output.win_value.item()
    your_advantage = -ai_value  # flip: positive = good for you

    print(f"\r{' ' * 50}\r", end='')  # clear "thinking" line

    sb, cell = decode_move(move)
    np_sb, np_cell = idx_to_np(sb), idx_to_np(cell)
    print(f"{BOLD}{MAGENTA}AI plays: board {np_sb}, cell {np_cell}{RESET}")

    # Show win estimate as a bar from your perspective
    if your_advantage > 0.1:
        est_color = GREEN
        est_label = "You're ahead"
    elif your_advantage < -0.1:
        est_color = RED
        est_label = "AI is ahead"
    else:
        est_label = "Even"
        est_color = YELLOW
    print(f"{DIM}  Time: {elapsed:.1f}s | {est_color}{est_label} ({your_advantage:+.2f}){RESET}")

    print(f"{DIM}  Top moves:{RESET}")
    for m, v in top_moves:
        msb, mc = decode_move(m)
        np_msb, np_mc = idx_to_np(msb), idx_to_np(mc)
        pct = 100 * v / total_visits if total_visits > 0 else 0
        bar = '█' * int(pct / 5)
        marker = ' ◄' if m == move else ''
        print(f"{DIM}    board {np_msb} cell {np_mc}: {v:4d} ({pct:5.1f}%) {bar}{marker}{RESET}")

    # Show Grad-CAM attention map
    try:
        heatmap = compute_gradcam(network, state, target_move=move, device='cpu')
        print()
        print(render_attention_map(state, heatmap, move))
    except Exception:
        pass  # skip if gradcam fails

    return move


def render_attention_map(state, heatmap, move=None):
    """Render Grad-CAM heatmap as ASCII art overlaid on the board.

    Uses block characters and color intensity to show where the AI is looking.
    """
    # Intensity levels: darker = less attention, brighter = more attention
    HEAT_CHARS = ' ░▒▓█'
    # Color gradient: blue (low) -> yellow (mid) -> red (high)
    HEAT_COLORS = [
        '\033[90m',   # dark gray (0.0-0.2)
        '\033[34m',   # blue     (0.2-0.4)
        '\033[33m',   # yellow   (0.4-0.6)
        '\033[91m',   # red      (0.6-0.8)
        '\033[91;1m', # bold red (0.8-1.0)
    ]

    lines = []
    lines.append(f"{BOLD}AI Attention Map:{RESET} {DIM}(brighter = more focus){RESET}")
    lines.append("")

    for block_row in range(3):
        for cell_row in range(3):
            row_parts = []
            for block_col in range(3):
                sb = block_row * 3 + block_col
                cells_str = []
                for cell_col in range(3):
                    cell_idx = cell_row * 3 + cell_col
                    row = block_row * 3 + cell_row
                    col = block_col * 3 + cell_col
                    val = heatmap[row, col].item()

                    # Map value to character and color
                    level = min(int(val * 5), 4)
                    char = HEAT_CHARS[level]
                    color = HEAT_COLORS[level]

                    # Show pieces on top of heatmap
                    piece = state.cells[sb, cell_idx]
                    flat_move = encode_move(sb, cell_idx)
                    if move is not None and flat_move == move:
                        cells_str.append(f"{BOLD}{MAGENTA}*{RESET}")
                    elif piece == 1:
                        cells_str.append(f"{GREEN}X{RESET}")
                    elif piece == -1:
                        cells_str.append(f"{RED}O{RESET}")
                    else:
                        cells_str.append(f"{color}{char}{RESET}")

                row_parts.append(' '.join(cells_str))

            lines.append('  ' + ' | '.join(row_parts))

        if block_row < 2:
            lines.append('  ------+-------+------')

    # Legend
    lines.append("")
    lines.append(f"  {DIM}Legend: {HEAT_COLORS[0]}░{RESET}{DIM}=low  "
                 f"{HEAT_COLORS[2]}▒{RESET}{DIM}=mid  "
                 f"{HEAT_COLORS[4]}█{RESET}{DIM}=high  "
                 f"{BOLD}{MAGENTA}*{RESET}{DIM}=AI's move{RESET}")

    return '\n'.join(lines)


def colorize_board(board_str, state):
    """Add ANSI colors to the ASCII board."""
    # Color X green, O red
    result = board_str
    result = result.replace(' X ', f' {GREEN}X{RESET} ')
    result = result.replace('*X ', f'*{GREEN}X{RESET} ')
    result = result.replace(' X*', f' {GREEN}X{RESET}*')
    result = result.replace(' O ', f' {RED}O{RESET} ')
    result = result.replace('*O ', f'*{RED}O{RESET} ')
    result = result.replace(' O*', f' {RED}O{RESET}*')
    # Highlight active sub-board markers
    result = result.replace('*', f'{YELLOW}*{RESET}')
    return result


def play_game(network, num_sims, human_player=1):
    """Main game loop."""
    state = create_initial_state()

    clear_screen()
    print_banner()
    print_board_guide()

    difficulty_name = [k for k, v in DIFFICULTY_SIMS.items() if v == num_sims]
    difficulty_name = difficulty_name[0] if difficulty_name else f'{num_sims} sims'
    print(f"{BOLD}Difficulty: {CYAN}{difficulty_name}{RESET}")
    print(f"{BOLD}You are: {GREEN}X (player 1){RESET}  |  {BOLD}AI is: {RED}O (player -1){RESET}")
    print(f"{DIM}{'─' * 54}{RESET}\n")

    while not state.is_terminal:
        # Display board
        board_str = render_board_ascii(state)
        print(colorize_board(board_str, state))
        print()

        if state.current_player == human_player:
            # Human turn
            legal = get_legal_moves(state)
            if state.active_sub_board != -1:
                np_sb = idx_to_np(state.active_sub_board)
                cells = sorted(idx_to_np(decode_move(m)[1]) for m in legal)
                print(f"{CYAN}Play in board {np_sb} — cells: {cells}{RESET}")
            else:
                print(f"{CYAN}Free choice — play in any open board{RESET}")
                print(f"{DIM}Type 'moves' to see all legal moves{RESET}")

            move = get_human_move(state)
            sb, cell = decode_move(move)
            print(f"{GREEN}You played: board {idx_to_np(sb)}, cell {idx_to_np(cell)}{RESET}\n")
        else:
            # AI turn
            move = ai_move(state, network, num_sims, human_player)
            print()

        state = apply_move(state, move)

    # Game over
    print(f"\n{'═' * 54}")
    board_str = render_board_ascii(state)
    print(colorize_board(board_str, state))

    if state.winner == human_player:
        print(f"\n{BOLD}{GREEN}{'═' * 54}")
        print(f"  YOU WIN!  Congratulations!")
        print(f"{'═' * 54}{RESET}")
    elif state.winner == -human_player:
        print(f"\n{BOLD}{RED}{'═' * 54}")
        print(f"  AI WINS!  Better luck next time.")
        print(f"{'═' * 54}{RESET}")
    else:
        print(f"\n{BOLD}{YELLOW}{'═' * 54}")
        print(f"  DRAW!  A well-fought game.")
        print(f"{'═' * 54}{RESET}")

    return state.winner


def main():
    parser = argparse.ArgumentParser(description='Play Ultimate TTT against AlphaZero AI')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                        default='medium', help='AI difficulty (default: medium)')
    parser.add_argument('--sims', type=int, default=None,
                        help='Custom MCTS simulations (overrides --difficulty)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    args = parser.parse_args()

    num_sims = args.sims if args.sims else DIFFICULTY_SIMS[args.difficulty]

    clear_screen()
    print_banner()

    network, has_checkpoint = load_network(args.checkpoint)
    network.eval()

    if not has_checkpoint and num_sims > 100:
        print(f"{YELLOW}Note: No trained model — reducing sims to 50 for speed.{RESET}")
        num_sims = 50

    print()

    while True:
        winner = play_game(network, num_sims)

        print()
        try:
            again = input(f"{BOLD}Play again? [Y/n] {RESET}").strip().lower()
            if again in ('n', 'no'):
                break
        except (EOFError, KeyboardInterrupt):
            break

    print(f"\n{DIM}Thanks for playing!{RESET}")


if __name__ == '__main__':
    main()
