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
network_mod = import_module('02_network.network')
search_mod = import_module('03_mcts.search')
trainer_mod = import_module('04_training.trainer')

GameState = board_mod.GameState
create_initial_state = board_mod.create_initial_state
decode_move = board_mod.decode_move
encode_move = board_mod.encode_move
get_legal_moves = rules_mod.get_legal_moves
apply_move = rules_mod.apply_move
render_board_ascii = viz_mod.render_board_ascii
UltimateTTTNetwork = network_mod.UltimateTTTNetwork
run_mcts = search_mod.run_mcts
select_move = search_mod.select_move

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
    'easy': 25,
    'medium': 100,
    'hard': 400,
}


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
│ 0 │ 1 │ 2 │              │ 0 │ 1 │ 2 │
├───┼───┼───┤              ├───┼───┼───┤
│ 3 │ 4 │ 5 │              │ 3 │ 4 │ 5 │
├───┼───┼───┤              ├───┼───┼───┤
│ 6 │ 7 │ 8 │              │ 6 │ 7 │ 8 │
└───┴───┴───┘              └───┴───┴───┘

{DIM}Enter moves as: sub_board cell  (e.g. "4 0" = center board, top-left)
Or as flat index 0-80      (e.g. "36" = same move, since 4*9+0=36)
Type "help" for this guide, "quit" to exit.{RESET}
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
        # Peek at checkpoint to get network config
        checkpoint = torch.load(cp_path, weights_only=False, map_location='cpu')
        state_dict = checkpoint['network_state_dict']
        # Infer channels from first conv weight shape
        first_key = [k for k in state_dict if 'input_conv.weight' in k][0]
        channels = state_dict[first_key].shape[0]
        # Infer num_blocks from number of trunk blocks
        num_blocks = sum(1 for k in state_dict if '.conv1.weight' in k and 'trunk' in k)

        net = UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
        net.load_state_dict(state_dict)
        iteration = checkpoint.get('iteration', '?')
        elo = checkpoint.get('elo', '?')
        print(f"{GREEN}Loaded checkpoint: {cp_path}{RESET}")
        print(f"{DIM}  Iteration: {iteration}, Elo: {elo}, Network: {channels}ch x {num_blocks}blocks{RESET}")
        return net, True
    else:
        # Fresh random network (smaller for faster play without training)
        net = UltimateTTTNetwork(channels=64, num_blocks=4)
        print(f"{YELLOW}No checkpoint found — using untrained network (random play){RESET}")
        return net, False


def format_legal_moves(state):
    """Format legal moves for display."""
    legal = get_legal_moves(state)
    if state.active_sub_board == -1:
        # Group by sub-board
        by_sb = {}
        for m in legal:
            sb, cell = decode_move(m)
            by_sb.setdefault(sb, []).append(cell)
        parts = []
        for sb in sorted(by_sb):
            cells = ','.join(str(c) for c in sorted(by_sb[sb]))
            parts.append(f"sb{sb}:[{cells}]")
        return ' '.join(parts)
    else:
        cells = [decode_move(m)[1] for m in legal]
        return f"sub-board {state.active_sub_board}, cells: {sorted(cells)}"


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

            if len(parts) == 1:
                # Flat index
                move = int(parts[0])
            elif len(parts) == 2:
                # sub_board cell
                sb, cell = int(parts[0]), int(parts[1])
                move = encode_move(sb, cell)
            else:
                print(f"{RED}Enter: sub_board cell (e.g. '4 0') or flat index (e.g. '36'){RESET}")
                continue

            if move not in legal:
                sb, cell = decode_move(move)
                print(f"{RED}Illegal move (sb={sb}, cell={cell}). ", end='')
                if state.active_sub_board != -1:
                    print(f"You must play in sub-board {state.active_sub_board}.{RESET}")
                else:
                    print(f"That cell is occupied or board is decided.{RESET}")
                continue

            return move

        except ValueError:
            print(f"{RED}Invalid input. Type 'help' for instructions.{RESET}")
        except EOFError:
            print(f"\n{DIM}Goodbye!{RESET}")
            sys.exit(0)


def ai_move(state, network, num_sims):
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

    # Get value estimate
    net_output = network.predict(state, device='cpu')
    win_value = net_output.win_value.item()

    print(f"\r{' ' * 50}\r", end='')  # clear "thinking" line

    sb, cell = decode_move(move)
    print(f"{BOLD}{MAGENTA}AI plays: sub-board {sb}, cell {cell}{RESET} {DIM}(flat: {move}){RESET}")
    print(f"{DIM}  Time: {elapsed:.1f}s | Win estimate: {win_value:+.2f} (AI perspective){RESET}")
    print(f"{DIM}  Top moves:{RESET}")
    for m, v in top_moves:
        msb, mc = decode_move(m)
        pct = 100 * v / total_visits if total_visits > 0 else 0
        bar = '█' * int(pct / 5)
        marker = ' ◄' if m == move else ''
        print(f"{DIM}    sb{msb} c{mc}: {v:4d} ({pct:5.1f}%) {bar}{marker}{RESET}")

    return move


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
                cells = sorted(decode_move(m)[1] for m in legal)
                print(f"{CYAN}Play in sub-board {state.active_sub_board} — cells: {cells}{RESET}")
            else:
                print(f"{CYAN}Free choice — play in any open sub-board{RESET}")
                print(f"{DIM}Type 'moves' to see all legal moves{RESET}")

            move = get_human_move(state)
            sb, cell = decode_move(move)
            print(f"{GREEN}You played: sub-board {sb}, cell {cell}{RESET}\n")
        else:
            # AI turn
            move = ai_move(state, network, num_sims)
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
