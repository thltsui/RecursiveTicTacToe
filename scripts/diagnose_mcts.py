#!/usr/bin/env python3
"""Diagnose what moves the model is actually making vs random."""

import sys, os, time, random

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from importlib import import_module

board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
network_mod = import_module('02_network.network')
search_mod = import_module('03_mcts.search')
policy_head_mod = import_module('02_network.policy_head')


def main():
    device = 'cpu'
    num_sims = 200

    # Load model
    cp = torch.load('checkpoints/best_model.pt', weights_only=False, map_location=device)
    sd = cp['network_state_dict']
    channels, num_blocks = 128, 8
    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(sd)
    net.eval()
    net.to(device)
    print(f"Loaded: {cp.get('iteration', '?')} | 128ch x 8blocks\n")

    # Play 1 game AI vs random, print first few moves
    state = board_mod.create_initial_state()
    moves_played = 0
    
    while not state.is_terminal and moves_played < 10:
        print(f"\n--- Move {moves_played+1} | Player {state.current_player} ---")
        sub = "FREE" if state.active_sub_board == -1 else str(state.active_sub_board)
        print(f"  Active sub-board: {sub}")
        # Print macro board
        for r in range(9):
            row = []
            for c in range(9):
                v = state.cells[r, c]
                row.append('X' if v == 1 else ('O' if v == -1 else '.'))
            print(f"  {''.join(row)}")
        
        if state.current_player == 1:
            # AI's turn
            root = search_mod.run_mcts(
                state, net,
                num_simulations=num_sims,
                dirichlet_epsilon=0.0,
                device=device,
            )
            
            # Show top 5 moves by visit count
            visits = root.get_visit_counts()
            sorted_moves = sorted(visits.items(), key=lambda x: x[1], reverse=True)
            print(f"  AI top moves (visit counts):")
            for move, count in sorted_moves[:5]:
                board_idx = move // 9
                cell_idx = move % 9
                print(f"    move={move} (sub-board {board_idx}, cell {cell_idx}) visits={count}")
            
            move = search_mod.select_move(root, temperature=0.0)
            board_idx = move // 9
            cell_idx = move % 9
            print(f"  AI CHOSE: move={move} (sub-board {board_idx}, cell {cell_idx})")
        else:
            # Random's turn
            legal_moves = rules_mod.get_legal_moves(state)
            move = random.choice(legal_moves)
            board_idx = move // 9
            cell_idx = move % 9
            print(f"  RANDOM: move={move} (sub-board {board_idx}, cell {cell_idx}) from {len(legal_moves)} legal moves")
        
        state = rules_mod.apply_move(state, move)
        moves_played += 1
    
    print(f"\n{'='*60}")
    print(f"Game status after {moves_played} moves:")
    print(f"  Terminal: {state.is_terminal}")
    print(f"  Winner: {state.winner}")
    print(f"  Macro board:\n{state}")
    
    if not state.is_terminal:
        # Show what's happening on the mini-boards
        for sb in range(9):
            sub = state.sub_boards[sb]
            print(f"  Sub-board {sb}: owner={state.sub_board_owners[sb]}")
            if sub is not None:
                for r in range(3):
                    row = []
                    for c in range(3):
                        v = sub[r, c]
                        row.append('X' if v == 1 else ('O' if v == -1 else '.'))
                    print(f"    {''.join(row)}")


if __name__ == '__main__':
    main()