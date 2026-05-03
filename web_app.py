#!/usr/bin/env python3
"""Web dashboard for playing Ultimate Tic-Tac-Toe against the AlphaZero AI."""

import sys
import os
import glob
import json

sys.path.insert(0, os.path.dirname(__file__))

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from importlib import import_module

board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
network_mod = import_module('02_network.network')
search_mod = import_module('03_mcts.search')

GameState = board_mod.GameState
create_initial_state = board_mod.create_initial_state
decode_move = board_mod.decode_move
encode_move = board_mod.encode_move
get_legal_moves = rules_mod.get_legal_moves
apply_move = rules_mod.apply_move

import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Global state ────────────────────────────────────────────────────────────

network = None
game_state = None
DIFFICULTY_SIMS = {'easy': 25, 'medium': 100, 'hard': 400}
current_sims = 200


def load_network(checkpoint_path=None):
    """Load the best available network."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        cp_path = checkpoint_path
    else:
        # Try v3 pure self-play first, then fall back
        v3_dir = 'checkpoints/large_v3_pure_self_play'
        if os.path.isdir(v3_dir):
            ckpts = sorted([f for f in os.listdir(v3_dir) if f.endswith('.pt') and 'checkpoint' in f])
            if ckpts:
                cp_path = os.path.join(v3_dir, ckpts[-1])
            else:
                cp_path = None
        else:
            pts = sorted(glob.glob('checkpoints/*.pt'))
            cp_path = pts[-1] if pts else None

    if not cp_path:
        print("ERROR: No checkpoint found!")
        return None

    print(f"Loading checkpoint: {cp_path}")
    checkpoint = torch.load(cp_path, weights_only=False, map_location='cpu')
    state_dict = checkpoint['network_state_dict']

    first_key = [k for k in state_dict if 'input_conv.weight' in k][0]
    channels = state_dict[first_key].shape[0]
    num_blocks = sum(1 for k in state_dict if '.conv1.weight' in k and 'trunk' in k)

    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(state_dict)
    net.eval()

    iteration = checkpoint.get('iteration', '?')
    print(f"  Loaded: {channels}ch x {num_blocks}blocks, iteration {iteration}")
    return net


def state_to_dict(state):
    """Serialize a GameState to a JSON-friendly dict."""
    return {
        'cells': state.cells.tolist(),
        'sub_board_results': state.sub_board_results.tolist(),
        'active_sub_board': int(state.active_sub_board),
        'current_player': int(state.current_player),
        'move_count': int(state.move_count),
        'is_terminal': bool(state.is_terminal),
        'winner': int(state.winner) if state.winner is not None else None,
        'legal_moves': [int(m) for m in get_legal_moves(state)] if not state.is_terminal else [],
    }


def dict_to_state(d):
    """Deserialize a dict back to a GameState."""
    return GameState(
        cells=np.array(d['cells'], dtype=np.int8),
        sub_board_results=np.array(d['sub_board_results'], dtype=np.int8),
        active_sub_board=int(d['active_sub_board']),
        current_player=int(d['current_player']),
        move_count=int(d['move_count']),
        is_terminal=bool(d['is_terminal']),
        winner=int(d['winner']) if d['winner'] is not None else None,
    )


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game. Optionally set difficulty."""
    global current_sims
    data = request.get_json(silent=True) or {}
    difficulty = data.get('difficulty', 'medium')
    current_sims = DIFFICULTY_SIMS.get(difficulty, 200)

    human_player = data.get('human_player', 1)  # 1 = human goes first
    state = create_initial_state()

    response = state_to_dict(state)
    response['difficulty'] = difficulty
    response['num_sims'] = current_sims

    # If human is player -1, AI goes first
    if human_player == -1:
        ai_result = do_ai_move(state)
        response = ai_result

    return jsonify(response)


@app.route('/api/move', methods=['POST'])
def human_move():
    """Human makes a move, then AI responds."""
    data = request.get_json()
    state = dict_to_state(data['state'])
    move = int(data['move'])

    # Validate move
    legal = get_legal_moves(state)
    if move not in legal:
        return jsonify({'error': 'Illegal move'}), 400

    # Apply human move
    state = apply_move(state, move)

    if state.is_terminal:
        return jsonify(state_to_dict(state))

    # AI responds
    return jsonify(do_ai_move(state))


def do_ai_move(state):
    """Run MCTS and apply AI's move."""
    root = search_mod.run_mcts(
        state, network,
        num_simulations=current_sims,
        dirichlet_epsilon=0.0,
        device='cpu',
    )
    ai_move = search_mod.select_move(root, temperature=0.0)

    # Get visit counts for top moves
    visits = root.get_visit_counts()
    total_visits = sum(visits.values())
    top_moves = sorted(visits.items(), key=lambda x: -x[1])[:5]

    # Get value estimate
    net_output = network.predict(state, device='cpu')
    ai_value = net_output.win_value.item()

    # Apply AI move
    state = apply_move(state, ai_move)

    result = state_to_dict(state)
    result['ai_move'] = int(ai_move)
    result['ai_value'] = float(ai_value)
    result['top_moves'] = [
        {
            'move': int(m),
            'sub_board': int(decode_move(m)[0]),
            'cell': int(decode_move(m)[1]),
            'visits': int(v),
            'pct': round(100 * v / total_visits, 1) if total_visits > 0 else 0,
        }
        for m, v in top_moves
    ]
    return result


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    network = load_network()
    if network is None:
        print("Failed to load network. Exiting.")
        sys.exit(1)
    print("\n✅ Server ready at http://localhost:5001")
    print("   Open in your browser to play!\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
