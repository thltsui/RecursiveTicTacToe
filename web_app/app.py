#!/usr/bin/env python3
"""Web dashboard for playing Ultimate Tic-Tac-Toe against the AlphaZero AI.

Run from the project root:
    uv run python -m web_app.app
Or directly:
    cd <project_root> && uv run python web_app/app.py
"""

import sys
import os
import glob
import json

# Ensure the project root is on sys.path so we can import game modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from importlib import import_module

board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
network_mod = import_module('02_network.network')
search_mod = import_module('03_mcts.search')
policy_head_mod = import_module('02_network.policy_head')

GameState = board_mod.GameState
create_initial_state = board_mod.create_initial_state
decode_move = board_mod.decode_move
encode_move = board_mod.encode_move
get_legal_moves = rules_mod.get_legal_moves
get_legal_move_mask = rules_mod.get_legal_move_mask
apply_move = rules_mod.apply_move
apply_legal_mask = policy_head_mod.apply_legal_mask

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
        # Prefer v5 fixed MCTS (latest), then v3, then generic
        search_dirs = [
            os.path.join(PROJECT_ROOT, 'checkpoints/large_v5_fixed_mcts'),
            os.path.join(PROJECT_ROOT, 'checkpoints/large_v3_pure_self_play'),
        ]
        cp_path = None
        for d in search_dirs:
            if os.path.isdir(d):
                best = os.path.join(d, 'best_model.pt')
                if os.path.exists(best):
                    cp_path = best
                    break
                ckpts = sorted([f for f in os.listdir(d) if f.endswith('.pt') and 'checkpoint' in f])
                if ckpts:
                    cp_path = os.path.join(d, ckpts[-1])
                    break

        if not cp_path:
            pts = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'checkpoints/*.pt')))
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


def get_analysis(state):
    """Run network + MCTS and return full analysis payload for a position."""
    # Raw network output
    net_output = network.predict(state, device='cpu')
    legal_mask = get_legal_move_mask(state)

    # Policy probabilities (softmax + legal mask)
    policy_probs = apply_legal_mask(net_output.policy_logits, legal_mask)
    policy_list = policy_probs.detach().cpu().tolist()

    # Opponent policy
    opp_probs = apply_legal_mask(net_output.opp_policy_logits, legal_mask)
    opp_list = opp_probs.detach().cpu().tolist()

    # Value outputs
    win_value = net_output.win_value.item()
    score_margin = net_output.score_margin.item()

    # Ownership (9 sub-boards)
    ownership = net_output.ownership.detach().cpu().tolist()

    # Run MCTS
    root = search_mod.run_mcts(
        state, network,
        num_simulations=current_sims,
        dirichlet_epsilon=0.0,
        device='cpu',
    )

    # MCTS visit counts and Q-values for all 81 cells
    visits_dict = root.get_visit_counts()
    total_visits = sum(visits_dict.values()) if visits_dict else 1

    mcts_visits = [0] * 81
    mcts_q_values = [0.0] * 81
    for move_idx in range(81):
        if move_idx in root.N:
            mcts_visits[move_idx] = root.N[move_idx]
        if move_idx in root.Q:
            mcts_q_values[move_idx] = root.Q[move_idx]

    # Top moves
    top_moves_raw = sorted(visits_dict.items(), key=lambda x: -x[1])[:8]
    top_moves = []
    for m, v in top_moves_raw:
        sb, cell = decode_move(m)
        top_moves.append({
            'move': int(m),
            'sub_board': int(sb),
            'cell': int(cell),
            'visits': int(v),
            'pct': round(100 * v / total_visits, 1) if total_visits > 0 else 0,
            'q_value': round(root.Q.get(m, 0.0), 4),
            'prior': round(policy_list[m], 4),
        })

    return {
        'policy_probs': policy_list,
        'opp_policy_probs': opp_list,
        'mcts_visits': mcts_visits,
        'mcts_q_values': mcts_q_values,
        'win_value': float(win_value),
        'score_margin': float(score_margin),
        'ownership': ownership,
        'top_moves': top_moves,
        'total_sims': int(total_visits),
    }, root


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


@app.route('/api/analyze', methods=['POST'])
def analyze_position():
    """Analyze a position without making a move. Returns full analysis."""
    data = request.get_json()
    state = dict_to_state(data['state'])

    if state.is_terminal:
        return jsonify({'error': 'Cannot analyze terminal position'}), 400

    analysis, _ = get_analysis(state)
    result = state_to_dict(state)
    result['analysis'] = analysis
    return jsonify(result)


def do_ai_move(state):
    """AI decides its move, then analyzes the resulting position for the human.

    Flow:
    1. Run MCTS from AI's perspective → pick AI's best move
    2. Apply AI's move → new state (now it's human's turn)
    3. If game is not over, run a SECOND analysis on the new position
       (from human's perspective) so the frontend shows the human's options
    4. Return: new state + AI's move + human-perspective analysis
    """
    # Step 1: AI decides its move (internal, not surfaced)
    ai_analysis, root = get_analysis(state)
    ai_move = search_mod.select_move(root, temperature=0.0)

    # Step 2: Apply AI's move
    state = apply_move(state, ai_move)

    result = state_to_dict(state)
    result['ai_move'] = int(ai_move)

    # Step 3: Analyze the resulting position from the HUMAN's perspective
    if not state.is_terminal:
        human_analysis, _ = get_analysis(state)
        result['analysis'] = human_analysis
    else:
        # Game is over — no analysis needed, but include final eval
        result['analysis'] = ai_analysis

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
