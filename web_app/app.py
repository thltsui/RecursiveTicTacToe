#!/usr/bin/env python3
"""Web dashboard for playing Ultimate Tic-Tac-Toe against the AlphaZero AI or another Human.

Run from the project root:
    uv run python -m web_app.app
Or directly:
    cd <project_root> && uv run python web_app/app.py
"""

import eventlet
eventlet.monkey_patch()

import sys
import os
import glob
import json
import copy
import uuid
from cachetools import TTLCache

# Ensure the project root is on sys.path so we can import game modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from importlib import import_module

# Configure torch for thread safety and optimal inference performance
torch.set_num_threads(2)

board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
model_factory_mod = import_module('02_network.model_factory')
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

import redis
redis_url = os.environ.get('REDIS_URL')
redis_client = redis.from_url(redis_url) if redis_url else None

socketio = SocketIO(app, cors_allowed_origins="*", message_queue=redis_url)

# ── Global state ────────────────────────────────────────────────────────────

network = None
network_metadata = {}
# The lightweight Transformer leaves enough latency headroom to give hard mode
# a materially deeper search while keeping easy and medium responsive.
DIFFICULTY_SIMS = {'easy': 80, 'medium': 200, 'hard': 800}
# Return a fast first analysis after the AI moves, then let the browser request
# the full difficulty budget without keeping the move response blocked.
ANALYSIS_PREVIEW_SIMS = 100
# A bounded mean backed by only one or two leaves looks deceptively precise.
# The API still returns every Q value for diagnostics, but tells the UI to hide
# sparse estimates until this many visits have accumulated.
MIN_Q_DISPLAY_VISITS = 10


def exploration_noise_for_difficulty(difficulty: str) -> bool:
    """Use deployment-time root noise only for intentionally easier play."""
    return difficulty == 'easy'

def c_puct_for_phase(move_count: int) -> float:
    """Exploration boost at both ends of the game -- helps PUCT surface
    low-prior-but-critical branches instead of concentrating hard on whatever
    currently has the best Q.

    Late game (move_count >= 45): boosted to 3.0. See debugging notes: the
    AI's value estimate correctly recognizes a losing position one move too
    late, traced to PUCT under-exploring low-prior-but-critical branches at
    c_puct=1.0. The tree is narrow here (few legal moves left), so a strong
    boost still concentrates visits usefully rather than spreading them too
    thin.

    Early game (move_count < 5, matching the training-side Dirichlet-noise
    boost window used in self-play): boosted to 2.0. Root search was found
    to almost never seriously re-examine certain openings (near-zero visits
    across most of a training run -- see debugging notes), the same
    prior-starvation mechanism as the endgame case. A more moderate boost is
    used here than late-game because branching factor is much higher at the
    opening (up to 81 legal moves vs. a handful late-game), so the same fixed
    simulation budget divided across a much wider set of candidates needs a
    gentler nudge -- pushing all the way to 3.0 here risked diluting visits
    too thin to meaningfully re-rank anything.

    All other plies: standard c_puct=1.0. This is an inference-only search
    tweak -- it changes how visits get allocated during MCTS, not what the
    underlying network believes, so it does not by itself fix miscalibrated
    raw value/policy estimates. See training-side fixes (dirichlet_epsilon
    boost + forced_opening_fraction) for that."""
    if move_count >= 45:
        return 3.0
    if move_count < 5:
        return 2.0
    return 1.0

# Local mapping to quickly find a user's room upon disconnect
sid_to_room = {}

# TTL Cache for multiplayer games (fallback if no Redis)
rooms_cache = TTLCache(maxsize=2048, ttl=7200)

def get_room(room_id):
    if redis_client:
        data = redis_client.get(f"room:{room_id}")
        if data:
            room_data = json.loads(data)
            room_data['state'] = dict_to_state(room_data['state'])
            return room_data
        return None
    return rooms_cache.get(room_id)

def save_room(room_id, room_data):
    if redis_client:
        # We need to serialize the GameState to dict before saving
        data_to_save = room_data.copy()
        data_to_save['state'] = state_to_dict(room_data['state'])
        redis_client.setex(f"room:{room_id}", 7200, json.dumps(data_to_save))
    else:
        rooms_cache[room_id] = room_data

def load_network(checkpoint_path=None):
    """Load the best available network (Transformer or CNN).

    Priority order:
      1. Explicit checkpoint_path argument.
      2. checkpoints/transformer_best.pt   (latest Transformer model).
      3. checkpoints/best_ever_model.pt    (legacy CNN champion).
      4. Legacy search in known checkpoint dirs.
      5. Any *.pt under checkpoints/.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        cp_path = checkpoint_path
    else:
        cp_path = None

        # 1. Prefer Transformer model if available
        transformer_best = os.path.join(PROJECT_ROOT, 'checkpoints/transformer_best.pt')
        if os.path.exists(transformer_best):
            cp_path = transformer_best

        # 2. Fall back to legacy CNN champion
        if not cp_path:
            best_ever = os.path.join(PROJECT_ROOT, 'checkpoints/best_ever_model.pt')
            if os.path.exists(best_ever):
                cp_path = best_ever

        if not cp_path:
            # Legacy search in named checkpoint dirs
            search_dirs = [
                os.path.join(PROJECT_ROOT, 'checkpoints/large_v4_deep_value'),
                os.path.join(PROJECT_ROOT, 'checkpoints/large_v5_fixed_mcts'),
                os.path.join(PROJECT_ROOT, 'checkpoints/large_v3_pure_self_play'),
            ]
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

    global network_metadata

    print(f"Loading checkpoint: {cp_path}")
    checkpoint = torch.load(cp_path, weights_only=False, map_location='cpu')
    iteration = checkpoint.get('iteration', '?')

    net, model_config = model_factory_mod.create_network_from_checkpoint(checkpoint)
    net.eval()
    if not bool(net.value_head.wdl_is_native.item()):
        print("  WARNING: migrated legacy scalar value head; W/D/L is uncalibrated")
    depth_label = 'layers' if model_config.architecture == 'transformer' else 'blocks'
    metadata_source = 'metadata' if checkpoint.get('model_config') else 'legacy inference'
    print(
        f"  Loaded {model_config.architecture.upper()}: "
        f"{model_config.channels}ch x {model_config.num_blocks}{depth_label}, "
        f"iteration {iteration} ({metadata_source})"
    )

    network_metadata = {
        'checkpoint': os.path.basename(cp_path),
        'architecture': model_config.architecture,
        'channels': model_config.channels,
        'layers': model_config.num_blocks,
        'iteration': iteration,
        'parameters': sum(parameter.numel() for parameter in net.parameters()),
        'wdl_calibrated': bool(net.value_head.wdl_is_calibrated.item()),
        'release_status': checkpoint.get(
            'release_status', 'experimental_portfolio_trial'
        ),
    }

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


def get_analysis(state, sims, add_noise=False):
    """Run network + MCTS and return full analysis payload for a position."""
    # Create an explicit copy of the state for thread safety
    safe_state = state.copy()

    with torch.inference_mode():
        # Raw network output
        net_output = network.predict(safe_state, device='cpu')
        legal_mask = get_legal_move_mask(safe_state)

        # Policy probabilities (softmax + legal mask)
        policy_probs = apply_legal_mask(net_output.policy_logits, legal_mask)
        policy_list = policy_probs.detach().cpu().tolist()

        # Opponent policy
        opp_probs = apply_legal_mask(net_output.opp_policy_logits, legal_mask)
        opp_list = opp_probs.detach().cpu().tolist()

        # Value outputs
        raw_wdl = net_output.wdl_probs.detach().cpu().numpy().astype(np.float64)
        score_margin = net_output.score_margin.item()

        # Ownership (9 sub-boards)
        ownership = net_output.ownership.detach().cpu().tolist()

        # Run MCTS
        c_puct = c_puct_for_phase(state.move_count)
        epsilon = 0.15 if add_noise else 0.0
        root = search_mod.run_mcts(
            safe_state, network,
            num_simulations=sims,
            dirichlet_epsilon=epsilon,
            device='cpu',
            c_puct=c_puct,
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

    # Report the W/D/L estimate for the move MCTS recommends, rather than an
    # average over deliberately explored bad moves.  This is the relevant
    # strong-play continuation for the player to move.
    evaluation_source = 'raw_network'
    search_wdl = raw_wdl
    if top_moves_raw:
        best_move = top_moves_raw[0][0]
        best_wdl = root.Q_WDL.get(best_move)
        if best_wdl is not None and float(np.sum(best_wdl)) > 0:
            search_wdl = np.asarray(best_wdl, dtype=np.float64)
            search_wdl = search_wdl / np.sum(search_wdl)
            evaluation_source = 'mcts_recommended_move'

    search_win_value = float(search_wdl[0] - search_wdl[2])
    value_head = network.value_head
    if not bool(value_head.wdl_is_native.item()):
        calibration_status = 'legacy_migrated_uncalibrated'
    elif bool(value_head.wdl_is_calibrated.item()):
        calibration_status = 'temperature_scaled'
    else:
        calibration_status = 'uncalibrated'

    return {
        'policy_probs': policy_list,
        'opp_policy_probs': opp_list,
        'mcts_visits': mcts_visits,
        'mcts_q_values': mcts_q_values,
        # Kept for API compatibility; this is P(win) - P(loss), not P(win).
        'win_value': search_win_value,
        'wdl_probs': {
            'win': float(search_wdl[0]),
            'draw': float(search_wdl[1]),
            'loss': float(search_wdl[2]),
        },
        'raw_wdl_probs': {
            'win': float(raw_wdl[0]),
            'draw': float(raw_wdl[1]),
            'loss': float(raw_wdl[2]),
        },
        'evaluation_source': evaluation_source,
        'calibration_status': calibration_status,
        'score_margin': float(score_margin),
        # Ownership is intentionally raw: P(current player owns board) is not
        # complementary across perspectives when a sub-board can draw.
        'ownership': ownership,
        'top_moves': top_moves,
        'total_sims': int(total_visits),
        'target_sims': int(sims),
        'is_preview': False,
        'min_q_display_visits': MIN_Q_DISPLAY_VISITS,
    }, root


# ── REST Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/model_info')
def model_info():
    """Describe the exact checkpoint serving this process."""
    return jsonify(network_metadata)


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game. Optionally set difficulty."""
    data = request.get_json(silent=True) or {}
    difficulty = data.get('difficulty', 'medium')
    sims = DIFFICULTY_SIMS.get(difficulty, 1000)

    human_player = data.get('human_player', 1)  # 1 = human goes first
    state = create_initial_state()

    response = state_to_dict(state)
    response['difficulty'] = difficulty
    response['num_sims'] = sims

    # If human is player -1, AI goes first
    if human_player == -1:
        ai_result = do_ai_move(state, sims, difficulty)
        response = ai_result
        response['difficulty'] = difficulty
        response['num_sims'] = sims

    return jsonify(response)


@app.route('/api/move', methods=['POST'])
def human_move():
    """Human makes a move, then AI responds."""
    data = request.get_json()
    state = dict_to_state(data['state'])
    move = int(data['move'])
    difficulty = data.get('difficulty', 'medium')
    sims = DIFFICULTY_SIMS.get(difficulty, 1000)

    # Validate move
    legal = get_legal_moves(state)
    if move not in legal:
        return jsonify({'error': 'Illegal move'}), 400

    # Apply human move
    state = apply_move(state, move)

    if state.is_terminal:
        return jsonify(state_to_dict(state))

    # AI responds
    ai_result = do_ai_move(state, sims, difficulty)
    return jsonify(ai_result)


@app.route('/api/analyze', methods=['POST'])
def analyze_position():
    """Analyze a position without making a move. Returns full analysis."""
    data = request.get_json()
    state = dict_to_state(data['state'])
    difficulty = data.get('difficulty', 'medium')
    sims = DIFFICULTY_SIMS.get(difficulty, 1000)

    if state.is_terminal:
        return jsonify({'error': 'Cannot analyze terminal position'}), 400

    analysis, _ = get_analysis(state, sims)
    result = state_to_dict(state)
    result['analysis'] = analysis
    return jsonify(result)


def do_ai_move(state, sims, difficulty='medium'):
    """AI decides its move, then analyzes the resulting position for the human.

    Flow:
    1. Run MCTS from AI's perspective → pick AI's best move
    2. Apply AI's move → new state (now it's human's turn)
    3. If game is not over, run a lightweight analysis on the new position
       (from human's perspective) so the frontend shows recommended moves instantly
    4. Return: new state + AI's move + human-perspective analysis
    """
    # Step 1: AI decides its move (internal, not surfaced)
    # Easy intentionally explores. Medium and Hard remain deterministic so
    # their larger simulation budgets translate into stronger play.
    ai_analysis, root = get_analysis(
        state,
        sims,
        add_noise=exploration_noise_for_difficulty(difficulty),
    )
    ai_move = search_mod.select_move(root, temperature=0.0)

    # Step 2: Apply AI's move
    state = apply_move(state, ai_move)

    result = state_to_dict(state)
    result['ai_move'] = int(ai_move)

    # Step 3: Analyze the resulting position from the HUMAN's perspective.
    # Return a fast preview so the move response stays responsive.  The client
    # follows it with /api/analyze when the selected difficulty has a deeper
    # budget, and replaces the preview only if the board is still unchanged.
    if not state.is_terminal:
        preview_sims = min(sims, ANALYSIS_PREVIEW_SIMS)
        human_analysis, _ = get_analysis(state, preview_sims)
        human_analysis['target_sims'] = int(sims)
        human_analysis['is_preview'] = preview_sims < sims
        result['analysis'] = human_analysis
    else:
        # Game is over — no analysis needed, but include final eval
        result['analysis'] = ai_analysis

    return result


# ── Socket.IO Event Handlers ──────────────────────────────────────────────────

@socketio.on('join')
def on_join(data):
    room_id = data.get('room_id')
    sid = request.sid

    if not room_id:
        # Create a new unique room ID
        room_id = str(uuid.uuid4())[:8]

    # Initialize room if it doesn't exist
    room = get_room(room_id)
    if not room:
        room = {
            'state': create_initial_state(),
            'player_x': None,
            'player_o': None,
            'spectators': [],
            'move_history': []
        }

    join_room(room_id)

    # Assign roles
    if room['player_x'] == sid or room['player_o'] == sid:
        # Rejoining
        role = 1 if room['player_x'] == sid else -1
    elif not room['player_x']:
        room['player_x'] = sid
        role = 1
    elif not room['player_o']:
        room['player_o'] = sid
        role = -1
    else:
        room['spectators'].append(sid)
        role = 0  # Spectator

    save_room(room_id, room)
    sid_to_room[sid] = room_id

    # Emit role assignment to joining user
    emit('room_joined', {
        'room_id': room_id,
        'role': role,
        'state': state_to_dict(room['state']),
        'player_x_present': room['player_x'] is not None,
        'player_o_present': room['player_o'] is not None
    })

    # Notify others in the room
    emit('player_joined', {
        'role': role,
        'player_x_present': room['player_x'] is not None,
        'player_o_present': room['player_o'] is not None
    }, to=room_id, include_self=False)

    # If both players are joined, notify room that game is starting/ready
    if room['player_x'] and room['player_o']:
        emit('game_ready', {
            'state': state_to_dict(room['state'])
        }, to=room_id)


@socketio.on('make_move')
def on_make_move(data):
    room_id = data.get('room_id')
    move = data.get('move')
    sid = request.sid

    room = get_room(room_id)
    if not room_id or not room:
        emit('error', {'message': 'Room not found'})
        return
    state = room['state']

    if state.is_terminal:
        emit('error', {'message': 'Game has already ended'})
        return

    # Determine role of the sender
    if sid == room['player_x']:
        sender_role = 1
    elif sid == room['player_o']:
        sender_role = -1
    else:
        emit('error', {'message': 'Spectators cannot make moves'})
        return

    # Verify turn
    if state.current_player != sender_role:
        emit('error', {'message': "It is not your turn"})
        return

    # Verify move validity
    legal = get_legal_moves(state)
    if move not in legal:
        emit('error', {'message': 'Illegal move'})
        return

    # Apply move
    new_state = apply_move(state, move)
    room['state'] = new_state
    room['move_history'].append(move)
    save_room(room_id, room)

    # Broadcast state update to everyone in the room
    emit('state_update', {
        'state': state_to_dict(new_state),
        'last_move': move
    }, to=room_id)


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    room_id = sid_to_room.get(sid)
    if not room_id:
        return
        
    room = get_room(room_id)
    if not room:
        return
        
    updated = False
    if room['player_x'] == sid:
        room['player_x'] = None
        updated = True
        emit('player_left', {'role': 1}, to=room_id, include_self=False)
    elif room['player_o'] == sid:
        room['player_o'] = None
        updated = True
        emit('player_left', {'role': -1}, to=room_id, include_self=False)
    elif sid in room['spectators']:
        room['spectators'].remove(sid)
        updated = True

    if updated:
        save_room(room_id, room)
        
    # Clean up local mapping
    if sid in sid_to_room:
        del sid_to_room[sid]


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    network = load_network()
    if network is None:
        print("Failed to load network. Exiting.")
        sys.exit(1)
    print("\n✅ Server ready at http://[::]:5001")
    print("   Open in your browser to play!\n")
    socketio.run(app, host='::', port=5001, debug=False)
