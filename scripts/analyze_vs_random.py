import sys, os
import torch
import random
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from importlib import import_module
board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
network_mod = import_module('02_network.network')
search_mod = import_module('03_mcts.search')

def render_board(state):
    # Print the board nicely
    # 0=empty, 1=X, -1=O
    chars = {0: '.', 1: 'X', -1: 'O'}
    for row in range(9):
        if row > 0 and row % 3 == 0:
            print("-" * 21)
        row_str = ""
        for col in range(9):
            if col > 0 and col % 3 == 0:
                row_str += "| "
            sb = (row // 3) * 3 + (col // 3)
            cell = (row % 3) * 3 + (col % 3)
            val = state.cells[sb, cell]
            row_str += chars[val] + " "
        print(row_str)
    print()

def main():
    checkpoint_path = 'checkpoints/large_v3_pure_self_play/checkpoint_iter00115_elo0.pt'
    device = 'cpu'
    
    cp = torch.load(checkpoint_path, weights_only=False, map_location=device)
    sd = cp['network_state_dict']
    channels = sd['input_conv.weight'].shape[0]
    num_blocks = sum(1 for k in sd if k.startswith('trunk.') and k.endswith('.conv1.weight'))
    
    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(sd)
    net.eval()
    net.to(device)
    
    state = board_mod.create_initial_state()
    
    print("=== AI (X) vs Random (O) ===")
    
    while not state.is_terminal:
        render_board(state)
        print(f"Active sub-board: {state.active_sub_board} (-1 means any)")
        if state.current_player == 1:
            print("--- AI Turn ---")
            root = search_mod.run_mcts(state, net, num_simulations=200, dirichlet_epsilon=0.0, device=device)
            # print top moves and value
            counts = root.get_visit_counts()
            top_moves = sorted(counts.items(), key=lambda x: -x[1])[:5]
            net_out = net.predict(state, device)
            print(f"Network Value Evaluation (from X's perspective): {net_out.win_value.item():.3f}")
            for m, v in top_moves:
                sb, c = board_mod.decode_move(m)
                print(f"  Move (sb={sb}, c={c}) visits={v}")
                
            move = search_mod.select_move(root, temperature=0.0)
            sb, c = board_mod.decode_move(move)
            print(f"AI plays: sb={sb}, cell={c}")
        else:
            print("--- Random Turn ---")
            legal = rules_mod.get_legal_moves(state)
            move = random.choice(legal)
            sb, c = board_mod.decode_move(move)
            print(f"Random plays: sb={sb}, cell={c}")
            
        state = rules_mod.apply_move(state, move)
        print("="*40)
        
    render_board(state)
    print(f"Game over! Winner: {state.winner}")

if __name__ == '__main__':
    main()
