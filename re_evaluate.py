import torch
from importlib import import_module

trainer_mod = import_module('04_training.trainer')
network_mod = import_module('02_network.network')
arena_mod = import_module('06_evaluation.arena')

config = trainer_mod.TrainingConfig(channels=192, num_blocks=10)

print("Loading networks...")
net_new = network_mod.UltimateTTTNetwork(channels=config.channels, num_blocks=config.num_blocks)
net_old = network_mod.UltimateTTTNetwork(channels=config.channels, num_blocks=config.num_blocks)

# Load checkpoints
meta_new = trainer_mod.load_checkpoint('checkpoints/large_v4_deep_value/checkpoint_iter00010_elo0.pt', net_new)
meta_old = trainer_mod.load_checkpoint('checkpoints/large_v4_deep_value/checkpoint_iter00005_elo0.pt', net_old)

net_new.eval()
net_old.eval()

print("Running Arena Evaluation: Iteration 10 vs Iteration 5...")
result = arena_mod.run_arena(
    network_new=net_new,
    network_old=net_old,
    num_games=50,
    num_simulations=100,  # Fast eval
    device='cpu'
)

print(f"Vs Iter 5: {result.wins}W/{result.losses}L/{result.draws}D (win_rate={result.win_rate:.3f})")
if result.new_is_better:
    print("Result: PASS! Iteration 10 is better.")
else:
    print("Result: FAIL. Iteration 10 is not better.")
