#!/usr/bin/env python3
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def analyze_buffer():
    v3_dir = 'checkpoints/large_v3_pure_self_play'
    buffer_path = os.path.join(v3_dir, 'replay_buffer.pt')
    
    if not os.path.exists(buffer_path):
        print(f"Buffer not found at {buffer_path}")
        return
        
    print(f"Loading replay buffer from {buffer_path}...")
    try:
        buffer_data = torch.load(buffer_path, weights_only=True)
    except Exception as e:
        buffer_data = torch.load(buffer_path, weights_only=False)
        
    records = buffer_data['records']
    current_size = len(records)
    
    wins, losses, draws = 0, 0, 0
    recent_wins, recent_losses, recent_draws = 0, 0, 0
    recent_count = min(5000, current_size)
    
    for i, rec in enumerate(records):
        val = rec['value_target'].item() if isinstance(rec['value_target'], torch.Tensor) else rec['value_target']
        if abs(val - 1.0) < 0.1:
            wins += 1
            if i >= current_size - recent_count: recent_wins += 1
        elif abs(val - (-1.0)) < 0.1:
            losses += 1
            if i >= current_size - recent_count: recent_losses += 1
        elif abs(val - (-0.5)) < 0.1 or abs(val - 0.0) < 0.1:
            draws += 1
            if i >= current_size - recent_count: recent_draws += 1
            
    other = current_size - (wins + losses + draws)
    
    print(f"--- Buffer Outcome Distribution (Last {current_size} states) ---")
    print(f"Wins  (+1.0) : {wins} ({(wins/current_size)*100:.1f}%)")
    print(f"Losses(-1.0) : {losses} ({(losses/current_size)*100:.1f}%)")
    print(f"Draws (-0.5) : {draws} ({(draws/current_size)*100:.1f}%)")
    if other > 0:
        print(f"Other values : {other}")
        
    if recent_count > 0:
        print(f"\n--- Recent Trend (Last {recent_count} states) ---")
        print(f"Wins  (+1.0) : {recent_wins} ({(recent_wins/recent_count)*100:.1f}%)")
        print(f"Losses(-1.0) : {recent_losses} ({(recent_losses/recent_count)*100:.1f}%)")
        print(f"Draws (-0.5) : {recent_draws} ({(recent_draws/recent_count)*100:.1f}%)")

if __name__ == '__main__':
    analyze_buffer()
