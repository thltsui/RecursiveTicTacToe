#!/bin/bash

# Navigate to the correct directory (just in case)
cd "$(dirname "$0")"

echo "Starting Ultimate Tic-Tac-Toe Training..."
echo "Logs will be written to training.log"
echo "You can monitor the progress with: tail -f training.log"

# Run the training script in the background
uv run train.py > training.log 2>&1 &

echo "Training job is now running in the background (PID $!)."
