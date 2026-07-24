#!/bin/bash
# Run from repo root: bash substack_push.sh
set -e
cd "$(dirname "$0")"

# Clear stale git locks (safe — only needed after a crashed process)
find .git -name "*.lock" -delete 2>/dev/null || true

git checkout Substack

git add substack/notebook_1_uttt_engine.ipynb substack/figures/ \
         substack/essay1_what_is_uttt.md \
         substack/essay1b_rl_background.md \
         substack/essay1c_bandits_mcts_intro.md

git -c user.email="drterencetsui@gmail.com" -c user.name="Terence Tsui" \
    commit -m "Substack: figure placeholders + notebook call-outs in essays; regenerate fig4"

git push origin Substack

echo "Done — Substack branch pushed."
