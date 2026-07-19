# Substack Series — Current Status

*(Written here because the chat display is broken in this session)*

## Completed this session

### P2 draft created — post 206509000
**Title:** From Zero to AlphaZero: Teaching an Agent to Play — From Q-Tables to Deep Q-Networks

Three sections with full working Python code:
1. **Tabular Q-Learning** — TabularQAgent, training loop, evaluation. Result: 54% vs random, plateaus immediately (3^81 state space)
2. **Neural DQN** — QNetwork (3 conv layers), ReplayBuffer, DQNAgent with target net + experience replay. Result: 73.5% ceiling after 20k episodes
3. **What fixes it** — each failure mapped to the AlphaZero solution (MCTS targets, tree search, self-play) → bridge to P3

### Settings titles updated on all drafts
All 6 drafts now show "From Zero to AlphaZero:" in both the article title and settings/email subject.

## Full series status

| Post | ID | Status | Title |
|------|-----|--------|-------|
| T0 | — | ✅ Published | Intro to UTTT |
| P1 | 201669322 | ✅ Published | From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python |
| T1 | — | ✅ Published | RL Landscape |
| T3 | — | ✅ Published | The Explore-Exploit Trade-off |
| T2a | 204591895 | 📝 Draft | The Credit-Assignment Problem |
| T2b | 204592058 | 📝 Draft | TD Learning to DQN |
| T2c | 204592175 | 📝 Draft | Policy Gradients to PPO |
| T4  | 204553282 | 📝 Draft | PUCT |
| T5  | 204919138 | 📝 Draft | Inside One MCTS Simulation |
| T6  | 204919315 | 📝 Draft | Randomness by Design |
| P2  | 206509000 | 📝 Draft | Teaching an Agent to Play (NEW) |

| P3  | 206510389 | 📝 Draft | The Full System — Network, MCTS, and Self-Play (NEW) |

## Series complete (all drafts)

All practitioner and theory posts are now drafted. Source essay saved at `substack/essay_p3_alphazero.md`.

## Google Search Console Setup (SEO fix)

**Problem:** tthl.substack.com not appearing in Google search.
**Root cause:** Substack doesn't auto-submit to Google for smaller publications.

**Fix (one-time, ~15 min):**
1. Go to https://search.google.com/search-console → Add property
2. Choose **URL prefix** (NOT Domain — you don't have DNS access to substack.com)
3. Enter `https://tthl.substack.com`
4. Choose **HTML tag** verification → copy the `<meta>` tag
5. In Substack: Settings → Advanced → Custom meta tags → paste it → Save
6. Back in Search Console → click Verify
7. Sitemaps → submit `https://tthl.substack.com/feed`
8. URL Inspection → paste each published post URL → Request Indexing

Do step 8 for all 4 published posts, then again after publishing PUCT.

---

## Latest action (T3 title fix)

T3 published title was "T3: The Slot Machine Problem — Where UCB Comes From" — now fixed to:
**"From Zero to AlphaZero: The Explore-Exploit Trade-off — The Bandit Algorithm Behind AlphaZero"**

Root cause: Substack stores two fields — `draft_title` (editor) and `title` (live). Previous session only updated `draft_title`. Fixed via PUT to `/api/v1/drafts/204915376` + publish call. No new email sent.

**Also fixed (new session):** P1 (post 201669322) — "P1: Building the Ultimate Tic-Tac-Toe Engine in Python" → "From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python". Original email_sent_at (2026-06-14) preserved. No re-email sent.
