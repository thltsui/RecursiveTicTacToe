# Ultimate Tic-Tac-Toe Web App — Revised Simple Implementation Plan

Based on your feedback, we are pivoting to the **simplest possible approach** for a novice to understand. We will **not** rewrite everything. Instead, we will build directly on your existing codebase.

## Why this is simpler for a novice:
1. **No Build Steps**: No Node.js, `npm`, Vite, or compiling required. You just run Python, and it works.
2. **Vanilla JavaScript**: Your existing `main.js` is pure, readable JavaScript. No React hooks, state management, or complex component lifecycles to learn.
3. **Familiar Backend**: You already have a working Flask app in `app.py`. We will just extend it.

---

## 🛠️ Proposed Changes

We will upgrade your current `app.py` and `main.js` to support everything in the original plan (multiplayer, thread safety, robust memory management) without changing the core stack.

### 1. Fix Critical Backend Flaws (Thread Safety & Memory)
We need to ensure the app doesn't crash or leak memory when multiple people play.
- **[MODIFY] `web_app/app.py`**:
  - Add `cachetools.TTLCache` to store game sessions so they automatically expire after 1 hour (prevents memory leaks).
  - Explicitly use `copy.deepcopy(state)` before sending states to the AI MCTS threads (prevents thread-safety crashes).
  - Set `torch.set_num_threads(1)` to prevent CPU thrashing when multiple AI requests arrive.

### 2. Add Multiplayer via WebSockets (Flask-SocketIO)
We will add real-time multiplayer using `Flask-SocketIO`, which integrates seamlessly with your existing Flask app and Vanilla JS.
- **[MODIFY] `web_app/app.py`**:
  - Integrate `SocketIO(app)`.
  - Add socket events: `on('join')`, `on('move')`.
  - Manage "Rooms" so two players can share a link and play against each other.
- **[MODIFY] `web_app/static/main.js`**:
  - Connect to the server using the Socket.IO client library (no npm required, just a `<script>` tag).
  - Listen for real-time board updates from the opponent.
  - Update the UI to show a "Share Link" when starting a multiplayer game.

### 3. Polish the Frontend UI
- **[MODIFY] `web_app/static/index.html` & `web_app/static/style.css`**:
  - Add a "Play vs Human (Online)" button next to the difficulty selector.
  - Add a small UI to copy the multiplayer share link.
  - Add a waiting screen ("Waiting for opponent to join...").

### 4. Simple Deployment (Fly.io)
- **[NEW] `Dockerfile`**: A very simple Dockerfile that just installs Python dependencies and runs the Flask app.
- **[NEW] `fly.toml`**: Configure Fly.io to serve the app, ensuring `auto_stop_machines = false` so active games aren't killed.

---

## Verification Plan
1. **Local Testing (vs AI)**: Run `python web_app/app.py`, open the browser, and verify the AI plays correctly, heatmaps work, and nothing crashes.
2. **Local Testing (Multiplayer)**: Open two browser tabs, click "Play vs Human", share the link to the second tab, and play a game in real-time.
3. **Thread Safety Check**: Rapidly click moves while the AI is thinking to ensure the backend doesn't crash due to state corruption.
