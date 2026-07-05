# Ultimate Tic-Tac-Toe Web App — Architecture & Implementation Plan

## Overview

A web application with two play modes:
1. **vs AI** — play against the trained AlphaZero-style network using MCTS
2. **vs Human (online)** — play against another person via a shared game link, in real time

The existing codebase already has all the logic needed: `GameState`, `run_mcts`, `select_move`, and `UltimateTTTNetwork.predict()`. The web app is a thin layer that exposes these over HTTP/WebSocket and provides a game board UI.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            React Frontend (TypeScript + Vite)        │   │
│  │  Board UI · Mode select · Win probability bar        │   │
│  │  Move history · Share link · MCTS heatmap overlay    │   │
│  └──────────────┬─────────────────────┬─────────────────┘   │
│                 │ REST (vs AI)         │ WebSocket (vs Human)│
└─────────────────│─────────────────────│─────────────────────┘
                  │                     │
┌─────────────────▼─────────────────────▼─────────────────────┐
│                  FastAPI Backend (Python)                    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Game Router │  │  AI Router   │  │  WS Room Manager │  │
│  │  (REST)      │  │  (REST)      │  │  (WebSocket)     │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│  ┌──────▼─────────────────▼────────────────────▼──────────┐ │
│  │              Game Engine Layer                          │ │
│  │  GameState · apply_move · get_legal_moves              │ │
│  │  encode_state · check_winner                           │ │
│  └──────────────────────────────┬──────────────────────── ┘ │
│                                 │                           │
│  ┌──────────────────────────────▼──────────────────────────┐ │
│  │              AI Layer                                   │ │
│  │  UltimateTTTNetwork (loaded at startup)                 │ │
│  │  run_mcts · select_move · win_value (for UI bar)        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Session Store                              │ │
│  │  In-memory dict (single instance)                      │ │
│  │  Redis (multi-instance / production)                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │ Deployed on Fly.io (Docker container)
           │ Model checkpoint bundled or loaded from R2/S3
```

---

## Phase 1 — Game Engine API (REST)

### Game State Serialisation

`GameState` is a Python dataclass with numpy arrays. It must be serialised to JSON for the API. Define a Pydantic model that mirrors it:

```python
class GameStateDTO(BaseModel):
    cells: list[list[int]]          # (9, 9) — int8 → int
    sub_board_results: list[int]    # (9,)
    active_sub_board: int           # -1 = free choice
    current_player: int             # 1 or -1
    move_count: int
    is_terminal: bool
    winner: int | None              # 1, -1, 0 (draw), None (ongoing)
```

Conversion helpers:

```python
def state_to_dto(state: GameState) -> GameStateDTO: ...
def dto_to_state(dto: GameStateDTO) -> GameState: ...
```

### REST Endpoints (vs AI mode)

```
POST  /game/new
      Response: { game_id: str, state: GameStateDTO, legal_moves: list[int] }

POST  /game/{game_id}/move
      Body:    { move_idx: int }
      Response: { state: GameStateDTO, legal_moves: list[int], game_over: bool }

POST  /game/{game_id}/ai-move
      Body:    { simulations?: int }   # default 200
      Response: {
        state: GameStateDTO,
        legal_moves: list[int],
        move_idx: int,
        visit_counts: dict[int, int],  # for heatmap overlay
        win_value: float               # current player's perspective [-1, 1]
      }

GET   /game/{game_id}
      Response: current state + legal moves

DELETE /game/{game_id}
      Cleanup (also auto-expire after 1 hour of inactivity)
```

### Key Implementation Notes

**MCTS runs on CPU** — `UltimateTTTNetwork` with 128 channels and 8 residual blocks is fast enough for CPU inference. At 200 simulations, expect ~1–2 seconds response time. 800 simulations is 5–8 seconds; expose `simulations` as a parameter and let the frontend offer "Fast / Strong" difficulty.

**MCTS is CPU-bound** — FastAPI is async; calling `run_mcts` directly in an async handler blocks the event loop. Wrap it:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@router.post("/game/{game_id}/ai-move")
async def ai_move(game_id: str, body: AIMoveRequest):
    state = session_store.get(game_id)
    loop = asyncio.get_event_loop()
    root = await loop.run_in_executor(
        executor,
        lambda: run_mcts(state, network, num_simulations=body.simulations)
    )
    move_idx = select_move(root, temperature=0.0)  # greedy in play mode
    ...
```

**Model loading** — load the checkpoint once at application startup, not per request:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.network = UltimateTTTNetwork(channels=128, num_blocks=8)
    checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
    app.state.network.load_state_dict(checkpoint["model_state_dict"])
    app.state.network.eval()
    yield
```

**Session store** — a simple `dict[str, GameState]` protected by a lock is fine for a single instance. Use `uuid4()` as game ID.

---

## Phase 2 — Multiplayer via WebSocket

### Room Model

Each online game has:
- A `game_id` (UUID, shared as a link)
- Player 1 slot (first to connect)
- Player 2 slot (second to connect, or AI in vs-AI mode)
- The current `GameState`
- A set of connected WebSocket connections

```python
@dataclass
class GameRoom:
    game_id: str
    state: GameState
    players: dict[str, WebSocket]   # { "p1": ws1, "p2": ws2 }
    player_assignment: dict[str, int]  # { ws_id: 1 or -1 }
    created_at: datetime
```

### WebSocket Message Protocol

All messages are JSON with a `type` field.

**Client → Server:**
```jsonc
{ "type": "join",  "game_id": "abc123" }
{ "type": "move",  "move_idx": 42 }
{ "type": "resign" }
```

**Server → Client:**
```jsonc
{ "type": "joined",  "player": 1, "state": {...} }
{ "type": "waiting" }                  // sent to P1 while waiting for P2
{ "type": "start",   "state": {...} }  // sent to both when P2 joins
{ "type": "state",   "state": {...}, "legal_moves": [...], "last_move": 42 }
{ "type": "game_over", "winner": 1, "state": {...} }
{ "type": "error",   "message": "Not your turn" }
{ "type": "opponent_disconnected" }
```

### WebSocket Endpoint

```
WS /ws/{game_id}
```

On connect: assign player role, send `joined` or `waiting`. On move: validate it's the sender's turn, apply move, broadcast new state to both players. On disconnect: notify opponent, mark room for cleanup.

### Creating a Multiplayer Game

```
POST /online/new
Response: { game_id: str, share_url: str }

# P1 connects: GET /online/{game_id}/join → redirect to WS
# P2 visits share URL → frontend auto-connects to same game_id
```

---

## Phase 3 — Frontend (React + TypeScript)

### Tech Stack

- **Vite** — fast dev server and build tool
- **React 18** + TypeScript
- **Tailwind CSS** — utility-first styling, easy responsive layout
- **Framer Motion** — smooth animations for piece placement and highlights
- **React Query** — for REST calls in vs-AI mode
- **Native WebSocket** (or a tiny wrapper) for multiplayer

No heavy game framework needed — the board is just a 9×9 grid with CSS.

### Component Tree

```
<App>
  <ModeSelect />                  ← landing page: "vs AI" | "vs Human (online)"
  <GameView mode="ai"|"online">
    <BoardLayout>
      <MetaBoard>                 ← 3×3 grid of sub-boards
        <SubBoard idx={0..8}>
          <Cell idx={0..8} />     ← 81 cells total
        </SubBoard>
      </MetaBoard>
      <Sidebar>
        <WinProbabilityBar />     ← live value estimate from AI
        <MoveHistory />
        <MCTSHeatmapToggle />     ← toggle visit-count overlay
        <DifficultySelect />      ← Fast (200 sims) | Strong (800 sims)
        <ShareLink />             ← online mode only
      </Sidebar>
    </BoardLayout>
    <GameOverModal />
  </GameView>
</App>
```

### Board Rendering

The 9×9 grid must visually communicate:
- **Active sub-board**: glowing border or blue highlight on the 3×3 region the current player must play in. When active_sub_board == -1, all sub-boards are highlighted.
- **Won sub-boards**: large X or O overlaid on the 3×3 region, with a coloured background.
- **Drawn sub-boards**: grey/hatched fill.
- **Last move**: subtle indicator (ring or shadow) on the last-played cell.
- **Legal cells**: clickable cells in the active sub-board; all others are non-interactive and slightly dimmed.
- **MCTS heatmap** (optional overlay): colour each legal cell from cool (low visit count) to hot (high visit count), using `visit_counts` from the AI response.

### Game State in React

```typescript
interface GameState {
  cells: number[][];          // 9×9
  subBoardResults: number[];  // 9
  activeSubBoard: number;     // -1 or 0–8
  currentPlayer: number;      // 1 or -1
  moveCount: number;
  isTerminal: boolean;
  winner: number | null;
}

interface UIState {
  gameId: string;
  legalMoves: number[];
  winValue: number;           // [-1, 1], from AI's last eval
  visitCounts: Record<number, number>;
  aiThinking: boolean;
  myPlayer: number | null;    // null = vs AI (human always moves)
}
```

### Move Flow (vs AI)

1. User clicks a cell → validate `move_idx` is in `legalMoves` → POST `/game/{id}/move`
2. Update board state from response
3. If not terminal: POST `/game/{id}/ai-move` (show spinner on board)
4. Update board state + win probability bar from response
5. Repeat

### Move Flow (vs Human / online)

1. User connects via WebSocket, receives player assignment (1 or -1)
2. On state update from server: re-render board, enable/disable cells based on `currentPlayer === myPlayer`
3. User clicks legal cell → send `{ type: "move", move_idx }` over WebSocket
4. Server validates, applies, broadcasts; both clients re-render

---

## Phase 4 — Infrastructure & Deployment

### Recommended Stack

| Component | Choice | Reason |
|---|---|---|
| Backend runtime | Python 3.11, FastAPI + uvicorn | Async, WebSocket support, easy |
| AI inference | PyTorch CPU | No GPU needed for 200–800 sims |
| Session store | In-memory dict (v1) → Redis (v2) | Start simple, scale later |
| Frontend build | Vite + React, output as static files | Fast builds, CDN-friendly |
| Static serving | Nginx sidecar or FastAPI `StaticFiles` | Serve React from same container |
| Containerisation | Docker (single image) | Reproducible, portable |
| Deployment | Fly.io | Free tier, WebSocket support, simple `fly.toml` |
| Model checkpoint | Bundled in Docker image (small model) | No external dependencies at runtime |
| Domain | Custom domain via Fly.io | Optional |

### Docker Image

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy game engine and model code
COPY 01_game/ ./01_game/
COPY 02_network/ ./02_network/
COPY 03_mcts/ ./03_mcts/
COPY checkpoints/best_model.pt ./checkpoints/

# Copy web app backend
COPY web_app/backend/ ./web_app/backend/

# Copy pre-built frontend
COPY web_app/frontend/dist/ ./web_app/frontend/dist/

EXPOSE 8080
CMD ["uvicorn", "web_app.backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### `requirements.txt` additions for the web app

```
fastapi>=0.111
uvicorn[standard]>=0.29
websockets>=12.0
pydantic>=2.0
torch>=2.2          # already a dep
numpy>=1.26         # already a dep
```

### Fly.io Configuration (`fly.toml`)

```toml
app = "uttt-app"
primary_region = "lhr"  # London — closest to target audience

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true

[[vm]]
  memory = "1gb"   # PyTorch needs ~500MB for model + runtime
  cpu_kind = "shared"
  cpus = 2         # 2 CPUs for concurrent MCTS threads
```

### Scaling Considerations

**Single instance is fine initially.** The bottleneck is MCTS CPU time, not memory or network. A shared 2-CPU Fly.io instance handles ~4 concurrent AI games without noticeable slowdown (each MCTS uses one thread via the executor).

**If traffic grows:**
- Add Redis for session storage so multiple instances can share game state
- Run a dedicated `mcts-worker` service with a job queue (e.g. Redis Queue) to decouple AI computation from the HTTP/WebSocket server
- Scale the worker horizontally (each worker handles MCTS; the API server is stateless)

**WebSocket stickiness:** Fly.io routes WebSocket connections to the same machine via session affinity by default. With Redis as session store, any machine can handle any connection.

---

## Phase 5 — AI Latency & Difficulty Tuning

MCTS response time on a 2-CPU shared instance:

| Simulations | Approximate time | Quality |
|---|---|---|
| 50  | ~0.3s | Weak, good for "hint" mode |
| 200 | ~1.0s | Good balance for casual play |
| 400 | ~2.0s | Strong, comfortable wait |
| 800 | ~4–5s | Very strong, slow |

**Recommendation:** default to 200 simulations with a "Strong AI" toggle that uses 800. Display a thinking indicator clearly so the user doesn't feel like it's broken.

The `win_value` from the value head (returned after every AI call) can be displayed as a live win-probability bar, which updates even after the human's move by sending it as part of the `/ai-move` response. This is a nice UX touch that makes the AI feel more transparent.

---

## File Structure

```
web_app/
├── PLAN.md                      ← this document
├── Dockerfile
├── backend/
│   ├── main.py                  ← FastAPI app, lifespan (model loading)
│   ├── routers/
│   │   ├── game.py              ← REST: new, move, ai-move, get, delete
│   │   └── online.py            ← WebSocket: join, move, resign
│   ├── models/
│   │   ├── dto.py               ← GameStateDTO, request/response Pydantic models
│   │   └── room.py              ← GameRoom dataclass for multiplayer
│   ├── services/
│   │   ├── game_service.py      ← state serialisation, apply_move wrappers
│   │   ├── ai_service.py        ← run_mcts wrapper, executor management
│   │   └── session_store.py     ← in-memory dict + cleanup
│   └── requirements.txt
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── App.tsx
        ├── components/
        │   ├── ModeSelect.tsx
        │   ├── MetaBoard.tsx
        │   ├── SubBoard.tsx
        │   ├── Cell.tsx
        │   ├── WinProbabilityBar.tsx
        │   ├── MCTSHeatmap.tsx
        │   ├── MoveHistory.tsx
        │   ├── GameOverModal.tsx
        │   └── ShareLink.tsx
        ├── hooks/
        │   ├── useGameVsAI.ts       ← REST game loop logic
        │   └── useGameOnline.ts     ← WebSocket game loop logic
        ├── types/
        │   └── game.ts              ← GameState, UIState, Message types
        └── utils/
            ├── board.ts             ← decode_move, encode_move in TS
            └── api.ts               ← fetch wrappers for backend endpoints
```

---

## Development Roadmap

### Milestone 1 — Backend + vs AI (1–2 weeks)
- [ ] `backend/services/game_service.py`: GameState ↔ DTO serialisation
- [ ] `backend/services/session_store.py`: in-memory sessions with TTL
- [ ] `backend/services/ai_service.py`: async wrapper for `run_mcts`
- [ ] `backend/routers/game.py`: REST endpoints
- [ ] `backend/main.py`: app setup, model loading at startup
- [ ] Manual testing via curl / Postman
- [ ] Basic React board (no styling) that can play vs AI end-to-end

### Milestone 2 — Frontend polish (1 week)
- [ ] Full board styling: sub-board highlights, active region glow, won sub-boards
- [ ] Win probability bar (updates after each AI move)
- [ ] Move history panel
- [ ] Game over modal with result and restart
- [ ] Difficulty selector (200 / 800 sims)
- [ ] Mobile responsive layout

### Milestone 3 — Multiplayer (1 week)
- [ ] `backend/routers/online.py`: WebSocket endpoint + room management
- [ ] `frontend/hooks/useGameOnline.ts`: WebSocket client
- [ ] Share link UI
- [ ] Handle disconnect / rejoin gracefully
- [ ] Room cleanup cron (purge rooms older than 2 hours)

### Milestone 4 — Deployment (2–3 days)
- [ ] Dockerfile (single image: backend + pre-built frontend)
- [ ] `fly.toml`
- [ ] CI: GitHub Actions → build frontend → build Docker image → deploy to Fly.io
- [ ] Smoke test deployed app end-to-end

### Milestone 5 — Optional extras
- [ ] MCTS visit-count heatmap overlay (toggle in sidebar)
- [ ] Game replay (step through move history)
- [ ] ELO leaderboard for online games
- [ ] Redis session store for multi-instance scaling
- [ ] Link from Substack essays to live demo

---

## Key Design Decisions

**No authentication.** Games are identified by UUID. Anyone with the game ID can join. This is fine for a hobby project and removes auth complexity entirely.

**No database.** Sessions are in-memory with TTL. If the server restarts, in-progress games are lost. This is acceptable — add a DB if replay or user accounts become wanted.

**Frontend served from the same container.** The React build output (`dist/`) is mounted as FastAPI `StaticFiles`. One container, one port, no CORS issues.

**PyTorch CPU only.** The model is small (128 channels, 8 blocks). CPU inference for 200 simulations takes ~1 second. No GPU required, which keeps deployment costs at zero on Fly.io's free tier.

**Temperature = 0 in play mode.** During self-play training, temperature adds diversity. When playing against a human, the AI should always pick its best move (greedy on visit counts). Set `temperature=0.0` in `select_move` for the vs-AI REST endpoint.
