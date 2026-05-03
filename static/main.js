/**
 * Ultimate Tic-Tac-Toe — Web Dashboard Client
 * Communicates with the Flask backend to play against the AlphaZero AI.
 */

// ── State ────────────────────────────────────────────────────────────────

let gameState = null;   // Current GameState dict from server
let humanPlayer = 1;    // Human is always Player 1 (X) for now
let difficulty = 'medium';
let isThinking = false;
let gameActive = false;

// ── DOM References ───────────────────────────────────────────────────────

const boardEl = document.getElementById('board');
const statusText = document.getElementById('status-text');
const statusIndicator = document.getElementById('status-indicator');
const evalFill = document.getElementById('eval-fill');
const evalValue = document.getElementById('eval-value');
const topMovesEl = document.getElementById('top-moves');
const resultOverlay = document.getElementById('result-overlay');
const resultIcon = document.getElementById('result-icon');
const resultTitle = document.getElementById('result-title');
const resultSubtitle = document.getElementById('result-subtitle');

// ── Initialization ───────────────────────────────────────────────────────

document.getElementById('btn-new-game').addEventListener('click', startNewGame);
document.getElementById('btn-play-again').addEventListener('click', () => {
    hideResult();
    startNewGame();
});

// Difficulty buttons
document.querySelectorAll('.diff-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        difficulty = btn.dataset.diff;
    });
});

buildBoardDOM();

// ── Board Construction ───────────────────────────────────────────────────

function buildBoardDOM() {
    boardEl.innerHTML = '';
    for (let sb = 0; sb < 9; sb++) {
        const subBoard = document.createElement('div');
        subBoard.className = 'sub-board';
        subBoard.id = `sb-${sb}`;

        for (let cell = 0; cell < 9; cell++) {
            const cellEl = document.createElement('div');
            cellEl.className = 'cell';
            cellEl.id = `cell-${sb}-${cell}`;
            cellEl.dataset.sb = sb;
            cellEl.dataset.cell = cell;
            cellEl.addEventListener('click', () => onCellClick(sb, cell));
            subBoard.appendChild(cellEl);
        }

        boardEl.appendChild(subBoard);
    }
}

// ── API Communication ────────────────────────────────────────────────────

async function startNewGame() {
    setStatus('Starting new game...', '');
    try {
        const res = await fetch('/api/new_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ difficulty, human_player: humanPlayer }),
        });
        const data = await res.json();
        gameState = data;
        gameActive = true;
        renderBoard();
        setStatus('Your turn — place X', 'your-turn');
    } catch (err) {
        setStatus('Error connecting to server', '');
        console.error(err);
    }
}

async function sendMove(move, stateToSend) {
    if (isThinking || !gameActive) return;

    isThinking = true;
    setStatus('AI is thinking', 'ai-turn');

    try {
        const res = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: stateToSend, move }),
        });

        if (!res.ok) {
            const err = await res.json();
            console.error('Move error:', err);
            isThinking = false;
            setStatus('Your turn — place X', 'your-turn');
            return;
        }

        const data = await res.json();
        gameState = data;
        renderBoard();

        // Highlight AI's last move
        if (data.ai_move !== undefined) {
            const aiSb = Math.floor(data.ai_move / 9);
            const aiCell = data.ai_move % 9;
            const aiCellEl = document.getElementById(`cell-${aiSb}-${aiCell}`);
            if (aiCellEl) aiCellEl.classList.add('ai-last-move');
        }

        // Update analysis
        if (data.ai_value !== undefined) {
            updateAnalysis(data.ai_value, data.top_moves);
        }

        // Check game over
        if (data.is_terminal) {
            gameActive = false;
            setTimeout(() => showResult(data.winner), 600);
            setStatus('Game over', 'game-over');
        } else {
            const boardHint = gameState.active_sub_board === -1 
                ? 'any board' 
                : `board ${gameState.active_sub_board}`;
            setStatus(`Your turn — play in ${boardHint}`, 'your-turn');
        }
    } catch (err) {
        console.error('Network error:', err);
        setStatus('Connection error — try again', '');
    }

    isThinking = false;
}

// ── Event Handlers ───────────────────────────────────────────────────────

function onCellClick(sb, cell) {
    if (!gameActive || isThinking) return;
    if (!gameState) return;

    const move = sb * 9 + cell;
    if (!gameState.legal_moves.includes(move)) return;

    // Send the unmodified state to the server with the move
    const stateToSend = JSON.parse(JSON.stringify(gameState));

    // Optimistically render human move on UI
    gameState.cells[sb][cell] = humanPlayer;
    gameState.legal_moves = []; // prevent double clicks
    renderBoard();

    sendMove(move, stateToSend);
}

// ── Rendering ────────────────────────────────────────────────────────────

function renderBoard() {
    if (!gameState) return;

    const legalSet = new Set(gameState.legal_moves || []);
    const activeSb = gameState.active_sub_board;

    for (let sb = 0; sb < 9; sb++) {
        const subBoardEl = document.getElementById(`sb-${sb}`);
        const sbResult = gameState.sub_board_results[sb];

        // Reset classes
        subBoardEl.className = 'sub-board';

        // Remove old overlays
        const oldOverlay = subBoardEl.querySelector('.sub-board-overlay');
        if (oldOverlay) oldOverlay.remove();

        if (sbResult === 1) {
            subBoardEl.classList.add('won-p1');
            addSubBoardOverlay(subBoardEl, '✕', 'p1');
        } else if (sbResult === -1) {
            subBoardEl.classList.add('won-p2');
            addSubBoardOverlay(subBoardEl, '○', 'p2');
        } else if (sbResult === 2) {
            subBoardEl.classList.add('drawn');
            addSubBoardOverlay(subBoardEl, '—', 'draw');
        } else if (!gameState.is_terminal && gameActive && !isThinking) {
            // Highlight active sub-boards
            if (activeSb === -1) {
                // Free choice: subtle glow on all open boards
                subBoardEl.classList.add('active');
            } else if (activeSb === sb) {
                // Single required board: strong pulsing ring
                subBoardEl.classList.add('active-single');
            }
        }

        for (let cell = 0; cell < 9; cell++) {
            const cellEl = document.getElementById(`cell-${sb}-${cell}`);
            const val = gameState.cells[sb][cell];
            const move = sb * 9 + cell;

            // Reset
            cellEl.className = 'cell';
            cellEl.textContent = '';

            if (val === 1) {
                cellEl.classList.add('p1');
                cellEl.textContent = '✕';
            } else if (val === -1) {
                cellEl.classList.add('p2');
                cellEl.textContent = '○';
            } else if (legalSet.has(move) && gameActive && !isThinking) {
                cellEl.classList.add('playable');
            }
        }
    }
}

function addSubBoardOverlay(parentEl, text, cls) {
    // Don't add duplicate
    if (parentEl.querySelector('.sub-board-overlay')) return;
    const overlay = document.createElement('div');
    overlay.className = `sub-board-overlay ${cls}`;
    overlay.textContent = text;
    parentEl.appendChild(overlay);
}

// ── Status ───────────────────────────────────────────────────────────────

function setStatus(text, indicatorClass) {
    statusText.textContent = text;
    statusIndicator.className = 'status-indicator';
    if (indicatorClass) statusIndicator.classList.add(indicatorClass);
}

// ── Analysis ─────────────────────────────────────────────────────────────

function updateAnalysis(aiValue, topMoves) {
    // aiValue is from AI's perspective: positive = AI ahead
    // For display: convert to human's perspective
    const humanEval = -aiValue;
    const pct = Math.max(5, Math.min(95, 50 + humanEval * 45));
    evalFill.style.width = pct + '%';

    if (humanEval > 0.1) {
        evalFill.style.background = 'linear-gradient(90deg, var(--cyan), var(--blue))';
        evalValue.style.color = 'var(--cyan)';
    } else if (humanEval < -0.1) {
        evalFill.style.background = 'linear-gradient(90deg, var(--magenta), #c94060)';
        evalValue.style.color = 'var(--magenta)';
    } else {
        evalFill.style.background = 'linear-gradient(90deg, var(--gold), #e0a800)';
        evalValue.style.color = 'var(--gold)';
    }
    evalValue.textContent = (humanEval >= 0 ? '+' : '') + humanEval.toFixed(2);

    // Top moves
    const label = topMovesEl.querySelector('.top-moves-label');
    topMovesEl.innerHTML = '';
    topMovesEl.appendChild(label || createLabel());

    if (topMoves) {
        const maxPct = Math.max(...topMoves.map(m => m.pct), 1);
        topMoves.forEach(m => {
            const row = document.createElement('div');
            row.className = 'top-move-row';

            const coord = document.createElement('span');
            coord.className = 'top-move-coord';
            coord.textContent = `(${m.sub_board},${m.cell})`;

            const bar = document.createElement('div');
            bar.className = 'top-move-bar';
            const fill = document.createElement('div');
            fill.className = 'top-move-bar-fill';
            fill.style.width = (m.pct / maxPct * 100) + '%';
            bar.appendChild(fill);

            const pct = document.createElement('span');
            pct.className = 'top-move-pct';
            pct.textContent = m.pct + '%';

            row.appendChild(coord);
            row.appendChild(bar);
            row.appendChild(pct);
            topMovesEl.appendChild(row);
        });
    }
}

function createLabel() {
    const label = document.createElement('div');
    label.className = 'top-moves-label';
    label.textContent = 'Top Moves';
    return label;
}

// ── Game Result ──────────────────────────────────────────────────────────

function showResult(winner) {
    if (winner === humanPlayer) {
        resultIcon.textContent = '🎉';
        resultTitle.textContent = 'You Win!';
        resultTitle.style.color = 'var(--cyan)';
        resultSubtitle.textContent = 'Incredible — you beat the AlphaZero agent!';
    } else if (winner === -humanPlayer) {
        resultIcon.textContent = '🤖';
        resultTitle.textContent = 'AI Wins';
        resultTitle.style.color = 'var(--magenta)';
        resultSubtitle.textContent = 'The machine prevails. Try again!';
    } else {
        resultIcon.textContent = '🤝';
        resultTitle.textContent = 'Draw';
        resultTitle.style.color = 'var(--gold)';
        resultSubtitle.textContent = 'A well-fought game — neither side could break through.';
    }
    resultOverlay.classList.add('visible');
}

function hideResult() {
    resultOverlay.classList.remove('visible');
}
