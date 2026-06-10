/**
 * Ultimate Tic-Tac-Toe — Web Dashboard Client
 * Communicates with the Flask backend to play against the AlphaZero AI.
 * Visualizes the AI's thinking in real-time.
 */

// ── State ────────────────────────────────────────────────────────────────

let gameState = null;
let humanPlayer = 1;
let difficulty = 'medium';
let isThinking = false;
let gameActive = false;
let currentAnalysis = null;
let vizMode = 'none'; // 'none', 'visits', 'policy', 'qvalue', 'opponent'

// ── Keypad Mapping ───────────────────────────────────────────────────────
// Array index (row-major, 0=top-left) → numpad label (7=top-left, 1=bottom-left)
const IDX_TO_KEYPAD = [7, 8, 9, 4, 5, 6, 1, 2, 3];

function keypadLabel(subBoard, cell) {
    return `${IDX_TO_KEYPAD[subBoard]}·${IDX_TO_KEYPAD[cell]}`;
}

function keypadBoard(subBoard) {
    return IDX_TO_KEYPAD[subBoard];
}

// ── DOM References ───────────────────────────────────────────────────────

const boardEl = document.getElementById('board');
const statusText = document.getElementById('status-text');
const statusIndicator = document.getElementById('status-indicator');
const evalPctHuman = document.getElementById('eval-pct-human');
const evalPctAi = document.getElementById('eval-pct-ai');
const evalBestHuman = document.getElementById('eval-best-human');
const evalBestAi = document.getElementById('eval-best-ai');
const evalBarFill = document.getElementById('eval-bar-fill');
const scoreMarginValue = document.getElementById('score-margin-value');
const simsValue = document.getElementById('sims-value');
const ownershipGrid = document.getElementById('ownership-grid');
const topMovesList = document.getElementById('top-moves-list');
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

// Visualization toggle buttons
document.querySelectorAll('.viz-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.viz-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        vizMode = btn.dataset.mode;
        renderHeatmap();
    });
});

buildBoardDOM();
buildOwnershipGrid();

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

            // Heatmap overlay element
            const hm = document.createElement('div');
            hm.className = 'cell-heatmap';
            hm.id = `hm-${sb}-${cell}`;
            const hmVal = document.createElement('span');
            hmVal.className = 'hm-val';
            hm.appendChild(hmVal);
            cellEl.appendChild(hm);

            subBoard.appendChild(cellEl);
        }

        boardEl.appendChild(subBoard);
    }
}

function buildOwnershipGrid() {
    ownershipGrid.innerHTML = '';
    for (let i = 0; i < 9; i++) {
        const cell = document.createElement('div');
        cell.className = 'own-cell';
        cell.id = `own-${i}`;
        const val = document.createElement('span');
        val.className = 'own-val';
        val.textContent = '—';
        cell.appendChild(val);
        ownershipGrid.appendChild(cell);
    }
}

// ── API Communication ────────────────────────────────────────────────────

async function startNewGame() {
    setStatus('Starting new game...', '');
    currentAnalysis = null;
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
        renderHeatmap();
        resetAnalysisPanel();
        setStatus('Your turn — place X', 'your-turn');
    } catch (err) {
        setStatus('Error connecting to server', '');
        console.error(err);
    }
}

async function sendMove(move, stateToSend) {
    if (isThinking || !gameActive) return;

    isThinking = true;
    setStatus('AI is thinking…', 'ai-turn');

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

        // Store analysis
        if (data.analysis) {
            currentAnalysis = data.analysis;
            updateAnalysisPanel(data.analysis);
        }

        renderBoard();
        renderHeatmap();

        // Highlight AI's last move
        if (data.ai_move !== undefined) {
            const aiSb = Math.floor(data.ai_move / 9);
            const aiCell = data.ai_move % 9;
            const aiCellEl = document.getElementById(`cell-${aiSb}-${aiCell}`);
            if (aiCellEl) aiCellEl.classList.add('ai-last-move');
        }

        // Check game over
        if (data.is_terminal) {
            gameActive = false;
            setTimeout(() => showResult(data.winner), 600);
            setStatus('Game over', 'game-over');
        } else {
            const boardHint = gameState.active_sub_board === -1
                ? 'any board'
                : `board ${keypadBoard(gameState.active_sub_board)}`;
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

    const stateToSend = JSON.parse(JSON.stringify(gameState));

    // Optimistically render human move on UI
    gameState.cells[sb][cell] = humanPlayer;
    gameState.legal_moves = [];
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

        subBoardEl.className = 'sub-board';

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
            if (activeSb === -1) {
                subBoardEl.classList.add('active');
            } else if (activeSb === sb) {
                subBoardEl.classList.add('active-single');
            }
        }

        for (let cell = 0; cell < 9; cell++) {
            const cellEl = document.getElementById(`cell-${sb}-${cell}`);
            const val = gameState.cells[sb][cell];
            const move = sb * 9 + cell;

            // Reset but preserve heatmap child
            cellEl.className = 'cell';
            // Remove text nodes but keep heatmap
            for (let i = cellEl.childNodes.length - 1; i >= 0; i--) {
                const child = cellEl.childNodes[i];
                if (child.nodeType === Node.TEXT_NODE) {
                    cellEl.removeChild(child);
                }
            }

            if (val === 1) {
                cellEl.classList.add('p1');
                cellEl.insertBefore(document.createTextNode('✕'), cellEl.firstChild);
            } else if (val === -1) {
                cellEl.classList.add('p2');
                cellEl.insertBefore(document.createTextNode('○'), cellEl.firstChild);
            } else if (legalSet.has(move) && gameActive && !isThinking) {
                cellEl.classList.add('playable');
            }
        }
    }
}

function addSubBoardOverlay(parentEl, text, cls) {
    if (parentEl.querySelector('.sub-board-overlay')) return;
    const overlay = document.createElement('div');
    overlay.className = `sub-board-overlay ${cls}`;
    overlay.textContent = text;
    parentEl.appendChild(overlay);
}

// ── Heatmap Rendering ────────────────────────────────────────────────────

function renderHeatmap() {
    if (!currentAnalysis || vizMode === 'none') {
        clearHeatmap();
        return;
    }

    const a = currentAnalysis;
    let values;
    let colorFn;
    let labelFn;

    switch (vizMode) {
        case 'visits':
            values = a.mcts_visits;
            const maxVisits = Math.max(...values, 1);
            colorFn = (v) => {
                const intensity = v / maxVisits;
                return `rgba(17, 138, 178, ${intensity * 0.7})`;
            };
            labelFn = (v) => v > 0 ? v.toString() : '';
            break;

        case 'policy':
            values = a.policy_probs;
            const maxPolicy = Math.max(...values, 0.001);
            colorFn = (v) => {
                const intensity = v / maxPolicy;
                return `rgba(6, 214, 160, ${intensity * 0.65})`;
            };
            labelFn = (v) => v > 0.01 ? (v * 100).toFixed(0) + '%' : '';
            break;

        case 'qvalue':
            values = a.mcts_q_values;
            colorFn = (v, idx) => {
                // Only color cells that have been visited
                if (a.mcts_visits[idx] === 0) return 'transparent';
                // Q-value: positive = good for current player, negative = bad
                if (v > 0) {
                    return `rgba(6, 214, 160, ${Math.min(Math.abs(v), 1) * 0.65})`;
                } else {
                    return `rgba(239, 71, 111, ${Math.min(Math.abs(v), 1) * 0.65})`;
                }
            };
            labelFn = (v, idx) => {
                if (a.mcts_visits[idx] === 0) return '';
                return v >= 0 ? '+' + v.toFixed(2) : v.toFixed(2);
            };
            break;

        case 'opponent':
            values = a.opp_policy_probs;
            const maxOpp = Math.max(...values, 0.001);
            colorFn = (v) => {
                const intensity = v / maxOpp;
                return `rgba(239, 71, 111, ${intensity * 0.6})`;
            };
            labelFn = (v) => v > 0.01 ? (v * 100).toFixed(0) + '%' : '';
            break;

        default:
            clearHeatmap();
            return;
    }

    for (let sb = 0; sb < 9; sb++) {
        for (let cell = 0; cell < 9; cell++) {
            const idx = sb * 9 + cell;
            const hm = document.getElementById(`hm-${sb}-${cell}`);
            if (!hm) continue;

            const val = values[idx];
            const color = typeof colorFn === 'function'
                ? colorFn(val, idx)
                : 'transparent';
            const label = typeof labelFn === 'function'
                ? labelFn(val, idx)
                : '';

            hm.style.background = color;
            hm.style.opacity = '1';
            const valEl = hm.querySelector('.hm-val');
            if (valEl) valEl.textContent = label;
        }
    }
}

function clearHeatmap() {
    for (let sb = 0; sb < 9; sb++) {
        for (let cell = 0; cell < 9; cell++) {
            const hm = document.getElementById(`hm-${sb}-${cell}`);
            if (!hm) continue;
            hm.style.background = 'transparent';
            hm.style.opacity = '0';
            const valEl = hm.querySelector('.hm-val');
            if (valEl) valEl.textContent = '';
        }
    }
}

// ── Analysis Panel ───────────────────────────────────────────────────────

function updateAnalysisPanel(analysis) {
    // Analysis is now from the HUMAN's perspective (current player).
    // win_value: positive = human ahead, negative = AI ahead
    const winVal = analysis.win_value;
    const humanWinPct = Math.round((1 + winVal) / 2 * 100);
    const aiWinPct = 100 - humanWinPct;

    // Dual eval columns
    evalPctHuman.textContent = humanWinPct + '%';
    evalPctAi.textContent = aiWinPct + '%';

    // Vertical balance bar (height = human's share, from bottom)
    const barPct = Math.max(5, Math.min(95, humanWinPct));
    evalBarFill.style.height = barPct + '%';
    if (humanWinPct > 55) {
        evalBarFill.style.background = 'linear-gradient(0deg, var(--cyan), rgba(6, 214, 160, 0.5))';
    } else if (humanWinPct < 45) {
        evalBarFill.style.background = 'linear-gradient(0deg, var(--magenta), rgba(239, 71, 111, 0.5))';
    } else {
        evalBarFill.style.background = 'linear-gradient(0deg, var(--cyan), var(--gold))';
    }

    // Best move for human = top MCTS move (highest visits in human-perspective analysis)
    if (analysis.top_moves && analysis.top_moves.length > 0) {
        const best = analysis.top_moves[0];
        evalBestHuman.textContent = keypadLabel(best.sub_board, best.cell);
    } else {
        evalBestHuman.textContent = '—';
    }

    // AI's likely response = opponent policy head (what the model predicts AI will play)
    if (analysis.opp_policy_probs) {
        let bestOppIdx = 0;
        let bestOppVal = -1;
        for (let i = 0; i < 81; i++) {
            if (analysis.opp_policy_probs[i] > bestOppVal) {
                bestOppVal = analysis.opp_policy_probs[i];
                bestOppIdx = i;
            }
        }
        const oppSb = Math.floor(bestOppIdx / 9);
        const oppCell = bestOppIdx % 9;
        evalBestAi.textContent = keypadLabel(oppSb, oppCell);
    } else {
        evalBestAi.textContent = '—';
    }

    // Score margin
    const margin = analysis.score_margin;
    const marginEl = scoreMarginValue;
    marginEl.textContent = (margin >= 0 ? '+' : '') + margin.toFixed(3);
    marginEl.style.color = margin > 0.05 ? 'var(--magenta)' : margin < -0.05 ? 'var(--cyan)' : 'var(--text-secondary)';

    // Sims
    simsValue.textContent = analysis.total_sims.toLocaleString();

    // Ownership map
    updateOwnership(analysis.ownership);

    // Top moves
    updateTopMoves(analysis.top_moves);
}

function resetAnalysisPanel() {
    evalPctHuman.textContent = '50%';
    evalPctAi.textContent = '50%';
    evalBestHuman.textContent = '—';
    evalBestAi.textContent = '—';
    evalBarFill.style.height = '50%';
    evalBarFill.style.background = 'linear-gradient(0deg, var(--cyan), var(--gold))';
    scoreMarginValue.textContent = '0.000';
    scoreMarginValue.style.color = 'var(--text-secondary)';
    simsValue.textContent = '—';
    topMovesList.innerHTML = '';

    for (let i = 0; i < 9; i++) {
        const cell = document.getElementById(`own-${i}`);
        if (cell) {
            cell.style.background = 'rgba(255, 255, 255, 0.03)';
            const val = cell.querySelector('.own-val');
            if (val) val.textContent = '—';
        }
    }
}

function updateOwnership(ownership) {
    if (!ownership || ownership.length < 9) return;

    for (let i = 0; i < 9; i++) {
        const cell = document.getElementById(`own-${i}`);
        if (!cell) continue;
        const val = cell.querySelector('.own-val');

        const o = ownership[i]; // 0..1 range, where 0.5 = contested

        // Check if sub-board is already decided
        if (gameState && gameState.sub_board_results[i] !== 0) {
            const result = gameState.sub_board_results[i];
            if (result === 1) {
                cell.style.background = 'rgba(6, 214, 160, 0.2)';
                if (val) val.textContent = '✕';
                val.style.color = 'var(--cyan)';
            } else if (result === -1) {
                cell.style.background = 'rgba(239, 71, 111, 0.2)';
                if (val) val.textContent = '○';
                val.style.color = 'var(--magenta)';
            } else {
                cell.style.background = 'rgba(255, 255, 255, 0.05)';
                if (val) val.textContent = '—';
                val.style.color = 'var(--text-dim)';
            }
            continue;
        }

        // Map ownership value to color
        const pctText = Math.round(o * 100);
        if (val) {
            val.textContent = pctText + '%';
            val.style.color = 'var(--text-primary)';
        }

        if (o > 0.6) {
            // AI-leaning (magenta)
            const intensity = (o - 0.5) * 2;
            cell.style.background = `rgba(239, 71, 111, ${intensity * 0.3})`;
        } else if (o < 0.4) {
            // Human-leaning (cyan)
            const intensity = (0.5 - o) * 2;
            cell.style.background = `rgba(6, 214, 160, ${intensity * 0.3})`;
        } else {
            // Contested (gold)
            cell.style.background = `rgba(255, 209, 102, 0.1)`;
        }
    }
}

function updateTopMoves(topMoves) {
    topMovesList.innerHTML = '';
    if (!topMoves || topMoves.length === 0) return;

    const maxPct = Math.max(...topMoves.map(m => m.pct), 1);

    topMoves.forEach((m, idx) => {
        const row = document.createElement('div');
        row.className = 'top-move-row';

        // Coordinate
        const coord = document.createElement('span');
        coord.className = 'top-move-coord';
        coord.textContent = keypadLabel(m.sub_board, m.cell);

        // Bar + percentage
        const barWrap = document.createElement('div');
        barWrap.className = 'top-move-bar-wrap';

        const bar = document.createElement('div');
        bar.className = 'top-move-bar';
        const fill = document.createElement('div');
        fill.className = `top-move-bar-fill rank-${idx}`;
        fill.style.width = (m.pct / maxPct * 100) + '%';
        bar.appendChild(fill);

        const pct = document.createElement('span');
        pct.className = 'top-move-pct';
        pct.textContent = m.pct + '%';

        barWrap.appendChild(bar);
        barWrap.appendChild(pct);

        // Q-Value
        const qVal = document.createElement('span');
        qVal.className = 'top-move-q';
        const q = m.q_value || 0;
        qVal.textContent = (q >= 0 ? '+' : '') + q.toFixed(3);
        qVal.style.color = q > 0.05 ? 'var(--cyan)' : q < -0.05 ? 'var(--magenta)' : 'var(--text-dim)';

        // Prior
        const prior = document.createElement('span');
        prior.className = 'top-move-prior';
        prior.textContent = ((m.prior || 0) * 100).toFixed(1) + '%';

        row.appendChild(coord);
        row.appendChild(barWrap);
        row.appendChild(qVal);
        row.appendChild(prior);

        // Hover: highlight cell on board
        row.addEventListener('mouseenter', () => {
            const cellEl = document.getElementById(`cell-${m.sub_board}-${m.cell}`);
            if (cellEl) cellEl.style.outline = '2px solid var(--gold)';
        });
        row.addEventListener('mouseleave', () => {
            const cellEl = document.getElementById(`cell-${m.sub_board}-${m.cell}`);
            if (cellEl) cellEl.style.outline = '';
        });

        topMovesList.appendChild(row);
    });
}

// ── Status ───────────────────────────────────────────────────────────────

function setStatus(text, indicatorClass) {
    statusText.textContent = text;
    statusIndicator.className = 'status-indicator';
    if (indicatorClass) statusIndicator.classList.add(indicatorClass);
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
