# Interlude: How an AI "Sees" the Board

In the previous essay, we established that solving Ultimate Tic-Tac-Toe at scale requires neural networks to estimate the value of a state. We noted that grid games naturally benefit from architectures like Convolutional Neural Networks (CNNs) and Transformers because they can recognize spatial patterns and long-distance threats. 

But there is a crucial missing piece: **how do we feed the game board into the network in the first place?**

If we were building an agent for standard 3x3 Tic-Tac-Toe, we could just pass it a flat array of 9 numbers: `1` for X, `-1` for O, and `0` for empty. 

Ultimate Tic-Tac-Toe is not so simple. The rules are highly contextual. The most important rule in the game—*you must play in the sub-board dictated by your opponent's last move*—is completely invisible if you just look at where the pieces are. A flat array of 81 numbers would force the neural network to spend an enormous amount of training time just trying to implicitly deduce the rules of the game before it could even begin learning strategy.

Instead, we borrow a concept from computer vision. Just as a color image is separated into Red, Green, and Blue (RGB) color channels, we encode our UTTT board into a **7-channel tensor** of dimensions `(7, 9, 9)`. 

Each channel is a 9x9 grid of binary values (0s and 1s) that explicitly isolates a specific, critical feature of the game state. By "spelling out" the game rules and macro-states across these channels, we encapsulate all the defining characteristics of a board so the ML model can learn efficiently.

Here is what the 7 channels look like in action, taken from a real game after 5 moves (it is currently O's turn to play):

<!-- Figure: figures/fig4_tensor_channels.png — "The 7-channel tensor encoding of a UTTT position after 5 moves (O to move). Each channel is a 9×9 binary plane. The encoding is always from the current player's perspective: Ch 0 shows O's pieces, Ch 1 shows X's pieces, Ch 2 highlights the sub-board O must play in, Ch 4 shows the sub-board X has already won, and Ch 6 is all-ones because it is O's turn." -->

### The Breakdown of the 7 Channels

Notice that the encoding is always **relative to the current player**. Because it is O's turn to move, Channel 0 represents O's pieces, and Channel 1 represents X's pieces. This symmetry means the neural network only ever has to learn how to play from "its own" perspective, rather than learning separate logic for being Player 1 versus Player 2.

Here is exactly what each channel tells the network:

**Channel 0: Current Player Pieces**
A literal map of where the current player has placed their pieces. The network uses this to find its own win threats and localized structures.

**Channel 1: Opponent Pieces**
A literal map of where the opponent has placed their pieces. The network uses this to identify incoming threats and blocking opportunities.

**Channel 2: Active Sub-board Mask**
This is perhaps the most important channel. It contains 1s only in the valid sub-board(s) where the current player is legally allowed to play. If the player is sent to a full or won board, giving them a "free choice" anywhere, this entire channel fills with 1s. This channel explicitly hands the network the "send rule," saving it from having to deduce where it is allowed to click.

**Channel 3: Sub-boards Won by Current Player**
If the current player wins a 3x3 sub-board, all 9 cells of that sub-board light up with 1s in this channel. This encapsulates the "macro-game" state, telling the network which major sectors it controls for the global win.

**Channel 4: Sub-boards Won by Opponent**
The exact inverse of Channel 3. In the figure above, X won the top-left sub-board in their first three moves, so all 9 cells in that top-left sector are lit up here, warning O that this territory is lost.

**Channel 5: Drawn Sub-boards**
When a sub-board fills up without a winner, it becomes a "dead zone." This channel lights up those regions so the network knows they can no longer be claimed by either side.

**Channel 6: Turn Indicator**
This channel is either entirely 0s (if it is Player 1's turn) or entirely 1s (if it is Player 2's turn). Because the rest of the board is encoded symmetrically, this channel serves as a subtle anchor, allowing the network to know exactly which phase of parity the game is in. 

### Why This Matters

This spatial, multi-layered encoding is the bridge between the raw rules of the game and the architecture of the neural network. By formatting the state this way, a Convolutional Neural Network can slide a 3x3 filter over the board and instantly recognize local tactical patterns, while a Transformer can attend to the Macro-board channels (Channels 3, 4, and 5) to connect distant strategic goals.

With the credit-assignment problem understood (Essay 1b) and our game state perfectly translated into the language of tensors, we are almost ready to build our agent. But first, we need to solve the final piece of the puzzle: how to explore the game tree without getting lost in its immense size.

---

Code: [Notebook 1 — The UTTT Game Engine](https://colab.research.google.com/github/thltsui/UlltimateTicTacToe/blob/Substack/substack/notebook_1_uttt_engine.ipynb) shows the full 7-channel `encode_state()` function in action, including the code that generated the visualizations above.
