import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from importlib import import_module

network_mod = import_module('02_network.network')
policy_head_mod = import_module('02_network.policy_head')
value_head_mod = import_module('02_network.value_head')

NetworkOutput = network_mod.NetworkOutput
PolicyHead = policy_head_mod.PolicyHead
ValueHead = value_head_mod.ValueHead

class TransformerTTTNetwork(nn.Module):
    """Transformer network for Ultimate TTT.

    Architecture:
        - Patch embed: 7 channels -> C channels (1x1 conv)
        - Reshape to (B, 81, C)
        - Add Positional Encoding
        - Transformer Encoder (N layers)
        - Reshape to (B, C, 9, 9)
        - Policy head & Value head (reused from CNN)
    """

    INPUT_CHANNELS = 7

    def __init__(
        self,
        channels: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        position_encoding: str = "absolute",
        value_channels: int = 32,
        value_hidden_size: int = 512,
        value_feature_size: int = 128,
    ):
        # We use num_blocks parameter as num_layers to match the interface of trainer.py
        super().__init__()
        num_layers = num_blocks

        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if position_encoding not in {"absolute", "hierarchical"}:
            raise ValueError(
                "position_encoding must be 'absolute' or 'hierarchical'"
            )

        self.channels = channels
        self.num_blocks = num_layers
        self.num_heads = num_heads
        self.ffn_multiplier = ffn_multiplier
        self.dropout = dropout
        self.position_encoding = position_encoding

        # Patch embedding: projecting each 1x1 spatial location across all 7 channels into `channels` dimensions
        self.patch_embed = nn.Conv2d(self.INPUT_CHANNELS, channels, kernel_size=1, bias=False)
        self.embed_bn = nn.BatchNorm2d(channels)
        
        # Hierarchical encoding shares what "top-left within a sub-board"
        # means across all nine boards while separately identifying the macro
        # board.  Legacy checkpoints retain their original absolute encoding.
        if position_encoding == "hierarchical":
            self.macro_pos_embed = nn.Parameter(torch.randn(1, 9, channels))
            self.cell_pos_embed = nn.Parameter(torch.randn(1, 9, channels))
            macro_indices = []
            cell_indices = []
            for row in range(9):
                for col in range(9):
                    macro_indices.append((row // 3) * 3 + (col // 3))
                    cell_indices.append((row % 3) * 3 + (col % 3))
            self.register_buffer(
                "macro_position_indices",
                torch.tensor(macro_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "cell_position_indices",
                torch.tensor(cell_indices, dtype=torch.long),
                persistent=False,
            )
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, 81, channels))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, 
            nhead=num_heads, 
            dim_feedforward=channels * ffn_multiplier,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dual heads (reusing the exact same ones from the CNN)
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(
            channels,
            conv_channels=value_channels,
            hidden_size=value_hidden_size,
            feature_size=value_feature_size,
        )

    def _position_embedding(self) -> torch.Tensor:
        if self.position_encoding == "absolute":
            return self.pos_embed
        return (
            self.macro_pos_embed[:, self.macro_position_indices]
            + self.cell_pos_embed[:, self.cell_position_indices]
        )

    def forward(self, x: torch.Tensor) -> NetworkOutput:
        B = x.shape[0]
        
        # 1. Patch Embedding
        h = self.patch_embed(x)        # (B, C, 9, 9)
        h = self.embed_bn(h)           # (B, C, 9, 9)
        
        # 2. Reshape to sequence of tokens
        h = h.flatten(2)               # (B, C, 81)
        h = h.transpose(1, 2)          # (B, 81, C)
        
        # 3. Add positional embedding
        h = h + self._position_embedding()
        
        # 4. Transformer
        h = self.transformer(h)        # (B, 81, C)
        
        # 5. Reshape back to spatial feature map for the heads
        h = h.transpose(1, 2)          # (B, C, 81)
        h = h.reshape(B, -1, 9, 9).contiguous()     # (B, C, 9, 9)
        
        # 6. Heads
        policy_logits, opp_policy_logits = self.policy_head(h)
        value_out = self.value_head(h)

        return NetworkOutput(
            policy_logits=policy_logits,
            opp_policy_logits=opp_policy_logits,
            wdl_logits=value_out.wdl_logits,
            wdl_probs=value_out.wdl_probs,
            win_value=value_out.win_value,
            score_margin=value_out.score_margin,
            ownership=value_out.ownership,
        )

    def predict(self, state, device: str = 'cpu') -> NetworkOutput:
        board_mod = import_module('01_game.board')
        encode_state = board_mod.encode_state

        tensor = encode_state(state)                    # (7, 9, 9)
        tensor = tensor.unsqueeze(0).to(device)         # (1, 7, 9, 9)

        self.eval()
        with torch.no_grad():
            output = self.forward(tensor)

        return NetworkOutput(
            policy_logits=output.policy_logits.squeeze(0),
            opp_policy_logits=output.opp_policy_logits.squeeze(0),
            wdl_logits=output.wdl_logits.squeeze(0),
            wdl_probs=output.wdl_probs.squeeze(0),
            win_value=output.win_value.squeeze(0),
            score_margin=output.score_margin.squeeze(0),
            ownership=output.ownership.squeeze(0),
        )

if __name__ == "__main__":
    print("=== Transformer Network ===")
    net = TransformerTTTNetwork()
    x = torch.randn(4, 7, 9, 9)
    out = net(x)
    print(f"Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    print("Forward pass successful.")
