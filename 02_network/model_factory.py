"""Validated model configuration and checkpoint-aware network construction.

Architecture is part of the model artifact.  Training, calibration, evaluation,
and deployment must all construct the same module from the same metadata rather
than guessing most settings from tensor shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


MODEL_CONFIG_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    """Complete architecture definition for a network checkpoint."""

    architecture: str = "cnn"
    channels: int = 128
    num_blocks: int = 8
    num_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    position_encoding: str = "absolute"
    value_channels: int = 32
    value_hidden_size: int = 512
    value_feature_size: int = 128

    def __post_init__(self) -> None:
        if self.architecture not in {"cnn", "transformer"}:
            raise ValueError("model.architecture must be 'cnn' or 'transformer'")
        for name in (
            "channels",
            "num_blocks",
            "value_channels",
            "value_hidden_size",
            "value_feature_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.position_encoding not in {"absolute", "hierarchical"}:
            raise ValueError(
                "model.position_encoding must be 'absolute' or 'hierarchical'"
            )
        if self.architecture == "transformer":
            if self.num_heads <= 0:
                raise ValueError("model.num_heads must be positive")
            if self.channels % self.num_heads != 0:
                raise ValueError("model.channels must be divisible by model.num_heads")
            if self.ffn_multiplier <= 0:
                raise ValueError("model.ffn_multiplier must be positive")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        """Parse the JSON model section and reject misspelled settings."""
        data = dict(raw)
        version = data.pop("config_version", MODEL_CONFIG_VERSION)
        if version != MODEL_CONFIG_VERSION:
            raise ValueError(
                f"unsupported model config_version {version}; "
                f"expected {MODEL_CONFIG_VERSION}"
            )
        if "num_layers" in data:
            if "num_blocks" in data:
                raise ValueError("set only one of model.num_layers or model.num_blocks")
            data["num_blocks"] = data.pop("num_layers")

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown model config field(s): {', '.join(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return stable metadata suitable for JSON and ``torch.save``."""
        result = asdict(self)
        result["config_version"] = MODEL_CONFIG_VERSION
        if self.architecture == "transformer":
            result["num_layers"] = result.pop("num_blocks")
        return result


def create_network(config: ModelConfig):
    """Construct the architecture described by ``config``."""
    if config.architecture == "transformer":
        from transformer.transformer_network import TransformerTTTNetwork

        network = TransformerTTTNetwork(
            channels=config.channels,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            ffn_multiplier=config.ffn_multiplier,
            dropout=config.dropout,
            position_encoding=config.position_encoding,
            value_channels=config.value_channels,
            value_hidden_size=config.value_hidden_size,
            value_feature_size=config.value_feature_size,
        )
    else:
        from .network import UltimateTTTNetwork

        network = UltimateTTTNetwork(
            channels=config.channels,
            num_blocks=config.num_blocks,
            value_channels=config.value_channels,
            value_hidden_size=config.value_hidden_size,
            value_feature_size=config.value_feature_size,
        )

    # Keep a canonical, serializable definition attached to the module.
    network.model_config = config.to_dict()
    return network


def model_config_for_network(network) -> ModelConfig:
    """Return explicit metadata for a constructed network."""
    stored = getattr(network, "model_config", None)
    if stored is not None:
        return ModelConfig.from_dict(stored)

    value_head = network.value_head
    common = dict(
        channels=int(network.channels),
        num_blocks=int(network.num_blocks),
        value_channels=int(value_head.conv.out_channels),
        value_hidden_size=int(value_head.fc1.out_features),
        value_feature_size=int(value_head.fc2.out_features),
    )
    if hasattr(network, "transformer"):
        return ModelConfig(
            architecture="transformer",
            num_heads=int(network.num_heads),
            ffn_multiplier=int(network.ffn_multiplier),
            dropout=float(network.dropout),
            position_encoding=str(network.position_encoding),
            **common,
        )
    return ModelConfig(architecture="cnn", **common)


def infer_legacy_model_config(state_dict: Mapping[str, Any]) -> ModelConfig:
    """Best-effort compatibility for checkpoints created before metadata.

    New checkpoints never use this path.  Head count and dropout are not
    encoded in a transformer's tensors, so the historical defaults are the
    only safe values available for legacy artifacts.
    """
    is_transformer = any("patch_embed" in key for key in state_dict)

    if is_transformer:
        embed = state_dict.get("pos_embed")
        if embed is None:
            embed = state_dict.get("macro_pos_embed")
        if embed is None:
            raise ValueError("cannot infer Transformer channels from checkpoint")
        channels = int(embed.shape[-1])
        num_blocks = sum(
            key.endswith(".self_attn.in_proj_weight")
            and "transformer.layers." in key
            for key in state_dict
        ) or 4
        linear1 = next(
            (
                tensor
                for key, tensor in state_dict.items()
                if key.endswith("transformer.layers.0.linear1.weight")
            ),
            None,
        )
        ffn_multiplier = (
            max(1, int(linear1.shape[0]) // channels) if linear1 is not None else 4
        )
        position_encoding = (
            "hierarchical" if "macro_pos_embed" in state_dict else "absolute"
        )
        architecture = "transformer"
    else:
        input_weight = next(
            (
                tensor
                for key, tensor in state_dict.items()
                if key.endswith("input_conv.weight")
            ),
            None,
        )
        if input_weight is None:
            raise ValueError("cannot infer CNN channels from checkpoint")
        channels = int(input_weight.shape[0])
        num_blocks = sum(
            key.endswith(".conv1.weight") and "trunk." in key
            for key in state_dict
        )
        ffn_multiplier = 4
        position_encoding = "absolute"
        architecture = "cnn"

    value_conv = state_dict.get("value_head.conv.weight")
    value_fc1 = state_dict.get("value_head.fc1.weight")
    value_fc2 = state_dict.get("value_head.fc2.weight")
    if value_conv is None or value_fc1 is None or value_fc2 is None:
        raise ValueError("cannot infer value-head dimensions from checkpoint")

    return ModelConfig(
        architecture=architecture,
        channels=channels,
        num_blocks=num_blocks,
        num_heads=4,
        ffn_multiplier=ffn_multiplier,
        dropout=0.1,
        position_encoding=position_encoding,
        value_channels=int(value_conv.shape[0]),
        value_hidden_size=int(value_fc1.shape[0]),
        value_feature_size=int(value_fc2.shape[0]),
    )


def model_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> ModelConfig:
    """Read authoritative architecture metadata, with legacy fallback."""
    raw = checkpoint.get("model_config")
    if raw is not None:
        return ModelConfig.from_dict(raw)
    state_dict = checkpoint.get("network_state_dict")
    if state_dict is None:
        raise ValueError("checkpoint has no network_state_dict")
    return infer_legacy_model_config(state_dict)


def create_network_from_checkpoint(checkpoint: Mapping[str, Any]):
    """Construct and populate a network from a loaded checkpoint mapping."""
    config = model_config_from_checkpoint(checkpoint)
    network = create_network(config)
    network.load_state_dict(checkpoint["network_state_dict"])
    return network, config
