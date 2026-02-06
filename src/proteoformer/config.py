"""
Configuration classes for Proteoformer models.

This module defines the configuration schema for Proteoformer, enabling
consistent model instantiation and serialization across training, evaluation,
and deployment workflows.
"""

from typing import Optional, Tuple
from transformers.configuration_utils import PretrainedConfig
from transformers import RoFormerConfig


class ProteoformerConfig(PretrainedConfig):
    """
    Configuration class for Proteoformer models.

    This configuration extends the standard transformer configuration with
    proteoform-specific parameters, enabling native support for PTM tokens
    and specialized training objectives.

    Attributes:
        vocab_size: Size of the vocabulary including PTM tokens. Default: 2000
        hidden_size: Dimensionality of encoder layers and pooler. Default: 768
        num_hidden_layers: Number of transformer encoder layers. Default: 12
        num_attention_heads: Number of attention heads per layer. Default: 12
        intermediate_size: Dimensionality of the feed-forward layer. Default: 3072
        max_position_embeddings: Maximum sequence length supported. Default: 2048
        rotary_value: Whether to apply RoPE to value vectors. Default: False
        hidden_act: Activation function in encoder/pooler. Default: "gelu"
        layer_norm_eps: Epsilon for layer normalization. Default: 1e-12
        hidden_dropout_prob: Dropout for fully connected layers. Default: 0.1
        attention_probs_dropout_prob: Dropout for attention probabilities. Default: 0.1
        initializer_range: Std dev for weight initialization. Default: 0.02
        pad_token_id: Token ID for padding. Default: 0
        mlm_probability: Masking probability for MLM training. Default: 0.15
        modification_token_ids: Token IDs reserved for PTM representations.
        roformer_kwargs: Additional kwargs for transformer backbone.

    Example:
        >>> config = ProteoformerConfig(
        ...     vocab_size=2000,
        ...     hidden_size=768,
        ...     num_hidden_layers=12,
        ...     modification_token_ids=(100, 101, 102)
        ... )
        >>> model = ProteoformerForMLM(config)
    """

    model_type = "proteoformer"

    def __init__(
        self,
        vocab_size: int = 2000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        rotary_value: bool = False,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        mlm_probability: float = 0.15,
        modification_token_ids: Optional[Tuple[int, ...]] = None,
        roformer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize Proteoformer configuration.

        Args:
            vocab_size: Vocabulary size including special and PTM tokens.
            hidden_size: Hidden dimension of transformer layers.
            num_hidden_layers: Number of transformer encoder layers.
            num_attention_heads: Number of attention heads (must divide hidden_size).
            intermediate_size: FFN intermediate dimension.
            max_position_embeddings: Maximum input sequence length.
            rotary_value: Apply rotary embeddings to values (default: keys/queries only).
            hidden_act: Activation function ("gelu", "relu", "silu", etc.).
            layer_norm_eps: Layer normalization epsilon.
            hidden_dropout_prob: Dropout probability for hidden layers.
            attention_probs_dropout_prob: Dropout probability for attention weights.
            initializer_range: Std dev for truncated normal weight initialization.
            pad_token_id: ID of the padding token.
            bos_token_id: ID of the beginning-of-sequence token (optional).
            eos_token_id: ID of the end-of-sequence token (optional).
            mlm_probability: Token masking probability for MLM pre-training.
            modification_token_ids: Tuple of token IDs for PTM representations.
            roformer_kwargs: Additional arguments passed to RoFormerConfig.
            **kwargs: Additional arguments for PretrainedConfig.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Core transformer architecture parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rotary_value = rotary_value
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        # Proteoform-specific parameters
        self.mlm_probability = mlm_probability
        self.modification_token_ids = modification_token_ids

        # Extended configuration options
        self.roformer_kwargs = roformer_kwargs or {}

        # Validate configuration
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert 0.0 <= self.mlm_probability <= 1.0, \
            "mlm_probability must be in [0, 1]"

    def get_roformer_config(self) -> RoFormerConfig:
        """
        Generate the backbone transformer configuration.

        Creates a RoFormerConfig from Proteoformer parameters for the
        underlying transformer encoder.

        Returns:
            RoFormerConfig configured with current model parameters.
        """
        base = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rotary_value": self.rotary_value,
            "hidden_act": self.hidden_act,
            "layer_norm_eps": self.layer_norm_eps,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        base.update(self.roformer_kwargs)
        return RoFormerConfig(**base)
