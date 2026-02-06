from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers import RoFormerModel

from .config import ProteoformerConfig


# ==============================================================================
# Model Output Classes
# ==============================================================================

@dataclass
class ProteoformerMLMOutput(ModelOutput):
    """
    Output class for Proteoformer MLM pre-training.

    Attributes:
        loss (Optional[torch.FloatTensor]): Masked language modeling loss.
            Only returned when `labels` is provided.
        logits (torch.FloatTensor): Prediction scores for each token in vocabulary.
            Shape: `(batch_size, sequence_length, vocab_size)`
        last_hidden_state (torch.FloatTensor): Sequence of hidden states from
            the final encoder layer.
            Shape: `(batch_size, sequence_length, hidden_size)`
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


@dataclass
class ProteoformerEmbeddingOutput(ModelOutput):
    """
    Output class for Proteoformer embedding extraction.

    Attributes:
        last_hidden_state (torch.FloatTensor): Sequence of hidden states from
            the final encoder layer. These embeddings capture PTM-aware
            proteoform representations.
            Shape: `(batch_size, sequence_length, hidden_size)`
        backbone_outputs (Optional[Union[Dict, Tuple]]): Complete outputs from
            the transformer backbone, including attention weights and
            intermediate hidden states if requested.
    """
    last_hidden_state: torch.FloatTensor = None
    backbone_outputs: Optional[Union[Dict, Tuple]] = None


# ==============================================================================
# Model Components
# ==============================================================================

class ProteoformerMLMHead(nn.Module):
    """
    Masked Language Model prediction head for Proteoformer.

    This head transforms encoder hidden states into vocabulary predictions,
    enabling the model to learn proteoform representations through the
    MLM pre-training objective.

    Architecture:
        1. Linear projection (hidden_size -> hidden_size)
        2. SiLU activation function
        3. Layer normalization
        4. Output projection (hidden_size -> vocab_size)

    Attributes:
        dense (nn.Linear): Hidden layer for feature transformation.
        layer_norm (nn.LayerNorm): Normalization layer.
        decoder (nn.Linear): Output projection to vocabulary space.
        bias (nn.Parameter): Output bias for vocabulary predictions.
    """

    def __init__(self, config: ProteoformerConfig):
        """
        Initialize the MLM head.

        Args:
            config (ProteoformerConfig): Model configuration containing
                hidden_size, vocab_size, and layer_norm_eps.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute vocabulary predictions from hidden states.

        Args:
            hidden_states (torch.Tensor): Encoder outputs.
                Shape: `(batch_size, sequence_length, hidden_size)`

        Returns:
            torch.Tensor: Logits for each token in the vocabulary.
                Shape: `(batch_size, sequence_length, vocab_size)`
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ProteoformerEncoder(PreTrainedModel):
    """
    Transformer encoder for Proteoformer.

    The encoder processes tokenized proteoform sequences and produces
    contextualized representations. It employs Rotary Position Embeddings (RoPE)
    for capturing positional information, which is particularly effective for
    modeling the sequential nature of proteoforms with modifications.

    Key Features:
        - Rotary Position Embeddings for relative position encoding
        - Multi-head self-attention for capturing residue interactions
        - Compatible with Hugging Face's PreTrainedModel interface

    Attributes:
        config (ProteoformerConfig): Model configuration.
        backbone (RoFormerModel): Transformer encoder backbone.

    Example:
        >>> encoder = ProteoformerEncoder(config)
        >>> outputs = encoder(input_ids, attention_mask=attention_mask)
        >>> embeddings = outputs["last_hidden_state"]
    """

    config_class = ProteoformerConfig

    def __init__(self, config: ProteoformerConfig):
        """
        Initialize the encoder.

        Args:
            config (ProteoformerConfig): Configuration object specifying
                model architecture and hyperparameters.
        """
        super().__init__(config)
        self.config = config
        self.backbone = RoFormerModel(config.get_roformer_config())
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode proteoform sequences into contextualized representations.

        Args:
            input_ids (torch.Tensor): Tokenized proteoform sequences.
                Shape: `(batch_size, sequence_length)`
            attention_mask (Optional[torch.Tensor]): Mask for padding tokens.
                Shape: `(batch_size, sequence_length)`
                1 for real tokens, 0 for padding.
            token_type_ids (Optional[torch.Tensor]): Segment IDs for multi-segment inputs.
                Shape: `(batch_size, sequence_length)`
            head_mask (Optional[torch.Tensor]): Mask to nullify specific attention heads.
            inputs_embeds (Optional[torch.Tensor]): Pre-computed embeddings as
                alternative to input_ids.
            output_attentions (Optional[bool]): Return attention weights if True.
            output_hidden_states (Optional[bool]): Return all hidden states if True.
            return_dict (Optional[bool]): Return dictionary output if True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "last_hidden_state": Final layer hidden states.
                  Shape: `(batch_size, sequence_length, hidden_size)`
                - "backbone_outputs": Complete backbone outputs including
                  pooler output, attentions, and hidden states if requested.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]
        return {"last_hidden_state": last_hidden_state, "backbone_outputs": outputs}


# ==============================================================================
# Pre-training and Inference Models
# ==============================================================================

class ProteoformerForMLM(PreTrainedModel):
    """
    Proteoformer model for Masked Language Model pre-training.

    This model learns proteoform representations by predicting randomly masked
    tokens in the input sequence. The pre-training objective enables the model
    to capture:
        - Amino acid sequence patterns
        - PTM-context relationships
        - Long-range dependencies in proteoform sequences

    The learned representations can be transferred to various downstream tasks
    including PTM site prediction, protein function annotation, and
    proteoform classification.

    Architecture:
        - ProteoformerEncoder: Transformer encoder with RoPE
        - ProteoformerMLMHead: Vocabulary prediction head

    Attributes:
        config (ProteoformerConfig): Model configuration.
        encoder (ProteoformerEncoder): Transformer encoder.
        mlm_head (ProteoformerMLMHead): MLM prediction head.

    Example:
        >>> model = ProteoformerForMLM(config)
        >>> outputs = model(input_ids, attention_mask=mask, labels=labels)
        >>> loss = outputs.loss  # Use for backpropagation
        >>> predictions = outputs.logits.argmax(dim=-1)
    """

    config_class = ProteoformerConfig

    def __init__(self, config: ProteoformerConfig):
        """
        Initialize the MLM model.

        Args:
            config (ProteoformerConfig): Configuration specifying model
                architecture, vocabulary size, and training parameters.
        """
        super().__init__(config)
        self.config = config
        self.encoder = ProteoformerEncoder(config)
        self.mlm_head = ProteoformerMLMHead(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Get the input embedding layer.

        Returns:
            nn.Embedding: Word embedding layer mapping token IDs to vectors.
        """
        return self.encoder.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """
        Set the input embedding layer.

        Args:
            value (nn.Embedding): New embedding layer to use.
        """
        self.encoder.backbone.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> ProteoformerMLMOutput:
        """
        Perform forward pass for MLM pre-training.

        Args:
            input_ids (torch.Tensor): Tokenized proteoform sequences with
                masked tokens.
                Shape: `(batch_size, sequence_length)`
            attention_mask (Optional[torch.Tensor]): Mask for padding tokens.
                Shape: `(batch_size, sequence_length)`
            token_type_ids (Optional[torch.Tensor]): Segment IDs.
                Shape: `(batch_size, sequence_length)`
            head_mask (Optional[torch.Tensor]): Attention head mask.
            inputs_embeds (Optional[torch.Tensor]): Pre-computed embeddings.
            labels (Optional[torch.Tensor]): Ground truth token IDs for MLM.
                Shape: `(batch_size, sequence_length)`
                Use -100 for tokens that should not contribute to loss.
            output_attentions (Optional[bool]): Return attention weights.
            output_hidden_states (Optional[bool]): Return all hidden states.
            return_dict (Optional[bool]): Return ProteoformerMLMOutput if True.

        Returns:
            ProteoformerMLMOutput: Model outputs containing:
                - loss: MLM loss (if labels provided)
                - logits: Vocabulary prediction scores
                - last_hidden_state: Final encoder hidden states
        """
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = enc["last_hidden_state"]
        prediction_scores = self.mlm_head(sequence_output)

        # Compute MLM loss if labels are provided
        mlm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # Ignores -100 indices by default
            mlm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores, sequence_output)
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return ProteoformerMLMOutput(
            loss=mlm_loss,
            logits=prediction_scores,
            last_hidden_state=sequence_output
        )

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, ProteoformerMLMOutput]:
        """
        Extract proteoform embeddings without gradient computation.

        This method provides a convenient interface for embedding extraction
        during inference, bypassing the MLM head for efficiency.

        Args:
            input_ids (torch.Tensor): Tokenized proteoform sequences.
                Shape: `(batch_size, sequence_length)`
            attention_mask (Optional[torch.Tensor]): Padding mask.
                Shape: `(batch_size, sequence_length)`
            return_dict (bool): If False, return only hidden states tensor.
                If True, return full ProteoformerMLMOutput.

        Returns:
            Union[torch.Tensor, ProteoformerMLMOutput]:
                - If return_dict=False: Hidden states tensor
                  Shape: `(batch_size, sequence_length, hidden_size)`
                - If return_dict=True: Full model output
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state if return_dict is False else outputs


class ProteoformerForEmbedding(PreTrainedModel):
    """
    Proteoformer model optimized for embedding extraction.

    This model variant is designed for inference tasks where only proteoform
    embeddings are needed. It excludes the MLM head, reducing memory footprint
    and computational overhead compared to ProteoformerForMLM.

    Use Cases:
        - Proteoform similarity search
        - PTM site prediction (as input features)
        - Proteoform clustering and visualization
        - Transfer learning for downstream tasks

    The model can load weights directly from a pre-trained ProteoformerForMLM
    checkpoint, automatically ignoring the MLM head parameters.

    Attributes:
        config (ProteoformerConfig): Model configuration.
        encoder (ProteoformerEncoder): Transformer encoder.

    Example:
        >>> # Load from pre-trained MLM model
        >>> model = ProteoformerForEmbedding.from_pretrained("path/to/checkpoint")
        >>> 
        >>> # Extract embeddings
        >>> with torch.no_grad():
        ...     outputs = model(input_ids, attention_mask=mask)
        ...     embeddings = outputs.last_hidden_state
        >>> 
        >>> # Get sequence-level representation (e.g., mean pooling)
        >>> seq_embedding = embeddings.mean(dim=1)
    """

    config_class = ProteoformerConfig

    def __init__(self, config: ProteoformerConfig):
        """
        Initialize the embedding model.

        Args:
            config (ProteoformerConfig): Configuration specifying
                encoder architecture and hyperparameters.
        """
        super().__init__(config)
        self.config = config
        self.encoder = ProteoformerEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Get the input embedding layer.

        Returns:
            nn.Embedding: Word embedding layer.
        """
        return self.encoder.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """
        Set the input embedding layer.

        Args:
            value (nn.Embedding): New embedding layer.
        """
        self.encoder.backbone.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ProteoformerEmbeddingOutput:
        """
        Extract proteoform embeddings from input sequences.

        Args:
            input_ids (torch.Tensor): Tokenized proteoform sequences.
                Shape: `(batch_size, sequence_length)`
            attention_mask (Optional[torch.Tensor]): Mask for padding tokens.
                Shape: `(batch_size, sequence_length)`
            token_type_ids (Optional[torch.Tensor]): Segment token IDs.
                Shape: `(batch_size, sequence_length)`
            head_mask (Optional[torch.Tensor]): Attention head mask.
            inputs_embeds (Optional[torch.Tensor]): Pre-computed embeddings.
            output_attentions (Optional[bool]): Return attention weights.
            output_hidden_states (Optional[bool]): Return all layer hidden states.

        Returns:
            ProteoformerEmbeddingOutput: Output containing:
                - last_hidden_state: PTM-aware token embeddings.
                  Shape: `(batch_size, sequence_length, hidden_size)`
                - backbone_outputs: Full encoder outputs including attention
                  weights and intermediate states if requested.
        """
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = enc["last_hidden_state"]
        backbone_outputs = enc["backbone_outputs"]

        return ProteoformerEmbeddingOutput(
            last_hidden_state=sequence_output,
            backbone_outputs=backbone_outputs
        )
