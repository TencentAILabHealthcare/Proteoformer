"""
Proteoformer: Proteoform sequence modeling toolkit.

This package provides tools for proteoform sequence tokenization and modeling
using transformer-based architectures with PTM-aware representations.
"""

__version__ = "1.0.0"

from .config import ProteoformerConfig
from .tokenization import ProteoformerTokenizer
from .net import (
    ProteoformerMLMHead,
    ProteoformerEncoder,
    ProteoformerForMLM,
    ProteoformerForEmbedding,
)
from .models import ProteoformerClassifier

__all__ = [
    # Configuration
    "ProteoformerConfig",
    # Tokenization
    "ProteoformerTokenizer",
    # Model components
    "ProteoformerMLMHead",
    "ProteoformerEncoder",
    "ProteoformerForMLM",
    "ProteoformerForEmbedding",
    "ProteoformerClassifier",
]