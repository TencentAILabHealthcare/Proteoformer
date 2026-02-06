import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ProteoformerClassifier(nn.Module):
    """
    Binary classifier for PTM site prediction.
    
    Uses a pretrained Proteoformer model as backbone, followed by a two-layer CNN
    classifier to predict whether a given protein sequence position is a PTM site.
    
    Architecture:
        Proteoformer -> Conv1d -> BN -> ReLU -> MaxPool -> Conv1d -> BN -> ReLU -> MaxPool -> AdaptiveMaxPool -> FC
    """
    
    def __init__(self, proteoformer_model, embedding_dim, hidden_size=512):
        """
        Args:
            proteoformer_model: Pretrained Proteoformer backbone model
            embedding_dim: Dimension of Proteoformer output embeddings
            hidden_size: Hidden dimension for CNN layers (default: 512)
        """
        super().__init__()
        self.proteoformer_model = proteoformer_model
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Two-layer CNN classifier for binary classification
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Global pooling to handle variable sequence lengths
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Final fully connected layer for binary classification
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, return_logits=False):
        """Forward pass through the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_logits: If True, return logits; if False, return probabilities
            
        Returns:
            logits (batch,) if return_logits=True - raw logits for BCEWithLogitsLoss
            probs (batch,) if return_logits=False - probabilities after sigmoid
        """
        outputs = self.proteoformer_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, embedding_dim)
        
        # Transpose for Conv1d: (batch, seq_len, embedding_dim) -> (batch, embedding_dim, seq_len)
        x = last_hidden.transpose(1, 2)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Global pooling to handle variable lengths: (batch, hidden_size, seq_len) -> (batch, hidden_size, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, hidden_size, 1) -> (batch, hidden_size)
        x = x.view(x.size(0), -1)
        
        # Dropout and fully connected layer
        x = self.dropout(x)
        logits = self.fc(x).squeeze(1)  # (batch, 1) -> (batch,)
        
        if return_logits:
            return logits
        else:
            # Return probabilities for inference (apply sigmoid)
            probs = torch.sigmoid(logits)
            return probs
        
class PTMFucntionalClassifier(nn.Module):
    """
    Classifier for predicting functional association between PTM pairs.
    
    Takes precomputed PTM embeddings and predicts whether two PTM sites
    are functionally associated using enhanced pair-wise features.
    
    Feature combination: [h1, h2, |h1 - h2|, h1 ⊙ h2]
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        """
        Initialize the classifier.
        
        Args:
            embedding_dim: Dimension of PTM embeddings
            hidden_dim: Hidden dimension for FC layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),  # Changed from 2* to 4* for enhanced features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )
        
    
    def forward(
        self,
        ptm1_embeddings: torch.Tensor,
        ptm2_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            ptm1_embeddings: PTM1 embeddings (batch_size, embedding_dim)
            ptm2_embeddings: PTM2 embeddings (batch_size, embedding_dim)
        
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        # Enhanced pair-wise features: [h1, h2, |h1 - h2|, h1 ⊙ h2]
        absolute_diff = torch.abs(ptm1_embeddings - ptm2_embeddings)
        elementwise_product = ptm1_embeddings * ptm2_embeddings
        
        combined_embeddings = torch.cat([
            ptm1_embeddings,       # h1: first PTM embedding
            ptm2_embeddings,       # h2: second PTM embedding
            absolute_diff,         # |h1 - h2|: capture distance
            elementwise_product    # h1 ⊙ h2: capture interaction
        ], dim=1)
        
        logits = self.classifier(combined_embeddings)
        
        return logits

class VariantPTMClassifier(nn.Module):
    """
    Binary classifier for variant PTM effect prediction using delta embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        """
        Initialize the classifier.
        
        Args:
            embedding_dim: Dimension of delta embeddings
            hidden_dim: Hidden dimension for FC layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )
        
        logger.info(f"Classifier initialized with embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, delta_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            delta_embeddings: Delta embeddings (batch_size, embedding_dim)
        
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        logits = self.classifier(delta_embeddings)
        return logits