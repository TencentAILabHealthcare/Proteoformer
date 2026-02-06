#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
π-Proteoformer Embedding Extraction Script

This script extracts embeddings from proteoform sequences using the π-Proteoformer model.
It supports both single sequence and batch processing from file input.

Usage:
    python extract_embedding.py --model_path /path/to/checkpoint --sequence "MYPK[PFMOD:21]TEIN"
    python extract_embedding.py --model_path /path/to/checkpoint --input_file sequences.txt --output_file embeddings.pt
"""

import argparse
import torch
import os
import sys
from typing import List, Optional

from torch.utils.data import DataLoader, Dataset

# Add the project root to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from proteoformer.tokenization import ProteoformerTokenizer
from proteoformer.net import ProteoformerForEmbedding


class ProteoformDataset(Dataset):
    """Dataset class for proteoform sequences."""
    
    def __init__(self, sequences: List[str], tokenizer: ProteoformerTokenizer, max_length: int = 1024):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of proteoform sequences
            tokenizer: ProteoformerTokenizer instance
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        return self.tokenizer(
            self.sequences[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for DataLoader."""
    return {
        'input_ids': torch.cat([item['input_ids'] for item in batch], dim=0),
        'attention_mask': torch.cat([item['attention_mask'] for item in batch], dim=0)
    }


def load_sequences_from_file(file_path: str) -> List[str]:
    """
    Load sequences from a text file (one sequence per line).
    
    Args:
        file_path: Path to the input file
        
    Returns:
        List of sequences
    """
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                sequences.append(line)
    return sequences


def extract_embeddings(
    model: torch.nn.Module,
    tokenizer: ProteoformerTokenizer,
    sequences: List[str],
    batch_size: int = 32,
    max_length: int = 1024,
    device: torch.device = None,
    pooling_method: str = "cls",
    show_progress: bool = True
) -> torch.Tensor:
    """
    Extract embeddings from sequences.
    
    Args:
        model: The proteoformer model
        tokenizer: The tokenizer
        sequences: List of sequences to process
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on
        pooling_method: Pooling method ("cls" or "mean")
        show_progress: Whether to show progress
        
    Returns:
        Tensor of embeddings [num_sequences, hidden_dim]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    dataset = ProteoformDataset(sequences, tokenizer, max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        shuffle=False
    )
    
    all_embeddings = []
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if show_progress:
                print(f"Processing batch {i + 1}/{total_batches}...")
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Get token-level embeddings
            token_embeddings = outputs.last_hidden_state
            
            if pooling_method == "cls":
                # Use [CLS] token embedding
                emb = token_embeddings[:, 0, :]
            elif pooling_method == "mean":
                # Mean pooling over all tokens (excluding padding)
                attention_mask = batch['attention_mask'].unsqueeze(-1)
                emb = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")
            
            all_embeddings.append(emb.cpu())
    
    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from proteoform sequences using π-Proteoformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embedding for a single sequence
  python extract_embedding.py --model_path /path/to/checkpoint --sequence "MYPK[PFMOD:21]TEIN"
  
  # Extract embeddings from a file
  python extract_embedding.py --model_path /path/to/checkpoint --input_file sequences.txt --output_file embeddings.pt
  
  # Use mean pooling instead of CLS token
  python extract_embedding.py --model_path /path/to/checkpoint --input_file sequences.txt --pooling mean
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/your/proteoformer/checkpoint",
        help="Path to the pretrained proteoformer checkpoint"
    )
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sequence",
        type=str,
        help="Single proteoform sequence to process"
    )
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Path to input file containing sequences (one per line)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the embeddings (as .pt file). If not specified, embeddings are printed to stdout"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "mean"],
        default="cls",
        help="Pooling method for sequence embedding (default: cls)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="proteoformer-pfmod",
        help="Name of the tokenizer to use (default: proteoformer-pfmod)"
    )
    parser.add_argument(
        "--save_sequences",
        action="store_true",
        help="Save sequences along with embeddings in output file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = ProteoformerTokenizer.from_pretrained(args.tokenizer_name)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = ProteoformerForEmbedding.from_pretrained(args.model_path)
    
    # Prepare sequences
    if args.sequence:
        sequences = [args.sequence]
        print(f"Processing single sequence: {args.sequence}")
    else:
        sequences = load_sequences_from_file(args.input_file)
        print(f"Loaded {len(sequences)} sequences from {args.input_file}")
    
    # Extract embeddings
    print(f"Extracting embeddings with {args.pooling} pooling...")
    embeddings = extract_embeddings(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        pooling_method=args.pooling,
        show_progress=not args.quiet
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save or print results
    if args.output_file:
        output_data = {"embeddings": embeddings}
        if args.save_sequences:
            output_data["sequences"] = sequences
        torch.save(output_data, args.output_file)
        print(f"Embeddings saved to: {args.output_file}")
    else:
        print("\nEmbeddings:")
        for i, (seq, emb) in enumerate(zip(sequences, embeddings)):
            print(f"\nSequence {i + 1}: {seq[:50]}{'...' if len(seq) > 50 else ''}")
            print(f"Embedding (first 10 dims): {emb[:10].tolist()}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
