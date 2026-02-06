#!/bin/bash

# =============================================================================
# π-Proteoformer Embedding Extraction Script
# =============================================================================
# This script runs the extract_embedding.py to generate embeddings from 
# proteoform sequences.
#
# Usage:
#   bash run_embedding.sh
#
# Before running, please modify the following variables according to your setup:
#   - MODEL_PATH: Path to your pretrained proteoformer checkpoint
#   - INPUT_FILE or SEQUENCE: Your input sequences
#   - OUTPUT_FILE: Where to save the embeddings
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Configuration - MODIFY THESE VARIABLES
# =============================================================================

# Path to the pretrained proteoformer checkpoint
# Replace this with your actual checkpoint path
MODEL_PATH="CHECKPOINT_PATH"

# Tokenizer name (default: proteoformer-pfmod)
TOKENIZER_NAME="proteoformer-pfmod"

# Batch size for processing
BATCH_SIZE=32

# Maximum sequence length
MAX_LENGTH=1024

# Pooling method: "cls" or "mean"
POOLING="mean"

# Device: "cuda" or "cpu" (leave empty for auto-detection)
DEVICE=""

# =============================================================================
# Example 1: Process a single sequence
# =============================================================================
echo "=== Example 1: Single Sequence Processing ==="

SEQUENCE="MYPK[PFMOD:21]TEINSEQUENCE"

python "${SCRIPT_DIR}/extract_embedding.py" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --sequence "${SEQUENCE}" \
    --pooling "${POOLING}" \
    --max_length ${MAX_LENGTH}

echo ""

# =============================================================================
# Example 2: Process sequences from a file
# =============================================================================
# Uncomment and modify the following section to process sequences from a file

# echo "=== Example 2: Batch Processing from File ==="
# 
# INPUT_FILE="/path/to/your/sequences.txt"
# OUTPUT_FILE="${SCRIPT_DIR}/output_embeddings.pt"
# 
# python "${SCRIPT_DIR}/extract_embedding.py" \
#     --model_path "${MODEL_PATH}" \
#     --tokenizer_name "${TOKENIZER_NAME}" \
#     --input_file "${INPUT_FILE}" \
#     --output_file "${OUTPUT_FILE}" \
#     --batch_size ${BATCH_SIZE} \
#     --max_length ${MAX_LENGTH} \
#     --pooling "${POOLING}" \
#     --save_sequences
# 
# echo "Embeddings saved to: ${OUTPUT_FILE}"

# =============================================================================
# Example 3: Process with GPU
# =============================================================================
# Uncomment to explicitly use GPU

# echo "=== Example 3: GPU Processing ==="
# 
# CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/extract_embedding.py" \
#     --model_path "${MODEL_PATH}" \
#     --tokenizer_name "${TOKENIZER_NAME}" \
#     --input_file "${INPUT_FILE}" \
#     --output_file "${OUTPUT_FILE}" \
#     --device cuda \
#     --batch_size ${BATCH_SIZE}

# =============================================================================
# Example 4: Use mean pooling
# =============================================================================
# Uncomment to use mean pooling instead of CLS token

# echo "=== Example 4: Mean Pooling ==="
# 
# python "${SCRIPT_DIR}/extract_embedding.py" \
#     --model_path "${MODEL_PATH}" \
#     --tokenizer_name "${TOKENIZER_NAME}" \
#     --input_file "${INPUT_FILE}" \
#     --output_file "${OUTPUT_FILE}_mean.pt" \
#     --pooling mean \
#     --batch_size ${BATCH_SIZE}

echo "=== Script completed ==="
