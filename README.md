# Proteoformer

A foundation model for comprehensive proteoform modeling and representation.

## Overview
**Proteoforms**—the diverse molecular forms of proteins arising from genetic variations, alternative splicing, and post-translational modifications (PTMs)—are the true functional units of the proteome. Unlike traditional protein models that focus solely on canonical sequences, **Proteoformer** is the first foundation model specifically designed to understand and represent the full complexity of proteoform diversity.

## Installation

### Option 1: Using Docker (Recommended)

We provide a pre-configured Docker image with all dependencies installed:

```bash
docker pull devinjzhu/proteoformer:v1.0
```

**Step 1: Clone the repository to your local machine**

```bash
# Clone the repository on your local machine first
git clone https://github.com/TencentAILabHealthcare/Proteoformer.git
```

**Step 2: Start the Docker container with GPU support and mount your local directory**

```bash
# Run the container interactively with GPU support
# Mount the cloned repository directory to /workspace in the container
# Mount the checkpoint directory if you put them in elseswhere
docker run -it --gpus all -v /path/to/Proteoformer:/workspace devinjzhu/proteoformer:v1.0
```

**Step 3: Navigate to the mounted directory inside the container**

```bash
# Inside the container, navigate to the mounted workspace
cd /workspace
```

**Step 4: Install the package**

```bash
pip install -e .
```

**Step 5: Verify the installation**

```bash
# Test the installation by importing the package
python -c "import proteoformer; print('Installation successful!')"
```

After successful installation, you can start using the model. See [Command-Line Embedding Extraction Scripts](#command-line-embedding-extraction-scripts) for ready-to-use examples.

### Option 2: Manual Installation

#### Requirements

- Python >= 3.10
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

#### Setup

1. Clone the repository:
```bash
git clone https://github.com/TencentAILabHealthcare/Proteoformer.git
cd Proteoformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from proteoformer.tokenization import ProteoformerTokenizer

# Load tokenizer
tokenizer = ProteoformerTokenizer.from_pretrained("proteoformer-pfmod")

# Tokenize a sequence with PTM annotations
sequence = "MYPK[PFMOD:21]TEIN[PFMOD:1]SEQUENCE"
tokens = tokenizer.tokenize(sequence)
print(tokens)
# Output: ['M', 'Y', 'P', 'K', '[PFMOD:21]', 'T', 'E', 'I', 'N', '[PFMOD:1]', 'S', 'E', 'Q', 'U', 'E', 'N', 'C', 'E']

# Encode to IDs
input_ids = tokenizer.encode(sequence, add_special_tokens=True)
```

### Batch Processing

```python
sequences = [
    "PROTEIN[PFMOD:1]SEQUENCE",
    "ANOTHER[PFMOD:21]PROTEIN",
    "MODIFIED[PFMOD:35]PEPTIDE"
]

# Tokenize and encode
encoded = tokenizer(
    sequences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print("Input IDs shape:", encoded['input_ids'].shape)
```

### Extracting Proteoform Embeddings

Extract sequence embeddings from π-Proteoformer for downstream tasks:

```python
import torch
from proteoformer.tokenization import ProteoformerTokenizer
from proteoformer.net import ProteoformerForEmbedding

# Load model and tokenizer
model_path = "/path/to/your/proteoformer/checkpoint"
tokenizer = ProteoformerTokenizer.from_pretrained("proteoformer-pfmod")
model = ProteoformerForEmbedding.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare input sequences with PTM annotations
sequences = [
    "MYPK[PFMOD:21]TEINSEQUENCE",  # Phosphorylated sequence
    "ANOTHER[PFMOD:1]PROTEIN",      # Acetylated sequence
]

# Tokenize sequences
inputs = tokenizer(
    sequences,
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors="pt"
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Extract embeddings
with torch.no_grad():
    outputs = model(**inputs)
    
    # Get the last hidden states (token-level embeddings)
    token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
    
    # Get sequence-level embedding using [CLS] token
    cls_embedding = token_embeddings[:, 0, :]  # Shape: [batch_size, hidden_dim]
    
    # Alternative: Mean pooling over all tokens (excluding padding)
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    mean_embedding = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

print(f"CLS Embedding shape: {cls_embedding.shape}")
print(f"Mean Pooled Embedding shape: {mean_embedding.shape}")
```

### Batch Embedding Extraction for Large Datasets

For processing large-scale proteoform datasets:

```python
import torch
from torch.utils.data import DataLoader, Dataset

class ProteoformDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=1024):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.tokenizer(
            self.sequences[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

def collate_fn(batch):
    return {
        'input_ids': torch.cat([item['input_ids'] for item in batch], dim=0),
        'attention_mask': torch.cat([item['attention_mask'] for item in batch], dim=0)
    }

# Create dataloader
dataset = ProteoformDataset(sequences, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Extract embeddings in batches
all_embeddings = []
with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_emb.cpu())

# Concatenate all embeddings
embeddings = torch.cat(all_embeddings, dim=0)
print(f"Total embeddings shape: {embeddings.shape}")
```

### Command-Line Embedding Extraction Scripts

For convenience, we provide ready-to-use command-line scripts in the `examples/` directory.

**Usage Examples:**

```bash
# Navigate to examples directory
cd examples/

# Example 1: Extract embedding for a single sequence
python extract_embedding.py \
    --model_path /path/to/your/proteoformer/checkpoint \
    --sequence "MYPK[PFMOD:21]TEINSEQUENCE"

# Example 2: Extract embeddings from a file and save to output
python extract_embedding.py \
    --model_path /path/to/your/proteoformer/checkpoint \
    --input_file /path/to/sequences.txt \
    --output_file embeddings.pt \
    --batch_size 32 \
    --save_sequences

# Example 3: Use mean pooling instead of CLS token
python extract_embedding.py \
    --model_path /path/to/your/proteoformer/checkpoint \
    --input_file /path/to/sequences.txt \
    --output_file embeddings_mean.pt \
    --pooling mean

# Example 4: Process with specific GPU
CUDA_VISIBLE_DEVICES=0 python extract_embedding.py \
    --model_path /path/to/your/proteoformer/checkpoint \
    --input_file /path/to/sequences.txt \
    --output_file embeddings.pt \
    --device cuda
```

#### Shell Script: `run_embedding.sh`

A convenient wrapper script with pre-configured examples:

```bash
# Make the script executable (if needed)
chmod +x run_embedding.sh

# Edit the script to set your MODEL_PATH, then run:
bash run_embedding.sh
```

**Configuration variables in `run_embedding.sh`:**

```bash
MODEL_PATH="/path/to/your/proteoformer/checkpoint"  # Your checkpoint path
TOKENIZER_NAME="proteoformer-pfmod"                 # Tokenizer name
BATCH_SIZE=32                                        # Batch size
MAX_LENGTH=1024                                      # Max sequence length
POOLING="cls"                                        # Pooling method
```

## Vocabulary

The tokenizer uses a vocabulary of 2000 tokens including:
- **Special tokens**: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- **Amino acid tokens**: Standard 20 amino acids
- **PTM tokens**: 1524 post-translational modification tokens (`[PFMOD:XXX]`)

Common PTM tokens:
| Token | Description |
|-------|-------------|
| `[PFMOD:1]` | Acetylation |
| `[PFMOD:21]` | Phosphorylation |

For detailed vocabulary documentation, see [resource/README.md](resource/README.md).

## License

This project is licensed under the Apache License 2.0.

Proteoformer is developed and maintained by **Tencent AI for Life Sciences Lab**. Copyright © 2026 Tencent AI for Life Sciences Lab. All rights reserved.

## Contact

For questions and issues, please open an issue on GitHub.