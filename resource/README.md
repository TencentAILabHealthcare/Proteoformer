# Proteoformer Vocabulary and Tokenization

This directory contains vocabulary files and resources for the Proteoformer tokenizer, which is designed to handle proteoform sequences with post-translational modifications (PTMs).

## Vocabulary Files

### proteoformer-pfmod-2000.json

The main vocabulary file containing:
- **vocab**: Token-to-ID mapping (2000 tokens total)
- **mod_dict**: Modification description mapping (1524 PTM annotations)

#### File Structure

```json
{
  "vocab": {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "A": 5,
    "C": 6,
    ...
    "[PFMOD:1]": 30,
    "[PFMOD:21]": 45,
    "[PFMOD:213]": 191,
    ...
  },
  "mod_dict": {
    "[PFMOD:1]": "Acetylation",
    "[PFMOD:21]": "Phosphorylation",
    "[PFMOD:213]": "ADP  Ribose addition",
    ...
  }
}
```

## Modification Dictionary (mod_dict)

The `mod_dict` provides a mapping between PFMOD tokens and their corresponding modification types, enabling quick lookup of modification descriptions.

### Key Statistics

- **Total tokens**: 2000
- **PTM tokens**: 1524

### Common Modifications

| Token | ID | Description |
|-------|-----|-------------|
| `[PFMOD:1]` | 30 | Acetylation |
| `[PFMOD:21]` | 45 | Phosphorylation |
| `[PFMOD:35]` | 57 | Oxidation or Hydroxylation |
| `[PFMOD:7]` | 36 | Deamidation |
| `[PFMOD:213]` | 191 | ADP  Ribose addition |

## Usage

### 1. Direct JSON Access

```python
import json

# Load vocabulary file
with open('proteoformer-pfmod-2000.json', 'r') as f:
    vocab_data = json.load(f)

# Access modification descriptions
mod_dict = vocab_data['mod_dict']
print(mod_dict['[PFMOD:213]'])  # Output: ADP  Ribose addition

# Get token ID
token_id = vocab_data['vocab']['[PFMOD:213]']
print(token_id)  # Output: 191
```

### 2. Using ProteoformerTokenizer

The `ProteoformerTokenizer` class automatically loads the `mod_dict` and provides convenient methods for working with modifications.

#### Basic Usage

```python
from proteoformer.tokenization import ProteoformerTokenizer

# Load tokenizer
tokenizer = ProteoformerTokenizer.from_pretrained("proteoformer-pfmod")

# Method 1: Query by token string
desc = tokenizer.get_mod_description('[PFMOD:213]')
print(desc)  # Output: ADP  Ribose addition

# Method 2: Query by token ID
token_id = tokenizer.convert_tokens_to_ids(['[PFMOD:213]'])[0]
desc = tokenizer.get_mod_description(token_id)
print(desc)  # Output: ADP  Ribose addition

# Method 3: Direct access to mod_dict
print(tokenizer.mod_dict['[PFMOD:1]'])  # Output: Acetylation
```

#### Tokenizing Proteoform Sequences

```python
# Tokenize a sequence with modifications
sequence = "MYPK[PFMOD:21]TEIN[PFMOD:1]SEQUENCE"
tokens = tokenizer.tokenize(sequence)
print(tokens)
# Output: ['M', 'Y', 'P', 'K', '[PFMOD:21]', 'T', 'E', 'I', 'N', '[PFMOD:1]', 'S', 'E', 'Q', 'U', 'E', 'N', 'C', 'E']

# Encode to IDs
input_ids = tokenizer.encode(sequence, add_special_tokens=True)
print(input_ids)

# Decode back to sequence
decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
print(decoded)
```

#### Displaying Modification Information

```python
sequence = "MYPK[PFMOD:21]TEIN[PFMOD:1]SEQUENCE"
tokens = tokenizer.tokenize(sequence)

print(f"Sequence: {sequence}")
print("Modifications found:")
for token in tokens:
    if token.startswith('[PFMOD:'):
        desc = tokenizer.get_mod_description(token)
        token_id = tokenizer.convert_tokens_to_ids([token])[0]
        print(f"  {token} (ID: {token_id}) -> {desc}")

# Output:
# Sequence: MYPK[PFMOD:21]TEIN[PFMOD:1]SEQUENCE
# Modifications found:
#   [PFMOD:21] (ID: 45) -> Phosphorylation
#   [PFMOD:1] (ID: 30) -> Acetylation
```

#### Searching for Specific Modification Types

```python
# Find all phosphorylation-related modifications
phospho_mods = {
    token: desc 
    for token, desc in tokenizer.mod_dict.items() 
    if 'Phosphorylation' in desc
}

print("Phosphorylation modifications:")
for token, desc in phospho_mods.items():
    token_id = tokenizer.convert_tokens_to_ids([token])[0]
    print(f"  {token} (ID: {token_id}) -> {desc}")

# Find all methylation-related modifications
methyl_mods = {
    token: desc 
    for token, desc in tokenizer.mod_dict.items() 
    if 'Methylation' in desc
}

print(f"\nFound {len(methyl_mods)} methylation modifications")
```

#### Batch Processing

```python
# Process multiple sequences
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
print("Attention mask shape:", encoded['attention_mask'].shape)

# Decode and show modifications
for i, seq in enumerate(sequences):
    print(f"\nSequence {i+1}: {seq}")
    tokens = tokenizer.tokenize(seq)
    for token in tokens:
        if token.startswith('[PFMOD:'):
            desc = tokenizer.get_mod_description(token)
            print(f"  - {token}: {desc}")
```

### 3. Advanced Usage

#### Custom Modification Analysis

```python
def analyze_modifications(sequence, tokenizer):
    """Analyze all modifications in a proteoform sequence."""
    tokens = tokenizer.tokenize(sequence)
    modifications = []
    
    for i, token in enumerate(tokens):
        if token.startswith('[PFMOD:'):
            desc = tokenizer.get_mod_description(token)
            token_id = tokenizer.convert_tokens_to_ids([token])[0]
            
            # Get the modified amino acid (previous token)
            aa = tokens[i-1] if i > 0 else None
            
            modifications.append({
                'position': i,
                'amino_acid': aa,
                'modification_token': token,
                'modification_id': token_id,
                'modification_type': desc
            })
    
    return modifications

# Example usage
sequence = "MYPK[PFMOD:21]TEIN[PFMOD:1]SEQUENCE"
mods = analyze_modifications(sequence, tokenizer)

print(f"Sequence: {sequence}")
print(f"Found {len(mods)} modifications:\n")
for mod in mods:
    print(f"Position {mod['position']}:")
    print(f"  Amino acid: {mod['amino_acid']}")
    print(f"  Modification: {mod['modification_token']}")
    print(f"  Type: {mod['modification_type']}")
    print(f"  Token ID: {mod['modification_id']}\n")
```

#### Modification Statistics

```python
def get_modification_statistics(tokenizer):
    """Get statistics about modification types in the vocabulary."""
    stats = {}
    
    for token, desc in tokenizer.mod_dict.items():
        # Categorize by common modification types
        if 'Phosphorylation' in desc:
            category = 'Phosphorylation'
        elif 'Acetylation' in desc:
            category = 'Acetylation'
        elif 'Methylation' in desc:
            category = 'Methylation'
        elif 'Ubiquitin' in desc:
            category = 'Ubiquitination'
        elif 'Oxidation' in desc:
            category = 'Oxidation'
        else:
            category = 'Other'
        
        if category not in stats:
            stats[category] = []
        stats[category].append((token, desc))
    
    return stats

# Example usage
stats = get_modification_statistics(tokenizer)

print("Modification Type Statistics:")
print("-" * 50)
for category, mods in sorted(stats.items()):
    print(f"\n{category}: {len(mods)} modifications")
    for token, desc in mods[:3]:  # Show first 3 examples
        print(f"  - {token}: {desc}")
    if len(mods) > 3:
        print(f"  ... and {len(mods) - 3} more")
```


### ProteoformerTokenizer

#### Attributes

- **vocab** (dict): Token-to-ID mapping
- **ids_to_tokens** (dict): ID-to-token mapping
- **mod_dict** (dict): PFMOD token to modification description mapping

#### Methods

##### `from_pretrained(pretrained_model_name_or_path)`

Load a pretrained tokenizer.

**Parameters:**
- `pretrained_model_name_or_path` (str): Model name or path
  - Predefined models: `"proteoformer-pfmod"`, `"proteoformer-base"`
  - Or path to directory containing vocabulary file

**Returns:**
- `ProteoformerTokenizer` instance

##### `tokenize(sequence)`

Tokenize a proteoform sequence.

**Parameters:**
- `sequence` (str): Proteoform sequence with modifications

**Returns:**
- `List[str]`: List of tokens

##### `encode(sequence, add_special_tokens=True, ...)`

Encode a sequence to token IDs.

**Parameters:**
- `sequence` (str): Proteoform sequence
- `add_special_tokens` (bool): Whether to add special tokens

**Returns:**
- `List[int]`: List of token IDs

##### `decode(token_ids, skip_special_tokens=True)`

Decode token IDs back to sequence.

**Parameters:**
- `token_ids` (List[int]): List of token IDs
- `skip_special_tokens` (bool): Whether to skip special tokens

**Returns:**
- `str`: Decoded sequence

##### `get_mod_description(token)`

Get modification description for a PFMOD token.

**Parameters:**
- `token` (Union[str, int]): PFMOD token string (e.g., `"[PFMOD:213]"`) or token ID

**Returns:**
- `Optional[str]`: Modification description, or `None` if not found

**Examples:**
```python
tokenizer.get_mod_description('[PFMOD:213]')  # Returns: 'ADP  Ribose addition'
tokenizer.get_mod_description(191)  # Returns: 'ADP  Ribose addition'
tokenizer.get_mod_description('[PFMOD:999]')  # Returns: None
```

## Notes

1. **Not all PFMOD IDs are in the vocabulary**: The vocabulary contains 1524 out of thousands of possible PFMOD modifications. Only modifications present in the vocabulary have corresponding descriptions.

2. **Token ID consistency**: Token IDs are fixed in the vocabulary file. Always use the same vocabulary file for encoding and decoding to ensure consistency.

3. **Special tokens**: The tokenizer includes special tokens:
   - `[PAD]` (ID: 0): Padding token
   - `[UNK]` (ID: 1): Unknown token
   - `[CLS]` (ID: 2): Classification token
   - `[SEP]` (ID: 3): Separator token
   - `[MASK]` (ID: 4): Mask token for masked language modeling

4. **Modification format**: Modifications must be in the format `[PFMOD:XXX]` where XXX is the PFMOD ID number.

## Data Source

The modification descriptions in `mod_dict` are derived from:
- **Mapping**: PFMOD ID → UniMod full name
- **Reference**: UniMod database (unimod.org)

Each modification entry includes:
- `pfmod_id`: PFMOD identifier
- `unimod_id`: UniMod identifier
- `unimod_title`: Short name
- `unimod_full_name`: Full description (used in mod_dict)
- `chemistry`: Chemical composition and mass information


