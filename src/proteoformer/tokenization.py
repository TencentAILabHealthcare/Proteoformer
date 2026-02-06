import re
import json
import os
from typing import List, Optional, Union, Dict, Any, Tuple
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class ProteoformerTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for proteoform sequences with amino acids and modifications.
    
    This tokenizer handles protein sequences where:
    - Single uppercase letters represent standard amino acids (e.g., 'A', 'M', 'K')
    - Square brackets represent post-translational modifications (e.g., '[PFMOD:1]')
    
    Example:
        >>> tokenizer = ProteoformerTokenizer.from_pretrained("proteoformer-base")
        >>> tokens = tokenizer.tokenize("M[PFMOD:1]AK")
        >>> # Returns: ['M', '[PFMOD:1]', 'A', 'K']
    
    Attributes:
        vocab: Dict mapping tokens to their integer IDs
        ids_to_tokens: Dict mapping integer IDs back to tokens
        mod_dict: Dict mapping PFMOD tokens to human-readable modification descriptions
        add_special_tokens: Whether to add [CLS] and [SEP] tokens during encoding
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        add_special_tokens: bool = False,
        **kwargs
    ):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_file: Path to the vocabulary JSON file containing 'vocab' and optionally 'mod_dict'
            unk_token: Token for unknown/out-of-vocabulary tokens
            sep_token: Separator token for sequence pairs
            pad_token: Padding token for batch alignment
            cls_token: Classification token prepended to sequences
            mask_token: Mask token for masked language modeling
            add_special_tokens: If True, add [CLS] and [SEP] tokens during encoding
        """
        self.vocab_file = vocab_file
        self.vocab = {}
        self.ids_to_tokens = {}
        self.mod_dict = {}
        self.add_special_tokens = add_special_tokens
        
        if vocab_file is not None:
            self._load_vocab(vocab_file)
        
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

    def _load_vocab(self, vocab_file: str):
        """
        Load vocabulary from a JSON file.
        
        Expected JSON format:
            {
                "vocab": {"token": id, ...},
                "mod_dict": {"[PFMOD:X]": "description", ...}  # optional
            }
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data["vocab"]
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        if "mod_dict" in vocab_data:
            self.mod_dict = vocab_data["mod_dict"]
        else:
            self.mod_dict = {}
    
    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dict of {token: index}"""
        return dict(self.vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a protein sequence string.
        
        Splits the sequence into:
        - Single uppercase letters (amino acids)
        - Content within square brackets (modifications like [PFMOD:1])
        
        Args:
            text: Input protein sequence (e.g., "M[PFMOD:1]AK")
            
        Returns:
            List of tokens (e.g., ['M', '[PFMOD:1]', 'A', 'K'])
        """
        pattern = re.compile(r'[A-Z]|\[[^\]]+\]')
        tokens = pattern.findall(text)
        tokens = [token.replace(' ', '') for token in tokens]
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token string to its vocabulary ID. Returns UNK ID if not found."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Converts a vocabulary ID to its token string. Returns UNK token if not found."""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Concatenates tokens into a single sequence string (no separators)."""
        return "".join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a JSON file.
        
        Args:
            save_directory: Directory path to save the vocabulary
            filename_prefix: Optional prefix for the filename
            
        Returns:
            Tuple containing the path to the saved vocabulary file
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        vocab_data = {
            "vocab": self.vocab,
            "special_tokens": list(self.special_tokens_map.values())
        }
        
        with open(vocab_file, "w", encoding="utf-8") as writer:
            json.dump(vocab_data, writer, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs by optionally adding [CLS] and [SEP] tokens.
        
        Behavior depends on self.add_special_tokens:
        - If False: returns token_ids_0 (+ token_ids_1 if provided)
        - If True: returns [CLS] + token_ids_0 + [SEP] (+ token_ids_1 + [SEP])
        """
        if not self.add_special_tokens:
            if token_ids_1 is None:
                return token_ids_0
            return token_ids_0 + token_ids_1
            
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Create a mask identifying special tokens (1) vs sequence tokens (0).
        
        Used for tasks like masked language modeling to avoid masking special tokens.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_special_tokens:
            if token_ids_1 is not None:
                return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1))
            return [0] * len(token_ids_0)

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for sequence pair tasks.
        
        Returns 0 for first sequence tokens, 1 for second sequence tokens.
        Used to distinguish between sentence A and sentence B in pair classification.
        """
        if not self.add_special_tokens:
            if token_ids_1 is None:
                return len(token_ids_0) * [0]
            return len(token_ids_0) * [0] + len(token_ids_1) * [1]
        
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    
    # ==================== Legacy Methods ====================
    # The following methods are kept for backward compatibility with older code.
    
    def tokenize_proteoform_sequence(self, sequence: str) -> List[str]:
        """Legacy method: Use tokenize() instead."""
        return self._tokenize(sequence)
    
    def encode_sequence(self, sequence: str) -> List[int]:
        """Legacy method: Use encode() instead."""
        tokens = self._tokenize(sequence)
        return self.convert_tokens_to_ids(tokens)
    
    def decode_sequence(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Legacy method: Use decode() instead."""
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_metrics_sequences(self, token_ids: List[int]) -> str:
        """
        Decode tokens to a space-separated string for metrics computation.
        
        Filters out special tokens and joins with spaces for comparison tasks.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        filtered_tokens = [token for token in tokens if token not in self.all_special_tokens]
        return " ".join(filtered_tokens)
    
    # ==================== Vocabulary Extension ====================
    
    def add_vocab(self, vocab: List[str]):
        """
        Add new tokens to the vocabulary.
        
        Args:
            vocab: List of new token strings to add
        """
        last_index = len(self.vocab)
        for token in vocab:
            if token not in self.vocab:
                self.vocab[token] = last_index
                self.ids_to_tokens[last_index] = token
                last_index += 1
    
    def add_placeholders(self, num_ph: int):
        """
        Add placeholder tokens to vocabulary for future use.
        
        Args:
            num_ph: Number of placeholder tokens to add (named <placeholder_0>, <placeholder_1>, ...)
        """
        last_index = len(self.vocab)
        for i in range(num_ph):
            placeholder_token = f"<placeholder_{i}>"
            if placeholder_token not in self.vocab:
                self.vocab[placeholder_token] = last_index
                self.ids_to_tokens[last_index] = placeholder_token
                last_index += 1
    
    def get_mod_description(self, token: Union[str, int]) -> Optional[str]:
        """
        Get the human-readable description for a modification token.
        
        Args:
            token: Either a PFMOD token string (e.g., "[PFMOD:213]") or its token ID
            
        Returns:
            Modification description if found (e.g., "ADP Ribose addition"), None otherwise
        """
        if isinstance(token, int):
            token = self.ids_to_tokens.get(token, None)
            if token is None:
                return None
        
        return self.mod_dict.get(token, None)
    
    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]] = None,
        text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: Optional[bool] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and encode sequence(s) for model input.
        
        Args:
            text: Single sequence or batch of sequences to encode
            text_pair: Optional second sequence(s) for pair tasks
            add_special_tokens: Override the tokenizer's default add_special_tokens setting
            **kwargs: Additional arguments (padding, truncation, max_length, etc.)
            
        Returns:
            BatchEncoding with 'input_ids' and 'attention_mask'
        """
        original_add_special_tokens = self.add_special_tokens
        if add_special_tokens is not None:
            self.add_special_tokens = add_special_tokens
        
        try:
            result = super().__call__(text=text, text_pair=text_pair, **kwargs)
            return result
        finally:
            self.add_special_tokens = original_add_special_tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Load a tokenizer from a predefined model name or local path.
        
        Args:
            pretrained_model_name_or_path: One of:
                - Predefined name: "proteoformer-base", "proteoformer-base-2000", "proteoformer-pfmod"
                - Directory path containing vocab.json
                - Direct path to a vocabulary JSON file
            
        Returns:
            ProteoformerTokenizer instance
            
        Raises:
            FileNotFoundError: If vocabulary file cannot be found
        """
        PREDEFINED_MODELS = {
            "proteoformer-base-2000": "proteoformer-base-2000.json",
            "proteoformer-pfmod": "proteoformer-pfmod-2000.json",
            "proteoformer-base": "proteoformer-base-2000.json",
        }
        
        vocab_file = None
        
        # Handle predefined model names
        if pretrained_model_name_or_path in PREDEFINED_MODELS:
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resource_dir = os.path.join(package_dir, "resource")
            vocab_file = os.path.join(resource_dir, PREDEFINED_MODELS[pretrained_model_name_or_path])
            
            if not os.path.exists(vocab_file):
                # Fallback: search up directory tree for resource folder
                current_dir = os.path.dirname(os.path.abspath(__file__))
                for _ in range(3):
                    current_dir = os.path.dirname(current_dir)
                    potential_resource_dir = os.path.join(current_dir, "resource")
                    potential_vocab_file = os.path.join(potential_resource_dir, PREDEFINED_MODELS[pretrained_model_name_or_path])
                    if os.path.exists(potential_vocab_file):
                        vocab_file = potential_vocab_file
                        break
                
                if not os.path.exists(vocab_file):
                    raise FileNotFoundError(
                        f"Could not find vocabulary file for model '{pretrained_model_name_or_path}'. "
                        f"Expected to find '{PREDEFINED_MODELS[pretrained_model_name_or_path]}' in the resource directory."
                    )
        
        # Handle local directory path
        elif os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_FILES_NAMES["vocab_file"])
            if not os.path.exists(vocab_file):
                vocab_file = pretrained_model_name_or_path
        else:
            # Assume direct file path
            vocab_file = pretrained_model_name_or_path
            
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
            
        return cls(vocab_file=vocab_file, **kwargs)