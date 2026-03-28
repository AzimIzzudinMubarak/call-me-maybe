"""Utility to map between token IDs and their string representations."""

import json
from typing import Dict, List, Optional, Tuple
from llm_sdk import Small_LLM_Model


class VocabMapper:
    """Maps token IDs to strings and vice versa for constrained decoding."""

    def __init__(self, model: Small_LLM_Model) -> None:
        """Initialize the vocabulary mapper."""
        self._model = model
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        """Load vocabulary from the tokenizer file."""
        tokenizer_path = self._model.get_path_to_tokenizer_file()

        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        # Qwen tokenizer stores vocab in 'model' -> 'vocab'
        vocab = tokenizer_data.get('model', {}).get('vocab', {})

        for token, token_id in vocab.items():
            self._id_to_token[token_id] = token
            self._token_to_id[token] = token_id

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert a token ID to its string representation."""
        return self._id_to_token.get(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a string to its token ID."""
        return self._token_to_id.get(token)

    def encode_text(self, text: str) -> List[int]:
        """
        Encode text into token IDs using the model's tokenizer.

        This is the CORRECT way to get token IDs for strings like function
        names.
        """
        encoded = self._model.encode(text)
        return encoded[0].tolist()  # Remove batch dimension

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        import torch
        tensor = torch.tensor([token_ids], device=self._model._device)
        return self._model.decode(tensor)

    def get_json_control_ids(self) -> Dict[str, int]:
        """Get token IDs for JSON structure characters."""
        control_chars = ['{', '}', '"', ':', ',', '[', ']']
        result = {}
        for char in control_chars:
            token_id = self.token_to_id(char)
            if token_id is not None:
                result[char] = token_id
        return result

    def get_function_name_token_sequences(self) -> Dict[str, List[int]]:
        """
        Get token ID sequences for each function name.

        Returns a dict mapping function name -> list of token IDs.
        """
        try:
            with open('data/input/functions_definition.json', 'r') as f:
                functions = json.load(f)

            result = {}
            for func in functions:
                func_name = func.get('name', '')
                token_sequence = self.encode_text(func_name)
                result[func_name] = token_sequence

            return result
        except FileNotFoundError:
            return {}

    def get_vocab_size(self) -> int:
        """Return the total vocabulary size."""
        return len(self._id_to_token)

    def find_tokens_with_prefix(self, prefix: str) -> List[Tuple[int, str]]:
        """Find all tokens that start with a given prefix."""
        results = []
        for token, token_id in self._token_to_id.items():
            if token.startswith(prefix):
                results.append((token_id, token))
        return sorted(results, key=lambda x: x[0])

    def find_tokens_containing(self, substring: str) -> List[Tuple[int, str]]:
        """Find all tokens that contain a given substring."""
        results = []
        for token, token_id in self._token_to_id.items():
            if substring in token:
                results.append((token_id, token))
        return sorted(results, key=lambda x: x[0])[:20]


def test_vocab_mapper() -> None:
    """Test the vocabulary mapper."""
    print("Initializing model and vocab mapper...")
    model = Small_LLM_Model()
    mapper = VocabMapper(model)

    print(f"Vocab size: {mapper.get_vocab_size()}")

    # Test JSON structure characters
    print("\nJSON Structure Characters:")
    json_ids = mapper.get_json_control_ids()
    for char, token_id in json_ids.items():
        print(f"   '{char}' -> ID: {token_id}")

    # Test how spaces are represented
    print("\nHow Spaces Are Represented:")
    test_strings = [" hello", "  hello", "a b"]
    for text in test_strings:
        tokens = mapper.encode_text(text)
        token_strs = [mapper.id_to_token(t) for t in tokens]
        print(f"   '{text}' -> {tokens} ({token_strs})")

    # Test function names (as token SEQUENCES, not single IDs)
    print("\nFunction Name Token Sequences:")
    func_sequences = mapper.get_function_name_token_sequences()
    for func_name, token_seq in func_sequences.items():
        token_strs = [mapper.id_to_token(t) for t in token_seq]
        print(f"   '{func_name}' -> {token_seq} ({token_strs})")

    # Find tokens starting with 'fn'
    print("\nTokens Starting with 'fn':")
    fn_tokens = mapper.find_tokens_with_prefix('fn')
    for token_id, token in fn_tokens[:10]:
        print(f"   ID {token_id}: '{token}'")

    # Find tokens containing underscore
    print("\nTokens Containing '_':")
    underscore_tokens = mapper.find_tokens_containing('_')
    for token_id, token in underscore_tokens[:10]:
        print(f"   ID {token_id}: '{token}'")


if __name__ == "__main__":
    test_vocab_mapper()
