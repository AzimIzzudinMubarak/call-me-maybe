import json
from typing import Optional
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


class Vocabulary:
    """Loads vocab and provides valid token masks for constrained decoding."""

    def __init__(
        self,
        functions: list[FunctionDefinition]
    ) -> None:
        self._model = Small_LLM_Model()
        self._id_to_token: dict[int, str] = {}
        self._token_to_id: dict[str, int] = {}
        self._load_vocab()

        self._function_sequences: dict[str, list[int]] = {
            fn.name: self.encode_text(fn.name)
            for fn in functions
        }

    def _load_vocab(self) -> None:
        """Load the vocabulary JSON file."""
        path = self._model.get_path_to_tokenizer_file()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})
        for token, token_id in vocab.items():
            self._id_to_token[token_id] = token
            self._token_to_id[token] = token_id

    def token_to_id(self, token: str) -> Optional[int]:
        """Get token ID for a string."""
        return self._token_to_id.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Get string for a token ID."""
        return self._id_to_token.get(token_id)

    def encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._model.encode(text)[0].tolist()

    def get_valid_next_tokens_for_function_name(
        self, position: int
    ) -> list[int]:
        """
        Return token IDs that are valid at this position across
        all function name sequences.
        """
        valid_token_ids: set[int] = set()

        for token_sequence in self._function_sequences.values():
            if position < len(token_sequence):
                valid_token_ids.add(token_sequence[position])

        return list(valid_token_ids)
