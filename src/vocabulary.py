import json
from typing import Optional
from llm_sdk import Small_LLM_Model


class Vocabulary:
    """Loads vocab and provides valid token masks for constrained decoding."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self._model = model
        self._id_to_token: dict[int, str] = {}
        self._token_to_id: dict[str, int] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        """Load the vocabulary JSON file."""
        path = self._model.get_path_to_tokenizer_file()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})
        for token, token_id in vocab.items():
            self._id_to_token[token_id] = token
            self._token_to_id[token] = token_id

    def token_str_to_id(self, token: str) -> Optional[int]:
        """Get token ID for a string."""
        return self._token_to_id.get(token)

    def token_id_to_str(self, token_id: int) -> Optional[str]:
        """Get string for a token ID."""
        return self._id_to_token.get(token_id)

    def encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._model.encode(text)[0].tolist()

    def get_number_tokens(self) -> list[int]:
        """Return token IDs that are valid parts of a number."""
        return [
            token_id
            for char in "0123456789.-"
            if (token_id := self.token_str_to_id(char)) is not None
        ]

    def get_string_tokens(self) -> list[int]:
        """Return all token IDs (strings can contain anything)."""
        return list(self._id_to_token.keys())
