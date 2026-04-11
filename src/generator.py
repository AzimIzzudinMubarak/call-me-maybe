import numpy as np
from llm_sdk import Small_LLM_Model

from src.vocabulary import Vocabulary
from src.models import FunctionDefinition, FunctionCall


class ConstrainedGenerator:
    """Generates valid JSON function calls using constrained decoding."""

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition]
    ) -> None:
        self._model = model
        self._vocab = Vocabulary(model)
        self._functions = functions

    def _mask_tokens(self, logits: np.ndarray, valid: list[int]) -> np.ndarray:
        """
        Set all token logits to -inf except the valid ones.
        This forces the model to only pick from valid tokens.
        """
        masked = np.full_like(logits, float('-inf'))
        masked[valid] = logits[valid]
        return masked

    def _select_function(self, input_ids: list[int]) -> FunctionDefinition:
        """
        Use constrained decoding to select the correct function
        for the given prompt.

        At each position:
        1. Get valid token IDs for this position across all remaining functions
        2. Mask all other logits to -inf
        3. Pick the highest logit (argmax)
        4. Eliminate functions that don't match the chosen token
        5. Repeat until only one function remains
        """

        remaining: dict[str, list[int]] = {
            fn.name: self._vocab.encode_text(fn.name)
            for fn in self._functions
        }

        position = 0

        while len(remaining) > 1:
            valid_token = [
                sequence[position]
                for sequence in remaining.values()
                if position < len(sequence)
            ]

            logits = np.array(self._model.get_logits_from_input_ids(input_ids),
                              dtype=np.float32)
            next_token = int(np.array(self._mask_tokens(logits, valid_token)))

            remaining = {
                name: sequence
                for name, sequence in remaining.items()
                if position < len(sequence)
                and sequence[position] == next_token
            }

        chosen_name = next(iter(remaining))
        return next(
            fn for fn in self._functions
            if fn.name == chosen_name
        )

    def _build_function_prompt(self, user_prompt: str) -> str:
        """
        Build a full prompt with instructions and available functions
        """
        functions_description = "\n".join([
            f"- {fn.name}: {fn.description}"
            for fn in self._functions
        ])

        return (
            f"You are a function calling assistant.\n"
            f"Given a user request, select the most appropriate function.\n"
            f"\nAvailable functions:\n{functions_description}"
            f"\nUser request: {user_prompt}"
            f"\nFunction name:"
        )

    def generate(self, prompt: str) -> FunctionCall:
        """
        Given a natural language prompt, return a validated function call.
        """

        instructed_prompt = self._build_function_prompt(prompt)
        input_ids = self._vocab.encode_text(instructed_prompt)

        chosen_function = self._select_function(input_ids)
