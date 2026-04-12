import json
import numpy as np
from typing import Any
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

    def _build_function_prompt(self, user_prompt: str) -> str:
        """
        Build a full prompt with instructions and available functions
        """
        func_prompt = "Select the correct function and extract arguments.\n\nFunctions:\n"
        for func in self._functions:
            func_prompt += f"  {func.name}: {func.description}\n"
            if func.parameters:
                param_str = ", ".join(
                    f"{k}=<{v.type}>" for k, v in func.parameters.items()
                )
                func_prompt += f"    Params: {param_str}\n"
        func_prompt += f"\nRequest: {user_prompt}\n\nOutput JSON: "

        return func_prompt

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
            masked_logits = self._mask_tokens(logits, valid_token)
            chosen_token_id = int(np.argmax(masked_logits))

            remaining = {
                name: sequence
                for name, sequence in remaining.items()
                if position < len(sequence)
                and sequence[position] == chosen_token_id
            }

            input_ids.append(chosen_token_id)
            position += 1

        chosen_name = next(iter(remaining))
        return next(
            fn for fn in self._functions
            if fn.name == chosen_name
        )

    def _build_argument_prompt(
        self,
        prompt: str,
        fn: FunctionDefinition,
        already_extracted: dict[str, Any]
    ) -> str:
        """
        Build a prompt that tells the model which argument to generate next.
        """

        already_extracted_parameter = "".join([
            f"- Parameter {name} ({fn.parameters[name].type}): {value}\n"
            for name, value in already_extracted.items()
        ])

        next_param = next(
            name for name in fn.parameters
            if name not in already_extracted
        )

        return (
            f"Extract the function arguments from the user request.\n"
            f"User request: {prompt}\n"
            f"Function: {fn.name}\n"
            f"Description: {fn.description}\n"
            f"{already_extracted_parameter}"
            f"The value of '{next_param}' found in the user request is:"
        )

    def _should_stop(self, param_type: str, generated_so_far: str, next_token: str) -> bool:
        """Decide whether generation should stop."""
        if param_type == "number":
            candidate = (generated_so_far + next_token.replace('Ġ', '')).strip()
            # allow these as valid starts of a number
            if candidate in ('-', '.', '-.'):
                return False
            try:
                float(candidate)
                return False  # valid number so far → continue
            except ValueError:
                return True  # not a valid number → stop
        elif param_type == "string":
            stop_tokens = {'"', '\n', 'Ċ', ',', '}'}
            return next_token in stop_tokens and len(generated_so_far) > 0
        return False

    def _generate_value(self, input_ids: list[int], param_type: str) -> str:
        generated_text = ""
        max_tokens = 50  # prevent infinite loop

        for _ in range(max_tokens):
            if param_type == "number":
                valid_tokens = self._vocab.get_number_tokens()
            elif param_type == "string":
                valid_tokens = self._vocab.get_string_tokens()
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            logits = np.array(self._model.get_logits_from_input_ids(input_ids),
                              dtype=np.float32)
            masked_logits = self._mask_tokens(logits, valid_tokens)

            chosen_token_id = int(np.argmax(masked_logits))
            chosen_token_str = self._vocab.token_id_to_str(chosen_token_id)

            if self._should_stop(param_type, generated_text, chosen_token_str):
                break

            generated_text += chosen_token_str
            input_ids.append(chosen_token_id)

        return generated_text.strip()

    def _extract_arguments(
            self,
            prompt: str,
            fn: FunctionDefinition
    ) -> dict[str, Any]:
        """Extract all arguments for the chosen function one by one."""
        extracted: dict[str, Any] = {}

        for param_name, param_def in fn.parameters.items():
            argument_prompt = self._build_argument_prompt(
                prompt, fn, extracted
            )
            input_ids = self._vocab.encode_text(argument_prompt)
            raw_value = self._generate_value(input_ids, param_def.type)

            if param_def.type == "number":
                extracted[param_name] = float(raw_value)
            elif param_def.type == "string":
                extracted[param_name] = raw_value

        return extracted

    def generate(self, prompt: str) -> FunctionCall:
        """
        Given a natural language prompt, return a validated function call.
        """

        instructed_function_prompt = self._build_function_prompt(prompt)
        input_ids = self._vocab.encode_text(instructed_function_prompt)
        chosen_function = self._select_function(input_ids)

        parameters = self._extract_arguments(prompt, chosen_function)

        return FunctionCall(
            prompt=prompt,
            name=chosen_function.name,
            parameters=parameters
        )
