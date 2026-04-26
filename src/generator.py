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

    def _build_function_prompt(self, user_prompt: str) -> str:
        """
        Build a full prompt with instructions and available functions
        so the LLM understands the task and can select the right function.
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

            logits = np.array(self._model.get_logits_from_input_ids(input_ids))
            masked_logits = np.full_like(logits, float('-inf'))
            masked_logits[valid_token] = logits[valid_token]
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
        Build a prompt that guides the model to generate the next argument value.
        """
        parameters_description = ", ".join([
            f"{name} ({param.type})"
            for name, param in fn.parameters.items()
        ])

        # build the partial JSON with already extracted values
        partial_json_parts = []
        for name, value in already_extracted.items():
            if isinstance(value, float):
                partial_json_parts.append(f'"{name}": {value}')
            else:
                partial_json_parts.append(f'"{name}": "{value}"')
        partial_json = ", ".join(partial_json_parts)

        if partial_json:
            json_so_far = f'{{{partial_json}, '
        else:
            json_so_far = '{'

        next_param = next(
            name for name in fn.parameters
            if name not in already_extracted
        )
        json_so_far += f'"{next_param}": '

        return (
            f"Extract the function arguments from the user request in JSON.\n"
            f"Prompt: {prompt.replace("'", '"')}\n"
            f"Function: {fn.name}\n"
            f"Description: {fn.description}\n"
            f"Parameters: {parameters_description}\n"
            f"Next parameter to extract: '{next_param}' from the user prompt.\n"
            f"Output JSON:\n"
            f"{json_so_far}"
        )

    def _generate_value(self, input_ids: list[int], param_type: str) -> str:
        """
        Run constrained generation loop for a single parameter value.
        Stops when the value is complete based on its type.
        """
        generated_text = ""
        max_tokens = 50

        for _ in range(max_tokens):
            logits = np.array(self._model.get_logits_from_input_ids(input_ids))
            chosen_token_id = int(np.argmax(logits))
            chosen_token_str = self._vocab.token_id_to_str(chosen_token_id)
            if param_type == "integer":
                if chosen_token_str and chosen_token_str not in "0123456789-.":
                    break
            elif param_type == "number":
                if chosen_token_str and chosen_token_str not in "0123456789-":
                    break
            elif param_type == "string":
                stop_char = {
                    "\",", "\"}Ċ", ",", "}Ċ",
                    "\",Ċ", "\"}}Ċ", "}\",Ċ"
                }
                print("chosen: ", chosen_token_str)
                if chosen_token_str and chosen_token_str in stop_char:
                    break
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            if chosen_token_str is None:
                break

            generated_text += chosen_token_str
            input_ids.append(chosen_token_id)
        return generated_text.strip()

    def _extract_arguments(
        self,
        prompt: str,
        fn: FunctionDefinition
    ) -> dict[str, Any]:
        """
        Extract all arguments one by one using a growing JSON prompt.
        At each iteration the prompt already contains the partial JSON
        so the model only needs to generate the next value.
        """
        extracted: dict[str, Any] = {}

        for param_name, param_def in fn.parameters.items():
            argument_prompt = self._build_argument_prompt(
                prompt, fn, extracted
            )
            print(f"\n\n{argument_prompt}\n\n")
            input_ids = self._vocab.encode_text(argument_prompt)
            raw_value = self._generate_value(input_ids, param_def.type)

            try:
                if param_def.type == "integer":
                    extracted[param_name] = int(raw_value)
                elif param_def.type == "number":
                    extracted[param_name] = float(raw_value)
                elif param_def.type == "string":
                    cleaned = raw_value.replace('Ġ', ' ').strip("\" ")
                    extracted[param_name] = cleaned
            except ValueError:
                print(f"Warning: could not convert '{raw_value}' for '{param_name}'")
                extracted[param_name] = 0.0 if param_def.type == "number" else ""

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
