import argparse
import json
from typing import Any
from pydantic import ValidationError

from src.models import FunctionDefinition, PromptInput


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding for small LLMs"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to functions definition JSON file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to output results JSON file"
    )
    return parser.parse_args()


def load_json_file(file_path: str) -> Any:
    """Load and parse a JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: Invalid JSON in {file_path}: {e}")


def validate_functions(raw: Any) -> list[FunctionDefinition]:
    """Validate and parse function definitions from raw JSON."""
    if not isinstance(raw, list):
        raise ValueError("Error: functions_definition.json must be a JSON "
                         "array")
    try:
        return [FunctionDefinition(**function) for function in raw]
    except (ValidationError, TypeError) as e:
        raise ValueError(f"Error: Invalid function definition structure: {e}")


def validate_prompts(raw: Any) -> list[PromptInput]:
    """Validate and extract prompts from raw JSON."""
    if not isinstance(raw, list):
        raise ValueError("Error: input file must be a JSON array")
    try:
        return [PromptInput(**prompt) for prompt in raw]
    except (KeyError, TypeError):
        raise ValueError("Error: each entry in input file must have a 'prompt'"
                         " key")
