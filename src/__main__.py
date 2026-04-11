import sys
from llm_sdk import Small_LLM_Model

from src.helpers import (parse_arguments, load_json_file, validate_functions,
                         validate_prompts)


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    print(f"\n📁 Functions definition: {args.functions_definition}")
    print(f"📁 Input file: {args.input}")
    print(f"📁 Output file: {args.output}")

    try:
        print("\nLoading input files...")
        raw_function_definitions = load_json_file(args.functions_definition)
        raw_prompts = load_json_file(args.input)
        function_definitions = validate_functions(raw_function_definitions)
        prompts = validate_prompts(raw_prompts)
    except Exception as e:
        print(e)
        sys.exit(1)

    print(f"Loaded {len(function_definitions)} function definitions")
    print(f"Loaded {len(prompts)} prompts")

    # Initialize model and decoder
    print("\n🤖 Initializing LLM model...")
    try:
        model = Small_LLM_Model()
        print(f"Model loaded on device: {model._device}")
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
