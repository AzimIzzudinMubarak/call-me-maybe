import sys
import json
import os
from llm_sdk import Small_LLM_Model

from src.helpers import (parse_arguments, load_json_file, validate_functions,
                         validate_prompts)
from src.generator import ConstrainedGenerator


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

    generator = ConstrainedGenerator(model, function_definitions)

    print("\n⚙️  Generating function calls...")
    results = []
    for i, prompt_input in enumerate(prompts):
        print(f"  [{i + 1}/{len(prompts)}] {prompt_input.prompt}")
        try:
            result = generator.generate(prompt_input.prompt)
            results.append(result.model_dump())
        except Exception as e:
            print(f"  Warning: failed to process prompt: {e}")

    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")
    except Exception as e:
        print(f"Failed to save output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
