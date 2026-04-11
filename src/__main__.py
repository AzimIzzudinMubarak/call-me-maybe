import sys

from src.helpers import parse_arguments, load_json_file


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    print(f"\n📁 Functions definition: {args.functions_definition}")
    print(f"📁 Input file: {args.input}")
    print(f"📁 Output file: {args.output}")

    try:
        print("\nLoading input files...")
        function_definitions = load_json_file(args.functions_definition)
        prompts = load_json_file(args.input)
    except Exception as e:
        print(e)
        sys.exit(1)

    print(f"Loaded {len(function_definitions)} function definitions")
    print(f"Loaded {len(prompts)} prompts")


if __name__ == "__main__":
    main()
