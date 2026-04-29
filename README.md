# Call Me Maybe

*This project has been created as part of the 42 curriculum by azmubara*

---

## Description

**Call Me Maybe** is a function calling tool that bridges the gap between natural language and machine-executable code. Given a plain-language request like `"What is the sum of 40 and 2?"`, the system does not return `42` — instead it produces a structured function call:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The core challenge is reliability: small language models (like the 0.6B-parameter Qwen3 model used here) spontaneously produce valid JSON only ~30% of the time when prompted naively. This project achieves near-perfect reliability using **constrained decoding** — a technique that guides the model token-by-token, masking logits for invalid tokens at each generation step rather than hoping the model gets it right on its own.

### Key capabilities

- **Function selection** via constrained decoding over function name tokens
- **Argument extraction** with type-aware generation loops (string, integer, number)
- **100% valid JSON output** guaranteed by structural constraints
- **Graceful error handling** for malformed inputs, missing files, and type conversion failures

---

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- The `llm_sdk/` package (copy it into the project root alongside `src/`)

### Installation

```bash
uv sync
```

This installs all dependencies declared in `pyproject.toml` (numpy, pydantic, and others).

### Running the program

```bash
uv run python -m src
```

By default the program reads from `data/input/` and writes to `data/output/`. You can override all paths with CLI flags:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Makefile targets

| Target | Description |
|--------|-------------|
| `make install` | Install dependencies via uv |
| `make run` | Run the main script |
| `make debug` | Run with Python's pdb debugger |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, etc. |
| `make lint` | Run flake8 and mypy with standard flags |
| `make lint-strict` | Run flake8 and mypy with `--strict` |

---

## Algorithm Explanation

The pipeline has two distinct constrained decoding phases.

### Phase 1 — Function selection

1. A prompt listing all available functions and the user request is built and tokenized.
2. Each candidate function name is also tokenized into a sequence of token IDs.
3. At each position the decoder collects the set of valid tokens — one per still-active function name at that position.
4. All other logits are masked to `-inf`; the highest remaining logit is selected (argmax).
5. Any function whose token at this position does not match the chosen token is eliminated.
6. This repeats until only one function remains.

Because the valid token set shrinks monotonically, the function name is always chosen character-by-character in a way the model finds most probable, while remaining strictly within the set of known function names.

### Phase 2 — Argument extraction

For each parameter in the chosen function:

1. A prompt is built that includes the original user request and a partial JSON block already containing previously extracted arguments.
2. The model generates tokens one at a time; each token is checked against the expected type:
   - **integer**: only digits and `-` are accepted.
   - **number**: only digits, `-`, and `.` are accepted.
   - **string**: generation stops at the next `"` character (end of JSON string value).
3. The accumulated characters are cast to the correct Python type (int, float, str).

---

## Design Decisions

**Vocabulary-based token mapping**: Rather than working at the character level, the Vocabulary class loads the tokenizer's raw vocab file to map between token IDs and their string representations. This makes it possible to check in O(1) whether a token contains a forbidden character for a given type.

**Prompt engineering for argument extraction**: Each argument is extracted with a fresh prompt that already contains the partial JSON built from prior arguments. This gives the model the strongest possible context for each next value, without needing multi-turn conversation state.

**Pydantic models throughout**: All input and output data structures (`FunctionDefinition`, `FunctionCall`, `PromptInput`) are validated with Pydantic, so schema violations are caught immediately at the boundary rather than deep in generation logic.

**No external constrained-decoding libraries**: The use of packages such as `outlines`, `dspy`, or HuggingFace `transformers` is explicitly forbidden. The constrained decoding logic is implemented from scratch using only `numpy` for logit manipulation.

---

## Performance Analysis

| Metric | Target | Notes |
|--------|--------|-------|
| JSON validity | 100% | Guaranteed by structural constraints — invalid tokens are masked, not just discouraged |
| Function selection accuracy | ≥ 90% | Dependent on prompt clarity and model confidence at disambiguation tokens |
| Argument extraction accuracy | ≥ 90% | Type-aware stopping prevents numeric overflow and string bleed |
| Speed | < 5 min for full test set | Token generation is sequential; no batching |

The Qwen3-0.6B model has only ~500 million parameters, yet constrained decoding elevates its reliability to levels comparable to much larger models for this structured task.

---

## Challenges Faced

**Token boundary mismatches**: Function names may not align neatly with token boundaries. A name like `fn_add_numbers` is tokenized into multiple subword tokens; the constrained decoder must handle positions within those sub-tokens correctly rather than assuming one token per character.

**Tokenizer quirks (leading spaces)**: BPE tokenizers encode a word differently depending on whether it appears at the start of a sequence or after a space. The `Ġ` prefix in the vocabulary represents a preceding space. Argument extraction strips these artifacts when converting raw token strings back to Python values.

**Numeric type stopping**: Integers and floats share digit tokens. The stopping condition is purely character-based (`.` is only allowed for `number`, not `integer`), so the generator correctly differentiates the two types without needing separate vocabularies.

**Edge case inputs**: Prompts with special characters, very large numbers, or ambiguous phrasing can cause the model to produce unexpected tokens. Type conversion failures are caught, logged as warnings, and replaced with safe defaults (`0.0` for numbers, `""` for strings) so the program never crashes.

---

## Testing Strategy

1. **Happy-path tests**: Run the provided example files and verify that output JSON matches expected function names and typed argument values.
2. **Edge-case prompts**: Test with empty strings, large numbers (e.g. `265 + 345`), special characters, and prompts that are semantically ambiguous between two functions.
3. **Malformed input files**: Supply invalid JSON and missing files to confirm graceful error messages and clean exits.

---

## Example Usage

### Input files

`data/input/function_calling_tests.json`:
```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

`data/input/functions_definition.json`:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": { "a": { "type": "number" }, "b": { "type": "number" } }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": { "name": { "type": "string" } }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": { "s": { "type": "string" } }
  }
]
```

### Output

`data/output/function_calls.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

---

## Resources

### Documentation and articles

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the base LLM used in this project
- [Pydantic documentation](https://docs.pydantic.dev/) — data validation library used throughout
- [numpy documentation](https://numpy.org/doc/) — used for logit manipulation
- [BPE tokenization explained](https://huggingface.co/learn/nlp-course/chapter6/5) — background on how tokens map to text
- [Structured generation / constrained decoding overview](https://lmsys.org/blog/2024-02-05-compressed-fsm/) — FSM-based approach to constrained decoding

### AI usage

AI assistance (Claude) was used for the following tasks during this project:

- **README drafting**: generating an initial structure and wording for this document based on the project code and subject PDF.
- **Debugging support**: suggesting fixes for token boundary edge cases during argument extraction.
- **Documentation**: writing docstrings for classes and functions.

All AI-generated content was reviewed, tested, and understood before being incorporated. No AI-generated code was used without full comprehension and manual validation.
