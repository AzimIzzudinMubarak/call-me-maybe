from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionParameter
from src.generator import ConstrainedGenerator

# define a few simple functions manually
functions = [
    FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers together and return their sum.",
        parameters={
            "a": FunctionParameter(type="number"),
            "b": FunctionParameter(type="number")
        }
    ),
    FunctionDefinition(
        name="fn_greet",
        description="Generate a greeting message for a person by name.",
        parameters={
            "name": FunctionParameter(type="string")
        }
    ),
    FunctionDefinition(
        name="fn_reverse_string",
        description="Reverse a string and return the reversed result.",
        parameters={
            "s": FunctionParameter(type="string")
        }
    ),
]

model = Small_LLM_Model()
generator = ConstrainedGenerator(model, functions)

# test cases — expected function + rough expected args
test_cases = [
    ("What is the sum of 2 and 3?",      "fn_add_numbers"),
    ("What is the sum of 265 and 345?",  "fn_add_numbers"),
    ("Greet john",                        "fn_greet"),
    ("Reverse the string 'hello'",        "fn_reverse_string"),
]

passed = 0
for prompt, expected_fn in test_cases:
    result = generator.generate(prompt)
    status = "✓" if result.name == expected_fn else "✗"
    if result.name == expected_fn:
        passed += 1
    print(f"{status} prompt:   {prompt}")
    print(f"  expected: {expected_fn}")
    print(f"  got:      {result.name}")
    print(f"  params:   {result.parameters}")
    print()

print(f"Result: {passed}/{len(test_cases)} passed")