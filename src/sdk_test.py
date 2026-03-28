"""Test script to explore the LLM SDK functionality."""

from llm_sdk import Small_LLM_Model


def test_sdk_basic() -> None:
    """Test basic SDK functionality."""
    print("🔍 Initializing Small_LLM_Model...")
    model = Small_LLM_Model()

    print(f"✅ Model loaded on device: {model._device}")
    print(f"✅ Model dtype: {model._dtype}")

    # Test encoding
    print("\n🔍 Testing encode()...")
    text = "Hello world"
    input_ids = model.encode(text)
    print(f"   Input: '{text}'")
    print(f"   Token IDs: {input_ids.tolist()}")

    # Test decoding
    print("\n🔍 Testing decode()...")
    decoded = model.decode(input_ids)
    print(f"   Decoded: '{decoded}'")

    # Test logits
    print("\n🔍 Testing get_logits_from_input_ids()...")
    ids_list = input_ids.tolist()[0]
    logits = model.get_logits_from_input_ids(ids_list)
    print(f"   Logits shape: {len(logits)} (vocab size)")
    print(f"   Top 5 logits: {sorted(logits, reverse=True)[:5]}")

    # Test vocab file access
    print("\n🔍 Getting vocab file path...")
    vocab_path = model.get_path_to_vocab_file()
    print(f"   Vocab file: {vocab_path}")

    # Test tokenizer file access
    print("\n🔍 Getting tokenizer file path...")
    tokenizer_path = model.get_path_to_tokenizer_file()
    print(f"   Tokenizer file: {tokenizer_path}")


if __name__ == "__main__":
    test_sdk_basic()
