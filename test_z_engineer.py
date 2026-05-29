from z_engineer import sanitize_prompt, split_batch


def test_sanitize_prompt():
    text = "<think>notes</think>\nPrompt: neon market street\nNegative prompt: blur"
    cleaned = sanitize_prompt(text)
    assert "<think>" not in cleaned
    assert "Negative prompt" not in cleaned
    assert cleaned == "neon market street"


def test_split_batch_separator():
    items = split_batch("a\n---\nb", True, "\\n---\\n")
    assert items == ["a", "b"]


def main():
    test_sanitize_prompt()
    test_split_batch_separator()
    print("ComfyUI Z-Engineer V6 node tests passed")


if __name__ == "__main__":
    main()
