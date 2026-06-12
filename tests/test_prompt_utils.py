"""Standalone tests for the ComfyUI-free parts of Z-Engineer.

Run from the repo root: python tests/test_prompt_utils.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "zengineer"))

import prompt_utils  # noqa: E402


def test_sanitize_prompt():
    text = "<think>notes</think>\nPrompt: neon market street\nNegative prompt: blur"
    cleaned = prompt_utils.sanitize_prompt(text)
    assert "<think>" not in cleaned
    assert "Negative prompt" not in cleaned
    assert cleaned == "neon market street"


def test_sanitize_unclosed_think():
    text = "<think>endless reasoning that never closes"
    assert prompt_utils.sanitize_prompt(text) == ""


def test_sanitize_chatml_tags():
    text = "<|im_start|>assistant\na quiet harbor at dawn<|im_end|>"
    assert prompt_utils.sanitize_prompt(text) == "a quiet harbor at dawn"


def test_split_batch_separator():
    items = prompt_utils.split_batch("a\n---\nb", True, "\\n---\\n")
    assert items == ["a", "b"]


def test_split_batch_disabled():
    items = prompt_utils.split_batch("a\nb", False, "\\n---\\n")
    assert items == ["a\nb"]


def test_seed_constraint_terms():
    terms = prompt_utils.seed_constraint_terms('a red umbrella and "OPEN 24H" sign with three cats')
    lowered = [t.lower() for t in terms]
    assert "red umbrella" in lowered
    assert "open 24h" in lowered
    assert "three cats" in lowered


def test_preserve_seed_constraints_appends_missing():
    seed = "two moons over a lonely tree"
    prompt = "A vast night landscape stretches to the horizon."
    result = prompt_utils.preserve_seed_constraints(seed, prompt)
    assert "two moons" in result.lower()
    assert "lonely tree" in result.lower()
    assert len(result.split()) >= 150


def test_build_chat_prompt():
    chat = prompt_utils.build_chat_prompt("SYS", "USER")
    assert chat.startswith("<|im_start|>system\nSYS<|im_end|>\n")
    assert chat.endswith("<|im_start|>assistant\n")
    assert "<|im_start|>user\nUSER<|im_end|>" in chat


def test_build_user_prompt_includes_seed():
    user = prompt_utils.build_user_prompt("a brass key on a green tablecloth")
    assert "Seed: a brass key on a green tablecloth" in user
    assert "brass key" in user


def test_parse_keep_terms():
    terms = prompt_utils.parse_keep_terms(" m4rty style,  neon glow ; m4rty style\nOzzy_v2 ")
    assert terms == ["m4rty style", "neon glow", "Ozzy_v2"]
    assert prompt_utils.parse_keep_terms("") == []
    assert prompt_utils.parse_keep_terms(None) == []


def test_enforce_keep_terms_appends_missing():
    result = prompt_utils.enforce_keep_terms(
        "A quiet alley at night.", ["m4rty style", "night"]
    )
    assert "m4rty style" in result
    assert result.lower().count("night") == 1
    assert result.endswith(".")


def test_enforce_keep_terms_keeps_casing():
    result = prompt_utils.enforce_keep_terms("a portrait", ["XJ-9_TriGGer"])
    assert "XJ-9_TriGGer" in result


def test_enforce_keep_terms_noop_when_present():
    prompt = "a portrait, m4rty style, soft light."
    assert prompt_utils.enforce_keep_terms(prompt, ["m4rty style"]) == prompt


def test_build_user_prompt_with_keep_terms():
    user = prompt_utils.build_user_prompt("a cat", ["m4rty style"])
    assert "m4rty style" in user
    assert "trigger words" in user


def test_keep_terms_survive_sanitize_pipeline():
    # Trigger word resembling a camera brand gets stripped by sanitize, then
    # restored by enforce_keep_terms (the node applies it last).
    raw = "A studio portrait, Canon EOS style, dramatic rim light."
    terms = prompt_utils.parse_keep_terms("Canon EOS")
    cleaned = prompt_utils.sanitize_prompt(raw)
    assert "Canon" not in cleaned
    restored = prompt_utils.enforce_keep_terms(cleaned, terms)
    assert "Canon EOS" in restored


def main():
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All Z-Engineer prompt_utils tests passed")


if __name__ == "__main__":
    main()
