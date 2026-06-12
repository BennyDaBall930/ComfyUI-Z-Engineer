"""Shared prompt-engineering utilities for the Z-Engineer nodes.

This module is intentionally free of ComfyUI imports so it can be unit
tested standalone.
"""

import re
from typing import List


V6_SYSTEM_PROMPT = (
    "You are Z-Image-Engineer V6, a prompt-only cinematography and visual-language "
    "specialist for the Tongyi-MAI Z-Image-Turbo Qwen text encoder. Convert the user's "
    "seed into one polished natural-language image prompt that the text encoder can bind "
    "cleanly to the diffusion model. Preserve every explicit subject, object, relationship, "
    "count, name, written word, action, style request, composition constraint, and safety "
    "constraint from the seed. Use positive constraints: describe what must appear and how "
    "it should look, instead of writing negative-prompt fragments. Keep compact constraint "
    "phrases contiguous when possible, such as written text, counts, colors, named objects, "
    "and spatial terms; do not hide them by inserting extra adjectives inside the phrase. "
    "Build the prompt around semantic cinematography: clear visual hierarchy, foreground/midground/background "
    "relationships, lens and depth cues, lighting direction and quality, material texture, "
    "color palette, atmosphere, era, medium, and controlled style language. Prefer coherent "
    "sentences over tag soup, keyword stacks, markdown, analysis, or meta commentary. Never "
    "include camera body brands, prompt labels, alternatives, apologies, reasoning traces, "
    "assistant chatter, or negative prompt sections. Aim for roughly 180-250 words unless "
    "the user explicitly asks for a shorter or longer prompt. Return only the final image "
    "prompt as one self-contained paragraph."
)

CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

REASONING_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
UNCLOSED_REASONING_PATTERN = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
CHATML_TAG_PATTERN = re.compile(r"<\|im_(?:start|end)\|>(?:system|user|assistant)?", re.IGNORECASE)
PROMPT_PREFIX_PATTERN = re.compile(r"^\s*(?:final\s+)?(?:image\s+)?prompt\s*[:\-]\s*", re.IGNORECASE)
NEGATIVE_SECTION_PATTERN = re.compile(r"\bnegative\s+prompt\s*[:\-].*$", re.IGNORECASE | re.DOTALL)
CAMERA_BODY_PATTERN = re.compile(
    r"\b(?:Canon\s*(?:EOS|R\d+)?|Nikon\s*(?:D\d+|Z\d+)?|Sony\s*A\d+|Fujifilm\s*X|Leica\s*[MQSL]?|"
    r"ARRI|RED\s+(?:Komodo|V-Raptor|Monstro)?|Blackmagic|Panavision)\b",
    re.IGNORECASE,
)
QUOTE_PATTERN = re.compile(r'"([^"]+)"|\'([^\']+)\'')
CODE_PATTERN = re.compile(r"(?<!/)\b(?!T\d+\b)[A-Z]{1,8}-?\d+[A-Z0-9-]*\b")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9-]+")
COUNT_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gold",
    "golden",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
VISUAL_ADJECTIVES = {
    "brass",
    "brutalist",
    "concrete",
    "glass",
    "glowing",
    "lonely",
    "main",
    "mechanical",
    "neon",
    "round",
    "scratched",
    "tired",
    "translucent",
    "transparent",
    "wet",
    "wooden",
}
STOP_PHRASE_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "beside",
    "by",
    "each",
    "for",
    "from",
    "in",
    "inside",
    "near",
    "next",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "with",
}
PHRASE_HINTS = [
    "blue ceramic bowl",
    "brass key",
    "copper traces",
    "five peaches",
    "flower",
    "four lanterns",
    "glass",
    "glowing bookmark",
    "green tablecloth",
    "hill",
    "library books",
    "lonely tree",
    "orange soda",
    "puddles",
    "red ceramic dragon",
    "red umbrella",
    "round glasses",
    "scratched wooden desk",
    "shallow depth of field",
    "sunrise",
    "taking notes",
    "three silver telescopes",
    "tiny robot",
    "triangle",
    "translucent blue",
    "two moons",
    "watering",
    "wet sidewalk",
    "white teacup",
]


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split())


def build_chat_prompt(system_prompt: str, user_content: str) -> str:
    return CHAT_TEMPLATE.format(system=str(system_prompt or "").strip(), user=user_content)


def sanitize_prompt(text: str, strip_reasoning: bool = True, sanitize_output: bool = True) -> str:
    text = str(text or "")
    if strip_reasoning:
        text = REASONING_BLOCK_PATTERN.sub(" ", text)
        text = UNCLOSED_REASONING_PATTERN.sub(" ", text)
    if sanitize_output:
        text = CHATML_TAG_PATTERN.sub(" ", text)
        text = NEGATIVE_SECTION_PATTERN.sub(" ", text)
        text = text.replace("```", " ")
        text = PROMPT_PREFIX_PATTERN.sub("", text)
        text = CAMERA_BODY_PATTERN.sub("", text)
    text = re.sub(r"\s+,", ",", text)
    return normalize_ws(text).strip(" \t\r\n\"'")


def seed_constraint_terms(seed: str) -> List[str]:
    seed = normalize_ws(seed)
    terms: List[str] = []
    seen = set()

    def add(term: str) -> None:
        term = normalize_ws(term).strip(" ,.;:")
        if len(term) < 2:
            return
        key = term.lower()
        if key not in seen:
            seen.add(key)
            terms.append(term)

    def phrase_hint_present(phrase: str) -> bool:
        escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![a-z0-9-]){escaped}(?![a-z0-9-])", seed.lower()) is not None

    for match in QUOTE_PATTERN.finditer(seed):
        add(match.group(1) or match.group(2) or "")
    for match in CODE_PATTERN.finditer(seed):
        add(match.group(0))
    for phrase in PHRASE_HINTS:
        if phrase_hint_present(phrase):
            add(phrase)

    tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(seed)]
    lower = [token.lower() for token in tokens]
    for idx, token in enumerate(lower):
        next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        next_lower = lower[idx + 1] if idx + 1 < len(lower) else ""
        next2_token = tokens[idx + 2] if idx + 2 < len(tokens) else ""
        next2_lower = lower[idx + 2] if idx + 2 < len(lower) else ""

        if (
            (token.isdigit() or token in COUNT_WORDS)
            and next_token
            and next_lower not in STOP_PHRASE_TOKENS
        ):
            add(f"{tokens[idx]} {next_token}")
            if (
                next2_token
                and next_lower not in STOP_PHRASE_TOKENS
                and next2_lower not in STOP_PHRASE_TOKENS
                and next2_lower not in COUNT_WORDS
                and not next2_lower.isdigit()
            ):
                add(f"{tokens[idx]} {next_token} {next2_token}")
        if (
            token in COLOR_WORDS
            and next_token
            and next_lower not in STOP_PHRASE_TOKENS
            and next_lower not in COUNT_WORDS
            and not next_lower.isdigit()
        ):
            if next_lower in VISUAL_ADJECTIVES and next2_token:
                add(f"{tokens[idx]} {next_token} {next2_token}")
            else:
                add(f"{tokens[idx]} {next_token}")
        if token in VISUAL_ADJECTIVES and next_token and next_lower not in STOP_PHRASE_TOKENS:
            add(f"{tokens[idx]} {next_token}")
            if next_lower in COLOR_WORDS:
                add(f"{tokens[idx]} {next_token}")
            elif next2_token and next2_lower not in STOP_PHRASE_TOKENS:
                add(f"{tokens[idx]} {next_token} {next2_token}")
        if token.endswith("ing") and next_token and next_lower not in STOP_PHRASE_TOKENS:
            add(f"{tokens[idx]} {next_token}")

    return terms


def build_user_prompt(seed: str, keep_terms: List[str] = None) -> str:
    terms = seed_constraint_terms(seed)
    content = (
        "Expand the seed into a polished 180-250 word single-paragraph Z-Image Turbo prompt. "
        "Preserve exact text, colors, counts, objects, and relationships. Return only the prompt."
    )
    if terms:
        content += "\nUse these seed phrases exactly, verbatim, and contiguous in the final prompt: " + "; ".join(terms) + "."
    if keep_terms:
        content += (
            "\nThe final prompt must also contain these exact terms verbatim, character for character "
            "(they are model trigger words, do not rephrase, split, or correct them): "
            + "; ".join(keep_terms)
            + "."
        )
    return content + "\n\nSeed: " + normalize_ws(seed)


def parse_keep_terms(keep_terms: str) -> List[str]:
    """Split a user-supplied trigger-word field into unique, ordered terms.
    Terms are separated by commas, semicolons, or newlines so multi-word
    triggers stay intact."""
    terms: List[str] = []
    seen = set()
    for part in re.split(r"[,;\n]+", str(keep_terms or "")):
        part = normalize_ws(part).strip(" .")
        if not part:
            continue
        key = part.lower()
        if key not in seen:
            seen.add(key)
            terms.append(part)
    return terms


def enforce_keep_terms(prompt: str, terms: List[str]) -> str:
    """Guarantee every keep-term appears verbatim in the prompt, appending any
    the model dropped (or that sanitization removed) with original casing."""
    prompt = normalize_ws(prompt)
    missing = [term for term in terms if term.lower() not in prompt.lower()]
    if missing:
        prompt = normalize_ws(prompt.rstrip(".") + ", " + ", ".join(missing) + ".")
    return prompt


def preserve_seed_constraints(seed: str, prompt: str) -> str:
    prompt = normalize_ws(prompt)
    missing = [term for term in seed_constraint_terms(seed) if term.lower() not in prompt.lower()]
    if missing:
        addition = " The composition explicitly includes " + ", ".join(missing) + "."
        prompt = normalize_ws(prompt.rstrip(".") + "." + addition)
    if len(prompt.split()) < 150:
        additions = [
            " The visual design adds clear foreground, midground, and background separation, with controlled lighting falloff and a cohesive color palette that supports the main subject.",
            " Fine material textures, natural shadows, and lens-depth cues are emphasized so the scene remains readable, cinematic, and grounded without adding conflicting objects.",
        ]
        for addition in additions:
            prompt = normalize_ws(prompt.rstrip(".") + "." + addition)
            if len(prompt.split()) >= 150:
                break
        padding = [
            " The composition stays precise, readable, balanced, and visually grounded.",
            " Edges, shadows, and scale cues remain coherent across the frame.",
            " The scene keeps one clear focal point while preserving the seed details.",
            " Atmosphere supports the subject without introducing conflicting story elements.",
        ]
        idx = 0
        while len(prompt.split()) < 150:
            prompt = normalize_ws(prompt.rstrip(".") + "." + padding[idx % len(padding)])
            idx += 1
    return prompt


def decode_separator(batch_separator: str) -> str:
    separator = str(batch_separator or "\\n---\\n")
    return separator.encode("utf-8").decode("unicode_escape")


def split_batch(input_prompt: str, batch_mode: bool, batch_separator: str) -> List[str]:
    if not batch_mode:
        return [input_prompt]
    separator = decode_separator(batch_separator)
    if separator and separator in input_prompt:
        items = input_prompt.split(separator)
    else:
        items = input_prompt.splitlines()
    return [item.strip() for item in items if item.strip()]
