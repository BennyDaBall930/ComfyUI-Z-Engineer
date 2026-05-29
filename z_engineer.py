import json
import logging
import re
from typing import List, Tuple

import requests


DEFAULT_API_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen3-4b-z-image-engineer-v4"
PREFERRED_MODELS = [
    "qwen3-4b-z-image-engineer-v4",
    "qwen/qwen3.6-27b",
    "granite-4.1-30b",
    "google/gemma-4-31b",
    "google/gemma-4-26b-a4b",
]

V6_SYSTEM_PROMPT = (
    "You are Z-Image-Engineer V6, a prompt-only cinematography and visual-language "
    "specialist for the Tongyi-MAI Z-Image-Turbo Qwen text encoder. Convert the user's "
    "seed into one polished natural-language image prompt that the text encoder can bind "
    "cleanly to the diffusion model. Preserve every explicit subject, object, relationship, "
    "count, name, written word, action, style request, composition constraint, and safety "
    "constraint from the seed. Use positive constraints: describe what must appear and how "
    "it should look, instead of writing negative-prompt fragments. Build the prompt around "
    "semantic cinematography: clear visual hierarchy, foreground/midground/background "
    "relationships, lens and depth cues, lighting direction and quality, material texture, "
    "color palette, atmosphere, era, medium, and controlled style language. Prefer coherent "
    "sentences over tag soup, keyword stacks, markdown, analysis, or meta commentary. Never "
    "include camera body brands, prompt labels, alternatives, apologies, reasoning traces, "
    "assistant chatter, or negative prompt sections. Return only the final image prompt as "
    "one self-contained paragraph."
)

REASONING_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
CHATML_TAG_PATTERN = re.compile(r"<\|im_(?:start|end)\|>(?:system|user|assistant)?", re.IGNORECASE)
PROMPT_PREFIX_PATTERN = re.compile(r"^\s*(?:final\s+)?(?:image\s+)?prompt\s*[:\-]\s*", re.IGNORECASE)
NEGATIVE_SECTION_PATTERN = re.compile(r"\bnegative\s+prompt\s*[:\-].*$", re.IGNORECASE | re.DOTALL)
CAMERA_BODY_PATTERN = re.compile(
    r"\b(?:Canon\s*(?:EOS|R\d+)?|Nikon\s*(?:D\d+|Z\d+)?|Sony\s*A\d+|Fujifilm\s*X|Leica\s*[MQSL]?|"
    r"ARRI|RED\s+(?:Komodo|V-Raptor|Monstro)?|Blackmagic|Panavision)\b",
    re.IGNORECASE,
)


def normalize_api_url(api_url: str) -> str:
    api_url = (api_url or DEFAULT_API_URL).strip().rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url[: -len("/chat/completions")]
    if not api_url.endswith("/v1"):
        api_url = f"{api_url}/v1"
    return api_url


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split())


def discover_lmstudio_models(api_url: str, timeout: int = 5) -> List[str]:
    endpoint = f"{normalize_api_url(api_url)}/models"
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        models = [str(item.get("id", "")).strip() for item in data.get("data", []) if item.get("id")]
        return models
    except Exception as exc:
        logging.warning("Z-Engineer model discovery failed: %s", exc)
        return []


def choose_model(requested_model: str, api_url: str, discover_models: bool) -> Tuple[str, List[str]]:
    requested_model = (requested_model or "").strip()
    available = discover_lmstudio_models(api_url) if discover_models else []
    if requested_model and requested_model.lower() != "auto":
        return requested_model, available
    lowered = {model.lower(): model for model in available}
    for preferred in PREFERRED_MODELS:
        if preferred.lower() in lowered:
            return lowered[preferred.lower()], available
    return (available[0] if available else DEFAULT_MODEL), available


def sanitize_prompt(text: str, strip_reasoning: bool = True, sanitize_output: bool = True) -> str:
    text = str(text or "")
    if strip_reasoning:
        text = REASONING_BLOCK_PATTERN.sub(" ", text)
    if sanitize_output:
        text = CHATML_TAG_PATTERN.sub(" ", text)
        text = NEGATIVE_SECTION_PATTERN.sub(" ", text)
        text = text.replace("```", " ")
        text = PROMPT_PREFIX_PATTERN.sub("", text)
        text = CAMERA_BODY_PATTERN.sub("", text)
    text = re.sub(r"\s+,", ",", text)
    return normalize_ws(text).strip(" \t\r\n\"'")


def split_batch(input_prompt: str, batch_mode: bool, batch_separator: str) -> List[str]:
    if not batch_mode:
        return [input_prompt]
    separator = batch_separator.encode("utf-8").decode("unicode_escape") if batch_separator else "\n---\n"
    if separator and separator in input_prompt:
        items = input_prompt.split(separator)
    else:
        items = input_prompt.splitlines()
    return [item.strip() for item in items if item.strip()]


class ZEngineer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Raw prompt or newline-separated batch...",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": V6_SYSTEM_PROMPT,
                        "placeholder": "Z-Engineer system prompt...",
                    },
                ),
                "api_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_API_URL,
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_MODEL,
                        "placeholder": "Use 'auto' to pick from LM Studio /v1/models",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 6606,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1200,
                        "min": 32,
                        "max": 4096,
                    },
                ),
                "batch_mode": ("BOOLEAN", {"default": False}),
                "batch_separator": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "\\n---\\n",
                    },
                ),
                "discover_models": ("BOOLEAN", {"default": True}),
                "strip_reasoning": ("BOOLEAN", {"default": True}),
                "sanitize_output": ("BOOLEAN", {"default": True}),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 120,
                        "min": 5,
                        "max": 1200,
                    },
                ),
                "retries": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 5,
                    },
                ),
                "error_mode": (["return_input", "return_error", "empty"], {"default": "return_input"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Z-Engineer"

    def _error_result(self, message: str, fallback_prompt: str, error_mode: str) -> str:
        logging.error("Z-Engineer error: %s", message)
        if error_mode == "return_error":
            return f"Z-Engineer error: {message}"
        if error_mode == "empty":
            return ""
        return fallback_prompt

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        api_url: str,
        model: str,
        seed: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: int,
        retries: int,
    ) -> str:
        endpoint = f"{normalize_api_url(api_url)}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "seed": int(seed),
            "stream": False,
        }

        last_exc = None
        for attempt in range(max(1, int(retries) + 1)):
            try:
                response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout_seconds)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                content = str(message.get("content", "") or "")
                if not content and message.get("reasoning_content"):
                    raise RuntimeError(
                        "model returned only reasoning_content; increase max_tokens or use a non-thinking/no-think model"
                    )
                return content
            except Exception as exc:
                last_exc = exc
                logging.warning("Z-Engineer request attempt %s failed: %s", attempt + 1, exc)
        raise RuntimeError(str(last_exc))

    def generate_prompt(
        self,
        input_prompt,
        system_prompt,
        api_url,
        model,
        seed,
        temperature,
        top_p,
        max_tokens,
        batch_mode,
        batch_separator,
        discover_models,
        strip_reasoning,
        sanitize_output,
        timeout_seconds,
        retries,
        error_mode,
    ):
        input_prompt = str(input_prompt or "").strip()
        if not input_prompt:
            return ("",)

        api_url = normalize_api_url(api_url)
        chosen_model, available_models = choose_model(model, api_url, discover_models)
        if available_models and chosen_model not in available_models:
            logging.info("Z-Engineer using custom model '%s'; LM Studio returned %s models.", chosen_model, len(available_models))
        else:
            logging.info("Z-Engineer using model '%s'.", chosen_model)

        prompts = split_batch(input_prompt, bool(batch_mode), str(batch_separator or ""))
        outputs = []
        separator = str(batch_separator or "\\n---\\n").encode("utf-8").decode("unicode_escape")
        for idx, prompt in enumerate(prompts):
            try:
                raw = self._call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt or V6_SYSTEM_PROMPT,
                    api_url=api_url,
                    model=chosen_model,
                    seed=int(seed) + idx,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
                outputs.append(sanitize_prompt(raw, bool(strip_reasoning), bool(sanitize_output)))
            except Exception as exc:
                outputs.append(self._error_result(str(exc), prompt, error_mode))

        return (separator.join(outputs) if batch_mode else outputs[0],)
