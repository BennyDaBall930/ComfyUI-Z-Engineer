"""Optional OpenAI-compatible API node (LM Studio, llama.cpp server, etc.).

This is the legacy integration path. The recommended path is the local
ZEngineerEnhance node, which needs no external server.
"""

import logging
from typing import List, Tuple

import requests

from .prompt_utils import (
    V6_SYSTEM_PROMPT,
    build_user_prompt,
    decode_separator,
    enforce_keep_terms,
    parse_keep_terms,
    preserve_seed_constraints,
    sanitize_prompt,
    split_batch,
)


DEFAULT_API_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "auto"
PREFERRED_MODELS = [
    "z-image-engineer-v6",
    "z-image-engineer-v6@q4_k_m",
    "z-image-engineer-v6@f16",
    "z-image-engineer-v6@mxfp4",
    "z-image-engineer-v6@q5_k_m",
    "z-image-engineer-v6@q6_k",
    "z-image-engineer-v6@q8_0",
    "z-image-engineer-v6@q3_k_m",
    "Z-Image-Engineer-V6",
    "Z-Image-Engineer-V6-GGUF",
    "z-image-engineer-v6-qwen3-native-step500",
    "z-image-engineer-v6-smart",
    "qwen3-4b-z-image-engineer-v4",
]


def normalize_api_url(api_url: str) -> str:
    api_url = (api_url or DEFAULT_API_URL).strip().rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url[: -len("/chat/completions")]
    if not api_url.endswith("/v1"):
        api_url = f"{api_url}/v1"
    return api_url


def discover_models(api_url: str, timeout: int = 5) -> List[str]:
    endpoint = f"{normalize_api_url(api_url)}/models"
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return [str(item.get("id", "")).strip() for item in data.get("data", []) if item.get("id")]
    except Exception as exc:
        logging.warning("Z-Engineer model discovery failed: %s", exc)
        return []


def choose_model(requested_model: str, api_url: str, discover: bool) -> Tuple[str, List[str]]:
    requested_model = (requested_model or "").strip()
    available = discover_models(api_url) if discover else []
    if requested_model and requested_model.lower() != "auto":
        return requested_model, available
    lowered = {model.lower(): model for model in available}
    for preferred in PREFERRED_MODELS:
        if preferred.lower() in lowered:
            return lowered[preferred.lower()], available
    for key, model in lowered.items():
        if "z-image-engineer" in key:
            return model, available
    return (available[0] if available else DEFAULT_MODEL), available


class ZEngineer:
    """Prompt enhancement through an OpenAI-compatible /chat/completions API."""

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
                        "placeholder": "Use 'auto' to pick from /v1/models",
                    },
                ),
                "seed": ("INT", {"default": 6606, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000}),
                "min_p": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 320, "min": 32, "max": 4096}),
                "batch_mode": ("BOOLEAN", {"default": False}),
                "batch_separator": ("STRING", {"multiline": False, "default": "\\n---\\n"}),
                "discover_models": ("BOOLEAN", {"default": True}),
                "enforce_seed_terms": ("BOOLEAN", {"default": True}),
                "strip_reasoning": ("BOOLEAN", {"default": True}),
                "sanitize_output": ("BOOLEAN", {"default": True}),
                "timeout_seconds": ("INT", {"default": 120, "min": 5, "max": 1200}),
                "retries": ("INT", {"default": 1, "min": 0, "max": 5}),
                "error_mode": (["return_input", "return_error", "empty"], {"default": "return_input"}),
            },
            "optional": {
                "keep_terms": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Comma-separated trigger words/phrases (e.g. LoRA triggers) kept verbatim in the output. Any the model drops are re-appended.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_NODE = True
    FUNCTION = "generate_prompt"
    CATEGORY = "Z-Engineer"
    DESCRIPTION = "Prompt enhancement through an external OpenAI-compatible server (LM Studio, llama.cpp, Ollama). Prefer the local Z-Engineer Prompt Enhancer node when the model is loaded inside ComfyUI."

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
        top_k: int,
        min_p: float,
        max_tokens: int,
        timeout_seconds: int,
        retries: int,
        keep_terms=None,
    ) -> str:
        endpoint = f"{normalize_api_url(api_url)}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(prompt, keep_terms)},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "seed": int(seed),
            "stream": False,
        }
        if int(top_k) > 0:
            payload["top_k"] = int(top_k)
        if float(min_p) > 0:
            payload["min_p"] = float(min_p)

        last_exc = None
        for attempt in range(max(1, int(retries) + 1)):
            try:
                response = requests.post(
                    endpoint,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout_seconds,
                )
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
        top_k,
        min_p,
        max_tokens,
        batch_mode,
        batch_separator,
        discover_models,
        enforce_seed_terms,
        strip_reasoning,
        sanitize_output,
        timeout_seconds,
        retries,
        error_mode,
        keep_terms="",
    ):
        input_prompt = str(input_prompt or "").strip()
        if not input_prompt:
            return {"ui": {"text": [""]}, "result": ("",)}

        api_url = normalize_api_url(api_url)
        chosen_model, available_models = choose_model(model, api_url, discover_models)
        if available_models and chosen_model not in available_models:
            logging.info(
                "Z-Engineer using custom model '%s'; server returned %s models.",
                chosen_model,
                len(available_models),
            )
        else:
            logging.info("Z-Engineer using model '%s'.", chosen_model)

        kept_terms = parse_keep_terms(keep_terms)
        prompts = split_batch(input_prompt, bool(batch_mode), str(batch_separator or ""))
        outputs = []
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
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    keep_terms=kept_terms,
                )
                cleaned = sanitize_prompt(raw, bool(strip_reasoning), bool(sanitize_output))
                if enforce_seed_terms:
                    cleaned = preserve_seed_constraints(prompt, cleaned)
                if kept_terms:
                    cleaned = enforce_keep_terms(cleaned, kept_terms)
                outputs.append(cleaned)
            except Exception as exc:
                outputs.append(self._error_result(str(exc), prompt, error_mode))

        if batch_mode:
            result = decode_separator(batch_separator).join(outputs)
        else:
            result = outputs[0]
        return {"ui": {"text": outputs}, "result": (result,)}
