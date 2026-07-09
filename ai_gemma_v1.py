from __future__ import annotations
import argparse
import base64
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
load_dotenv()

OUTPUT_DIR = Path("./data/output")

SUPPORTED_PROVIDERS = ("gemini", "openai", "claude", "gemma")
DEFAULT_ALL_PROVIDERS = ("gemini", "openai", "claude")
ALL_PROVIDERS_VALUE = "all"
DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", ALL_PROVIDERS_VALUE).strip().lower() or ALL_PROVIDERS_VALUE
DEFAULT_GEMMA_HF_MODEL = "google/gemma-4-31B-it"

DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": os.getenv("GEMINI_MODEL", os.getenv("AI_MODEL", "gemini-3.1-pro-preview")),
    "openai": os.getenv("OPENAI_MODEL", os.getenv("AI_MODEL", "gpt-5.5")),
    "claude": os.getenv("CLAUDE_MODEL", os.getenv("AI_MODEL", "claude-opus-4-8")),
    "gemma": os.getenv("GEMMA_MODEL", os.getenv("AI_MODEL", DEFAULT_GEMMA_HF_MODEL)),
}

DEFAULT_FALLBACK_MODELS_BY_PROVIDER = {
    "gemini": os.getenv("GEMINI_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["gemini"])),
    "openai": os.getenv("OPENAI_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["openai"])),
    "claude": os.getenv("CLAUDE_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["claude"])),
    "gemma": os.getenv("GEMMA_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["gemma"])),
}

DEFAULT_AI_RETRIES = int(os.getenv("AI_MAX_RETRIES", os.getenv("GEMINI_MAX_RETRIES", "5")))
DEFAULT_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", os.getenv("GEMINI_RETRY_BASE_DELAY", "2.0")))
DEFAULT_FILE_PROCESSING_TIMEOUT = int(os.getenv("GEMINI_FILE_PROCESSING_TIMEOUT", "300"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("AI_MAX_OUTPUT_TOKENS", "8192"))
DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER = {
    "gemini": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
    "openai": int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
    "claude": int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
    "gemma": int(os.getenv("GEMMA_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
}
DEFAULT_GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "256"))
DEFAULT_CLAUDE_TEXT_SINGLE_MAX_CHARS = int(os.getenv("CLAUDE_TEXT_SINGLE_MAX_CHARS", "350000"))
DEFAULT_CLAUDE_TEXT_CHUNK_CHARS = int(os.getenv("CLAUDE_TEXT_CHUNK_CHARS", "120000"))
DEFAULT_CLAUDE_TEXT_MIN_CHUNK_CHARS = int(os.getenv("CLAUDE_TEXT_MIN_CHUNK_CHARS", "30000"))
DEFAULT_GEMMA_TEXT_SINGLE_MAX_CHARS = int(os.getenv("GEMMA_TEXT_SINGLE_MAX_CHARS", "220000"))
DEFAULT_GEMMA_TEXT_CHUNK_CHARS = int(os.getenv("GEMMA_TEXT_CHUNK_CHARS", "220000"))
DEFAULT_GEMMA_TEXT_MIN_CHUNK_CHARS = int(os.getenv("GEMMA_TEXT_MIN_CHUNK_CHARS", "10000"))
DEFAULT_GEMMA_RUNTIME = os.getenv("GEMMA_RUNTIME", "transformers").strip().lower() or "transformers"
DEFAULT_GEMMA_DEVICE_MAP = os.getenv("GEMMA_DEVICE_MAP", "auto").strip() or "auto"
DEFAULT_GEMMA_TORCH_DTYPE = os.getenv("GEMMA_TORCH_DTYPE", os.getenv("GEMMA_DTYPE", "auto")).strip() or "auto"
DEFAULT_GEMMA_EXTRACTION_MODE = os.getenv("GEMMA_EXTRACTION_MODE", "layout").strip().lower() or "layout"
DEFAULT_GEMMA_MERGE_MODE = os.getenv("GEMMA_MERGE_MODE", "local").strip().lower() or "local"
DEFAULT_GEMMA_MERGE_STRATEGY = os.getenv("GEMMA_MERGE_STRATEGY", "batched").strip().lower() or "batched"
DEFAULT_GEMMA_MERGE_BATCH_CHARS = int(os.getenv("GEMMA_MERGE_BATCH_CHARS", "60000"))
DEFAULT_GEMMA_LEVEL_REFERENCE = (
    os.getenv("GEMMA_LEVEL_REFERENCE", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_GEMMA_LEVEL_VERIFY = (
    os.getenv("GEMMA_LEVEL_VERIFY", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE = (
    os.getenv(
        "GEMMA_LEVEL_VERIFY_SCOPE",
        "final" if DEFAULT_GEMMA_LEVEL_VERIFY else "none",
    ).strip().lower()
    or ("final" if DEFAULT_GEMMA_LEVEL_VERIFY else "none")
)
DEFAULT_GEMMA_DOCUMENT_STYLE_REFERENCE = (
    os.getenv("GEMMA_DOCUMENT_STYLE_REFERENCE", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_CLUSTERS = int(os.getenv("GEMMA_DOCUMENT_STYLE_MAX_CLUSTERS", "12"))
DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_EXAMPLES = int(os.getenv("GEMMA_DOCUMENT_STYLE_MAX_EXAMPLES", "80"))
DEFAULT_WRITE_PARSED_PDF = (
    os.getenv("WRITE_PARSED_PDF", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_WRITE_INPUT_CHUNKS = (
    os.getenv("WRITE_INPUT_CHUNKS", os.getenv("WRITE_CHUNK_RESULTS", "true")).strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_WRITE_STAGE_FILES = (
    os.getenv("WRITE_STAGE_FILES", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_GEMMA_VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("GEMMA_VLLM_TENSOR_PARALLEL_SIZE", os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
DEFAULT_GEMMA_VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("GEMMA_VLLM_GPU_MEMORY_UTILIZATION", os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")))
DEFAULT_GEMMA_VLLM_MAX_MODEL_LEN = os.getenv("GEMMA_VLLM_MAX_MODEL_LEN", os.getenv("VLLM_MAX_MODEL_LEN", "")).strip()
DEFAULT_GEMMA_VLLM_TRUST_REMOTE_CODE = (
    os.getenv("GEMMA_VLLM_TRUST_REMOTE_CODE", os.getenv("VLLM_TRUST_REMOTE_CODE", "false")).strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_GEMMA_VLLM_SERVER_BASE_URL = (
    os.getenv("GEMMA_VLLM_SERVER_BASE_URL", os.getenv("VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000/v1")).strip()
    or "http://127.0.0.1:8000/v1"
)
DEFAULT_GEMMA_VLLM_SERVER_API_KEY = (
    os.getenv("GEMMA_VLLM_SERVER_API_KEY", os.getenv("VLLM_API_KEY", "EMPTY")).strip()
    or "EMPTY"
)
DEFAULT_GEMMA_VLLM_SERVER_TIMEOUT = int(os.getenv("GEMMA_VLLM_SERVER_TIMEOUT", "3600"))

TOC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer"},
                    "chapter": {"type": "string"},
                    "page": {"type": "integer"},
                    "level_reason": {"type": "string"},
                },
                "required": ["level", "chapter", "page"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "chapters"],
    "additionalProperties": False,
}

LAYOUT_METADATA_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "_layout_x": ("_layout_x", "layout_x", "x", "left_indent"),
    "_layout_y": ("_layout_y", "layout_y", "y", "vertical_position"),
    "_layout_font_size": ("_layout_font_size", "layout_font_size", "font_size", "s"),
    "_layout_size_ratio": ("_layout_size_ratio", "layout_size_ratio", "size_ratio", "r"),
    "_layout_text": ("_layout_text", "layout_text", "source_text"),
    "_layout_page": ("_layout_page", "layout_page"),
    "_layout_bold": ("_layout_bold", "layout_bold", "bold", "b"),
    "_layout_italic": ("_layout_italic", "layout_italic", "italic", "i"),
    "_source_order": ("_source_order", "source_order"),
}

DEFAULT_USER_PROMPT = """
Create a table of contents from real body hierarchy titles only.
Exclude covers, prefaces, existing TOC pages/listings, indexes, references, page numbers, repeated headers/footers, captions, questions, and body sentences.
Keep exact source titles/numbering and actual PDF viewer page numbers.
""".strip()

RETRYABLE_ERROR_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "RATE LIMIT",
    "500",
    "INTERNAL",
    "502",
    "503",
    "UNAVAILABLE",
    "HIGH DEMAND",
    "OVERLOADED",
    "TRY AGAIN LATER",
    "504",
    "DEADLINE_EXCEEDED",
    "TIMEOUT",
    "TIMED OUT",
    "TEMPORAR",
    "CONNECTION",
)

CONFIGURATION_ERROR_MARKERS = (
    "API KEY",
    "API_KEY",
    "UNAUTHENTICATED",
    "PERMISSION_DENIED",
    "UNAUTHORIZED",
    "401",
    "403",
    "PIP INSTALL",
    "NO API KEY",
    "MISSING API KEY",
)

SCHEMA_CONFIG_ERROR_MARKERS = (
    "RESPONSE_SCHEMA",
    "RESPONSE_JSON_SCHEMA",
    "THINKING",
    "BUDGET",
    "JSON_SCHEMA",
    "UNKNOWN FIELD",
    "UNRECOGNIZED FIELD",
    "INVALID FIELD",
    "EXTRA INPUTS ARE NOT PERMITTED",
    "UNSUPPORTED",
    "NOT SUPPORTED",
)


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_provider(provider: str | None) -> str:
    provider = clean_text(provider).lower()
    aliases = {
        "anthropic": "claude",
        "open-ai": "openai",
        "google": "gemini",
        "local": "gemma",
    }
    provider = aliases.get(provider, provider)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider} / available: {', '.join(SUPPORTED_PROVIDERS)}")
    return provider


def resolve_providers(provider_option: str | None) -> list[str]:
    provider_text = clean_text(provider_option).lower() or DEFAULT_PROVIDER
    aliases = {
        "*": ALL_PROVIDERS_VALUE,
        "multi": ALL_PROVIDERS_VALUE,
        "all": ALL_PROVIDERS_VALUE,
        "anthropic": "claude",
        "open-ai": "openai",
        "google": "gemini",
        "local": "gemma",
    }
    provider_text = aliases.get(provider_text, provider_text)

    if provider_text == ALL_PROVIDERS_VALUE:
        return list(DEFAULT_ALL_PROVIDERS)

    providers: list[str] = []
    for item in provider_text.split(","):
        provider = normalize_provider(item)
        if provider not in providers:
            providers.append(provider)

    if not providers:
        raise ValueError("No providers to run.")

    return providers


def default_model_for(provider: str) -> str:
    return DEFAULT_MODEL_BY_PROVIDER[provider]


def default_fallback_models_for(provider: str) -> str:
    return DEFAULT_FALLBACK_MODELS_BY_PROVIDER[provider]


def default_max_output_tokens_for(provider: str) -> int:
    return DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER[provider]


def normalize_model_for_provider(provider: str, model: str) -> str:
    """Normalize user shorthand model names to provider-specific API model names."""
    model = clean_text(model)

    if provider == "gemini" and model:
        if model.startswith("models/"):
            model = model.split("/", 1)[1]
        if not model.startswith("gemini-"):
            known_suffixes = (
                "3.1-pro-preview",
                "3.1-pro",
                "2.5-pro",
                "2.5-flash",
                "2.0-flash",
                "1.5-pro",
                "1.5-flash",
            )
            if model in known_suffixes:
                model = f"gemini-{model}"

    if provider == "gemma" and model:
        if model.startswith("models/"):
            model = model.split("/", 1)[1]
        if model in {"latest", "highest", "max"}:
            model = "gemma4:31b"
        elif model in {"e2b", "e4b", "12b", "26b", "31b"}:
            model = f"gemma4:{model}"
        elif model == "27b":
            model = "gemma3:27b"

    return model


def normalize_model_list_for_provider(provider: str, models_text: str | Iterable[str] | None) -> str:
    models = parse_model_list("", models_text)
    return ",".join(normalize_model_for_provider(provider, model) for model in models)


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()

    if args.prompt:
        return args.prompt.strip()

    return DEFAULT_USER_PROMPT


def build_prompt(user_prompt: str, max_depth: int) -> str:
    return f"""
You are an expert at creating tables of contents for PDF textbooks and lecture materials.
Return a compact table-of-contents JSON object using only the source document.

[User Prompt]
{user_prompt.strip()}

[Output Format]
Output exactly one JSON object in the format below. Do not output explanations, Markdown, or code fences.

{{
  "title": "Document title",
  "chapters": [
    {{"level": 1, "chapter": "Major section title", "page": 1, "level_reason": "Concise reason this is a top-level section."}},
    {{"level": 2, "chapter": "Subsection title", "page": 3, "level_reason": "Concise reason this is a subsection under the previous level 1 entry."}}
  ]
}}

[Required Rules]
- Include only real body hierarchy titles, sorted in PDF page order.
- Exclude body sentences, examples, questions, captions, references, existing TOC pages/listings, and repeated headers/footers.
- Use exact source titles and existing numbering only. Do not invent titles or numbering.
- Use levels 1 to {max_depth}; page is the actual PDF viewer page number.
- Include a concise level_reason for every entry.
- Output JSON only.
""".strip()


def get_api_key(provider: str) -> str:
    env_name_by_provider = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_name = env_name_by_provider[provider]
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(f"{env_name} is required. Add {env_name}=... to your .env file.")
    return api_key


def message_of(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def attach_error_debug(error: Exception, **debug: Any) -> Exception:
    existing = getattr(error, "_ai_debug", None)
    if isinstance(existing, dict):
        existing.update(debug)
        debug = existing
    try:
        setattr(error, "_ai_debug", debug)
    except Exception:
        pass
    return error


def error_debug_info(error: Exception) -> dict[str, Any]:
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        debug = getattr(current, "_ai_debug", None)
        if isinstance(debug, dict):
            return debug
        current = current.__cause__ or current.__context__

    return {}


def is_cuda_out_of_memory_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return "CUDA OUT OF MEMORY" in message or "OUTOFMEMORYERROR" in message


def is_json_parse_error(error: Exception) -> bool:
    if isinstance(error, json.JSONDecodeError):
        return True
    message = message_of(error).upper()
    return (
        "JSONDECODEERROR" in message
        or "EXPECTING ',' DELIMITER" in message
        or "EXTRA DATA" in message
        or "UNTERMINATED STRING" in message
    )


def empty_torch_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return

    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def update_processing_metadata(args: argparse.Namespace, **metadata: Any) -> None:
    existing = getattr(args, "_ai_processing_metadata", None)
    if not isinstance(existing, dict):
        existing = {}

    existing.update(metadata)
    setattr(args, "_ai_processing_metadata", existing)


def is_retryable_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def is_configuration_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in CONFIGURATION_ERROR_MARKERS)


def is_schema_config_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in SCHEMA_CONFIG_ERROR_MARKERS)


def is_prompt_too_long_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return "PROMPT IS TOO LONG" in message or ("TOKENS" in message and "MAXIMUM" in message)


def is_stream_required_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return "STREAMING IS REQUIRED" in message


def with_retry(
    label: str,
    func: Callable[[], Any],
    max_retries: int,
    base_delay: float,
) -> Any:
    max_retries = max(1, int(max_retries))
    base_delay = max(0.1, float(base_delay))
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as error:
            last_error = error

            if is_configuration_error(error):
                raise

            if not is_retryable_error(error) or attempt >= max_retries:
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
            delay += random.uniform(0, min(1.0, base_delay))
            print(
                f"{label} temporary error, retry {attempt}/{max_retries - 1} "
                f"(after {delay:.1f}s): {error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    if last_error:
        raise last_error

    raise RuntimeError(f"{label} failed")


def parse_model_list(primary_model: str, fallback_models: str | Iterable[str] | None) -> list[str]:
    models: list[str] = []

    def add(value: str | None) -> None:
        value = clean_text(value)
        if value and value not in models:
            models.append(value)

    add(primary_model)

    if fallback_models is None:
        fallback_models = ""

    if isinstance(fallback_models, str):
        for item in fallback_models.split(","):
            add(item)
    else:
        for item in fallback_models:
            add(str(item))

    return models


def make_ascii_upload_name(pdf_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf_path.stem)
    safe_stem = safe_stem.strip("._-") or "input_pdf"
    return f"{safe_stem}.pdf"


def prepare_pdf_for_processing(pdf_path: Path) -> tuple[Path, str]:
    """Copy PDFs with non-ASCII names to an ASCII-safe temp path for processing."""
    if not pdf_path.exists():
        return pdf_path, pdf_path.name

    ascii_name = make_ascii_upload_name(pdf_path)
    if ascii_name == pdf_path.name and pdf_path.suffix.lower() == ".pdf":
        return pdf_path, pdf_path.name

    temp_dir = Path(tempfile.gettempdir()) / "ai_gemma_ascii_pdfs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / ascii_name
    if not target_path.exists() or target_path.stat().st_size != pdf_path.stat().st_size:
        shutil.copy2(pdf_path, target_path)
    return target_path, pdf_path.name


def strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def extract_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    end = text.rfind("}")
    if end > start:
        return text[start:end + 1]
    return None


def tokenize_json_like(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    index = 0

    while index < len(text):
        char = text[index]

        if char.isspace():
            index += 1
            continue

        if char in "{}[]:,":
            tokens.append((char, char))
            index += 1
            continue

        if char == '"':
            start = index
            index += 1
            escape = False
            while index < len(text):
                current = text[index]
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    index += 1
                    break
                index += 1
            tokens.append(("string", text[start:index]))
            continue

        start = index
        while index < len(text) and not text[index].isspace() and text[index] not in '{}[]:,"':
            index += 1
        tokens.append(("atom", text[start:index]))

    return tokens


def repair_json_text(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text.strip())
    tokens = tokenize_json_like(text)
    output: list[str] = []
    stack: list[dict[str, str]] = []

    def mark_value_complete() -> None:
        if stack:
            stack[-1]["state"] = "comma_or_end"

    def append_value(token_kind: str, token_value: str) -> None:
        output.append(token_value)
        if token_kind == "{":
            stack.append({"kind": "object", "state": "key_or_end"})
        elif token_kind == "[":
            stack.append({"kind": "array", "state": "value_or_end"})
        else:
            mark_value_complete()

    index = 0
    while index < len(tokens):
        token_kind, token_value = tokens[index]

        if not stack:
            append_value(token_kind, token_value)
            index += 1
            continue

        container = stack[-1]
        state = container["state"]
        kind = container["kind"]

        if state == "comma_or_end":
            closing = "}" if kind == "object" else "]"
            next_state = "key_or_end" if kind == "object" else "value_or_end"

            if token_kind == ",":
                output.append(token_value)
                container["state"] = next_state
                index += 1
                continue

            if token_kind == closing:
                output.append(token_value)
                stack.pop()
                mark_value_complete()
                index += 1
                continue

            output.append(",")
            container["state"] = next_state
            continue

        if kind == "object" and state == "key_or_end":
            if token_kind == "}":
                output.append(token_value)
                stack.pop()
                mark_value_complete()
                index += 1
                continue

            if token_kind == ",":
                index += 1
                continue

            if token_kind == "atom":
                output.append(json.dumps(token_value, ensure_ascii=False))
            else:
                output.append(token_value)
            container["state"] = "colon"
            index += 1
            continue

        if kind == "object" and state == "colon":
            if token_kind == ":":
                output.append(token_value)
                container["state"] = "value"
                index += 1
                continue

            output.append(":")
            container["state"] = "value"
            continue

        if kind == "object" and state == "value":
            if token_kind == ",":
                index += 1
                continue

            append_value(token_kind, token_value)
            index += 1
            continue

        if kind == "array" and state == "value_or_end":
            if token_kind == "]":
                output.append(token_value)
                stack.pop()
                mark_value_complete()
                index += 1
                continue

            if token_kind == ",":
                index += 1
                continue

            append_value(token_kind, token_value)
            index += 1
            continue

        output.append(token_value)
        index += 1

    return "".join(output)


def parse_json_response_text(text: str, provider: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    if not text:
        raise ValueError(f"{provider} response is empty.")

    candidates: list[str] = [text]
    extracted = extract_json_object_text(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in list(candidates):
        repaired = repair_json_text(candidate)
        if repaired and repaired not in candidates:
            candidates.append(repaired)

    last_error: Exception | None = None
    data: Any = None

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as error:
            last_error = error
    else:
        if provider == "gemma":
            loose_data = parse_loose_toc_json_text(text)
            if loose_data is not None:
                return loose_data
        if last_error:
            raise last_error
        raise ValueError(f"Could not find a JSON object in the {provider} response.")

    if not isinstance(data, dict):
        raise ValueError(f"The top-level JSON value in the {provider} response must be an object.")

    return data


def json_string_value(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except Exception:
        return value.replace('\\"', '"').replace("\\n", " ").strip()


def parse_loose_toc_json_text(text: str) -> dict[str, Any] | None:
    """Best-effort recovery for Gemma responses with missing commas between flat TOC objects."""
    title = "Document title"
    title_match = re.search(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if title_match:
        title = clean_text(json_string_value(title_match.group(1))) or title

    chapters: list[dict[str, Any]] = []
    for match in re.finditer(r'\{[^{}]*"chapter"[^{}]*\}', text, flags=re.DOTALL):
        object_text = match.group(0)
        repaired = repair_json_text(object_text)
        parsed: dict[str, Any] | None = None
        if repaired:
            try:
                candidate = json.loads(repaired)
                if isinstance(candidate, dict):
                    parsed = candidate
            except Exception:
                parsed = None

        if parsed is None:
            chapter_match = re.search(r'"chapter"\s*:\s*"((?:\\.|[^"\\])*)"', object_text)
            if not chapter_match:
                continue
            level_match = re.search(r'"level"\s*:\s*(-?\d+)', object_text)
            page_match = re.search(r'"page"\s*:\s*(-?\d+)', object_text)
            parsed = {
                "level": int(level_match.group(1)) if level_match else 1,
                "chapter": json_string_value(chapter_match.group(1)),
                "page": int(page_match.group(1)) if page_match else 1,
            }

        if clean_text(parsed.get("chapter")):
            chapters.append(parsed)

    if not chapters:
        return None

    return {
        "title": title,
        "chapters": chapters,
    }


def parsed_response_to_dict(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, str):
        return parse_json_response_text(parsed, provider="parsed")

    for method_name in ("model_dump", "dict"):
        method = getattr(parsed, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return value

    return None


def extract_gemini_text(response: Any) -> str:
    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))

    if parts:
        return "\n".join(parts)

    return str(getattr(response, "text", "") or "")


def parse_json_response(response: Any, provider: str) -> dict[str, Any]:
    if provider == "gemini":
        return parse_json_response_text(extract_gemini_text(response), provider)

    parsed = getattr(response, "parsed", None)
    parsed_dict = parsed_response_to_dict(parsed)
    if parsed_dict is not None:
        return parsed_dict

    if provider == "openai":
        text = str(getattr(response, "output_text", "") or "")
        if text:
            return parse_json_response_text(text, provider)

    if provider == "claude":
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "") or ""))
        return parse_json_response_text("\n".join(parts), provider)

    text = extract_gemini_text(response)
    return parse_json_response_text(text, provider)


def first_present_metadata_value(source: dict[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        if alias in source and source.get(alias) is not None:
            return source.get(alias)
    return None


def metadata_float_value(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except Exception:
        return None


def metadata_int_value(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def normalize_toc_duplicate_title(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"^\s*\d{2,}\s*[∙·ㆍ.\-–—|:]\s*", "", text)
    text = re.sub(r"\s*[∙·ㆍ.\-–—|:]\s*\d+\s*$", "", text)
    if len(text.split()) >= 3:
        text = re.sub(r"^\s*\d{2,}\s+", "", text)
        text = re.sub(r"\s+\d+\s*$", "", text)
    return re.sub(r"\s+", "", text).lower()


def is_marker_only_toc_title(title: Any) -> bool:
    text = clean_text(title)
    if not text:
        return False
    return bool(re.fullmatch(
        r"(?:chapter|part|section|unit)\s*\d+[A-Za-z]?|"
        r"제\s*\d+\s*[장절편부]|"
        r"\d+(?:\.\d+)*[.)]?",
        text,
        re.IGNORECASE,
    ))


def is_repeated_running_title_candidate(chapter: dict[str, Any]) -> bool:
    title = clean_text(chapter.get("chapter"))
    y = metadata_float_value(chapter.get("_layout_y"))
    font_size = metadata_float_value(chapter.get("_layout_font_size"))
    size_ratio = metadata_float_value(chapter.get("_layout_size_ratio"))
    bold = metadata_int_value(chapter.get("_layout_bold")) or 0
    italic = metadata_int_value(chapter.get("_layout_italic")) or 0
    if y is None:
        return False

    near_page_edge = y <= 125.0 or y >= 720.0
    weak_style = (
        (size_ratio is not None and size_ratio <= 1.12)
        or (font_size is not None and font_size <= 11.0)
    )
    return near_page_edge and ((weak_style and not bold and not italic) or is_marker_only_toc_title(title))


def toc_duplicate_keep_key(chapter: dict[str, Any]) -> tuple[int, float, float, int, int]:
    running_like = 1 if is_repeated_running_title_candidate(chapter) else 0
    font_size = metadata_float_value(chapter.get("_layout_font_size")) or 0.0
    size_ratio = metadata_float_value(chapter.get("_layout_size_ratio")) or 0.0
    page = metadata_int_value(chapter.get("page")) or 999999
    source_order = metadata_int_value(chapter.get("_source_order")) or metadata_int_value(chapter.get("_local_merge_order")) or 999999
    return running_like, -font_size, -size_ratio, page, source_order


def dedupe_toc_chapters(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_title: dict[str, list[int]] = {}
    for index, chapter in enumerate(chapters):
        normalized = normalize_toc_duplicate_title(chapter.get("chapter"))
        if normalized:
            by_title.setdefault(normalized, []).append(index)

    remove_indexes: set[int] = set()
    for indexes in by_title.values():
        if len(indexes) < 2:
            continue

        ordered = sorted(
            indexes,
            key=lambda index: (
                metadata_int_value(chapters[index].get("page")) or 999999,
                metadata_int_value(chapters[index].get("_source_order")) or index,
                index,
            ),
        )

        running_like_indexes = [index for index in ordered if is_repeated_running_title_candidate(chapters[index])]
        marker_only_repeated = any(is_marker_only_toc_title(chapters[index].get("chapter")) for index in ordered)
        if len(running_like_indexes) >= 2:
            non_running = [index for index in ordered if index not in running_like_indexes]
            if non_running:
                keep_index = min(non_running, key=lambda index: toc_duplicate_keep_key(chapters[index]))
                remove_indexes.update(index for index in running_like_indexes if index != keep_index)
            else:
                remove_indexes.update(running_like_indexes)

        if marker_only_repeated:
            keep_index = min(ordered, key=lambda index: toc_duplicate_keep_key(chapters[index]))
            remove_indexes.update(index for index in ordered if index != keep_index)

        kept_for_close_pages: list[int] = []
        for index in ordered:
            if index in remove_indexes:
                continue
            chapter = chapters[index]
            page = metadata_int_value(chapter.get("page"))
            level = metadata_int_value(chapter.get("level"))
            duplicate_of: int | None = None
            for kept_index in kept_for_close_pages:
                kept = chapters[kept_index]
                kept_page = metadata_int_value(kept.get("page"))
                kept_level = metadata_int_value(kept.get("level"))
                if page is None or kept_page is None or level != kept_level:
                    continue
                if abs(page - kept_page) <= 1:
                    duplicate_of = kept_index
                    break

            if duplicate_of is None:
                kept_for_close_pages.append(index)
                continue

            better_index = min((duplicate_of, index), key=lambda item: toc_duplicate_keep_key(chapters[item]))
            worse_index = index if better_index == duplicate_of else duplicate_of
            remove_indexes.add(worse_index)
            if better_index == index:
                kept_for_close_pages = [item for item in kept_for_close_pages if item != duplicate_of]
                kept_for_close_pages.append(index)

    return [chapter for index, chapter in enumerate(chapters) if index not in remove_indexes]


def copy_toc_layout_metadata(source: dict[str, Any], target: dict[str, Any]) -> None:
    for target_key, aliases in LAYOUT_METADATA_FIELD_ALIASES.items():
        value = first_present_metadata_value(source, aliases)
        if value is None:
            continue

        if target_key in {"_layout_x", "_layout_y", "_layout_font_size", "_layout_size_ratio"}:
            number = metadata_float_value(value)
            if number is not None:
                target[target_key] = number
            continue

        if target_key in {"_layout_page", "_source_order", "_layout_bold", "_layout_italic"}:
            number = metadata_int_value(value)
            if number is not None:
                target[target_key] = number
            continue

        text = clean_text(value)
        if text:
            target[target_key] = text


def validate_toc(data: dict[str, Any], fallback_title: str, max_depth: int) -> dict[str, Any]:
    title = clean_text(data.get("title")) or fallback_title
    chapters: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    raw_chapters = data.get("chapters", [])
    if not isinstance(raw_chapters, list):
        raw_chapters = []

    for item in raw_chapters:
        if not isinstance(item, dict):
            continue

        chapter = clean_text(item.get("chapter"))
        if not chapter:
            continue

        try:
            level = int(item.get("level", 1))
        except Exception:
            level = 1

        try:
            page = int(item.get("page", 1))
        except Exception:
            page = 1

        level = max(1, min(level, max_depth))
        page = max(1, page)
        key = (re.sub(r"\s+", "", chapter).lower(), page)

        if key in seen:
            continue

        seen.add(key)
        chapter_item = {
            "level": level,
            "chapter": chapter,
            "page": page,
        }
        level_reason = clean_text(item.get("level_reason"))
        if level_reason:
            chapter_item["level_reason"] = level_reason
        copy_toc_layout_metadata(item, chapter_item)
        chapters.append(chapter_item)

    chapters = dedupe_toc_chapters(chapters)
    return {
        "title": title,
        "chapters": chapters,
    }


# ----------------------------- Gemini -----------------------------

def import_gemini():
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("Gemini API requires `pip install google-genai`.") from error

    return genai, types


def create_gemini_client(api_key: str):
    genai, types = import_gemini()
    return genai.Client(api_key=api_key), types


def upload_pdf_to_gemini(client, pdf_path: Path, retries: int, base_delay: float):
    safe_name = make_ascii_upload_name(pdf_path)

    def do_upload():
        with tempfile.TemporaryDirectory(prefix="gemini_pdf_upload_") as temp_dir:
            upload_path = Path(temp_dir) / safe_name
            shutil.copy2(pdf_path, upload_path)

            try:
                return client.files.upload(
                    file=str(upload_path),
                    config={
                        "mime_type": "application/pdf",
                        "display_name": safe_name,
                    },
                )
            except TypeError:
                return client.files.upload(file=str(upload_path))

    print("  Gemini PDF upload started...", flush=True)
    uploaded = with_retry(
        label=f"Gemini PDF upload({pdf_path.name})",
        func=do_upload,
        max_retries=retries,
        base_delay=base_delay,
    )
    print("  Gemini PDF upload completed", flush=True)
    return uploaded


def file_state_name(uploaded_file: Any) -> str:
    state = getattr(uploaded_file, "state", None)
    if state is None:
        return ""

    name = getattr(state, "name", None)
    if name:
        return str(name).upper()

    text = str(state).upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip()


def wait_for_gemini_file_ready(client, uploaded_file: Any, timeout_seconds: int) -> Any:
    started = time.time()
    current = uploaded_file

    while True:
        state = file_state_name(current)

        if not state or state == "ACTIVE":
            return current

        if "FAILED" in state:
            raise RuntimeError(f"Gemini file processing failed: {state}")

        if time.time() - started > timeout_seconds:
            raise TimeoutError(
                f"Gemini file processing did not finish within {timeout_seconds}s. Last state: {state}"
            )

        print(f"  Gemini file processing: {state}", file=sys.stderr, flush=True)
        time.sleep(5)
        current = client.files.get(name=current.name)


def build_gemini_generation_configs(
    types,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
    thinking_budget: int | None,
) -> list[Any]:
    base: dict[str, Any] = {
        "response_mime_type": "application/json",
        "temperature": float(temperature),
    }

    if max_output_tokens:
        base["max_output_tokens"] = int(max_output_tokens)

    configs: list[Any] = []
    base_variants: list[dict[str, Any]] = []

    if thinking_budget is not None:
        thinking_base = dict(base)
        try:
            thinking_base["thinking_config"] = types.ThinkingConfig(
                thinking_budget=int(thinking_budget),
                include_thoughts=False,
            )
        except Exception:
            thinking_base["thinking_config"] = {
                "thinking_budget": int(thinking_budget),
                "include_thoughts": False,
            }
        base_variants.append(thinking_base)

    base_variants.append(base)

    for base_config in base_variants:
        if use_schema:
            try:
                kwargs = dict(base_config)
                kwargs["response_schema"] = TOC_SCHEMA
                configs.append(types.GenerateContentConfig(**kwargs))
            except Exception:
                pass

            schema_config = dict(base_config)
            schema_config["response_schema"] = TOC_SCHEMA
            configs.append(schema_config)

            json_schema_config = dict(base_config)
            json_schema_config["response_json_schema"] = TOC_SCHEMA
            configs.append(json_schema_config)

        configs.append(base_config)
    return configs


def generate_gemini_once(
    client,
    model: str,
    uploaded_file: Any,
    prompt: str,
    types,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
    thinking_budget: int | None,
):
    configs = build_gemini_generation_configs(
        types=types,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        use_schema=use_schema,
        thinking_budget=thinking_budget,
    )

    last_error: Exception | None = None

    for index, config in enumerate(configs):
        try:
            return client.models.generate_content(
                model=model,
                contents=[prompt, uploaded_file],
                config=config,
            )
        except Exception as error:
            last_error = error

            if index < len(configs) - 1 and is_schema_config_error(error):
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("Gemini response generation failed")


def generate_toc_from_pdf_gemini(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    client, types = create_gemini_client(api_key=get_api_key("gemini"))
    uploaded_file = upload_pdf_to_gemini(
        client=client,
        pdf_path=pdf_path,
        retries=args.ai_retries,
        base_delay=args.ai_retry_base_delay,
    )

    uploaded_file = wait_for_gemini_file_ready(
        client=client,
        uploaded_file=uploaded_file,
        timeout_seconds=args.file_processing_timeout,
    )

    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    try:
        for index, model in enumerate(models):
            model = normalize_model_for_provider("gemini", model)

            if index > 0:
                print(f"Trying Gemini fallback model: {model}", file=sys.stderr, flush=True)

            parse_retry_prompt = prompt

            for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
                try:
                    if parse_attempt > 1:
                        print(
                            f"  Retrying Gemini generation after JSON parse failure {parse_attempt}/{args.ai_retries}: {model}",
                            file=sys.stderr,
                            flush=True,
                        )
                        parse_retry_prompt = (
                            prompt
                            + "\n\n[Important Retry Instruction]\n"
                            + "The previous response failed JSON parsing due to invalid JSON syntax. "
                            + "This time, output exactly one valid JSON object that Python json.loads() can parse directly. "
                            + "Do not omit commas. Do not output Markdown, explanations, or code fences."
                        )

                    print(f"  Gemini TOC generation started: {model}", flush=True)
                    response = with_retry(
                        label=f"Gemini generation({model})",
                        func=lambda model=model, parse_retry_prompt=parse_retry_prompt: generate_gemini_once(
                            client=client,
                            model=model,
                            uploaded_file=uploaded_file,
                            prompt=parse_retry_prompt,
                            types=types,
                            temperature=args.temperature,
                            max_output_tokens=args.max_output_tokens,
                            use_schema=not args.no_schema,
                            thinking_budget=args.gemini_thinking_budget,
                        ),
                        max_retries=args.ai_retries,
                        base_delay=args.ai_retry_base_delay,
                    )
                    print(f"  Gemini TOC generation completed: {model}", flush=True)

                    raw_text = extract_gemini_text(response)
                    try:
                        parsed = parse_json_response_text(raw_text, provider="gemini")
                    except Exception as parse_error:
                        last_error = parse_error
                        if args.write_raw:
                            model_name_for_file = safe_filename_part(model)
                            raw_file = (
                                Path(args.output_dir)
                                / f"{pdf_path.stem}_gemini_{model_name_for_file}_raw_response_failed_attempt{parse_attempt}.txt"
                            )
                            raw_file.parent.mkdir(parents=True, exist_ok=True)
                            raw_file.write_text(raw_text, encoding="utf-8")
                            print(f"Raw response saved: {raw_file}", flush=True)

                        if parse_attempt >= max(1, int(args.ai_retries)):
                            raise

                        continue

                    return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model

                except Exception as error:
                    last_error = error
                    if is_configuration_error(error):
                        raise

                    # Retry parse failures with the same model, then move API/model failures to the next fallback model.
                    if isinstance(error, json.JSONDecodeError) and parse_attempt < max(1, int(args.ai_retries)):
                        continue

                    print(f"Gemini model failed: {model} / {error}", file=sys.stderr, flush=True)
                    break

        if last_error:
            raise last_error
        raise RuntimeError("No Gemini models to try.")

    finally:
        if args.delete_uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
                print("Gemini uploaded file deleted", file=sys.stderr, flush=True)
            except Exception as error:
                print(f"Failed to delete Gemini uploaded file: {error}", file=sys.stderr, flush=True)


# ----------------------------- OpenAI -----------------------------

def create_openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI API requires `pip install openai`.") from error
    return OpenAI(api_key=api_key)


def build_openai_response_payload(
    model: str,
    prompt: str,
    file_id: str,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "store": False,
    }

    # Some OpenAI reasoning models, including some gpt-5-family models, do not support temperature.
    # OpenAI requests therefore omit temperature by default.
    # Use prompts/schema constraints to control output stability when needed.

    if max_output_tokens:
        base["max_output_tokens"] = int(max_output_tokens)

    payloads: list[dict[str, Any]] = []

    if use_schema:
        schema_payload = dict(base)
        schema_payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "toc_schema",
                "schema": TOC_SCHEMA,
                "strict": True,
            }
        }
        payloads.append(schema_payload)

        json_object_payload = dict(base)
        json_object_payload["text"] = {"format": {"type": "json_object"}}
        payloads.append(json_object_payload)

    payloads.append(base)
    return payloads


def generate_openai_once(
    client,
    model: str,
    prompt: str,
    file_id: str,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
):
    payloads = build_openai_response_payload(
        model=model,
        prompt=prompt,
        file_id=file_id,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        use_schema=use_schema,
    )

    last_error: Exception | None = None
    for index, payload in enumerate(payloads):
        try:
            return client.responses.create(**payload)
        except Exception as error:
            last_error = error
            if index < len(payloads) - 1 and is_schema_config_error(error):
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("OpenAI response generation failed")


def generate_toc_from_pdf_openai(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    client = create_openai_client(api_key=get_api_key("openai"))
    uploaded_file = None
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    # OpenAI file/context input currently validates the filename extension case-sensitively.
    # A real PDF named "*.PDF" can fail with "supported format ... but got .PDF".
    # Upload through a temporary ASCII-safe lowercase ".pdf" filename while preserving
    # the original filename for output paths.
    safe_name = make_ascii_upload_name(pdf_path)

    try:
        print("  OpenAI PDF upload started...", flush=True)
        with tempfile.TemporaryDirectory(prefix="openai_pdf_upload_") as temp_dir:
            upload_path = Path(temp_dir) / safe_name
            shutil.copy2(pdf_path, upload_path)

            def do_upload():
                with upload_path.open("rb") as file_obj:
                    return client.files.create(file=file_obj, purpose="user_data")

            uploaded_file = with_retry(
                label=f"OpenAI PDF upload({pdf_path.name})",
                func=do_upload,
                max_retries=args.ai_retries,
                base_delay=args.ai_retry_base_delay,
            )
        print("  OpenAI PDF upload completed", flush=True)

        for index, model in enumerate(models):
            if index > 0:
                print(f"Trying OpenAI fallback model: {model}", file=sys.stderr, flush=True)

            parse_retry_prompt = prompt

            for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
                try:
                    if parse_attempt > 1:
                        print(
                            f"  Retrying OpenAI generation after JSON parse failure {parse_attempt}/{args.ai_retries}: {model}",
                            file=sys.stderr,
                            flush=True,
                        )
                        parse_retry_prompt = (
                            prompt
                            + "\n\n[Important Retry Instruction]\n"
                            + "The previous response failed JSON parsing due to invalid JSON syntax. "
                            + "This time, output exactly one valid JSON object that Python json.loads() can parse directly. "
                            + "Do not omit commas between array items, and close the JSON object completely even if the output is long. "
                            + "Do not output Markdown, explanations, or code fences."
                        )

                    print(f"  OpenAI TOC generation started: {model}", flush=True)
                    response = with_retry(
                        label=f"OpenAI generation({model})",
                        func=lambda model=model, parse_retry_prompt=parse_retry_prompt: generate_openai_once(
                            client=client,
                            model=model,
                            prompt=parse_retry_prompt,
                            file_id=uploaded_file.id,
                            temperature=args.temperature,
                            max_output_tokens=args.max_output_tokens,
                            use_schema=not args.no_schema,
                        ),
                        max_retries=args.ai_retries,
                        base_delay=args.ai_retry_base_delay,
                    )
                    print(f"  OpenAI TOC generation completed: {model}", flush=True)

                    raw_text = str(getattr(response, "output_text", "") or "")
                    try:
                        parsed = parse_json_response(response, provider="openai")
                    except Exception as parse_error:
                        last_error = parse_error
                        if args.write_raw:
                            model_name_for_file = safe_filename_part(model)
                            raw_file = (
                                Path(args.output_dir)
                                / f"{pdf_path.stem}_openai_{model_name_for_file}_raw_response_failed_attempt{parse_attempt}.txt"
                            )
                            raw_file.parent.mkdir(parents=True, exist_ok=True)
                            raw_file.write_text(raw_text, encoding="utf-8")
                            print(f"Raw response saved: {raw_file}", flush=True)

                        if parse_attempt >= max(1, int(args.ai_retries)):
                            raise

                        continue

                    return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model

                except Exception as error:
                    last_error = error
                    if is_configuration_error(error):
                        raise

                    if parse_attempt < max(1, int(args.ai_retries)) and (
                        isinstance(error, json.JSONDecodeError)
                        or isinstance(error, ValueError)
                    ):
                        continue

                    print(f"OpenAI model failed: {model} / {error}", file=sys.stderr, flush=True)
                    break

        if last_error:
            raise last_error
        raise RuntimeError("No OpenAI models to try.")

    finally:
        if args.delete_uploaded_file and uploaded_file is not None:
            try:
                client.files.delete(uploaded_file.id)
                print("OpenAI uploaded file deleted", file=sys.stderr, flush=True)
            except Exception as error:
                print(f"Failed to delete OpenAI uploaded file: {error}", file=sys.stderr, flush=True)


# ----------------------------- Claude -----------------------------

def create_claude_client(api_key: str):
    try:
        from anthropic import Anthropic
    except ImportError as error:
        raise RuntimeError("Claude API requires `pip install anthropic`.") from error
    return Anthropic(api_key=api_key)


def generate_claude_once(
    client,
    model: str,
    pdf_path: Path,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None,
):
    pdf_bytes = pdf_path.read_bytes()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    with client.messages.stream(
        model=model,
        max_tokens=int(max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS),
        # Some newer claude-opus/sonnet models deprecate temperature.
        # Sending it can cause a 400 invalid_request_error; use the prompt for stability.
        system=(
            "You are an expert at creating tables of contents for PDF textbooks and lecture materials. "
            "Output exactly one valid JSON object. Do not output Markdown or explanations."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    ) as stream:
        return stream.get_final_message()


def claude_response_text(response: Any) -> str:
    raw_text_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            raw_text_parts.append(str(getattr(block, "text", "") or ""))
    return "\n".join(raw_text_parts)


def generate_claude_text_once(
    client,
    model: str,
    prompt_text: str,
    max_output_tokens: int | None,
):
    with client.messages.stream(
        model=model,
        max_tokens=int(max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS),
        system=(
            "You are an expert at creating tables of contents for PDF textbooks and lecture materials. "
            "Output exactly one valid JSON object. Do not output Markdown or explanations."
        ),
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
    ) as stream:
        return stream.get_final_message()


def clean_extracted_pdf_text(text: str) -> str:
    text = str(text or "").replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def extract_pdf_text_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError("PDF text extraction requires `pip install pdfplumber`.") from error

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = clean_extracted_pdf_text(page.extract_text() or "")
            if text:
                pages.append((page_index, text))

    if not pages:
        raise RuntimeError("No extractable text was found in the PDF. Scanned PDFs require OCR.")

    return pages


def median_float(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return float(default)

    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2.0


def dominant_text_value(values: Iterable[Any]) -> str:
    counts: dict[str, int] = {}
    order: list[str] = []
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        if text not in counts:
            counts[text] = 0
            order.append(text)
        counts[text] += 1

    if not counts:
        return ""

    return max(order, key=lambda item: counts[item])


def font_style_flags(fontname: str) -> tuple[int, int]:
    fontname = clean_text(fontname).lower()
    bold_markers = ("bold", "black", "heavy", "demi", "semibold", "semi-bold", "medium")
    italic_markers = ("italic", "oblique", "slant")
    return (
        int(any(marker in fontname for marker in bold_markers)),
        int(any(marker in fontname for marker in italic_markers)),
    )


def compact_font_name(fontname: str) -> str:
    fontname = clean_text(fontname)
    if "+" in fontname:
        fontname = fontname.split("+", 1)[1]
    fontname = re.sub(r"[^A-Za-z0-9_-]+", "", fontname)
    return fontname[:24]


def font_id_for(fontname: str, font_ids: dict[str, str]) -> str:
    fontname = compact_font_name(fontname)
    if not fontname:
        return "f0"
    if fontname not in font_ids:
        font_ids[fontname] = f"f{len(font_ids) + 1}"
    return font_ids[fontname]


def page_body_font_size(page: Any) -> float:
    sizes = [
        float(char.get("size"))
        for char in getattr(page, "chars", []) or []
        if char.get("size") is not None and clean_text(char.get("text"))
    ]
    return median_float(sizes, default=10.0) or 10.0


def line_text_from_chars(chars: list[dict[str, Any]], fallback_text: Any) -> str:
    text = clean_extracted_pdf_text(fallback_text)
    if text:
        return text

    return clean_extracted_pdf_text("".join(str(char.get("text") or "") for char in chars))


def format_layout_line(
    line: dict[str, Any],
    body_size: float,
    font_ids: dict[str, str],
) -> str:
    chars = [char for char in line.get("chars", []) or [] if isinstance(char, dict)]
    text = line_text_from_chars(chars, line.get("text"))
    if not text:
        return ""

    if chars:
        size = median_float((char.get("size") for char in chars), default=body_size)
        x0 = min(float(char.get("x0", line.get("x0", 0.0)) or 0.0) for char in chars)
        top = min(float(char.get("top", line.get("top", 0.0)) or 0.0) for char in chars)
        fontname = dominant_text_value(char.get("fontname") for char in chars)
    else:
        size = float(line.get("size") or body_size or 10.0)
        x0 = float(line.get("x0") or 0.0)
        top = float(line.get("top") or 0.0)
        fontname = clean_text(line.get("fontname"))

    bold, italic = font_style_flags(fontname)
    ratio = size / body_size if body_size else 1.0
    font_id = font_id_for(fontname, font_ids)
    return f"[L s={size:.1f} r={ratio:.2f} x={x0:.0f} y={top:.0f} b={bold} i={italic} f={font_id}] {text}"


def extract_pdf_layout_pages(pdf_path: Path) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError("PDF layout extraction requires `pip install pdfplumber`.") from error

    pages: list[tuple[int, str]] = []
    font_ids: dict[str, str] = {}
    line_count = 0
    plain_chars = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            body_size = page_body_font_size(page)
            lines: list[dict[str, Any]]
            try:
                lines = page.extract_text_lines(layout=False, return_chars=True)
            except Exception:
                lines = []

            formatted_lines: list[str] = []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                formatted = format_layout_line(line=line, body_size=body_size, font_ids=font_ids)
                if formatted:
                    formatted_lines.append(formatted)
                    line_count += 1
                    plain_chars += len(formatted.split("] ", 1)[-1])

            if not formatted_lines:
                fallback_text = clean_extracted_pdf_text(page.extract_text() or "")
                if fallback_text:
                    formatted_lines.append(f"[L s={body_size:.1f} r=1.00 x=0 y=0 b=0 i=0 f=f0] {fallback_text}")
                    line_count += fallback_text.count("\n") + 1
                    plain_chars += len(fallback_text)

            if formatted_lines:
                pages.append((page_index, "\n".join(formatted_lines)))

    if not pages:
        raise RuntimeError("No extractable text was found in the PDF. Scanned PDFs require OCR.")

    metadata = {
        "gemma_extraction_mode": "layout",
        "layout_line_count": line_count,
        "layout_font_count": len(font_ids),
        "plain_extracted_chars": plain_chars,
    }
    return pages, metadata


def extract_gemma_pdf_pages(pdf_path: Path, args: argparse.Namespace) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    mode = clean_text(getattr(args, "gemma_extraction_mode", DEFAULT_GEMMA_EXTRACTION_MODE)).lower()
    if mode == "text":
        pages = extract_pdf_text_pages(pdf_path)
        plain_chars = sum(len(text) for _, text in pages)
        return pages, {
            "gemma_extraction_mode": "text",
            "plain_extracted_chars": plain_chars,
        }

    return extract_pdf_layout_pages(pdf_path)


def split_page_text_for_claude(page_number: int, text: str, max_chars: int) -> list[str]:
    prefix = f"[PAGE {page_number}]\n"
    max_body_chars = max(1000, max_chars - len(prefix) - 1)
    parts: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_body_chars, len(text))
        if end < len(text):
            newline = text.rfind("\n", start + max_body_chars // 2, end)
            if newline > start:
                end = newline + 1

        body = text[start:end].strip()
        if body:
            parts.append(prefix + body)
        start = max(end, start + 1)

    return parts


def chunk_pdf_text_pages(
    pages: list[tuple[int, str]],
    max_chars: int,
) -> list[dict[str, Any]]:
    max_chars = max(10000, int(max_chars))
    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_chars = 0
    current_start_page: int | None = None
    current_end_page: int | None = None

    def flush() -> None:
        nonlocal current_parts, current_chars, current_start_page, current_end_page
        if not current_parts:
            return
        chunks.append({
            "index": len(chunks) + 1,
            "start_page": current_start_page,
            "end_page": current_end_page,
            "text": "\n\n".join(current_parts),
        })
        current_parts = []
        current_chars = 0
        current_start_page = None
        current_end_page = None

    for page_number, page_text in pages:
        for part in split_page_text_for_claude(page_number, page_text, max_chars):
            part_len = len(part) + 2
            if current_parts and current_chars + part_len > max_chars:
                flush()

            if current_start_page is None:
                current_start_page = page_number
            current_end_page = page_number
            current_parts.append(part)
            current_chars += part_len

    flush()
    return chunks


def parsed_pdf_text_for_file(
    pdf_path: Path,
    pages: list[tuple[int, str]],
    extraction_metadata: dict[str, Any],
    source_pdf_path: Path | None = None,
) -> str:
    source_path = source_pdf_path or pdf_path
    lines = [
        "# Parsed PDF Text",
        f"source_pdf: {source_path}",
        f"processed_pdf: {pdf_path}",
        f"extraction_metadata: {json.dumps(extraction_metadata, ensure_ascii=False, sort_keys=True)}",
        "",
    ]
    for page_number, text in pages:
        lines.append(f"[PAGE {page_number}]")
        lines.append(str(text or ""))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_pdf_text(
    pdf_path: Path,
    args: argparse.Namespace,
    pages: list[tuple[int, str]],
    extraction_metadata: dict[str, Any],
) -> Path | None:
    if not getattr(args, "write_parsed_pdf", DEFAULT_WRITE_PARSED_PDF):
        return None

    source_pdf_text = str(getattr(args, "_ai_current_input_pdf_path", "") or "").strip()
    source_pdf_path = Path(source_pdf_text) if source_pdf_text else pdf_path
    run_timestamp = clean_text(getattr(args, "_ai_run_timestamp", "")) or timestamp_for_filename()
    provider_part = safe_filename_part(clean_text(getattr(args, "provider", "")) or "provider")
    pdf_part = safe_filename_part(source_pdf_path.stem or pdf_path.stem)
    parsed_dir = Path(args.output_dir) / "parsed_pdfs"
    parsed_file = unique_output_path(parsed_dir / f"{pdf_part}_{provider_part}_{run_timestamp}_parsed_pdf.txt")
    parsed_file.parent.mkdir(parents=True, exist_ok=True)
    parsed_file.write_text(
        parsed_pdf_text_for_file(
            pdf_path=pdf_path,
            pages=pages,
            extraction_metadata=extraction_metadata,
            source_pdf_path=source_pdf_path,
        ),
        encoding="utf-8",
    )
    update_processing_metadata(args, parsed_pdf_file=str(parsed_file))
    print(f"  Parsed PDF text saved: {parsed_file}", flush=True)
    return parsed_file


def write_input_chunk(
    *,
    provider: str,
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    chunk: dict[str, Any],
    total_chunks: int,
) -> Path | None:
    if not getattr(args, "write_input_chunks", DEFAULT_WRITE_INPUT_CHUNKS):
        return None

    source_pdf_text = str(getattr(args, "_ai_current_input_pdf_path", "") or "").strip()
    source_pdf_path = Path(source_pdf_text) if source_pdf_text else pdf_path
    run_timestamp = clean_text(getattr(args, "_ai_run_timestamp", "")) or timestamp_for_filename()
    provider_part = safe_filename_part(provider or clean_text(getattr(args, "provider", "")) or "provider")
    pdf_part = safe_filename_part(source_pdf_path.stem or pdf_path.stem)
    chunk_part = safe_filename_part(str(chunk.get("index", "chunk")))
    start_page = chunk.get("start_page")
    end_page = chunk.get("end_page")
    chunk_text = str(chunk.get("text") or "")

    chunk_dir = Path(args.output_dir) / "input_chunks" / f"{pdf_part}_{provider_part}_{run_timestamp}"
    chunk_file = unique_output_path(chunk_dir / f"chunk_{chunk_part}_pages_{start_page}-{end_page}.json")
    chunk_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp": timestamp_for_metadata(),
        "stage": "before_gemma_toc_generation",
        "provider": provider,
        "model": model,
        "source_pdf": str(source_pdf_path),
        "processed_pdf": str(pdf_path),
        "chunk": chunk.get("index"),
        "total_chunks": total_chunks,
        "start_page": start_page,
        "end_page": end_page,
        "input_chars": len(chunk_text),
        "text": chunk_text,
    }

    save_json(chunk_file, payload)

    input_chunk_files = getattr(args, "_ai_input_chunk_files", None)
    if not isinstance(input_chunk_files, list):
        input_chunk_files = []
    input_chunk_files.append(str(chunk_file))
    setattr(args, "_ai_input_chunk_files", input_chunk_files)
    update_processing_metadata(
        args,
        input_chunk_dir=str(chunk_file.parent),
        input_chunk_files=input_chunk_files,
    )
    return chunk_file


def write_stage_file(
    *,
    args: argparse.Namespace,
    pdf_path: Path,
    provider: str,
    model: str,
    stage: str,
    payload: dict[str, Any],
    chunk: Any = None,
) -> Path | None:
    if not getattr(args, "write_stage_files", DEFAULT_WRITE_STAGE_FILES):
        return None

    source_pdf_text = str(getattr(args, "_ai_current_input_pdf_path", "") or "").strip()
    source_pdf_path = Path(source_pdf_text) if source_pdf_text else pdf_path
    run_timestamp = clean_text(getattr(args, "_ai_run_timestamp", "")) or timestamp_for_filename()
    provider_part = safe_filename_part(provider or clean_text(getattr(args, "provider", "")) or "provider")
    pdf_part = safe_filename_part(source_pdf_path.stem or pdf_path.stem)
    stage_part = safe_filename_part(stage)
    chunk_part = f"_chunk_{safe_filename_part(str(chunk))}" if chunk is not None else ""

    stage_index = int(getattr(args, "_ai_stage_index", 0) or 0) + 1
    setattr(args, "_ai_stage_index", stage_index)

    stage_dir = Path(args.output_dir) / "stages" / f"{pdf_part}_{provider_part}_{run_timestamp}"
    stage_file = unique_output_path(stage_dir / f"{stage_index:03d}_{stage_part}{chunk_part}.json")
    stage_payload: dict[str, Any] = {
        "timestamp": timestamp_for_metadata(),
        "stage_index": stage_index,
        "stage": stage,
        "provider": provider,
        "model": model,
        "source_pdf": str(source_pdf_path),
        "processed_pdf": str(pdf_path),
    }
    if chunk is not None:
        stage_payload["chunk"] = chunk
    stage_payload.update({str(key): jsonable_debug_value(value) for key, value in payload.items()})
    save_json(stage_file, stage_payload)

    stage_files = getattr(args, "_ai_stage_files", None)
    if not isinstance(stage_files, list):
        stage_files = []
    stage_files.append(str(stage_file))
    setattr(args, "_ai_stage_files", stage_files)
    update_processing_metadata(
        args,
        stage_dir=str(stage_file.parent),
        stage_files=stage_files,
    )
    return stage_file


def build_claude_text_prompt(base_prompt: str, pdf_name: str, text: str) -> str:
    return f"""
{base_prompt}

[Claude Long PDF Text Mode]
Create the TOC using only the [Extracted PDF Text] below instead of an attached PDF.
Use each text block's [PAGE n] marker to determine page numbers.

[PDF File Name]
{pdf_name}

[Extracted PDF Text]
{text}
""".strip()


def build_claude_chunk_prompt(base_prompt: str, pdf_name: str, chunk: dict[str, Any], total_chunks: int) -> str:
    return f"""
{base_prompt}

[Claude Long PDF Chunk Mode]
The text below is one part of the full PDF.
Include only chapter, section, and subsection titles that are actually visible within this range.
Do not infer TOC entries outside this range.
Use [PAGE n] markers to determine page numbers.

[PDF File Name]
{pdf_name}

[Chunk]
{chunk["index"]}/{total_chunks}, pages {chunk["start_page"]}-{chunk["end_page"]}

[Extracted PDF Text]
{chunk["text"]}
""".strip()


def build_claude_merge_prompt(
    base_prompt: str,
    pdf_name: str,
    partial_tocs: list[dict[str, Any]],
    max_depth: int,
) -> str:
    partial_json = json.dumps(partial_tocs, ensure_ascii=False, indent=2)
    return f"""
{base_prompt}

[Chunked TOC Merge Instruction]
Create one final TOC JSON object using only the [Partial TOC Candidates] below instead of the attached PDF.
- Keep only one copy of duplicate entries.
- Preserve ascending page order and source appearance order.
- Remove covers, prefaces, existing TOC pages, indexes, references, and repeated headers/footers.
- Use only levels from 1 to {max_depth}.
- Do not invent chapter titles.
- Preserve original chapter/section titles exactly, including Korean text when the source title is Korean.

[PDF File Name]
{pdf_name}

[Partial TOC Candidates]
{partial_json}
""".strip()


def build_claude_json_retry_prompt(base_prompt: str, error: Exception, raw_text: str) -> str:
    raw_preview = clean_text(raw_text)[:500]
    return (
        base_prompt
        + "\n\n[Important Retry Instruction]\n"
        + f"The previous response failed JSON parsing. Error: {type(error).__name__}: {error}\n"
        + ("Previous response preview: " + raw_preview + "\n" if raw_preview else "The previous response was empty.\n")
        + "This time, output exactly one valid JSON object that Python json.loads() can parse directly. "
        + "Do not output Markdown, explanations, or code fences. Never return an empty response. "
        + 'If there are no entries, output {"title": "Document title", "chapters": []}.'
    )


def request_claude_json_text(
    client,
    model: str,
    prompt_text: str,
    args: argparse.Namespace,
    label: str,
) -> tuple[dict[str, Any], str]:
    retry_count = max(1, int(args.ai_retries))
    last_error: Exception | None = None
    last_raw_text = ""

    for parse_attempt in range(1, retry_count + 1):
        request_prompt = prompt_text
        if parse_attempt > 1 and last_error is not None:
            print(
                f"  Retrying {label} after JSON parse failure {parse_attempt}/{retry_count}: {last_error}",
                file=sys.stderr,
                flush=True,
            )
            request_prompt = build_claude_json_retry_prompt(prompt_text, last_error, last_raw_text)

        response = with_retry(
            label=f"{label} generation",
            func=lambda request_prompt=request_prompt: generate_claude_text_once(
                client=client,
                model=model,
                prompt_text=request_prompt,
                max_output_tokens=args.max_output_tokens,
            ),
            max_retries=args.ai_retries,
            base_delay=args.ai_retry_base_delay,
        )
        raw_text = claude_response_text(response)

        try:
            return parse_json_response_text(raw_text, provider="claude"), raw_text
        except Exception as error:
            last_error = error
            last_raw_text = raw_text
            if parse_attempt >= retry_count:
                raise

    if last_error:
        raise last_error
    raise RuntimeError(f"{label} failed")


def page_range_from_chunk_text(text: str, fallback_start: Any, fallback_end: Any) -> tuple[Any, Any]:
    pages = [int(match.group(1)) for match in re.finditer(r"\[PAGE\s+(\d+)\]", str(text))]
    if pages:
        return min(pages), max(pages)
    return fallback_start, fallback_end


def split_large_claude_segment(segment: str, max_chars: int) -> list[str]:
    segment = segment.strip()
    if len(segment) <= max_chars:
        return [segment] if segment else []

    page_match = re.match(r"(\[PAGE\s+\d+\]\n?)", segment)
    prefix = page_match.group(1) if page_match else ""
    body = segment[len(prefix):] if prefix else segment
    max_body_chars = max(1000, max_chars - len(prefix) - 1)

    parts: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + max_body_chars, len(body))
        if end < len(body):
            newline = body.rfind("\n", start + max_body_chars // 2, end)
            if newline > start:
                end = newline + 1
        part = body[start:end].strip()
        if part:
            parts.append((prefix + part).strip())
        start = max(end, start + 1)

    return parts


def split_claude_chunk_text(chunk: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    text = str(chunk.get("text") or "")
    max_chars = max(5000, int(max_chars))
    starts = [match.start() for match in re.finditer(r"(?m)^\[PAGE\s+\d+\]", text)]

    if starts:
        starts.append(len(text))
        segments: list[str] = []
        for index in range(len(starts) - 1):
            segment = text[starts[index]:starts[index + 1]].strip()
            segments.extend(split_large_claude_segment(segment, max_chars))
    else:
        segments = split_large_claude_segment(text, max_chars)

    subchunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current_parts, current_chars
        if not current_parts:
            return
        sub_text = "\n\n".join(current_parts)
        start_page, end_page = page_range_from_chunk_text(
            sub_text,
            chunk.get("start_page"),
            chunk.get("end_page"),
        )
        subchunks.append({
            "index": f"{chunk.get('index')}.{len(subchunks) + 1}",
            "start_page": start_page,
            "end_page": end_page,
            "text": sub_text,
        })
        current_parts = []
        current_chars = 0

    for segment in segments:
        segment_len = len(segment) + 2
        if current_parts and current_chars + segment_len > max_chars:
            flush()
        current_parts.append(segment)
        current_chars += segment_len

    flush()
    return subchunks


def process_claude_text_chunk(
    client,
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    prompt: str,
    chunk: dict[str, Any],
    total_chunks: int,
    raw_parts: list[dict[str, Any]],
    depth: int = 0,
) -> list[dict[str, Any]]:
    chunk_label = str(chunk["index"])
    chunk_prompt = build_claude_chunk_prompt(prompt, pdf_path.name, chunk, total_chunks)
    print(
        f"  Claude chunk {chunk_label}/{total_chunks} request: pages {chunk['start_page']}-{chunk['end_page']}",
        flush=True,
    )

    try:
        parsed, raw_text = request_claude_json_text(
            client=client,
            model=model,
            prompt_text=chunk_prompt,
            args=args,
            label=f"Claude chunk {chunk_label}/{total_chunks}",
        )
        partial_toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        raw_parts.append({
            "chunk": chunk["index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "raw_response": raw_text,
        })
        print(f"  Claude chunk {chunk_label} completed: {len(partial_toc.get('chapters', []))} chapters", flush=True)
        return [{
            "chunk": chunk["index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "toc": partial_toc,
        }]
    except Exception as error:
        min_chunk_chars = max(5000, int(getattr(args, "claude_text_min_chunk_chars", DEFAULT_CLAUDE_TEXT_MIN_CHUNK_CHARS)))
        if len(str(chunk.get("text") or "")) <= min_chunk_chars or depth >= 4:
            raw_parts.append({
                "chunk": chunk["index"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "error": message_of(error),
            })
            raise

        subchunk_chars = max(min_chunk_chars, len(str(chunk.get("text") or "")) // 2)
        subchunks = split_claude_chunk_text(chunk, max_chars=subchunk_chars)
        if len(subchunks) <= 1:
            raise

        print(
            f"  Claude chunk {chunk_label} failed; retrying with {len(subchunks)} subchunks: {error}",
            file=sys.stderr,
            flush=True,
        )
        partials: list[dict[str, Any]] = []
        for subchunk in subchunks:
            partials.extend(
                process_claude_text_chunk(
                    client=client,
                    model=model,
                    pdf_path=pdf_path,
                    args=args,
                    prompt=prompt,
                    chunk=subchunk,
                    total_chunks=total_chunks,
                    raw_parts=raw_parts,
                    depth=depth + 1,
                )
            )
        return partials

def int_sort_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def float_sort_value(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def first_float_sort_value(item: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = float_sort_value(item.get(key))
        if value is not None:
            return value
    return None


def local_merge_sort_key(item: dict[str, Any]) -> tuple[int, int, float, float, int, int]:
    page = int_sort_value(item.get("page"), default=1)
    level = int_sort_value(item.get("level"), default=1)
    y = first_float_sort_value(item, ("_layout_y", "layout_y", "vertical_position"))
    x = first_float_sort_value(item, ("_layout_x", "layout_x", "left_indent"))
    source_order = metadata_int_value(item.get("_source_order"))
    local_merge_order = metadata_int_value(item.get("_local_merge_order"))

    if source_order is not None:
        return (
            page,
            0,
            float(source_order),
            y if y is not None else 999999.0,
            x if x is not None else 999999.0,
            level,
        )

    if local_merge_order is not None:
        return (
            page,
            1,
            float(local_merge_order),
            y if y is not None else 999999.0,
            x if x is not None else 999999.0,
            level,
        )

    if y is None:
        return (page, 3, 999999.0, x if x is not None else 999999.0, 999999.0, level)

    return (page, 2, y, x if x is not None else 999999.0, 999999.0, level)


def merge_partial_tocs_locally(partial_tocs: list[dict[str, Any]], fallback_title: str, max_depth: int) -> dict[str, Any]:
    title = fallback_title
    chapters: list[dict[str, Any]] = []

    for partial in partial_tocs:
        toc = partial.get("toc")
        if not isinstance(toc, dict):
            continue
        if title == fallback_title and clean_text(toc.get("title")):
            title = clean_text(toc.get("title"))
        for chapter in toc.get("chapters", []) or []:
            if isinstance(chapter, dict):
                chapter_item = dict(chapter)
                chapter_item.setdefault("_local_merge_order", len(chapters))
                chapters.append(chapter_item)

    chapters.sort(key=local_merge_sort_key)
    return validate_toc({"title": title, "chapters": chapters}, fallback_title=fallback_title, max_depth=max_depth)


def generate_toc_from_pdf_claude_text(
    client,
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    prompt: str,
) -> tuple[dict[str, Any], str, str]:
    print("  Claude PDF text extraction started...", flush=True)
    pages = extract_pdf_text_pages(pdf_path)
    extracted_chars = sum(len(text) for _, text in pages)
    print(f"  Claude PDF text extraction completed: {len(pages)} pages, {extracted_chars} chars", flush=True)

    single_max_chars = max(10000, int(args.claude_text_single_max_chars))
    chunk_chars = max(10000, int(args.claude_text_chunk_chars))

    if extracted_chars <= single_max_chars:
        full_text = "\n\n".join(f"[PAGE {page_number}]\n{text}" for page_number, text in pages)
        text_prompt = build_claude_text_prompt(prompt, pdf_path.name, full_text)
        print(f"  Claude text TOC generation started: {model}", flush=True)
        parsed, raw_text = request_claude_json_text(
            client=client,
            model=model,
            prompt_text=text_prompt,
            args=args,
            label=f"Claude text({model})",
        )
        print(f"  Claude text TOC generation completed: {model}", flush=True)
        return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model

    chunks = chunk_pdf_text_pages(pages, max_chars=chunk_chars)
    print(f"  Claude text chunk processing started: {len(chunks)} chunks", flush=True)
    partial_tocs: list[dict[str, Any]] = []
    raw_parts: list[dict[str, Any]] = []

    for chunk in chunks:
        partial_tocs.extend(
            process_claude_text_chunk(
                client=client,
                model=model,
                pdf_path=pdf_path,
                args=args,
                prompt=prompt,
                chunk=chunk,
                total_chunks=len(chunks),
                raw_parts=raw_parts,
            )
        )

    merge_prompt = build_claude_merge_prompt(prompt, pdf_path.name, partial_tocs, args.max_depth)
    print(f"  Claude chunked TOC merge started: {model}", flush=True)
    try:
        parsed, final_raw_text = request_claude_json_text(
            client=client,
            model=model,
            prompt_text=merge_prompt,
            args=args,
            label=f"Claude chunked TOC merge({model})",
        )
        toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
    except Exception as merge_error:
        print(
            f"Claude chunked TOC merge failed; falling back to local merge: {merge_error}",
            file=sys.stderr,
            flush=True,
        )
        toc = merge_partial_tocs_locally(partial_tocs, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        final_raw_text = json.dumps(
            {
                "mode": "local_merge_after_claude_merge_failure",
                "error": message_of(merge_error),
                "title": toc.get("title"),
                "chapters": len(toc.get("chapters", [])),
            },
            ensure_ascii=False,
            indent=2,
        )

    raw_bundle = {
        "mode": "claude_text_chunk_fallback",
        "model": model,
        "pdf": pdf_path.name,
        "extracted_pages": len(pages),
        "extracted_chars": extracted_chars,
        "chunk_count": len(chunks),
        "chunk_raw_responses": raw_parts,
        "final_raw_response": final_raw_text,
    }
    return toc, json.dumps(raw_bundle, ensure_ascii=False, indent=2), model


def generate_toc_from_pdf_claude(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    client = create_claude_client(api_key=get_api_key("claude"))
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    for index, model in enumerate(models):
        try:
            if index > 0:
                print(f"Trying Claude fallback model: {model}", file=sys.stderr, flush=True)

            if args.claude_force_text:
                print(f"  Using Claude text extraction mode: {model}", flush=True)
                return generate_toc_from_pdf_claude_text(
                    client=client,
                    model=model,
                    pdf_path=pdf_path,
                    args=args,
                    prompt=prompt,
                )

            print(f"  Claude TOC generation started: {model}", flush=True)
            response = with_retry(
                label=f"Claude generation({model})",
                func=lambda model=model: generate_claude_once(
                    client=client,
                    model=model,
                    pdf_path=pdf_path,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                ),
                max_retries=args.ai_retries,
                base_delay=args.ai_retry_base_delay,
            )
            print(f"  Claude TOC generation completed: {model}", flush=True)

            raw_text = claude_response_text(response)
            parsed = parse_json_response(response, provider="claude")
            return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model

        except Exception as error:
            last_error = error
            if is_configuration_error(error):
                raise
            if is_prompt_too_long_error(error) or is_stream_required_error(error):
                print(
                    f"Claude long-input limit reached; switching to text extraction fallback: {model}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    return generate_toc_from_pdf_claude_text(
                        client=client,
                        model=model,
                        pdf_path=pdf_path,
                        args=args,
                        prompt=prompt,
                    )
                except Exception as fallback_error:
                    last_error = fallback_error
                    if is_configuration_error(fallback_error):
                        raise
                    print(f"Claude text fallback failed: {model} / {fallback_error}", file=sys.stderr, flush=True)
                    continue
            print(f"Claude model failed: {model} / {error}", file=sys.stderr, flush=True)

    if last_error:
        raise last_error
    raise RuntimeError("No Claude models to try.")


# ----------------------------- Gemma -----------------------------

_GEMMA_TRANSFORMERS_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_GEMMA_VLLM_CACHE: dict[tuple[str, str, int, float, str, bool], dict[str, Any]] = {}
GEMMA_RUNTIMES = {"transformers", "vllm", "vllm-server"}

GEMMA_SYSTEM_PROMPT = (
    "You are an expert at creating tables of contents for PDF textbooks and lecture materials. "
    "Output exactly one valid JSON object. Do not output Markdown or explanations."
)


def normalize_gemma_runtime(runtime: str | None) -> str:
    runtime = clean_text(runtime).lower() or DEFAULT_GEMMA_RUNTIME
    aliases = {
        "hf": "transformers",
        "huggingface": "transformers",
        "hugging-face": "transformers",
        "transformer": "transformers",
        "vllm-offline": "vllm",
        "vllm_offline": "vllm",
        "server": "vllm-server",
        "vllm_server": "vllm-server",
        "vllm-openai": "vllm-server",
        "vllm_openai": "vllm-server",
        "openai-compatible": "vllm-server",
        "openai_compatible": "vllm-server",
    }
    runtime = aliases.get(runtime, runtime)
    if runtime not in GEMMA_RUNTIMES:
        raise ValueError("--gemma-runtime must be transformers, vllm, or vllm-server.")
    return runtime


def normalize_gemma_merge_mode(mode: str | None) -> str:
    mode = clean_text(mode).lower() or DEFAULT_GEMMA_MERGE_MODE
    aliases = {
        "code": "local",
        "python": "local",
        "offline": "local",
        "llm": "gemma",
        "ai": "gemma",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"local", "gemma"}:
        raise ValueError("--gemma-merge-mode must be either local or gemma.")
    return mode


def normalize_gemma_merge_strategy(strategy: str | None) -> str:
    strategy = clean_text(strategy).lower() or DEFAULT_GEMMA_MERGE_STRATEGY
    aliases = {
        "batch": "batched",
        "batches": "batched",
        "chunked": "batched",
        "safe": "batched",
        "one": "single",
        "once": "single",
        "legacy": "single",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in {"single", "batched"}:
        raise ValueError("--gemma-merge-strategy must be either single or batched.")
    return strategy


def normalize_gemma_level_verify_scope(scope: str | None) -> str:
    scope = clean_text(scope).lower() or DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE
    aliases = {
        "0": "none",
        "false": "none",
        "no": "none",
        "off": "none",
        "disable": "none",
        "disabled": "none",
        "skip": "none",
        "1": "final",
        "true": "final",
        "yes": "final",
        "on": "final",
        "merge": "final",
        "merged": "final",
        "after-merge": "final",
        "after_merge": "final",
        "final-only": "final",
        "final_only": "final",
        "chunks": "chunk",
        "chunked": "chunk",
        "per-chunk": "chunk",
        "per_chunk": "chunk",
        "initial": "chunk",
        "all": "both",
        "everywhere": "both",
        "chunk-final": "both",
        "chunk_final": "both",
    }
    scope = aliases.get(scope, scope)
    if scope not in {"none", "chunk", "final", "both"}:
        raise ValueError("--gemma-level-verify-scope must be one of: none, chunk, final, both.")
    return scope


def gemma_level_verify_enabled(args: argparse.Namespace, scope: str) -> bool:
    verify_scope = normalize_gemma_level_verify_scope(
        getattr(args, "gemma_level_verify_scope", DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE)
    )
    return verify_scope == "both" or verify_scope == scope


def normalize_gemma_extraction_mode(mode: str | None) -> str:
    mode = clean_text(mode).lower() or DEFAULT_GEMMA_EXTRACTION_MODE
    aliases = {
        "plain": "text",
        "simple": "text",
        "pdfplumber": "layout",
        "layout-text": "layout",
        "layout_text": "layout",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"layout", "text"}:
        raise ValueError("--gemma-extraction-mode must be either layout or text.")
    return mode


def normalize_gemma_hf_model(model: str) -> str:
    model = clean_text(model)
    if model.startswith("models/"):
        model = model.split("/", 1)[1]

    lower = model.lower()
    aliases = {
        "latest": DEFAULT_GEMMA_HF_MODEL,
        "highest": DEFAULT_GEMMA_HF_MODEL,
        "max": DEFAULT_GEMMA_HF_MODEL,
        "31b": DEFAULT_GEMMA_HF_MODEL,
        "gemma4:31b": DEFAULT_GEMMA_HF_MODEL,
        "gemma-4-31b-it": DEFAULT_GEMMA_HF_MODEL,
        "12b": "google/gemma-4-12B-it",
        "gemma4:12b": "google/gemma-4-12B-it",
        "gemma-4-12b-it": "google/gemma-4-12B-it",
        "26b": "google/gemma-4-26B-A4B-it",
        "gemma4:26b": "google/gemma-4-26B-A4B-it",
        "gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B-it",
        "e4b": "google/gemma-4-E4B-it",
        "gemma4:e4b": "google/gemma-4-E4B-it",
        "gemma-4-e4b-it": "google/gemma-4-E4B-it",
        "e2b": "google/gemma-4-E2B-it",
        "gemma4:e2b": "google/gemma-4-E2B-it",
        "gemma-4-e2b-it": "google/gemma-4-E2B-it",
        "27b": "google/gemma-3-27b-it",
        "gemma3:27b": "google/gemma-3-27b-it",
        "gemma-3-27b-it": "google/gemma-3-27b-it",
    }
    if lower in aliases:
        return aliases[lower]

    if lower.startswith("google/"):
        return model

    if lower.startswith("gemma-"):
        return f"google/{model}"

    return model


def preflight_pdf_text_extraction() -> None:
    try:
        __import__("pdfplumber")
    except ImportError as error:
        raise RuntimeError("PDF text extraction requires `pip install pdfplumber`.") from error


def import_gemma_transformers_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    for module_name in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        raise RuntimeError(
            "Missing packages required by the Gemma Transformers runtime: "
            + ", ".join(missing)
            + ". Run `pip install -U transformers torch accelerate safetensors huggingface_hub` first."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Gemma Transformers runtime could not import the required model classes. "
            "Run `pip install -U transformers torch accelerate safetensors huggingface_hub` first."
        ) from error

    return torch, AutoModelForCausalLM, AutoTokenizer


def import_gemma_vllm_dependencies() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "Missing packages required by the Gemma vLLM runtime. "
            "Run `pip install -U vllm` first. vLLM also requires a compatible GPU/CUDA environment."
        ) from error

    return LLM, SamplingParams


def gemma_model_needs_gemma4_config(model: str) -> bool:
    model = clean_text(model).lower()
    return "gemma-4" in model or "gemma4" in model


def preflight_gemma_transformers(models: Iterable[str] | None = None) -> None:
    import_gemma_transformers_dependencies()

    if not models or not any(gemma_model_needs_gemma4_config(model) for model in models):
        return

    try:
        from transformers import Gemma4Config
        assert Gemma4Config is not None
    except Exception as error:
        error_text = message_of(error)
        if "torchvision::nms" in error_text or "torchvision" in error_text:
            raise RuntimeError(
                "Gemma 4 config import failed because torchvision is installed but incompatible "
                "with the current torch build. This script does not need torchvision for text-only "
                "Gemma inference. Run `python -m pip uninstall -y torchvision`, or reinstall a "
                "torchvision wheel that matches your torch/CUDA build."
            ) from error

        raise RuntimeError(
            "Your installed transformers package does not support Gemma 4 yet "
            "(Gemma4Config is unavailable or broken). "
            "Install the latest Transformers source build with: "
            "`python -m pip install --upgrade --force-reinstall "
            "git+https://github.com/huggingface/transformers.git`"
        ) from error


def preflight_gemma_vllm() -> None:
    import_gemma_vllm_dependencies()


def preflight_gemma_vllm_server(args: argparse.Namespace) -> None:
    base_url = clean_text(getattr(args, "gemma_vllm_server_base_url", DEFAULT_GEMMA_VLLM_SERVER_BASE_URL))
    if not base_url:
        raise RuntimeError("--gemma-vllm-server-base-url is required for --gemma-runtime vllm-server.")


def preflight_gemma_provider(args: argparse.Namespace) -> None:
    provider_args = build_provider_args(args, "gemma")
    preflight_pdf_text_extraction()
    models = [
        normalize_gemma_hf_model(model)
        for model in parse_model_list(provider_args.model, provider_args.ai_fallback_models)
    ]
    runtime = normalize_gemma_runtime(getattr(provider_args, "gemma_runtime", DEFAULT_GEMMA_RUNTIME))
    if runtime == "vllm":
        preflight_gemma_vllm()
    elif runtime == "vllm-server":
        preflight_gemma_vllm_server(provider_args)
    else:
        preflight_gemma_transformers(models=models)


def gemma_hf_token_kwargs() -> dict[str, str]:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    return {"token": token} if token else {}


def torch_dtype_from_text(torch_module: Any, dtype_text: str) -> Any:
    dtype_text = clean_text(dtype_text).lower()
    if not dtype_text:
        return None
    if dtype_text == "auto":
        return "auto"

    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    torch_dtype_name = aliases.get(dtype_text, dtype_text)
    return getattr(torch_module, torch_dtype_name, dtype_text)


def first_transformers_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            parameter_device = getattr(parameter, "device", None)
            if parameter_device is not None and str(parameter_device) != "meta":
                return parameter_device

    return None


def normalize_device_text(device: Any) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"

    text = clean_text(device)
    if text.isdigit():
        return f"cuda:{text}"
    return text


def device_text_is_cuda(device: Any) -> bool:
    text = normalize_device_text(device).lower()
    return text == "cuda" or text.startswith("cuda:")


def transformers_model_device_texts(model: Any, primary_device: Any) -> list[str]:
    devices: list[str] = []

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            device_text = normalize_device_text(device)
            if device_text and device_text not in devices:
                devices.append(device_text)

    primary_device_text = normalize_device_text(primary_device)
    if primary_device_text and primary_device_text != "meta" and primary_device_text not in devices:
        devices.append(primary_device_text)

    return devices


def cuda_device_names(torch_module: Any) -> list[str]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return []

    try:
        device_count = int(cuda.device_count())
    except Exception:
        return []

    names: list[str] = []
    for index in range(max(0, device_count)):
        try:
            names.append(str(cuda.get_device_name(index)))
        except Exception:
            names.append(f"cuda:{index}")
    return names


def gemma_transformers_runtime_metadata(
    torch_module: Any,
    loaded_model: Any,
    model: str,
    device_map: str,
    dtype_text: str,
) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    try:
        cuda_available = bool(cuda.is_available()) if cuda is not None else False
    except Exception:
        cuda_available = False

    try:
        cuda_device_count = int(cuda.device_count()) if cuda is not None else 0
    except Exception:
        cuda_device_count = 0

    primary_device = first_transformers_device(loaded_model)
    model_devices = transformers_model_device_texts(loaded_model, primary_device)
    gpu_used = any(device_text_is_cuda(device) for device in model_devices)

    return {
        "gemma_model": model,
        "gemma_device_map": device_map,
        "gemma_torch_dtype": dtype_text,
        "gpu_available": cuda_available,
        "gpu_used": gpu_used,
        "gpu_device_count": cuda_device_count,
        "gpu_devices": cuda_device_names(torch_module),
        "model_primary_device": normalize_device_text(primary_device) or None,
        "model_devices": model_devices,
        "hf_device_map_present": isinstance(getattr(loaded_model, "hf_device_map", None), dict),
    }


def print_gemma_runtime_metadata(metadata: dict[str, Any]) -> None:
    devices = metadata.get("model_devices") or []
    device_text = ", ".join(str(device) for device in devices) if devices else "unknown"
    print(
        f"  Gemma/Transformers device check: gpu_used={metadata.get('gpu_used')} / devices={device_text}",
        flush=True,
    )
    if not metadata.get("gpu_available"):
        print("  Warning: torch cannot see CUDA, so Gemma/Transformers will not use GPU.", file=sys.stderr, flush=True)
    elif not metadata.get("gpu_used"):
        print(
            "  Warning: CUDA is available, but the loaded Gemma model is not on a CUDA device.",
            file=sys.stderr,
            flush=True,
        )


def move_transformers_inputs_to_device(inputs: Any, device: Any) -> Any:
    if device is None:
        return inputs

    to_method = getattr(inputs, "to", None)
    if callable(to_method):
        return to_method(device)

    if isinstance(inputs, dict):
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            value_to = getattr(value, "to", None)
            moved[key] = value_to(device) if callable(value_to) else value
        return moved

    return inputs


def build_gemma_chat_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_gemma_vllm_prompt(tokenizer: Any, prompt: str) -> str:
    messages = build_gemma_chat_messages(prompt)

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered = apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass

    return f"System: {GEMMA_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"


def load_gemma_transformers_model(model: str, args: argparse.Namespace) -> dict[str, Any]:
    device_map = clean_text(getattr(args, "gemma_device_map", DEFAULT_GEMMA_DEVICE_MAP)) or "auto"
    dtype_text = clean_text(getattr(args, "gemma_torch_dtype", DEFAULT_GEMMA_TORCH_DTYPE)) or "auto"
    cache_key = (model, device_map, dtype_text)

    if cache_key in _GEMMA_TRANSFORMERS_CACHE:
        return _GEMMA_TRANSFORMERS_CACHE[cache_key]

    torch, AutoModelForCausalLM, AutoTokenizer = import_gemma_transformers_dependencies()

    token_kwargs = gemma_hf_token_kwargs()
    print(f"  Gemma/Transformers model loading started: {model}", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, **token_kwargs)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load Gemma tokenizer: {model}. "
            "Check Hugging Face model access and HF_TOKEN/huggingface-cli login state. "
            f"Cause: {error}"
        ) from error

    load_kwargs: dict[str, Any] = dict(token_kwargs)
    if device_map:
        load_kwargs["device_map"] = device_map

    dtype_value = torch_dtype_from_text(torch, dtype_text)
    if dtype_value is not None:
        load_kwargs["dtype"] = dtype_value

    try:
        loaded_model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
        fallback_kwargs = dict(load_kwargs)
        dtype_value = fallback_kwargs.pop("dtype", None)
        if dtype_value is not None:
            fallback_kwargs["torch_dtype"] = dtype_value
        loaded_model = AutoModelForCausalLM.from_pretrained(model, **fallback_kwargs)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load Gemma transformers model: {model}. "
            "Check the transformers version, Hugging Face access, and GPU/CPU memory. "
            f"Cause: {error}"
        ) from error

    eval_method = getattr(loaded_model, "eval", None)
    if callable(eval_method):
        eval_method()

    runtime_metadata = gemma_transformers_runtime_metadata(
        torch_module=torch,
        loaded_model=loaded_model,
        model=model,
        device_map=device_map,
        dtype_text=dtype_text,
    )

    bundle = {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": loaded_model,
        "runtime_metadata": runtime_metadata,
    }
    _GEMMA_TRANSFORMERS_CACHE[cache_key] = bundle
    print(f"  Gemma/Transformers model loading completed: {model}", flush=True)
    print_gemma_runtime_metadata(runtime_metadata)
    return bundle


def generate_gemma_once_transformers(
    model: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bundle = load_gemma_transformers_model(model=model, args=args)
    torch = bundle["torch"]
    tokenizer = bundle["tokenizer"]
    loaded_model = bundle["model"]

    messages = build_gemma_chat_messages(prompt)

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as error:
        raise RuntimeError(f"Failed to build Gemma transformers inputs: {error}") from error

    device = first_transformers_device(loaded_model)
    inputs = move_transformers_inputs_to_device(inputs, device)

    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    input_length = int(input_ids.shape[-1]) if input_ids is not None else 0

    max_new_tokens = int(args.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER["gemma"])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, max_new_tokens),
    }

    temperature = float(args.temperature)
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    try:
        with torch.inference_mode():
            outputs = loaded_model.generate(**inputs, **generation_kwargs)
    except Exception as error:
        raise RuntimeError(f"Gemma transformers generation failed: {error}") from error

    generated_ids = outputs[0][input_length:] if input_length else outputs[0]
    try:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(generated_ids)

    return {"response": text, "_runtime": bundle.get("runtime_metadata")}


def load_gemma_vllm_model(model: str, args: argparse.Namespace) -> dict[str, Any]:
    dtype_text = clean_text(getattr(args, "gemma_torch_dtype", DEFAULT_GEMMA_TORCH_DTYPE)) or "auto"
    tensor_parallel_size = max(
        1,
        int(getattr(args, "gemma_vllm_tensor_parallel_size", DEFAULT_GEMMA_VLLM_TENSOR_PARALLEL_SIZE)),
    )
    gpu_memory_utilization = float(
        getattr(args, "gemma_vllm_gpu_memory_utilization", DEFAULT_GEMMA_VLLM_GPU_MEMORY_UTILIZATION)
    )
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("--gemma-vllm-gpu-memory-utilization must be greater than 0 and less than or equal to 1.")

    max_model_len_text = clean_text(getattr(args, "gemma_vllm_max_model_len", DEFAULT_GEMMA_VLLM_MAX_MODEL_LEN))
    trust_remote_code = bool(
        getattr(args, "gemma_vllm_trust_remote_code", DEFAULT_GEMMA_VLLM_TRUST_REMOTE_CODE)
    )
    cache_key = (
        model,
        dtype_text,
        tensor_parallel_size,
        gpu_memory_utilization,
        max_model_len_text,
        trust_remote_code,
    )

    if cache_key in _GEMMA_VLLM_CACHE:
        return _GEMMA_VLLM_CACHE[cache_key]

    LLM, SamplingParams = import_gemma_vllm_dependencies()

    load_kwargs: dict[str, Any] = {
        "model": model,
        "dtype": dtype_text,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len_text:
        load_kwargs["max_model_len"] = int(max_model_len_text)

    print(f"  Gemma/vLLM model loading started: {model}", flush=True)
    try:
        llm = LLM(**load_kwargs)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load Gemma vLLM model: {model}. "
            "Check vLLM installation, Hugging Face access, and GPU memory. "
            f"Cause: {error}"
        ) from error

    tokenizer = None
    get_tokenizer = getattr(llm, "get_tokenizer", None)
    if callable(get_tokenizer):
        try:
            tokenizer = get_tokenizer()
        except Exception:
            tokenizer = None

    runtime_metadata = {
        "gemma_runtime": "vllm",
        "gemma_model": model,
        "gemma_torch_dtype": dtype_text,
        "gemma_vllm_tensor_parallel_size": tensor_parallel_size,
        "gemma_vllm_gpu_memory_utilization": gpu_memory_utilization,
        "gemma_vllm_max_model_len": int(max_model_len_text) if max_model_len_text else None,
        "gemma_vllm_trust_remote_code": trust_remote_code,
    }

    bundle = {
        "llm": llm,
        "tokenizer": tokenizer,
        "SamplingParams": SamplingParams,
        "runtime_metadata": runtime_metadata,
    }
    _GEMMA_VLLM_CACHE[cache_key] = bundle
    print(f"  Gemma/vLLM model loading completed: {model}", flush=True)
    print(
        "  Gemma/vLLM config: "
        f"tensor_parallel_size={tensor_parallel_size} / dtype={dtype_text} / "
        f"gpu_memory_utilization={gpu_memory_utilization}",
        flush=True,
    )
    return bundle


def build_gemma_vllm_sampling_params(tokenizer: Any, SamplingParams: Any, args: argparse.Namespace) -> Any:
    max_tokens = max(1, int(args.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER["gemma"]))
    temperature = max(0.0, float(args.temperature))
    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        sampling_kwargs["stop_token_ids"] = [eos_token_id]

    try:
        return SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("stop_token_ids", None)
        return SamplingParams(**sampling_kwargs)


def generate_gemma_vllm_texts(
    model: str,
    prompts: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    bundle = load_gemma_vllm_model(model=model, args=args)
    llm = bundle["llm"]
    tokenizer = bundle.get("tokenizer")
    SamplingParams = bundle["SamplingParams"]

    request_prompts = [build_gemma_vllm_prompt(tokenizer, prompt) for prompt in prompts]
    sampling_params = build_gemma_vllm_sampling_params(tokenizer, SamplingParams, args)

    try:
        try:
            outputs = llm.generate(request_prompts, sampling_params=sampling_params, use_tqdm=False)
        except TypeError:
            outputs = llm.generate(request_prompts, sampling_params=sampling_params)
    except Exception as error:
        raise RuntimeError(f"Gemma vLLM generation failed: {error}") from error

    if not outputs or len(outputs) != len(prompts):
        raise RuntimeError("Gemma vLLM generation returned no outputs.")

    texts: list[str] = []
    for output in outputs:
        candidates = getattr(output, "outputs", None)
        if not candidates:
            raise RuntimeError("Gemma vLLM generation returned an empty candidate list.")
        texts.append(str(getattr(candidates[0], "text", "") or ""))

    return texts, dict(bundle.get("runtime_metadata") or {})


def generate_gemma_once_vllm(
    model: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    texts, runtime_metadata = generate_gemma_vllm_texts(model=model, prompts=[prompt], args=args)
    text = texts[0] if texts else ""
    return {"response": text, "_runtime": runtime_metadata}


def gemma_vllm_server_chat_url(base_url: str) -> str:
    url = clean_text(base_url).rstrip("/")
    if not url:
        raise ValueError("--gemma-vllm-server-base-url is empty.")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def generate_gemma_once_vllm_server(
    model: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_url = clean_text(getattr(args, "gemma_vllm_server_base_url", DEFAULT_GEMMA_VLLM_SERVER_BASE_URL))
    api_key = clean_text(getattr(args, "gemma_vllm_server_api_key", DEFAULT_GEMMA_VLLM_SERVER_API_KEY))
    timeout = max(1, int(getattr(args, "gemma_vllm_server_timeout", DEFAULT_GEMMA_VLLM_SERVER_TIMEOUT)))
    max_tokens = max(1, int(args.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER["gemma"]))
    temperature = max(0.0, float(args.temperature))

    payload = {
        "model": model,
        "messages": build_gemma_chat_messages(prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = gemma_vllm_server_chat_url(base_url)
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemma vLLM server request failed: HTTP {error.code} {error.reason}. "
            f"URL={url}. Body: {error_body[:1000]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Gemma vLLM server is not reachable at {url}. "
            "Start the vLLM OpenAI-compatible server first, or check --gemma-vllm-server-base-url. "
            f"Cause: {error}"
        ) from error

    try:
        data = json.loads(response_text)
    except Exception as error:
        raise RuntimeError(
            f"Gemma vLLM server returned non-JSON response from {url}: {response_text[:1000]}"
        ) from error

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Gemma vLLM server returned no choices: {response_text[:1000]}")

    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    text = ""
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(
                clean_text(part.get("text") if isinstance(part, dict) else part)
                for part in content
                if clean_text(part.get("text") if isinstance(part, dict) else part)
            )
        else:
            text = str(content or "")
    if not text:
        text = str(first_choice.get("text") or "")
    if not text:
        raise RuntimeError(f"Gemma vLLM server returned an empty response: {response_text[:1000]}")

    runtime_metadata = {
        "gemma_runtime": "vllm-server",
        "gemma_model": model,
        "gemma_vllm_server_base_url": base_url,
        "gemma_vllm_server_timeout": timeout,
    }
    return {"response": text, "_runtime": runtime_metadata}


def generate_gemma_once(
    model: str,
    prompt: str,
    args: argparse.Namespace,
):
    runtime = normalize_gemma_runtime(getattr(args, "gemma_runtime", DEFAULT_GEMMA_RUNTIME))
    update_processing_metadata(args, gemma_runtime=runtime)
    if runtime == "vllm":
        return generate_gemma_once_vllm(
            model=model,
            prompt=prompt,
            args=args,
        )
    if runtime == "vllm-server":
        return generate_gemma_once_vllm_server(
            model=model,
            prompt=prompt,
            args=args,
        )

    return generate_gemma_once_transformers(
        model=model,
        prompt=prompt,
        args=args,
    )


def gemma_response_text(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("response") or "")
    return str(response or "")


def compact_toc_example_title(value: Any, max_chars: int = 80) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def normalize_toc_match_text(value: Any) -> str:
    return re.sub(r"\s+", "", clean_text(value)).lower()


def parse_layout_line(line: str) -> dict[str, Any] | None:
    match = re.match(r"^\[L\s+([^\]]+)\]\s*(.*)$", str(line or ""))
    if not match:
        return None

    raw_fields, title_text = match.groups()
    metadata: dict[str, Any] = {"text": clean_text(title_text)}

    for key, raw_value in re.findall(r"([A-Za-z]+)=([^\s\]]+)", raw_fields):
        value = clean_text(raw_value)
        try:
            if key in {"s", "r", "x", "y"}:
                number = float(value)
                metadata[{
                    "s": "font_size",
                    "r": "size_ratio",
                    "x": "left_indent",
                    "y": "vertical_position",
                }[key]] = round(number, 2)
            elif key in {"b", "i"}:
                metadata[{"b": "bold", "i": "italic"}[key]] = int(float(value))
            elif key == "f":
                metadata["font_id"] = value
        except Exception:
            continue

    return metadata


def find_layout_for_chapter_title(chunk_text: str, title: str) -> dict[str, Any]:
    normalized_title = normalize_toc_match_text(title)
    if not normalized_title:
        return {}

    fallback_match: dict[str, Any] = {}
    for line_index, line in enumerate(str(chunk_text or "").splitlines()):
        layout = parse_layout_line(line)
        if not layout:
            continue
        layout["line_index"] = line_index

        line_title = clean_text(layout.get("text"))
        normalized_line = normalize_toc_match_text(line_title)
        if not normalized_line:
            continue

        if normalized_line == normalized_title:
            return layout
        if normalized_title in normalized_line or normalized_line in normalized_title:
            if not fallback_match:
                fallback_match = layout

    return fallback_match


def toc_level_reference_entry(
    chapter_title: str,
    chunk_text: str,
    chapter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"title": compact_toc_example_title(chapter_title)}
    layout = layout_from_toc_metadata(chapter) if isinstance(chapter, dict) else {}
    if not layout:
        layout = find_layout_for_chapter_title(chunk_text=chunk_text, title=chapter_title)
    for key in (
        "font_size",
        "size_ratio",
        "left_indent",
        "bold",
        "italic",
        "font_id",
    ):
        if key in layout:
            entry[key] = layout[key]
    return entry


def toc_level_reference_style_outlier(
    entry: dict[str, Any],
    median_font_size: float,
    median_size_ratio: float,
) -> bool:
    font_size = metadata_float_value(entry.get("font_size"))
    size_ratio = metadata_float_value(entry.get("size_ratio"))
    if font_size is None or size_ratio is None:
        return False

    much_smaller_font = median_font_size >= 12.0 and font_size < median_font_size * 0.70
    much_smaller_ratio = median_size_ratio >= 1.25 and size_ratio < median_size_ratio * 0.75
    body_style_in_title_level = (
        median_size_ratio >= 1.50
        and size_ratio <= 1.12
        and font_size < median_font_size * 0.90
    )
    much_larger_cover_like = (
        median_font_size >= 8.0
        and median_size_ratio >= 0.8
        and font_size > median_font_size * 1.80
        and size_ratio > median_size_ratio * 1.80
    )
    return body_style_in_title_level or (much_smaller_font and much_smaller_ratio) or much_larger_cover_like


def filter_toc_level_reference_examples(
    examples: list[dict[str, Any]],
    max_examples_per_level: int,
) -> list[dict[str, Any]]:
    styled_examples = [
        example
        for example in examples
        if metadata_float_value(example.get("font_size")) is not None
        and metadata_float_value(example.get("size_ratio")) is not None
    ]
    if len(styled_examples) < 3:
        return examples[:max_examples_per_level]

    median_font_size = median_float(
        metadata_float_value(example.get("font_size"))
        for example in styled_examples
    )
    median_size_ratio = median_float(
        metadata_float_value(example.get("size_ratio"))
        for example in styled_examples
    )
    filtered = [
        example
        for example in examples
        if not toc_level_reference_style_outlier(example, median_font_size, median_size_ratio)
    ]
    if not filtered:
        filtered = examples
    return filtered[:max_examples_per_level]


def format_toc_level_reference_entry(entry: dict[str, Any]) -> str:
    title = clean_text(entry.get("title"))
    style_parts: list[str] = []
    if "font_size" in entry:
        style_parts.append(f"s={entry['font_size']}")
    if "size_ratio" in entry:
        style_parts.append(f"r={entry['size_ratio']}")
    if "left_indent" in entry:
        style_parts.append(f"x={entry['left_indent']}")
    if "bold" in entry:
        style_parts.append(f"b={entry['bold']}")
    if "italic" in entry:
        style_parts.append(f"i={entry['italic']}")
    if clean_text(entry.get("font_id")):
        style_parts.append(f"f={clean_text(entry.get('font_id'))}")

    style_text = f" style({', '.join(style_parts)})" if style_parts else ""
    return f'title="{title}"{style_text}' if title else style_text.strip()


def level_role_text(level: int) -> str:
    if level == 1:
        return "상위 장/대단원"
    if level == 2:
        return "중간 절/하위 단원"
    if level == 3:
        return "소절"
    return "세부 항목"


def layout_reason_style_text(layout: dict[str, Any]) -> str:
    parts: list[str] = []
    labels = (
        ("font_size", "s"),
        ("size_ratio", "r"),
        ("left_indent", "x"),
        ("vertical_position", "y"),
        ("bold", "b"),
        ("italic", "i"),
        ("font_id", "f"),
    )
    for key, label in labels:
        value = layout.get(key)
        if value is None or clean_text(value) == "":
            continue
        parts.append(f"{label}={value}")
    return ", ".join(parts)


def layout_from_toc_metadata(chapter: dict[str, Any]) -> dict[str, Any]:
    layout: dict[str, Any] = {}
    for target_key, aliases in LAYOUT_METADATA_FIELD_ALIASES.items():
        value = first_present_metadata_value(chapter, aliases)
        if value is None:
            continue

        if target_key == "_layout_x":
            number = metadata_float_value(value)
            if number is not None:
                layout["left_indent"] = number
        elif target_key == "_layout_y":
            number = metadata_float_value(value)
            if number is not None:
                layout["vertical_position"] = number
        elif target_key == "_layout_font_size":
            number = metadata_float_value(value)
            if number is not None:
                layout["font_size"] = number
        elif target_key == "_layout_size_ratio":
            number = metadata_float_value(value)
            if number is not None:
                layout["size_ratio"] = number
        elif target_key == "_layout_text":
            text = clean_text(value)
            if text:
                layout["text"] = text
        elif target_key == "_source_order":
            number = metadata_int_value(value)
            if number is not None:
                layout["line_index"] = number
        elif target_key == "_layout_page":
            number = metadata_int_value(value)
            if number is not None:
                layout["page"] = number
        elif target_key == "_layout_bold":
            number = metadata_int_value(value)
            if number is not None:
                layout["bold"] = number
        elif target_key == "_layout_italic":
            number = metadata_int_value(value)
            if number is not None:
                layout["italic"] = number

    return layout


def copy_layout_sort_metadata(
    chapter: dict[str, Any],
    layout: dict[str, Any],
    fallback_order: int,
    overwrite: bool = False,
) -> None:
    def assign(key: str, value: Any) -> None:
        if value is None:
            return
        if overwrite or key not in chapter:
            chapter[key] = value

    assign("_source_order", int_sort_value(layout.get("line_index"), default=fallback_order))
    if "page" in layout:
        assign("_layout_page", layout["page"])
    if "vertical_position" in layout:
        assign("_layout_y", layout["vertical_position"])
    if "left_indent" in layout:
        assign("_layout_x", layout["left_indent"])
    if "font_size" in layout:
        assign("_layout_font_size", layout["font_size"])
    if "size_ratio" in layout:
        assign("_layout_size_ratio", layout["size_ratio"])
    if "bold" in layout:
        assign("_layout_bold", layout["bold"])
    if "italic" in layout:
        assign("_layout_italic", layout["italic"])
    if clean_text(layout.get("text")):
        assign("_layout_text", clean_text(layout.get("text")))


def style_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    score = 0.0
    compared = False
    for key, weight in (
        ("font_size", 1.0),
        ("size_ratio", 8.0),
        ("left_indent", 0.08),
    ):
        if key not in left or key not in right:
            continue
        try:
            score += abs(float(left[key]) - float(right[key])) * weight
            compared = True
        except Exception:
            continue

    for key, penalty in (("bold", 2.0), ("italic", 1.0), ("font_id", 1.0)):
        if key not in left or key not in right:
            continue
        compared = True
        if clean_text(left[key]) != clean_text(right[key]):
            score += penalty

    return score if compared else 9999.0


def closest_level_reference(
    layout: dict[str, Any],
    level_reference: dict[int, list[dict[str, Any]]] | None,
) -> tuple[int | None, dict[str, Any] | None]:
    if not layout or not level_reference:
        return None, None

    best_level: int | None = None
    best_entry: dict[str, Any] | None = None
    best_score = 9999.0
    for level, entries in level_reference.items():
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            score = style_distance(layout, entry)
            if score < best_score:
                best_score = score
                best_level = level
                best_entry = entry

    if best_score >= 9999.0:
        return None, None
    return best_level, best_entry


def build_level_reason(
    level: int,
    chapter_title: str,
    chunk_text: str,
    level_reference: dict[int, list[dict[str, Any]]] | None = None,
    layout: dict[str, Any] | None = None,
) -> str:
    layout = layout if isinstance(layout, dict) else find_layout_for_chapter_title(chunk_text=chunk_text, title=chapter_title)
    role = level_role_text(level)
    reason_parts = [f"level {level}: {role} 계층으로 분류됨"]

    style_text = layout_reason_style_text(layout)
    if style_text:
        reason_parts.append(f"현재 줄 layout({style_text})")
    else:
        reason_parts.append("현재 chunk에서 대응 layout tag를 찾지 못해 제목 계층과 주변 문맥 기준")

    closest_level, closest_entry = closest_level_reference(layout, level_reference)
    if closest_level is not None:
        closest_title = clean_text((closest_entry or {}).get("title"))
        if closest_level == level:
            suffix = f' "{closest_title}"' if closest_title else ""
            reason_parts.append(f"이전 level {level} 예시{suffix}와 가장 가까운 style 패턴")
        else:
            reason_parts.append(
                f"이전 style 기준으로는 level {closest_level}와 가장 가까우나, 현재 문서의 제목 계층/문맥상 level {level}로 유지"
            )
    else:
        reason_parts.append("이전 level style 기준 없음")

    return "; ".join(reason_parts) + "."


def add_level_reasons_to_toc(
    toc: dict[str, Any],
    chunk_text: str,
    level_reference: dict[int, list[dict[str, Any]]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    chapters = toc.get("chapters")
    if not isinstance(chapters, list):
        return toc

    for chapter_index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            continue
        try:
            level = int(chapter.get("level", 1))
        except Exception:
            level = 1
        title = clean_text(chapter.get("chapter"))
        if not title:
            continue
        layout = layout_from_toc_metadata(chapter)
        if not layout:
            layout = find_layout_for_chapter_title(chunk_text=chunk_text, title=title)
        copy_layout_sort_metadata(chapter, layout, fallback_order=chapter_index)
        if clean_text(chapter.get("level_reason")) and not overwrite:
            continue
        chapter["level_reason"] = build_level_reason(
            level=level,
            chapter_title=title,
            chunk_text=chunk_text,
            level_reference=level_reference,
            layout=layout,
        )
    return toc


def partial_toc_level_reason_map(partial_tocs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    reason_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for partial in partial_tocs:
        toc = partial.get("toc") if isinstance(partial, dict) else None
        if not isinstance(toc, dict):
            continue
        chapters = toc.get("chapters", [])
        if not isinstance(chapters, list):
            continue
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            reason = clean_text(chapter.get("level_reason"))
            title = normalize_toc_match_text(chapter.get("chapter"))
            if not reason or not title:
                continue
            try:
                page = int(chapter.get("page", 1))
            except Exception:
                page = 1
            try:
                level = int(chapter.get("level", 1))
            except Exception:
                level = 1
            reason_by_key.setdefault((title, page), {"reason": reason, "level": level})
    return reason_by_key


def partial_toc_layout_sort_map(partial_tocs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    layout_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for partial in partial_tocs:
        toc = partial.get("toc") if isinstance(partial, dict) else None
        if not isinstance(toc, dict):
            continue
        chapters = toc.get("chapters", [])
        if not isinstance(chapters, list):
            continue
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            title = normalize_toc_match_text(chapter.get("chapter"))
            if not title:
                continue
            try:
                page = int(chapter.get("page", 1))
            except Exception:
                page = 1

            metadata: dict[str, Any] = {}
            for key in (
                "_layout_y",
                "_layout_x",
                "_layout_font_size",
                "_layout_size_ratio",
                "_layout_text",
                "_layout_page",
                "_source_order",
                "_local_merge_order",
            ):
                if key in chapter:
                    metadata[key] = chapter[key]
            if metadata:
                layout_by_key.setdefault((title, page), metadata)
    return layout_by_key


def restore_level_reasons_from_partials(toc: dict[str, Any], partial_tocs: list[dict[str, Any]]) -> dict[str, Any]:
    reason_by_key = partial_toc_level_reason_map(partial_tocs)
    if not reason_by_key:
        return toc

    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return toc

    for chapter in chapters:
        if not isinstance(chapter, dict) or clean_text(chapter.get("level_reason")):
            continue
        title = normalize_toc_match_text(chapter.get("chapter"))
        if not title:
            continue
        try:
            page = int(chapter.get("page", 1))
        except Exception:
            page = 1
        reason_entry = reason_by_key.get((title, page))
        if not reason_entry:
            continue
        try:
            if int(chapter.get("level", 1)) != int(reason_entry.get("level", 1)):
                continue
        except Exception:
            continue
        reason = clean_text(reason_entry.get("reason"))
        if reason:
            chapter["level_reason"] = reason
    return toc


def restore_layout_sort_metadata_from_partials(toc: dict[str, Any], partial_tocs: list[dict[str, Any]]) -> dict[str, Any]:
    layout_by_key = partial_toc_layout_sort_map(partial_tocs)
    if not layout_by_key:
        return toc

    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return toc

    for index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            continue
        title = normalize_toc_match_text(chapter.get("chapter"))
        if not title:
            continue
        try:
            page = int(chapter.get("page", 1))
        except Exception:
            page = 1
        metadata = layout_by_key.get((title, page))
        if not metadata:
            continue
        for key, value in metadata.items():
            chapter.setdefault(key, value)
        chapter.setdefault("_local_merge_order", index)
    return toc


def sort_toc_with_restored_layout(
    toc: dict[str, Any],
    partial_tocs: list[dict[str, Any]],
    fallback_title: str,
    max_depth: int,
) -> dict[str, Any]:
    restore_layout_sort_metadata_from_partials(toc, partial_tocs)
    chapters = toc.get("chapters", [])
    if isinstance(chapters, list):
        for index, chapter in enumerate(chapters):
            if isinstance(chapter, dict):
                chapter.setdefault("_local_merge_order", index)
        chapters.sort(key=lambda item: local_merge_sort_key(item) if isinstance(item, dict) else (999999,))
    return validate_toc(toc, fallback_title=fallback_title, max_depth=max_depth)


def update_toc_level_reference(
    level_reference: dict[int, list[dict[str, Any]]],
    toc: dict[str, Any],
    max_depth: int,
    chunk_text: str = "",
    max_examples_per_level: int = 5,
) -> None:
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue

        try:
            level = int(chapter.get("level", 1))
        except Exception:
            continue
        if level < 1 or level > max_depth:
            continue

        chapter_title = clean_text(chapter.get("chapter"))
        title = compact_toc_example_title(chapter_title)
        if not title:
            continue

        entry = toc_level_reference_entry(chapter_title=chapter_title, chunk_text=chunk_text, chapter=chapter)
        examples = level_reference.setdefault(level, [])
        normalized = re.sub(r"\s+", "", title).lower()
        if any(normalize_toc_match_text(existing.get("title")) == normalized for existing in examples):
            continue
        examples.append(entry)
        level_reference[level] = filter_toc_level_reference_examples(
            examples,
            max_examples_per_level=max_examples_per_level,
        )


def build_toc_level_reference(
    toc: dict[str, Any],
    max_depth: int,
    chunk_text: str = "",
) -> dict[int, list[dict[str, Any]]]:
    level_reference: dict[int, list[dict[str, Any]]] = {}
    update_toc_level_reference(
        level_reference,
        toc,
        max_depth=max_depth,
        chunk_text=chunk_text,
    )
    return level_reference


def serialize_toc_level_reference(
    level_reference: dict[int, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(level): examples
        for level, examples in sorted(level_reference.items())
    }


def format_toc_level_reference(level_reference: dict[int, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for level in sorted(level_reference):
        examples = [
            format_toc_level_reference_entry(example)
            for example in level_reference[level]
            if isinstance(example, dict)
        ]
        examples = [example for example in examples if clean_text(example)]
        if not examples:
            continue
        lines.append(f"- level {level}: {'; '.join(examples)}")

    if not lines:
        return ""

    return "\n".join([
        "[Previous TOC Level Reference]",
        "Use only for level consistency; do not copy titles unless visible in this chunk.",
        *lines,
    ])


def build_gemma_text_prompt(
    base_prompt: str,
    pdf_name: str,
    text: str,
    document_style_text: str = "",
) -> str:
    document_style_block = f"\n\n{document_style_text}" if clean_text(document_style_text) else ""
    return f"""
{base_prompt}

[Gemma Text/Layout Mode]
Create the TOC using only the [Extracted PDF Text] below instead of an attached PDF.
Use each text block's [PAGE n] marker to determine page numbers.
Rules:
- Output only body hierarchy titles: chapters, sections, subsections.
- Ignore existing TOC/Contents pages and listings, repeated headers/footers, body sentences, captions, questions, and references.
- Layout tags may appear as [L s=... r=... x=... y=... b=... i=... f=...]. Use them only for level inference; never copy the tag.
- Include level_reason and copy _layout_page/_layout_x/_layout_y/_layout_font_size/_layout_size_ratio/_layout_text when a reliable [L] title line exists.
- Return compact valid JSON only.
{document_style_block}

[PDF File Name]
{pdf_name}

[Extracted PDF Text]
{text}
""".strip()


def build_gemma_chunk_prompt(
    base_prompt: str,
    pdf_name: str,
    chunk: dict[str, Any],
    total_chunks: int,
    level_reference_text: str = "",
    document_style_text: str = "",
) -> str:
    level_reference_block = f"\n\n{level_reference_text}" if clean_text(level_reference_text) else ""
    document_style_block = f"\n\n{document_style_text}" if clean_text(document_style_text) else ""
    return f"""
{base_prompt}

[Gemma Text/Layout Chunk Mode]
The text below is one part of the full PDF.
Use [PAGE n] markers to determine page numbers.
Rules:
- Include only body hierarchy titles visible in this chunk. Do not infer entries outside this page range.
- Ignore existing TOC/Contents pages and listings, repeated headers/footers, body sentences, captions, questions, and references.
- Use [L s/r/x/y/b/i/f] layout only for level inference; never copy the tag.
- Include level_reason and copy _layout_page/_layout_x/_layout_y/_layout_font_size/_layout_size_ratio/_layout_text when reliable.
- Return compact valid JSON only.
{document_style_block}
{level_reference_block}

[PDF File Name]
{pdf_name}

[Chunk]
{chunk["index"]}/{total_chunks}, pages {chunk["start_page"]}-{chunk["end_page"]}

[Extracted PDF Text]
{chunk["text"]}
""".strip()


def build_gemma_merge_prompt(
    base_prompt: str,
    pdf_name: str,
    partial_tocs: list[dict[str, Any]],
    max_depth: int,
    include_level_reason: bool = True,
    document_style_text: str = "",
) -> str:
    partial_json = json.dumps(partial_tocs, ensure_ascii=False, indent=2)
    document_style_block = f"\n\n{document_style_text}" if clean_text(document_style_text) else ""
    level_reason_instruction = (
        "- Preserve level_reason when it exists. If an entry has no level_reason, add a concise reason based only on the candidate entry and nearby level pattern."
        if include_level_reason
        else "- For this merge response, omit level_reason to keep the JSON compact. The program will restore level_reason from chunk results after merge."
    )
    return f"""
{base_prompt}

[Chunked TOC Merge Instruction]
Create one final TOC JSON object using only the [Partial TOC Candidates] below instead of the attached PDF.
- Keep one copy of duplicates; remove cover, TOC, index, reference, repeated header/footer, caption, question, and body-sentence entries.
- Preserve source page/order and use only levels 1 to {max_depth}.
- Do not invent chapter titles.
- Preserve original body hierarchy titles exactly.
- Correct obvious level inconsistencies using layout metadata and nearby hierarchy.
{level_reason_instruction}
{document_style_block}

[PDF File Name]
{pdf_name}

[Partial TOC Candidates]
{partial_json}
""".strip()


def toc_chapter_style_key(chapter: dict[str, Any]) -> tuple[float | None, float | None]:
    font_size = metadata_float_value(chapter.get("_layout_font_size"))
    size_ratio = metadata_float_value(chapter.get("_layout_size_ratio"))
    if font_size is not None:
        font_size = round(font_size, 1)
    if size_ratio is not None:
        size_ratio = round(size_ratio, 2)
    return font_size, size_ratio


def format_style_key(style_key: tuple[float | None, float | None]) -> str:
    font_size, size_ratio = style_key
    parts: list[str] = []
    if font_size is not None:
        parts.append(f"font_size={font_size}")
    if size_ratio is not None:
        parts.append(f"size_ratio={size_ratio}")
    return ", ".join(parts) if parts else "no layout style"


def style_font_size(style_key: tuple[float | None, float | None]) -> float:
    font_size, _ = style_key
    return float(font_size) if font_size is not None else 0.0


def style_size_ratio(style_key: tuple[float | None, float | None]) -> float:
    _, size_ratio = style_key
    return float(size_ratio) if size_ratio is not None else 0.0


def leading_decimal_number_path(title: Any) -> tuple[int, ...] | None:
    text = clean_text(title)
    match = re.match(r"^\s*(\d+(?:\.\d+)+)\.?", text)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except Exception:
        return None


def format_number_path(path: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in path)


def format_gemma_numbering_gap_summary(toc: dict[str, Any], max_gaps: int = 20) -> str:
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return ""

    rows_by_prefix: dict[tuple[int, ...], dict[int, dict[str, Any]]] = {}
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = clean_text(chapter.get("chapter"))
        path = leading_decimal_number_path(title)
        if not path or len(path) < 2:
            continue
        try:
            level = int(chapter.get("level", 1))
        except Exception:
            level = 1
        try:
            page = int(chapter.get("page", 1))
        except Exception:
            page = None
        prefix = path[:-1]
        rows_by_prefix.setdefault(prefix, {})[path[-1]] = {
            "title": title,
            "path": path,
            "level": level,
            "page": page,
        }

    gaps: list[str] = []
    for prefix, rows_by_number in sorted(rows_by_prefix.items()):
        numbers = sorted(rows_by_number)
        if len(numbers) < 2:
            continue
        for previous_number, next_number in zip(numbers, numbers[1:]):
            if next_number <= previous_number + 1:
                continue
            before = rows_by_number[previous_number]
            after = rows_by_number[next_number]
            for missing_number in range(previous_number + 1, next_number):
                missing_path = (*prefix, missing_number)
                before_page = f", page={before['page']}" if before.get("page") else ""
                after_page = f", page={after['page']}" if after.get("page") else ""
                gaps.append(
                    f'- possible missing numbered body hierarchy title "{format_number_path(missing_path)}" between "{compact_toc_example_title(before["title"], max_chars=70)}"{before_page} and "{compact_toc_example_title(after["title"], max_chars=70)}"{after_page}. Search Source Text/Layout or Partial TOC Candidates for this grounded body title before finalizing.'
                )
                if len(gaps) >= max_gaps:
                    break
            if len(gaps) >= max_gaps:
                break
        if len(gaps) >= max_gaps:
            break

    if not gaps:
        return ""
    return "\n".join([
        "[Observed Numbering Sequence Gaps]",
        "These are not automatic rules. They are completeness-audit clues. Restore a missing numbered body hierarchy title only if it is grounded in Source Text/Layout or Partial TOC Candidates.",
        *gaps,
    ])


def toc_decimal_number_paths(toc: dict[str, Any]) -> set[tuple[int, ...]]:
    paths: set[tuple[int, ...]] = set()
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return paths

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        path = leading_decimal_number_path(chapter.get("chapter"))
        if path:
            paths.add(path)
    return paths


def toc_normalized_titles(toc: dict[str, Any]) -> set[str]:
    titles: set[str] = set()
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return titles

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = normalize_toc_match_text(chapter.get("chapter"))
        if title:
            titles.add(title)
    return titles


def is_existing_toc_line_candidate(title: str) -> bool:
    text = clean_text(title)
    if not text:
        return False
    if re.search(r"\.{3,}\s*\d+\s*$", text):
        return True
    if re.search(r"\s[·ㆍ.]{3,}\s*\d+\s*$", text):
        return True
    return False


def missing_numbered_candidate_layout_text(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    if candidate.get("page"):
        parts.append(f"page={candidate['page']}")
    if candidate.get("line_index") is not None:
        parts.append(f"line={candidate['line_index']}")
    if candidate.get("font_size") is not None:
        parts.append(f"s={candidate['font_size']}")
    if candidate.get("size_ratio") is not None:
        parts.append(f"r={candidate['size_ratio']}")
    if candidate.get("left_indent") is not None:
        parts.append(f"x={candidate['left_indent']}")
    if candidate.get("vertical_position") is not None:
        parts.append(f"y={candidate['vertical_position']}")
    return ", ".join(parts)


def is_probable_non_heading_candidate(title: str) -> bool:
    text = clean_text(title)
    if not text:
        return True
    if is_existing_toc_line_candidate(text):
        return True
    if len(text) > 180:
        return True
    if re.fullmatch(r"[\d\s.,;:()\-_/]+", text):
        return True
    if re.match(r"^[•●▪▫\-–—*]\s+", text):
        return True
    if re.match(r"^(table|figure|fig\.|box|표|그림)\s*[\dIVXivx가-힣.:-]", text, re.IGNORECASE):
        return True
    if len(text) > 80 and re.search(r"(다|요|니다|였다|한다|있다|된다|\.|;|:)$", text):
        return True
    return False


def has_structural_title_marker(title: str) -> bool:
    text = clean_text(title)
    if not text:
        return False
    patterns = (
        r"^제\s*\d+\s*[장절편부]\b",
        r"^\d+(?:\.\d+)+\b",
        r"^\d+\s*[.)]",
        r"^\([^)]+\)",
        r"^[A-Za-z]\s*[.)]",
        r"^[가-힣]\s*[.)]",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def is_ancillary_heading_candidate(title: str) -> bool:
    text = clean_text(title)
    if not text:
        return False
    patterns = (
        r"^\[?\s*(?:사례|예시|보기|case|example|box)\s*\d*",
        r"^(?:표|그림|figure|fig\.|table)\s*[\dIVXivx가-힣.:-]",
        r"^(?:기본\s*문제|심화\s*문제|연습\s*문제|문제|참고문헌|references?)\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def is_visual_material_caption(title: str) -> bool:
    text = clean_text(title)
    if not text:
        return False
    return bool(re.match(
        r"^(?:표|그림|figure|fig\.|table|box)\s*[\dIVXivx가-힣.:-]",
        text,
        re.IGNORECASE,
    ))


def toc_heading_style_profiles(toc: dict[str, Any]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return profiles

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        font_size = metadata_float_value(chapter.get("_layout_font_size"))
        size_ratio = metadata_float_value(chapter.get("_layout_size_ratio"))
        if font_size is None and size_ratio is None:
            continue
        try:
            level = int(chapter.get("level", 1))
        except Exception:
            level = 1
        profiles.append({
            "level": level,
            "font_size": font_size,
            "size_ratio": size_ratio,
            "left_indent": metadata_float_value(chapter.get("_layout_x")),
            "title": clean_text(chapter.get("chapter")),
        })
    return profiles


def layout_matches_existing_heading_style(layout: dict[str, Any], profiles: list[dict[str, Any]]) -> bool:
    font_size = metadata_float_value(layout.get("font_size"))
    size_ratio = metadata_float_value(layout.get("size_ratio"))
    if font_size is None and size_ratio is None:
        return False

    for profile in profiles:
        profile_font = metadata_float_value(profile.get("font_size"))
        profile_ratio = metadata_float_value(profile.get("size_ratio"))
        font_match = (
            font_size is not None
            and profile_font is not None
            and abs(font_size - profile_font) <= 1.0
        )
        ratio_match = (
            size_ratio is not None
            and profile_ratio is not None
            and abs(size_ratio - profile_ratio) <= 0.12
        )
        if font_match and (ratio_match or profile_ratio is None):
            return True
        if ratio_match and (font_match or profile_font is None):
            return True
    return False


def style_heading_candidate_score(layout: dict[str, Any], profiles: list[dict[str, Any]]) -> float:
    font_size = metadata_float_value(layout.get("font_size")) or 0.0
    size_ratio = metadata_float_value(layout.get("size_ratio")) or 1.0
    bold = 1 if metadata_int_value(layout.get("bold")) else 0
    italic = 1 if metadata_int_value(layout.get("italic")) else 0

    score = 0.0
    score += max(0.0, size_ratio - 1.0) * 5.0
    score += 1.4 if bold else 0.0
    score += 0.5 if italic else 0.0
    if layout_matches_existing_heading_style(layout, profiles):
        score += 2.0
    if font_size >= 14.0:
        score += 0.5
    return score


def is_style_heading_candidate(title: str, layout: dict[str, Any], profiles: list[dict[str, Any]]) -> bool:
    if is_probable_non_heading_candidate(title):
        return False
    if not layout:
        return False

    size_ratio = metadata_float_value(layout.get("size_ratio")) or 1.0
    bold = 1 if metadata_int_value(layout.get("bold")) else 0
    italic = 1 if metadata_int_value(layout.get("italic")) else 0
    if size_ratio <= 1.08 and not bold and not italic and not has_structural_title_marker(title):
        return False
    if layout_matches_existing_heading_style(layout, profiles):
        return True
    if size_ratio >= 1.18:
        return True
    if bold and size_ratio >= 1.03:
        return True
    if italic and size_ratio >= 1.08:
        return True
    return False


def add_missing_numbered_candidate(
    candidates_by_path: dict[tuple[int, ...], dict[str, Any]],
    existing_paths: set[tuple[int, ...]],
    existing_titles: set[str],
    source: str,
    title: Any,
    page: Any = None,
    chunk: Any = None,
    layout: dict[str, Any] | None = None,
    line_index: int | None = None,
) -> None:
    clean_title = clean_text(title)
    if not clean_title or is_existing_toc_line_candidate(clean_title):
        return

    path = leading_decimal_number_path(clean_title)
    if not path:
        return
    if path in existing_paths:
        return
    if normalize_toc_match_text(clean_title) in existing_titles:
        return

    candidate = candidates_by_path.get(path)
    if candidate and candidate.get("source") == "source_text":
        return

    layout = layout if isinstance(layout, dict) else {}
    row: dict[str, Any] = {
        "source": source,
        "number": format_number_path(path),
        "title": clean_title,
    }
    page_number = metadata_int_value(page)
    if page_number is not None:
        row["page"] = page_number
    if chunk is not None:
        row["chunk"] = chunk
    if line_index is not None:
        row["line_index"] = line_index
    for source_key, target_key in (
        ("font_size", "font_size"),
        ("size_ratio", "size_ratio"),
        ("left_indent", "left_indent"),
        ("vertical_position", "vertical_position"),
        ("bold", "bold"),
        ("italic", "italic"),
        ("font_id", "font_id"),
    ):
        if source_key in layout:
            row[target_key] = layout[source_key]

    candidates_by_path[path] = row


def source_text_layout_candidates(source_text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    current_page: int | None = None
    for line_index, raw_line in enumerate(str(source_text or "").splitlines()):
        line = clean_text(raw_line)
        if not line:
            continue
        page_match = re.match(r"^\[PAGE\s+(\d+)\]$", line)
        if page_match:
            current_page = metadata_int_value(page_match.group(1))
            continue
        layout = parse_layout_line(line)
        if not layout:
            continue
        title = clean_text(layout.get("text"))
        if not title:
            continue
        candidates.append({
            "source": "source_text",
            "title": title,
            "page": current_page,
            "line_index": line_index,
            "layout": layout,
        })
    return candidates


def gemma_full_text_from_pages(pages: list[tuple[int, str]]) -> str:
    return "\n\n".join(f"[PAGE {page_number}]\n{text}" for page_number, text in pages)


def document_style_cluster_key(layout: dict[str, Any]) -> tuple[float, float, int, int, str, int]:
    font_size = metadata_float_value(layout.get("font_size")) or 0.0
    size_ratio = metadata_float_value(layout.get("size_ratio")) or 1.0
    left_indent = metadata_float_value(layout.get("left_indent")) or 0.0
    bold = 1 if metadata_int_value(layout.get("bold")) else 0
    italic = 1 if metadata_int_value(layout.get("italic")) else 0
    font_id = clean_text(layout.get("font_id")) or "f?"
    x_band = int(round(left_indent / 10.0) * 10)
    return round(font_size, 1), round(size_ratio, 2), bold, italic, font_id, x_band


def document_style_page_range(pages: Iterable[Any]) -> str:
    numbers = sorted({
        page_number
        for page in pages
        if (page_number := metadata_int_value(page)) is not None
    })
    if not numbers:
        return ""
    if len(numbers) == 1:
        return str(numbers[0])
    return f"{numbers[0]}-{numbers[-1]}"


def document_style_sample_texts(samples: list[str], max_samples: int = 3) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        text = compact_toc_example_title(sample, max_chars=80)
        normalized = normalize_toc_match_text(text)
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        output.append(text)
        if len(output) >= max_samples:
            break
    return output


def document_style_cluster_summary(cluster: dict[str, Any]) -> dict[str, Any]:
    x_values = [metadata_float_value(value) for value in cluster.get("_x_values", [])]
    x_values = [value for value in x_values if value is not None]
    y_values = [metadata_float_value(value) for value in cluster.get("_y_values", [])]
    y_values = [value for value in y_values if value is not None]
    pages = sorted(cluster.get("_pages", set()))
    return {
        "font_size": cluster.get("font_size"),
        "size_ratio": cluster.get("size_ratio"),
        "bold": cluster.get("bold"),
        "italic": cluster.get("italic"),
        "font_id": cluster.get("font_id"),
        "left_indent_band": cluster.get("left_indent_band"),
        "median_left_indent": round(median_float(x_values), 1) if x_values else None,
        "median_vertical_position": round(median_float(y_values), 1) if y_values else None,
        "line_count": cluster.get("line_count", 0),
        "page_count": len(pages),
        "page_range": document_style_page_range(pages),
        "samples": document_style_sample_texts(cluster.get("_samples", [])),
    }


def document_style_line_is_title_like(title: str, layout: dict[str, Any]) -> bool:
    if is_probable_non_heading_candidate(title) or is_ancillary_heading_candidate(title):
        return False
    if len(clean_text(title)) > 140:
        return False

    size_ratio = metadata_float_value(layout.get("size_ratio")) or 1.0
    font_size = metadata_float_value(layout.get("font_size")) or 0.0
    bold = 1 if metadata_int_value(layout.get("bold")) else 0
    italic = 1 if metadata_int_value(layout.get("italic")) else 0
    return (
        size_ratio >= 1.12
        or font_size >= 14.0
        or bool(bold)
        or bool(italic)
        or (has_structural_title_marker(title) and size_ratio >= 1.05)
    )


def layout_candidate_has_visual_material_context(
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    vertical_window: float = 80.0,
) -> bool:
    layout = candidate.get("layout") if isinstance(candidate.get("layout"), dict) else {}
    page_number = metadata_int_value(candidate.get("page"))
    line_index = metadata_int_value(candidate.get("line_index"))
    y = metadata_float_value(layout.get("vertical_position"))
    for neighbor in candidates:
        if neighbor is candidate:
            continue
        neighbor_page = metadata_int_value(neighbor.get("page"))
        if page_number is not None and neighbor_page is not None and neighbor_page != page_number:
            continue
        neighbor_title = clean_text(neighbor.get("title"))
        if not is_visual_material_caption(neighbor_title):
            continue
        neighbor_layout = neighbor.get("layout") if isinstance(neighbor.get("layout"), dict) else {}
        neighbor_y = metadata_float_value(neighbor_layout.get("vertical_position"))
        neighbor_line_index = metadata_int_value(neighbor.get("line_index"))
        close_by_y = y is not None and neighbor_y is not None and abs(neighbor_y - y) <= vertical_window
        close_by_line = (
            line_index is not None
            and neighbor_line_index is not None
            and abs(neighbor_line_index - line_index) <= 8
        )
        if close_by_y or close_by_line:
            return True
    return False


def build_gemma_document_style_reference(
    pages: list[tuple[int, str]],
    *,
    enabled: bool = True,
    max_clusters: int = DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_CLUSTERS,
    max_examples: int = DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_EXAMPLES,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "prompt_text": ""}

    full_text = gemma_full_text_from_pages(pages)
    layout_lines = source_text_layout_candidates(full_text)
    if not layout_lines:
        return {
            "enabled": True,
            "available": False,
            "reason": "No compact layout-tagged lines were found.",
            "prompt_text": "",
        }

    clusters: dict[tuple[float, float, int, int, str, int], dict[str, Any]] = {}
    title_like_examples: list[dict[str, Any]] = []
    for candidate in layout_lines:
        layout = candidate.get("layout") if isinstance(candidate.get("layout"), dict) else {}
        title = clean_text(candidate.get("title"))
        if not title:
            continue
        key = document_style_cluster_key(layout)
        font_size, size_ratio, bold, italic, font_id, x_band = key
        cluster = clusters.setdefault(key, {
            "font_size": font_size,
            "size_ratio": size_ratio,
            "bold": bold,
            "italic": italic,
            "font_id": font_id,
            "left_indent_band": x_band,
            "line_count": 0,
            "_pages": set(),
            "_x_values": [],
            "_y_values": [],
            "_samples": [],
        })
        cluster["line_count"] += 1
        page_number = metadata_int_value(candidate.get("page"))
        if page_number is not None:
            cluster["_pages"].add(page_number)
        left_indent = metadata_float_value(layout.get("left_indent"))
        if left_indent is not None:
            cluster["_x_values"].append(left_indent)
        vertical_position = metadata_float_value(layout.get("vertical_position"))
        if vertical_position is not None:
            cluster["_y_values"].append(vertical_position)
        if len(cluster["_samples"]) < 8 and not is_probable_non_heading_candidate(title):
            cluster["_samples"].append(title)

        if (
            document_style_line_is_title_like(title, layout)
            and not layout_candidate_has_visual_material_context(candidate, layout_lines)
        ):
            title_like_examples.append({
                "page": page_number,
                "line_index": metadata_int_value(candidate.get("line_index")),
                "title": compact_toc_example_title(title, max_chars=90),
                "font_size": metadata_float_value(layout.get("font_size")),
                "size_ratio": metadata_float_value(layout.get("size_ratio")),
                "left_indent": metadata_float_value(layout.get("left_indent")),
                "vertical_position": metadata_float_value(layout.get("vertical_position")),
                "bold": 1 if metadata_int_value(layout.get("bold")) else 0,
                "italic": 1 if metadata_int_value(layout.get("italic")) else 0,
                "font_id": clean_text(layout.get("font_id")),
                "has_marker": has_structural_title_marker(title),
            })

    cluster_summaries = [document_style_cluster_summary(cluster) for cluster in clusters.values()]
    probable_body_styles = sorted(
        (
            cluster
            for cluster in cluster_summaries
            if (metadata_float_value(cluster.get("size_ratio")) or 1.0) <= 1.12
            and not metadata_int_value(cluster.get("bold"))
            and not metadata_int_value(cluster.get("italic"))
        ),
        key=lambda item: int(item.get("line_count") or 0),
        reverse=True,
    )[:5]
    stronger_styles = sorted(
        (
            cluster
            for cluster in cluster_summaries
            if (metadata_float_value(cluster.get("size_ratio")) or 1.0) > 1.08
            or metadata_int_value(cluster.get("bold"))
            or metadata_int_value(cluster.get("italic"))
        ),
        key=lambda item: (
            0 if int(item.get("line_count") or 0) >= 2 or int(item.get("page_count") or 0) >= 2 else 1,
            -(metadata_float_value(item.get("size_ratio")) or 0.0),
            -(metadata_float_value(item.get("font_size")) or 0.0),
            -int(item.get("line_count") or 0),
        ),
    )[:max(1, int(max_clusters))]

    title_like_examples = sorted(
        title_like_examples,
        key=lambda item: (
            int(item.get("page") or 0),
            int(item.get("line_index") or 0),
        ),
    )[:max(1, int(max_examples))]

    page_numbers = {
        page_number
        for candidate in layout_lines
        if (page_number := metadata_int_value(candidate.get("page"))) is not None
    }
    summary = {
        "enabled": True,
        "available": True,
        "page_count": len(page_numbers),
        "layout_line_count": len(layout_lines),
        "style_cluster_count": len(cluster_summaries),
        "probable_body_styles": probable_body_styles,
        "stronger_or_marked_style_clusters": stronger_styles,
        "title_like_examples": title_like_examples,
    }
    summary["prompt_text"] = format_gemma_document_style_reference(summary)
    return summary


def format_document_style_cluster_for_prompt(cluster: dict[str, Any]) -> str:
    samples = cluster.get("samples") if isinstance(cluster.get("samples"), list) else []
    sample_text = "; ".join(f'"{sample}"' for sample in samples[:3] if clean_text(sample))
    sample_suffix = f"; samples={sample_text}" if sample_text else ""
    return (
        f's={cluster.get("font_size")} r={cluster.get("size_ratio")} '
        f'x~{cluster.get("median_left_indent")} b={cluster.get("bold")} i={cluster.get("italic")} '
        f'f={clean_text(cluster.get("font_id"))} lines={cluster.get("line_count")} '
        f'pages={cluster.get("page_range")}{sample_suffix}'
    )


def format_gemma_document_style_reference(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary.get("available"):
        return ""

    body_lines = [
        f"- {format_document_style_cluster_for_prompt(cluster)}"
        for cluster in summary.get("probable_body_styles", [])
        if isinstance(cluster, dict)
    ]
    style_lines = [
        f"- {format_document_style_cluster_for_prompt(cluster)}"
        for cluster in summary.get("stronger_or_marked_style_clusters", [])
        if isinstance(cluster, dict)
    ]
    sections = [
        "[Document Style]",
        f'- layout_lines={summary.get("layout_line_count")} style_clusters={summary.get("style_cluster_count")} pages={summary.get("page_count")}',
        "- Use these compact style hints only for level consistency; do not copy sample text from this block.",
    ]
    if body_lines:
        sections.append("\nBody styles:")
        sections.extend(body_lines[:3])
    if style_lines:
        sections.append("\nCandidate title styles:")
        sections.extend(style_lines[:8])
    return "\n".join(sections)


def source_title_line_evidence(title: str, page: Any, source_text: str, max_examples: int = 3) -> dict[str, Any]:
    normalized_title = normalize_toc_match_text(title)
    if not normalized_title or not clean_text(source_text):
        return {}

    page_number = metadata_int_value(page)
    exact_examples: list[str] = []
    embedded_examples: list[str] = []
    exact_count = 0
    embedded_count = 0
    for candidate in source_text_layout_candidates(source_text):
        candidate_page = metadata_int_value(candidate.get("page"))
        if page_number is not None and candidate_page is not None and candidate_page != page_number:
            continue
        line_title = clean_text(candidate.get("title"))
        normalized_line = normalize_toc_match_text(line_title)
        if not normalized_line:
            continue
        layout = candidate.get("layout") if isinstance(candidate.get("layout"), dict) else {}
        location = missing_numbered_candidate_layout_text({
            "page": candidate_page,
            "line_index": candidate.get("line_index"),
            "font_size": metadata_float_value(layout.get("font_size")),
            "size_ratio": metadata_float_value(layout.get("size_ratio")),
            "left_indent": metadata_float_value(layout.get("left_indent")),
            "vertical_position": metadata_float_value(layout.get("vertical_position")),
        })
        example = f'{location}: "{compact_toc_example_title(line_title, max_chars=90)}"' if location else f'"{compact_toc_example_title(line_title, max_chars=90)}"'
        if normalized_line == normalized_title:
            exact_count += 1
            if len(exact_examples) < max_examples:
                exact_examples.append(example)
        elif normalized_title in normalized_line:
            embedded_count += 1
            if len(embedded_examples) < max_examples:
                embedded_examples.append(example)

    evidence: dict[str, Any] = {
        "exact_count": exact_count,
        "embedded_count": embedded_count,
    }
    if exact_examples:
        evidence["exact_examples"] = exact_examples
    if embedded_examples:
        evidence["embedded_examples"] = embedded_examples
    return evidence


def source_layout_for_title(title: str, page: Any, source_text: str) -> dict[str, Any]:
    normalized_title = normalize_toc_match_text(title)
    if not normalized_title or not clean_text(source_text):
        return {}

    page_number = metadata_int_value(page)
    embedded_match: dict[str, Any] = {}
    for candidate in source_text_layout_candidates(source_text):
        candidate_page = metadata_int_value(candidate.get("page"))
        if page_number is not None and candidate_page is not None and candidate_page != page_number:
            continue
        line_title = clean_text(candidate.get("title"))
        normalized_line = normalize_toc_match_text(line_title)
        if not normalized_line:
            continue
        layout = candidate.get("layout") if isinstance(candidate.get("layout"), dict) else {}
        layout = dict(layout)
        layout["line_index"] = candidate.get("line_index")
        layout["page"] = candidate_page
        if normalized_line == normalized_title:
            return layout
        if normalized_title in normalized_line and not embedded_match:
            embedded_match = layout
    return embedded_match


def source_visual_context_for_title(
    title: str,
    page: Any,
    source_text: str,
    *,
    vertical_window: float = 80.0,
    max_examples: int = 4,
) -> dict[str, Any]:
    normalized_title = normalize_toc_match_text(title)
    if not normalized_title or not clean_text(source_text):
        return {}

    page_number = metadata_int_value(page)
    candidates = source_text_layout_candidates(source_text)
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_page = metadata_int_value(candidate.get("page"))
        if page_number is not None and candidate_page is not None and candidate_page != page_number:
            continue
        line_title = clean_text(candidate.get("title"))
        normalized_line = normalize_toc_match_text(line_title)
        if not normalized_line:
            continue
        if normalized_line == normalized_title or normalized_title in normalized_line:
            matches.append(candidate)

    if not matches:
        return {}

    visual_examples: list[str] = []
    small_label_examples: list[str] = []
    for match in matches[:3]:
        match_page = metadata_int_value(match.get("page"))
        match_layout = match.get("layout") if isinstance(match.get("layout"), dict) else {}
        match_y = metadata_float_value(match_layout.get("vertical_position"))
        match_line_index = metadata_int_value(match.get("line_index"))
        for neighbor in candidates:
            if neighbor is match:
                continue
            neighbor_page = metadata_int_value(neighbor.get("page"))
            if match_page is not None and neighbor_page is not None and neighbor_page != match_page:
                continue
            neighbor_layout = neighbor.get("layout") if isinstance(neighbor.get("layout"), dict) else {}
            neighbor_y = metadata_float_value(neighbor_layout.get("vertical_position"))
            neighbor_line_index = metadata_int_value(neighbor.get("line_index"))
            close_by_y = (
                match_y is not None
                and neighbor_y is not None
                and abs(neighbor_y - match_y) <= vertical_window
            )
            close_by_line = (
                match_line_index is not None
                and neighbor_line_index is not None
                and abs(neighbor_line_index - match_line_index) <= 8
            )
            if not close_by_y and not close_by_line:
                continue

            neighbor_title = clean_text(neighbor.get("title"))
            if not neighbor_title:
                continue
            location = missing_numbered_candidate_layout_text({
                "page": neighbor_page,
                "line_index": neighbor_line_index,
                "font_size": metadata_float_value(neighbor_layout.get("font_size")),
                "size_ratio": metadata_float_value(neighbor_layout.get("size_ratio")),
                "left_indent": metadata_float_value(neighbor_layout.get("left_indent")),
                "vertical_position": metadata_float_value(neighbor_layout.get("vertical_position")),
            })
            example = (
                f'{location}: "{compact_toc_example_title(neighbor_title, max_chars=90)}"'
                if location
                else f'"{compact_toc_example_title(neighbor_title, max_chars=90)}"'
            )
            if is_visual_material_caption(neighbor_title):
                if len(visual_examples) < max_examples:
                    visual_examples.append(example)
                continue
            size_ratio = metadata_float_value(neighbor_layout.get("size_ratio"))
            font_size = metadata_float_value(neighbor_layout.get("font_size"))
            if (
                size_ratio is not None
                and size_ratio <= 0.9
                and font_size is not None
                and font_size <= 9.0
                and not has_structural_title_marker(neighbor_title)
            ):
                if len(small_label_examples) < max_examples:
                    small_label_examples.append(example)

    result: dict[str, Any] = {
        "visual_caption_count": len(visual_examples),
        "small_label_count": len(small_label_examples),
    }
    if visual_examples:
        result["visual_caption_examples"] = visual_examples
    if small_label_examples:
        result["small_label_examples"] = small_label_examples
    return result


def source_evidence_has_ancillary_context(evidence: dict[str, Any]) -> bool:
    examples = list(evidence.get("exact_examples", []) or []) + list(evidence.get("embedded_examples", []) or [])
    return any(
        re.search(
            r"(^|[\s\"':])(?:사례|예시|보기|case|example|box)\s*\d+|"
            r"(^|[\s\"':])(?:표|그림|figure|fig\.|table)\s*(?:<|\d|[IVXivx]+)[\dIVXivx.:-]*|"
            r"(?:기본\s*문제|심화\s*문제|연습\s*문제|참고문헌|references?)",
            clean_text(example),
            re.IGNORECASE,
        )
        for example in examples
    )


def possible_invalid_heading_candidates(
    toc: dict[str, Any],
    source_text: str = "",
    max_candidates: int = 40,
) -> list[dict[str, Any]]:
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return []

    candidates: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = clean_text(chapter.get("chapter"))
        if not title:
            continue
        try:
            level = int(chapter.get("level", 1))
        except Exception:
            level = 1
        layout = layout_from_toc_metadata(chapter)
        if not layout and clean_text(source_text):
            layout = source_layout_for_title(title, chapter.get("page"), source_text)
        size_ratio = metadata_float_value(layout.get("size_ratio"))
        bold = 1 if metadata_int_value(layout.get("bold")) else 0
        italic = 1 if metadata_int_value(layout.get("italic")) else 0
        weak_body_style = size_ratio is not None and size_ratio <= 1.08 and not bold and not italic
        very_weak_body_style = size_ratio is not None and size_ratio <= 1.05 and not bold and not italic
        has_marker = has_structural_title_marker(title)
        level_reason = clean_text(chapter.get("level_reason"))
        restored = "completeness audit" in level_reason.lower()
        ancillary = is_ancillary_heading_candidate(title)
        evidence = source_title_line_evidence(title, chapter.get("page"), source_text) if clean_text(source_text) else {}
        embedded_only = (
            int(evidence.get("embedded_count", 0) or 0) > 0
            and int(evidence.get("exact_count", 0) or 0) == 0
        )
        ancillary_context = source_evidence_has_ancillary_context(evidence)
        visual_context = source_visual_context_for_title(title, chapter.get("page"), source_text) if clean_text(source_text) else {}
        visual_material_context = (
            int(visual_context.get("visual_caption_count", 0) or 0) > 0
            or (
                int(visual_context.get("small_label_count", 0) or 0) >= 2
                and not has_marker
            )
        )
        reasons: list[str] = []
        if ancillary:
            reasons.append("title itself looks like a case/example/table/figure/problem/reference label")
        if ancillary_context:
            reasons.append("source context looks like case/example/table/figure/problem/reference material")
        if visual_material_context and not has_marker:
            reasons.append("nearby source layout looks like figure/table/diagram material, not hierarchy")
        if embedded_only and weak_body_style:
            reasons.append("title appears embedded in longer body lines, not as a standalone body hierarchy title")
        if weak_body_style and not has_marker and level >= 4:
            reasons.append("body-size or near-body-size style without bold/italic or structural marker at a deep level")
        if restored and weak_body_style and not has_marker:
            reasons.append("restored during completeness audit despite weak markerless body-size style")
        if very_weak_body_style and not has_marker and level >= 5:
            reasons.append("level 5 markerless entry uses body-size style")

        if not reasons:
            continue

        page_number = metadata_int_value(chapter.get("page"))
        row: dict[str, Any] = {
            "title": title,
            "level": level,
            "page": page_number,
            "reasons": reasons,
            "level_reason": level_reason,
            "font_size": metadata_float_value(layout.get("font_size")),
            "size_ratio": size_ratio,
            "left_indent": metadata_float_value(layout.get("left_indent")),
            "vertical_position": metadata_float_value(layout.get("vertical_position")),
            "bold": bold,
            "italic": italic,
            "source_evidence": evidence,
        }
        if visual_context:
            row["source_visual_context"] = visual_context
        candidates.append(row)

    candidates = sorted(
        candidates,
        key=lambda item: (
            int(item.get("page") or 0),
            float(item.get("vertical_position") or 0.0),
            int(item.get("level") or 0),
        ),
    )
    return candidates[:max_candidates]


def format_gemma_possible_invalid_heading_candidates(
    toc: dict[str, Any],
    source_text: str = "",
    max_candidates: int = 40,
) -> str:
    candidates = possible_invalid_heading_candidates(
        toc,
        source_text=source_text,
        max_candidates=max_candidates,
    )
    if not candidates:
        return ""

    lines = [
        "[Observed Possible Invalid Heading Candidates]",
        "These are removal-review clues, not automatic removals. Remove a candidate only if it is not a real chapter/section/subsection title in this document hierarchy.",
    ]
    for candidate in candidates:
        location = missing_numbered_candidate_layout_text(candidate)
        location_text = f", {location}" if location else ""
        reason_text = "; ".join(clean_text(reason) for reason in candidate.get("reasons", []) if clean_text(reason))
        source_evidence = candidate.get("source_evidence") if isinstance(candidate.get("source_evidence"), dict) else {}
        evidence_parts: list[str] = []
        if source_evidence:
            evidence_parts.append(f"source_exact={source_evidence.get('exact_count', 0)}")
            evidence_parts.append(f"source_embedded={source_evidence.get('embedded_count', 0)}")
            embedded_examples = source_evidence.get("embedded_examples") or []
            if embedded_examples:
                evidence_parts.append(f"embedded_example={compact_toc_example_title(embedded_examples[0], max_chars=120)}")
        visual_context = candidate.get("source_visual_context")
        if isinstance(visual_context, dict):
            visual_examples = visual_context.get("visual_caption_examples") or []
            small_label_examples = visual_context.get("small_label_examples") or []
            if visual_examples:
                evidence_parts.append(
                    "nearby_visual_caption="
                    + compact_toc_example_title(visual_examples[0], max_chars=120)
                )
            if small_label_examples:
                evidence_parts.append(
                    "nearby_small_label="
                    + compact_toc_example_title(small_label_examples[0], max_chars=120)
                )
        evidence_text = f" / {'; '.join(evidence_parts)}" if evidence_parts else ""
        lines.append(
            f'- level={candidate.get("level")}{location_text}: title="{compact_toc_example_title(candidate.get("title"), max_chars=100)}" / reasons={reason_text}{evidence_text}'
        )
    return "\n".join(lines)


def add_source_text_numbered_candidates(
    candidates_by_path: dict[tuple[int, ...], dict[str, Any]],
    existing_paths: set[tuple[int, ...]],
    existing_titles: set[str],
    source_text: str,
) -> None:
    current_page: int | None = None
    for line_index, raw_line in enumerate(str(source_text or "").splitlines()):
        line = clean_text(raw_line)
        if not line:
            continue
        page_match = re.match(r"^\[PAGE\s+(\d+)\]$", line)
        if page_match:
            current_page = metadata_int_value(page_match.group(1))
            continue

        layout = parse_layout_line(line)
        title = clean_text(layout.get("text")) if layout else line
        add_missing_numbered_candidate(
            candidates_by_path=candidates_by_path,
            existing_paths=existing_paths,
            existing_titles=existing_titles,
            source="source_text",
            title=title,
            page=current_page,
            layout=layout,
            line_index=line_index,
        )


def add_partial_toc_numbered_candidates(
    candidates_by_path: dict[tuple[int, ...], dict[str, Any]],
    existing_paths: set[tuple[int, ...]],
    existing_titles: set[str],
    partial_tocs: list[dict[str, Any]] | None,
) -> None:
    if not partial_tocs:
        return

    for partial in partial_tocs:
        if not isinstance(partial, dict):
            continue
        toc = partial.get("toc")
        chapters = toc.get("chapters", []) if isinstance(toc, dict) else []
        if not isinstance(chapters, list):
            continue
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            layout = layout_from_toc_metadata(chapter)
            add_missing_numbered_candidate(
                candidates_by_path=candidates_by_path,
                existing_paths=existing_paths,
                existing_titles=existing_titles,
                source="partial_toc",
                title=chapter.get("chapter"),
                page=chapter.get("page"),
                chunk=partial.get("chunk"),
                layout=layout,
                line_index=metadata_int_value(chapter.get("_source_order")),
            )


def format_gemma_missing_numbered_heading_candidates(
    toc: dict[str, Any],
    source_text: str = "",
    partial_tocs: list[dict[str, Any]] | None = None,
    max_candidates: int = 30,
) -> str:
    existing_paths = toc_decimal_number_paths(toc)
    existing_titles = toc_normalized_titles(toc)
    candidates_by_path: dict[tuple[int, ...], dict[str, Any]] = {}

    if clean_text(source_text):
        add_source_text_numbered_candidates(
            candidates_by_path=candidates_by_path,
            existing_paths=existing_paths,
            existing_titles=existing_titles,
            source_text=source_text,
        )
    add_partial_toc_numbered_candidates(
        candidates_by_path=candidates_by_path,
        existing_paths=existing_paths,
        existing_titles=existing_titles,
        partial_tocs=partial_tocs,
    )

    if not candidates_by_path:
        return ""

    lines = [
        "[Observed Missing Numbered Heading Candidates]",
        "These catch omissions even when the missing body hierarchy title is the final item in a numbered sequence. They are clues, not automatic additions. Restore a candidate only if it is a real visible chapter/section/subsection title and not an existing TOC line, running header/footer, body sentence, caption, question, reference, or duplicate.",
    ]
    for _, candidate in sorted(candidates_by_path.items())[:max_candidates]:
        location = missing_numbered_candidate_layout_text(candidate)
        location_text = f", {location}" if location else ""
        chunk_text = f", chunk={candidate['chunk']}" if candidate.get("chunk") is not None else ""
        lines.append(
            f'- {candidate["source"]}{chunk_text}{location_text}: number="{candidate["number"]}", title="{compact_toc_example_title(candidate["title"], max_chars=100)}"'
        )
    return "\n".join(lines)


def format_gemma_missing_style_heading_candidates(
    toc: dict[str, Any],
    source_text: str = "",
    max_candidates: int = 40,
) -> str:
    if not clean_text(source_text):
        return ""

    existing_titles = toc_normalized_titles(toc)
    profiles = toc_heading_style_profiles(toc)
    candidates: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for candidate in source_text_layout_candidates(source_text):
        title = clean_text(candidate.get("title"))
        normalized_title = normalize_toc_match_text(title)
        if not title or normalized_title in existing_titles or normalized_title in seen_titles:
            continue
        layout = candidate.get("layout")
        if not isinstance(layout, dict):
            continue
        if not is_style_heading_candidate(title, layout, profiles):
            continue

        row = {
            "source": candidate.get("source"),
            "title": title,
            "page": candidate.get("page"),
            "line_index": candidate.get("line_index"),
            "score": style_heading_candidate_score(layout, profiles),
            "font_size": metadata_float_value(layout.get("font_size")),
            "size_ratio": metadata_float_value(layout.get("size_ratio")),
            "left_indent": metadata_float_value(layout.get("left_indent")),
            "vertical_position": metadata_float_value(layout.get("vertical_position")),
            "bold": metadata_int_value(layout.get("bold")),
            "italic": metadata_int_value(layout.get("italic")),
            "font_id": clean_text(layout.get("font_id")),
        }
        candidates.append(row)
        seen_titles.add(normalized_title)

    if not candidates:
        return ""

    candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("score") or 0.0),
            int(item.get("page") or 0),
            int(item.get("line_index") or 0),
        ),
    )[:max_candidates]
    candidates = sorted(candidates, key=lambda item: (int(item.get("page") or 0), int(item.get("line_index") or 0)))

    lines = [
        "[Observed Missing Style Heading Candidates]",
        "These catch omissions when body hierarchy titles do not use numbering. They are layout-based clues from Source Text/Layout. Restore a candidate only if its font size/ratio/bold/indent/spacing pattern clearly belongs to this document's body title hierarchy and it is not body text, a caption, question, reference, running header/footer, duplicate, or existing TOC line.",
    ]
    for candidate in candidates:
        location = missing_numbered_candidate_layout_text(candidate)
        location_text = f", {location}" if location else ""
        bold_text = f", b={candidate['bold']}" if candidate.get("bold") is not None else ""
        italic_text = f", i={candidate['italic']}" if candidate.get("italic") is not None else ""
        font_text = f", f={candidate['font_id']}" if candidate.get("font_id") else ""
        lines.append(
            f'- source_text{location_text}{bold_text}{italic_text}{font_text}: title="{compact_toc_example_title(candidate["title"], max_chars=110)}"'
        )
    return "\n".join(lines)


def format_gemma_level_style_summary(toc: dict[str, Any], max_conflicts: int = 20) -> str:
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return ""

    by_level_style: dict[int, dict[tuple[float | None, float | None], dict[str, Any]]] = {}
    by_style_level_count: dict[tuple[float | None, float | None], dict[int, int]] = {}
    chapter_rows: list[dict[str, Any]] = []

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = clean_text(chapter.get("chapter"))
        if not title:
            continue
        try:
            level = int(chapter.get("level", 1))
        except Exception:
            level = 1
        style_key = toc_chapter_style_key(chapter)
        if style_key == (None, None):
            continue
        x_value = metadata_float_value(chapter.get("_layout_x"))
        page = metadata_int_value(chapter.get("page"))

        level_styles = by_level_style.setdefault(level, {})
        style_entry = level_styles.setdefault(
            style_key,
            {
                "count": 0,
                "examples": [],
                "x_values": [],
            },
        )
        style_entry["count"] += 1
        if len(style_entry["examples"]) < 3:
            style_entry["examples"].append(title)
        if x_value is not None:
            style_entry["x_values"].append(x_value)

        style_counts = by_style_level_count.setdefault(style_key, {})
        style_counts[level] = style_counts.get(level, 0) + 1
        chapter_rows.append({
            "level": level,
            "title": title,
            "page": page,
            "style_key": style_key,
            "x": x_value,
        })

    if not by_level_style:
        return ""

    lines = ["[Observed TOC Level Style Evidence]"]
    lines.append(
        "Use this only as evidence for the correction pass. It summarizes the levels already present in the current TOC; it is not a rule table."
    )
    for level in sorted(by_level_style):
        lines.append(f"Level {level} observed styles:")
        level_styles = sorted(
            by_level_style[level].items(),
            key=lambda item: (-int(item[1].get("count", 0)), format_style_key(item[0])),
        )
        for style_key, info in level_styles[:5]:
            x_values = [float(value) for value in info.get("x_values", []) if value is not None]
            x_text = ""
            if x_values:
                x_text = f", left_indent_range={min(x_values):.1f}-{max(x_values):.1f}"
            examples = "; ".join(compact_toc_example_title(title, max_chars=60) for title in info.get("examples", []))
            lines.append(
                f"- {format_style_key(style_key)}{x_text}: count={info.get('count', 0)}, examples={examples}"
            )

    conflicts: list[str] = []
    level_one_styles = by_level_style.get(1, {})
    if len(level_one_styles) >= 2:
        dominant_level_one_style, dominant_level_one_info = max(
            level_one_styles.items(),
            key=lambda item: (
                style_font_size(item[0]),
                style_size_ratio(item[0]),
                int(item[1].get("count", 0)),
            ),
        )
        dominant_font = style_font_size(dominant_level_one_style)
        dominant_ratio = style_size_ratio(dominant_level_one_style)
        dominant_examples = "; ".join(
            compact_toc_example_title(title, max_chars=60)
            for title in dominant_level_one_info.get("examples", [])
        )
        lines.append(
            f"Level 1 dominant/highest visual style: {format_style_key(dominant_level_one_style)}, examples={dominant_examples}"
        )
        for row in chapter_rows:
            if int(row["level"]) != 1:
                continue
            style_key = row["style_key"]
            if style_key == dominant_level_one_style:
                continue
            font_ratio = style_font_size(style_key) / dominant_font if dominant_font else 1.0
            size_ratio_ratio = style_size_ratio(style_key) / dominant_ratio if dominant_ratio else 1.0
            if font_ratio > 0.82 and size_ratio_ratio > 0.82:
                continue
            page_text = f", page={row['page']}" if row.get("page") else ""
            x_text = f", x={row['x']:.1f}" if row.get("x") is not None else ""
            conflicts.append(
                f'- title="{compact_toc_example_title(row["title"], max_chars=90)}"{page_text}: current level 1 uses smaller style {format_style_key(style_key)}{x_text}, while true level 1 appears to use {format_style_key(dominant_level_one_style)}. Audit as a likely lower-level body hierarchy title unless document-specific evidence proves it is a separate top-level title.'
            )
            if len(conflicts) >= max_conflicts:
                break

    for row in chapter_rows:
        if len(conflicts) >= max_conflicts:
            break
        style_key = row["style_key"]
        current_level = int(row["level"])
        style_counts = by_style_level_count.get(style_key, {})
        if not style_counts:
            continue
        best_level, best_count = max(style_counts.items(), key=lambda item: (item[1], -item[0]))
        current_count = style_counts.get(current_level, 0)
        current_level_styles = by_level_style.get(current_level, {})
        dominant_current_count = max((int(info.get("count", 0)) for info in current_level_styles.values()), default=0)
        current_style_count = int(current_level_styles.get(style_key, {}).get("count", 0))
        if best_level == current_level and current_style_count >= dominant_current_count:
            continue
        if best_level != current_level and best_count < max(2, current_count + 1):
            continue
        page_text = f", page={row['page']}" if row.get("page") else ""
        x_text = f", x={row['x']:.1f}" if row.get("x") is not None else ""
        conflicts.append(
            f'- title="{compact_toc_example_title(row["title"], max_chars=90)}"{page_text}: current level {current_level}, style {format_style_key(style_key)}{x_text} is more consistent with observed level {best_level} (style count {best_count} vs current level count {current_count}).'
        )
        if len(conflicts) >= max_conflicts:
            break

    if conflicts:
        lines.append("Potential style-level conflicts to audit carefully:")
        lines.extend(conflicts)

    return "\n".join(lines)


def build_gemma_level_review_prompt(
    base_prompt: str,
    pdf_name: str,
    toc: dict[str, Any],
    max_depth: int,
    stage_label: str,
    source_text: str = "",
    partial_tocs: list[dict[str, Any]] | None = None,
    level_reference_text: str = "",
    document_style_text: str = "",
) -> str:
    toc_json = json.dumps(compact_toc_for_gemma_level_review(toc), ensure_ascii=False, indent=2)
    style_summary = format_gemma_level_style_summary(toc)
    style_block = f"\n\n{style_summary}" if clean_text(style_summary) else ""
    invalid_heading_summary = format_gemma_possible_invalid_heading_candidates(
        toc,
        source_text=source_text,
    )
    invalid_heading_block = f"\n\n{invalid_heading_summary}" if clean_text(invalid_heading_summary) else ""
    reference_block = f"\n\n{level_reference_text}" if clean_text(level_reference_text) else ""
    document_style_block = f"\n\n{document_style_text}" if clean_text(document_style_text) else ""

    return f"""
{base_prompt}

[Gemma TOC Level Verification and Correction]
Review the [Current TOC JSON] and return one corrected TOC JSON object.
Rules:
- Correct level values using visible hierarchy and _layout_* metadata.
- Remove duplicates and clearly invalid non-title entries.
- Do not add new entries unless the current JSON already contains a clear duplicate/merge error.
- Keep only body hierarchy titles; remove any remaining existing TOC listing, repeated header/footer, caption, question, reference, or body sentence.
- Preserve correct title text, page, order, and _layout_* metadata.
- Use only levels 1 to {max_depth}. Return compact valid JSON only.
{document_style_block}
{reference_block}
{style_block}
{invalid_heading_block}

[PDF File Name]
{pdf_name}

[Review Stage]
{stage_label}

[Current TOC JSON]
{toc_json}
""".strip()


def build_gemma_json_retry_prompt(base_prompt: str, error: Exception, raw_text: str) -> str:
    raw_preview = clean_text(raw_text)[:500]
    return (
        base_prompt
        + "\n\n[Retry]\n"
        + f"Previous response was not valid JSON: {type(error).__name__}: {error}\n"
        + (f"Preview: {raw_preview}\n" if raw_preview else "")
        + 'Return one valid JSON object only. If empty, use {"title":"Document title","chapters":[]}.'
    )


def request_gemma_json_text(
    model: str,
    prompt_text: str,
    args: argparse.Namespace,
    label: str,
) -> tuple[dict[str, Any], str]:
    retry_count = max(1, int(args.ai_retries))
    last_error: Exception | None = None
    last_raw_text = ""

    for parse_attempt in range(1, retry_count + 1):
        request_prompt = prompt_text
        if parse_attempt > 1 and last_error is not None:
            print(
                f"  Retrying {label} after JSON parse failure {parse_attempt}/{retry_count}: {last_error}",
                file=sys.stderr,
                flush=True,
            )
            request_prompt = build_gemma_json_retry_prompt(prompt_text, last_error, last_raw_text)

        response = with_retry(
            label=f"{label} generation",
            func=lambda request_prompt=request_prompt: generate_gemma_once(
                model=model,
                prompt=request_prompt,
                args=args,
            ),
            max_retries=args.ai_retries,
            base_delay=args.ai_retry_base_delay,
        )
        if isinstance(response, dict) and isinstance(response.get("_runtime"), dict):
            update_processing_metadata(args, **response["_runtime"])

        raw_text = gemma_response_text(response)

        try:
            return parse_json_response_text(raw_text, provider="gemma"), raw_text
        except Exception as error:
            attach_error_debug(
                error,
                provider="gemma",
                model=model,
                label=label,
                parse_attempt=parse_attempt,
                retry_count=retry_count,
                raw_response=raw_text,
                raw_response_chars=len(raw_text),
            )
            last_error = error
            last_raw_text = raw_text
            if parse_attempt >= retry_count:
                raise

    if last_error:
        raise last_error
    raise RuntimeError(f"{label} failed")


def toc_chapter_pages(toc: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return pages

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        try:
            pages.append(int(chapter.get("page", 0)))
        except Exception:
            continue
    return [page for page in pages if page > 0]


def partial_toc_pages(partial_tocs: list[dict[str, Any]]) -> list[int]:
    pages: list[int] = []
    for partial in partial_tocs:
        toc = partial.get("toc") if isinstance(partial, dict) else None
        if isinstance(toc, dict):
            pages.extend(toc_chapter_pages(toc))
    return pages


def partial_toc_chapter_count(partial_tocs: list[dict[str, Any]]) -> int:
    count = 0
    for partial in partial_tocs:
        toc = partial.get("toc") if isinstance(partial, dict) else None
        chapters = toc.get("chapters", []) if isinstance(toc, dict) else []
        if isinstance(chapters, list):
            count += sum(1 for chapter in chapters if isinstance(chapter, dict))
    return count


def compact_toc_for_gemma_merge(toc: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "title": clean_text(toc.get("title")),
        "chapters": [],
    }
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return compact

    compact_chapters: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        item = {
            "level": chapter.get("level"),
            "chapter": clean_text(chapter.get("chapter")),
            "page": chapter.get("page"),
        }
        if item["chapter"]:
            compact_chapters.append(item)
    compact["chapters"] = compact_chapters
    return compact


def compact_partial_toc_for_gemma_merge(partial: dict[str, Any]) -> dict[str, Any]:
    toc = partial.get("toc") if isinstance(partial, dict) else None
    return {
        "chunk": partial.get("chunk") if isinstance(partial, dict) else None,
        "start_page": partial.get("start_page") if isinstance(partial, dict) else None,
        "end_page": partial.get("end_page") if isinstance(partial, dict) else None,
        "toc": compact_toc_for_gemma_merge(toc if isinstance(toc, dict) else {}),
    }


def compact_partial_tocs_for_gemma_merge(partial_tocs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_partial_toc_for_gemma_merge(partial) for partial in partial_tocs]


def compact_toc_for_gemma_level_review(toc: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "title": clean_text(toc.get("title")),
        "chapters": [],
    }
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return compact

    metadata_keys = (
        "level_reason",
        "_layout_page",
        "_layout_x",
        "_layout_y",
        "_layout_font_size",
        "_layout_size_ratio",
        "_layout_text",
        "_source_order",
        "_local_merge_order",
    )
    compact_chapters: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        item: dict[str, Any] = {
            "level": chapter.get("level"),
            "chapter": clean_text(chapter.get("chapter")),
            "page": chapter.get("page"),
        }
        if not item["chapter"]:
            continue
        for key in metadata_keys:
            value = chapter.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = clean_text(value)
                if not value:
                    continue
            item[key] = value
        compact_chapters.append(item)
    compact["chapters"] = compact_chapters
    return compact


def compact_partial_toc_for_gemma_level_review(partial: dict[str, Any]) -> dict[str, Any]:
    toc = partial.get("toc") if isinstance(partial, dict) else None
    return {
        "chunk": partial.get("chunk") if isinstance(partial, dict) else None,
        "start_page": partial.get("start_page") if isinstance(partial, dict) else None,
        "end_page": partial.get("end_page") if isinstance(partial, dict) else None,
        "toc": compact_toc_for_gemma_level_review(toc if isinstance(toc, dict) else {}),
    }


def compact_partial_tocs_for_gemma_level_review(partial_tocs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_partial_toc_for_gemma_level_review(partial) for partial in partial_tocs]


def toc_level_by_match_key(toc: dict[str, Any]) -> dict[tuple[str, int], int]:
    level_by_key: dict[tuple[str, int], int] = {}
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return level_by_key

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = normalize_toc_match_text(chapter.get("chapter"))
        if not title:
            continue
        try:
            page = int(chapter.get("page", 1))
            level = int(chapter.get("level", 1))
        except Exception:
            continue
        level_by_key[(title, page)] = level
    return level_by_key


def count_toc_level_changes(before: dict[str, Any], after: dict[str, Any]) -> int:
    before_levels = toc_level_by_match_key(before)
    after_levels = toc_level_by_match_key(after)
    return sum(
        1
        for key, after_level in after_levels.items()
        if key in before_levels and before_levels[key] != after_level
    )


def toc_chapter_by_match_key(toc: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    chapter_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return chapter_by_key

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = normalize_toc_match_text(chapter.get("chapter"))
        if not title:
            continue
        try:
            page = int(chapter.get("page", 1))
        except Exception:
            page = 1
        chapter_by_key[(title, page)] = chapter
    return chapter_by_key


def toc_review_chapter_snapshot(chapter: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "level",
        "chapter",
        "page",
        "level_reason",
        "_layout_page",
        "_layout_x",
        "_layout_y",
        "_layout_font_size",
        "_layout_size_ratio",
        "_layout_text",
        "_source_order",
        "_local_merge_order",
    )
    snapshot: dict[str, Any] = {}
    for key in keys:
        if key not in chapter:
            continue
        value = chapter.get(key)
        if isinstance(value, str):
            value = clean_text(value)
        if value is not None and value != "":
            snapshot[key] = value
    return snapshot


def toc_review_change_sort_key(item: dict[str, Any]) -> tuple[int, int, float, str]:
    chapter = item.get("after") if isinstance(item.get("after"), dict) else item.get("before")
    if not isinstance(chapter, dict):
        chapter = item
    page = metadata_int_value(chapter.get("page")) or metadata_int_value(item.get("page")) or 0
    source_order = metadata_int_value(chapter.get("_source_order")) or metadata_int_value(chapter.get("_local_merge_order")) or 0
    layout_y = metadata_float_value(chapter.get("_layout_y")) or 0.0
    title = clean_text(chapter.get("chapter") or item.get("chapter"))
    return page, source_order, layout_y, title


def build_toc_review_change_report(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_by_key = toc_chapter_by_match_key(before)
    after_by_key = toc_chapter_by_match_key(after)
    before_keys = set(before_by_key)
    after_keys = set(after_by_key)

    level_changes: list[dict[str, Any]] = []
    for key in sorted(before_keys & after_keys):
        before_chapter = before_by_key[key]
        after_chapter = after_by_key[key]
        try:
            before_level = int(before_chapter.get("level", 1))
            after_level = int(after_chapter.get("level", 1))
        except Exception:
            continue
        if before_level == after_level:
            continue
        level_changes.append({
            "chapter": clean_text(after_chapter.get("chapter") or before_chapter.get("chapter")),
            "page": metadata_int_value(after_chapter.get("page")) or metadata_int_value(before_chapter.get("page")),
            "before_level": before_level,
            "after_level": after_level,
            "before_level_reason": clean_text(before_chapter.get("level_reason")),
            "after_level_reason": clean_text(after_chapter.get("level_reason")),
            "before": toc_review_chapter_snapshot(before_chapter),
            "after": toc_review_chapter_snapshot(after_chapter),
        })

    added_entries = [
        {"after": toc_review_chapter_snapshot(after_by_key[key])}
        for key in after_keys - before_keys
    ]
    removed_entries = [
        {"before": toc_review_chapter_snapshot(before_by_key[key])}
        for key in before_keys - after_keys
    ]

    level_changes.sort(key=toc_review_change_sort_key)
    added_entries.sort(key=toc_review_change_sort_key)
    removed_entries.sort(key=toc_review_change_sort_key)
    return {
        "level_change_count": len(level_changes),
        "added_count": len(added_entries),
        "removed_count": len(removed_entries),
        "level_changes": level_changes,
        "added_entries": added_entries,
        "removed_entries": removed_entries,
    }


def toc_review_change_report_has_changes(change_report: dict[str, Any]) -> bool:
    return bool(
        int(change_report.get("level_change_count", 0) or 0)
        or int(change_report.get("added_count", 0) or 0)
        or int(change_report.get("removed_count", 0) or 0)
    )


def record_gemma_level_review_change_report(
    args: argparse.Namespace,
    *,
    stage_label: str,
    input_chapters: int,
    output_chapters: int,
    change_report: dict[str, Any],
) -> bool:
    if not toc_review_change_report_has_changes(change_report):
        return False

    reports = getattr(args, "_gemma_level_review_change_reports", None)
    if not isinstance(reports, list):
        reports = []

    reports.append({
        "review_stage": stage_label,
        "input_chapters": input_chapters,
        "output_chapters": output_chapters,
        **change_report,
    })
    setattr(args, "_gemma_level_review_change_reports", reports)
    update_processing_metadata(
        args,
        gemma_level_review_change_report_count=len(reports),
        gemma_level_review_change_summary_pending=True,
    )
    return True


def write_gemma_level_review_change_summary(
    *,
    args: argparse.Namespace,
    pdf_path: Path,
    provider: str,
    model: str,
) -> Path | None:
    existing_file = clean_text(getattr(args, "_gemma_level_review_change_summary_file", ""))
    if existing_file:
        return Path(existing_file)

    reports = getattr(args, "_gemma_level_review_change_reports", None)
    if not isinstance(reports, list) or not reports:
        return None

    total_level_changes = sum(int(report.get("level_change_count", 0) or 0) for report in reports)
    total_added = sum(int(report.get("added_count", 0) or 0) for report in reports)
    total_removed = sum(int(report.get("removed_count", 0) or 0) for report in reports)
    summary = {
        "review_count": len(reports),
        "level_change_count": total_level_changes,
        "added_count": total_added,
        "removed_count": total_removed,
        "reviews": reports,
    }
    summary_file = write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider=provider,
        model=model,
        stage="level_review_changes_summary",
        payload=summary,
    )
    if summary_file is not None:
        setattr(args, "_gemma_level_review_change_summary_file", str(summary_file))

    update_processing_metadata(
        args,
        gemma_level_review_change_summary={
            "review_count": len(reports),
            "level_change_count": total_level_changes,
            "added_count": total_added,
            "removed_count": total_removed,
            "file": str(summary_file) if summary_file is not None else None,
        },
        gemma_level_review_change_file=str(summary_file) if summary_file is not None else None,
        gemma_level_review_change_summary_pending=False,
    )
    return summary_file


def write_gemma_invalid_heading_candidates_file(
    *,
    args: argparse.Namespace,
    pdf_path: Path,
    provider: str,
    model: str,
    stage_label: str,
    toc: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Path | None:
    if not candidates:
        return None

    chapters = toc.get("chapters", [])
    candidate_file = write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider=provider,
        model=model,
        stage="possible_invalid_heading_candidates",
        chunk=stage_label,
        payload={
            "review_stage": stage_label,
            "input_chapters": len(chapters) if isinstance(chapters, list) else 0,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
    )

    candidate_files = getattr(args, "_gemma_invalid_heading_candidate_files", None)
    if not isinstance(candidate_files, list):
        candidate_files = []
    if candidate_file is not None:
        candidate_files.append(str(candidate_file))
        setattr(args, "_gemma_invalid_heading_candidate_files", candidate_files)

    update_processing_metadata(
        args,
        gemma_invalid_heading_candidate_file_count=len(candidate_files),
        gemma_invalid_heading_candidate_files=candidate_files,
    )
    return candidate_file


def restore_missing_toc_metadata_from_source(toc: dict[str, Any], source_toc: dict[str, Any]) -> dict[str, Any]:
    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    source_chapters = source_toc.get("chapters", [])
    if isinstance(source_chapters, list):
        for chapter in source_chapters:
            if not isinstance(chapter, dict):
                continue
            title = normalize_toc_match_text(chapter.get("chapter"))
            if not title:
                continue
            try:
                page = int(chapter.get("page", 1))
            except Exception:
                page = 1
            source_by_key[(title, page)] = chapter

    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        return toc

    metadata_keys = (
        "level_reason",
        "_layout_page",
        "_layout_x",
        "_layout_y",
        "_layout_font_size",
        "_layout_size_ratio",
        "_layout_text",
        "_source_order",
        "_local_merge_order",
    )
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = normalize_toc_match_text(chapter.get("chapter"))
        if not title:
            continue
        try:
            page = int(chapter.get("page", 1))
        except Exception:
            page = 1
        source_chapter = source_by_key.get((title, page))
        if not source_chapter:
            continue
        for key in metadata_keys:
            if key in chapter:
                continue
            if key == "level_reason":
                try:
                    if int(source_chapter.get("level", 1)) != int(chapter.get("level", 1)):
                        continue
                except Exception:
                    continue
            value = source_chapter.get(key)
            if value is not None:
                chapter[key] = value
    return toc


def record_gemma_level_review_status(args: argparse.Namespace, status: dict[str, Any]) -> None:
    stats = dict(getattr(args, "_gemma_level_review_stats", None) or {})
    stats.setdefault("enabled", True)
    stats["attempts"] = int(stats.get("attempts", 0)) + 1
    if status.get("success"):
        stats["success"] = int(stats.get("success", 0)) + 1
    else:
        stats["failed"] = int(stats.get("failed", 0)) + 1
    stats["level_changes"] = int(stats.get("level_changes", 0)) + int(status.get("level_changes", 0) or 0)
    setattr(args, "_gemma_level_review_stats", stats)
    update_processing_metadata(args, gemma_level_review=stats)


def review_gemma_toc_levels(
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    prompt: str,
    toc: dict[str, Any],
    stage_label: str,
    source_text: str = "",
    partial_tocs: list[dict[str, Any]] | None = None,
    level_reference_text: str = "",
    document_style_text: str = "",
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if normalize_gemma_level_verify_scope(
        getattr(args, "gemma_level_verify_scope", DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE)
    ) == "none":
        return toc, "", {"enabled": False, "success": False, "skipped": True}

    label = f"Gemma level review({stage_label})"
    invalid_heading_candidates = possible_invalid_heading_candidates(
        toc,
        source_text=source_text,
    )
    invalid_heading_candidates_file = write_gemma_invalid_heading_candidates_file(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
        stage_label=stage_label,
        toc=toc,
        candidates=invalid_heading_candidates,
    )
    review_prompt = build_gemma_level_review_prompt(
        base_prompt=prompt,
        pdf_name=pdf_path.name,
        toc=toc,
        max_depth=args.max_depth,
        stage_label=stage_label,
        source_text=source_text,
        partial_tocs=partial_tocs,
        level_reference_text=level_reference_text,
        document_style_text=document_style_text,
    )
    print(f"  {label} started", flush=True)
    try:
        parsed, raw_text = request_gemma_json_text(
            model=model,
            prompt_text=review_prompt,
            args=args,
            label=label,
        )
        reviewed_toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        restore_missing_toc_metadata_from_source(reviewed_toc, toc)
        change_report = build_toc_review_change_report(toc, reviewed_toc)
        level_changes = int(change_report.get("level_change_count", 0) or 0)
        input_chapters = len(toc.get("chapters", []) or [])
        output_chapters = len(reviewed_toc.get("chapters", []) or [])
        level_change_report_recorded = record_gemma_level_review_change_report(
            args,
            stage_label=stage_label,
            input_chapters=input_chapters,
            output_chapters=output_chapters,
            change_report=change_report,
        )
        status = {
            "enabled": True,
            "success": True,
            "stage": stage_label,
            "input_chapters": input_chapters,
            "output_chapters": output_chapters,
            "level_changes": level_changes,
            "added_entries": int(change_report.get("added_count", 0) or 0),
            "removed_entries": int(change_report.get("removed_count", 0) or 0),
            "level_change_report_recorded": level_change_report_recorded,
            "possible_invalid_heading_candidate_count": len(invalid_heading_candidates),
            "possible_invalid_heading_candidates": invalid_heading_candidates,
            "possible_invalid_heading_candidates_file": str(invalid_heading_candidates_file) if invalid_heading_candidates_file is not None else None,
        }
        record_gemma_level_review_status(args, status)
        print(
            f"  {label} completed: {status['output_chapters']} chapters / level changes {level_changes}",
            flush=True,
        )
        return reviewed_toc, raw_text, status
    except Exception as error:
        status = {
            "enabled": True,
            "success": False,
            "stage": stage_label,
            "error": message_of(error),
            "level_changes": 0,
            "possible_invalid_heading_candidate_count": len(invalid_heading_candidates),
            "possible_invalid_heading_candidates": invalid_heading_candidates,
            "possible_invalid_heading_candidates_file": str(invalid_heading_candidates_file) if invalid_heading_candidates_file is not None else None,
        }
        record_gemma_level_review_status(args, status)
        print(f"  {label} failed; keeping previous TOC: {error}", file=sys.stderr, flush=True)
        return toc, "", status


def split_partial_tocs_for_gemma_merge(
    partial_tocs: list[dict[str, Any]],
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    max_chars = max(10000, int(max_chars))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def batch_chars(batch: list[dict[str, Any]]) -> int:
        compact_batch = compact_partial_tocs_for_gemma_merge(batch)
        return len(json.dumps(compact_batch, ensure_ascii=False))

    for partial in partial_tocs:
        candidate = [*current, partial]
        if current and batch_chars(candidate) > max_chars:
            batches.append(current)
            current = [partial]
        else:
            current = candidate

    if current:
        batches.append(current)
    return batches


def partial_toc_page_bounds(partial_tocs: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    pages = partial_toc_pages(partial_tocs)
    if pages:
        return min(pages), max(pages)

    starts: list[int] = []
    ends: list[int] = []
    for partial in partial_tocs:
        if not isinstance(partial, dict):
            continue
        try:
            starts.append(int(partial.get("start_page")))
        except Exception:
            pass
        try:
            ends.append(int(partial.get("end_page")))
        except Exception:
            pass
    if starts or ends:
        values = [*starts, *ends]
        return min(values), max(values)
    return None, None


def gemma_merge_appears_tail_truncated(toc: dict[str, Any], source_partials: list[dict[str, Any]]) -> bool:
    source_pages = partial_toc_pages(source_partials)
    output_pages = toc_chapter_pages(toc)
    if not source_pages or not output_pages:
        return False

    source_max = max(source_pages)
    output_max = max(output_pages)
    if output_max >= source_max:
        return False

    missing_tail_pages = [page for page in source_pages if page > output_max + 3]
    if source_max - output_max >= 10 and len(missing_tail_pages) >= 3:
        return True

    source_count = partial_toc_chapter_count(source_partials)
    output_count = len(toc.get("chapters", []) or [])
    return source_count >= 50 and output_count < max(10, source_count // 5) and source_max - output_max >= 5


def merge_gemma_partial_tocs_once(
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    prompt: str,
    partial_tocs: list[dict[str, Any]],
    label: str,
    document_style_text: str = "",
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    compact_partials = compact_partial_tocs_for_gemma_merge(partial_tocs)
    merge_prompt = build_gemma_merge_prompt(
        prompt,
        pdf_path.name,
        compact_partials,
        args.max_depth,
        include_level_reason=False,
        document_style_text=document_style_text,
    )
    parsed, raw_text = request_gemma_json_text(
        model=model,
        prompt_text=merge_prompt,
        args=args,
        label=label,
    )
    toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
    restore_level_reasons_from_partials(toc, partial_tocs)
    toc = sort_toc_with_restored_layout(
        toc,
        partial_tocs=partial_tocs,
        fallback_title=pdf_path.stem,
        max_depth=args.max_depth,
    )

    source_min_page, source_max_page = partial_toc_page_bounds(partial_tocs)
    output_pages = toc_chapter_pages(toc)
    output_max_page = max(output_pages) if output_pages else None
    status = {
        "label": label,
        "source_chapters": partial_toc_chapter_count(partial_tocs),
        "output_chapters": len(toc.get("chapters", []) or []),
        "source_min_page": source_min_page,
        "source_max_page": source_max_page,
        "output_max_page": output_max_page,
        "used_local_fallback": False,
    }

    if gemma_merge_appears_tail_truncated(toc, partial_tocs):
        print(
            f"  {label} appears tail-truncated; using local merge for this batch",
            file=sys.stderr,
            flush=True,
        )
        toc = merge_partial_tocs_locally(partial_tocs, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        restore_level_reasons_from_partials(toc, partial_tocs)
        output_pages = toc_chapter_pages(toc)
        status.update({
            "output_chapters": len(toc.get("chapters", []) or []),
            "output_max_page": max(output_pages) if output_pages else None,
            "used_local_fallback": True,
            "fallback_reason": "gemma_merge_tail_truncated",
        })

    return toc, raw_text, status


def merge_gemma_partial_tocs_batched(
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    prompt: str,
    partial_tocs: list[dict[str, Any]],
    document_style_text: str = "",
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    batch_chars = max(10000, int(getattr(args, "gemma_merge_batch_chars", DEFAULT_GEMMA_MERGE_BATCH_CHARS)))
    batches = split_partial_tocs_for_gemma_merge(partial_tocs, max_chars=batch_chars)
    batch_partials: list[dict[str, Any]] = []
    batch_raw_parts: list[dict[str, Any]] = []
    print(
        f"  Gemma chunked TOC batched merge started: {model} / batches={len(batches)} / batch_chars={batch_chars}",
        flush=True,
    )

    for index, batch in enumerate(batches, start=1):
        label = f"Gemma chunked TOC merge batch {index}/{len(batches)}({model})"
        start_page, end_page = partial_toc_page_bounds(batch)
        print(
            f"  Gemma merge batch {index}/{len(batches)} request: pages {start_page}-{end_page}",
            flush=True,
        )
        try:
            batch_toc, raw_text, status = merge_gemma_partial_tocs_once(
                model=model,
                pdf_path=pdf_path,
                args=args,
                prompt=prompt,
                partial_tocs=batch,
                label=label,
                document_style_text=document_style_text,
            )
        except Exception as error:
            print(
                f"  Gemma merge batch {index}/{len(batches)} failed; using local merge for this batch: {error}",
                file=sys.stderr,
                flush=True,
            )
            batch_toc = merge_partial_tocs_locally(batch, fallback_title=pdf_path.stem, max_depth=args.max_depth)
            restore_level_reasons_from_partials(batch_toc, batch)
            raw_text = json.dumps(
                {
                    "mode": "local_merge_after_gemma_batch_failure",
                    "error": message_of(error),
                    "batch": index,
                    "title": batch_toc.get("title"),
                    "chapters": len(batch_toc.get("chapters", []) or []),
                },
                ensure_ascii=False,
                indent=2,
            )
            status = {
                "label": label,
                "source_chapters": partial_toc_chapter_count(batch),
                "output_chapters": len(batch_toc.get("chapters", []) or []),
                "source_min_page": start_page,
                "source_max_page": end_page,
                "output_max_page": max(toc_chapter_pages(batch_toc)) if toc_chapter_pages(batch_toc) else None,
                "used_local_fallback": True,
                "fallback_reason": "gemma_batch_merge_error",
            }

        batch_raw_parts.append({
            **status,
            "raw_response": raw_text,
        })
        batch_partials.append({
            "chunk": f"merge_batch_{index}",
            "start_page": start_page,
            "end_page": end_page,
            "toc": batch_toc,
        })
        write_stage_file(
            args=args,
            pdf_path=pdf_path,
            provider="gemma",
            model=model,
            stage="gemma_merge_batch_completed",
            chunk=index,
            payload={
                "batch": index,
                "batch_count": len(batches),
                "start_page": start_page,
                "end_page": end_page,
                "status": status,
                "toc": batch_toc,
                "raw_response": raw_text,
            },
        )
        print(
            f"  Gemma merge batch {index} completed: {len(batch_toc.get('chapters', []) or [])} chapters",
            flush=True,
        )

    if len(batch_partials) == 1:
        final_toc = batch_partials[0]["toc"]
    else:
        final_toc = merge_partial_tocs_locally(batch_partials, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        restore_level_reasons_from_partials(final_toc, batch_partials)
        restore_level_reasons_from_partials(final_toc, partial_tocs)

    final_raw_text = json.dumps(
        {
            "mode": "gemma_batched_merge_with_local_final_stitch",
            "batch_chars": batch_chars,
            "batch_count": len(batches),
            "final_chapters": len(final_toc.get("chapters", []) or []),
            "batches": batch_raw_parts,
        },
        ensure_ascii=False,
        indent=2,
    )
    metadata = {
        "gemma_merge_strategy": "batched",
        "gemma_merge_batch_chars": batch_chars,
        "gemma_merge_batch_count": len(batches),
        "gemma_merge_batch_local_fallback_count": sum(
            1 for item in batch_raw_parts if item.get("used_local_fallback")
        ),
    }
    return final_toc, final_raw_text, metadata


def process_gemma_text_chunk(
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    prompt: str,
    chunk: dict[str, Any],
    total_chunks: int,
    raw_parts: list[dict[str, Any]],
    level_reference: dict[int, list[dict[str, Any]]] | None = None,
    document_style_text: str = "",
    depth: int = 0,
) -> list[dict[str, Any]]:
    chunk_label = str(chunk["index"])
    use_level_reference = bool(
        getattr(
            args,
            "_gemma_chunk_level_reference_enabled",
            getattr(args, "gemma_level_reference", DEFAULT_GEMMA_LEVEL_REFERENCE),
        )
    )
    level_reference_text = format_toc_level_reference(level_reference or {}) if use_level_reference else ""
    chunk_prompt = build_gemma_chunk_prompt(
        prompt,
        pdf_path.name,
        chunk,
        total_chunks,
        level_reference_text=level_reference_text,
        document_style_text=document_style_text,
    )
    print(
        f"  Gemma chunk {chunk_label}/{total_chunks} request: pages {chunk['start_page']}-{chunk['end_page']}",
        flush=True,
    )
    input_chunk_file = write_input_chunk(
        provider="gemma",
        model=model,
        pdf_path=pdf_path,
        args=args,
        chunk=chunk,
        total_chunks=total_chunks,
    )
    if input_chunk_file is not None:
        print(f"  Gemma input chunk {chunk_label} saved: {input_chunk_file}", flush=True)
    write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
        stage="chunk_input_ready",
        chunk=chunk_label,
        payload={
            "total_chunks": total_chunks,
            "start_page": chunk.get("start_page"),
            "end_page": chunk.get("end_page"),
            "input_chars": len(str(chunk.get("text") or "")),
            "input_chunk_file": str(input_chunk_file) if input_chunk_file is not None else None,
        },
    )

    try:
        parsed, raw_text = request_gemma_json_text(
            model=model,
            prompt_text=chunk_prompt,
            args=args,
            label=f"Gemma chunk {chunk_label}/{total_chunks}",
        )
        partial_toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        add_level_reasons_to_toc(
            partial_toc,
            chunk_text=str(chunk.get("text") or ""),
            level_reference=level_reference if use_level_reference else None,
        )
        write_stage_file(
            args=args,
            pdf_path=pdf_path,
            provider="gemma",
            model=model,
            stage="chunk_toc_generated",
            chunk=chunk_label,
            payload={
                "total_chunks": total_chunks,
                "start_page": chunk.get("start_page"),
                "end_page": chunk.get("end_page"),
                "toc": partial_toc,
                "raw_response": raw_text,
            },
        )
        level_review_raw_text = ""
        level_review_status: dict[str, Any] = {}
        if gemma_level_verify_enabled(args, "chunk"):
            partial_toc, level_review_raw_text, level_review_status = review_gemma_toc_levels(
                model=model,
                pdf_path=pdf_path,
                args=args,
                prompt=prompt,
                toc=partial_toc,
                stage_label=f"chunk {chunk_label}/{total_chunks}",
                source_text=str(chunk.get("text") or ""),
                level_reference_text=level_reference_text,
                document_style_text=document_style_text,
            )
            write_stage_file(
                args=args,
                pdf_path=pdf_path,
                provider="gemma",
                model=model,
                stage="chunk_level_reviewed",
                chunk=chunk_label,
                payload={
                    "total_chunks": total_chunks,
                    "start_page": chunk.get("start_page"),
                    "end_page": chunk.get("end_page"),
                    "level_review": level_review_status,
                    "toc": partial_toc,
                    "raw_response": level_review_raw_text,
                },
            )
        raw_part = {
            "chunk": chunk["index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "raw_response": raw_text,
        }
        if level_review_status:
            raw_part["level_review"] = level_review_status
        if level_review_raw_text:
            raw_part["level_review_raw_response"] = level_review_raw_text
        raw_parts.append(raw_part)
        if use_level_reference and level_reference is not None:
            update_toc_level_reference(
                level_reference,
                partial_toc,
                max_depth=args.max_depth,
                chunk_text=str(chunk.get("text") or ""),
            )
        print(f"  Gemma chunk {chunk_label} completed: {len(partial_toc.get('chapters', []))} chapters", flush=True)
        return [{
            "chunk": chunk["index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "toc": partial_toc,
        }]
    except Exception as error:
        if is_cuda_out_of_memory_error(error):
            empty_torch_cuda_cache()

        if is_configuration_error(error):
            raw_parts.append({
                "chunk": chunk["index"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "error": message_of(error),
            })
            raise

        min_chunk_chars = max(5000, int(getattr(args, "gemma_text_min_chunk_chars", DEFAULT_GEMMA_TEXT_MIN_CHUNK_CHARS)))
        if is_json_parse_error(error):
            min_chunk_chars = 1500
        max_depth = 6 if is_json_parse_error(error) else 4
        if len(str(chunk.get("text") or "")) <= min_chunk_chars or depth >= max_depth:
            raw_parts.append({
                "chunk": chunk["index"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "error": message_of(error),
            })
            raise

        subchunk_chars = max(min_chunk_chars, len(str(chunk.get("text") or "")) // 2)
        subchunks = split_claude_chunk_text(chunk, max_chars=subchunk_chars)
        if len(subchunks) <= 1:
            raise

        print(
            f"  Gemma chunk {chunk_label} failed; retrying with {len(subchunks)} subchunks: {error}",
            file=sys.stderr,
            flush=True,
        )
        write_stage_file(
            args=args,
            pdf_path=pdf_path,
            provider="gemma",
            model=model,
            stage="chunk_split_retry",
            chunk=chunk_label,
            payload={
                "start_page": chunk.get("start_page"),
                "end_page": chunk.get("end_page"),
                "input_chars": len(str(chunk.get("text") or "")),
                "subchunk_count": len(subchunks),
                "error": message_of(error),
            },
        )
        partials: list[dict[str, Any]] = []
        for subchunk in subchunks:
            partials.extend(
                process_gemma_text_chunk(
                    model=model,
                    pdf_path=pdf_path,
                    args=args,
                    prompt=prompt,
                    chunk=subchunk,
                    total_chunks=total_chunks,
                    raw_parts=raw_parts,
                    level_reference=level_reference,
                    document_style_text=document_style_text,
                    depth=depth + 1,
                )
            )
        return partials


def generate_toc_from_pdf_gemma_text(
    model: str,
    pdf_path: Path,
    args: argparse.Namespace,
    prompt: str,
) -> tuple[dict[str, Any], str, str]:
    print("  Gemma PDF text/layout extraction started...", flush=True)
    raw_pages, extraction_metadata = extract_gemma_pdf_pages(pdf_path, args)
    raw_extracted_chars = sum(len(text) for _, text in raw_pages)
    extraction_mode = extraction_metadata.get("gemma_extraction_mode", "text")
    pages = list(raw_pages)
    extracted_chars = sum(len(text) for _, text in pages)
    full_text = gemma_full_text_from_pages(pages)
    document_style_summary = build_gemma_document_style_reference(
        pages,
        enabled=bool(getattr(args, "gemma_document_style_reference", DEFAULT_GEMMA_DOCUMENT_STYLE_REFERENCE)),
        max_clusters=max(1, int(getattr(args, "gemma_document_style_max_clusters", DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_CLUSTERS))),
        max_examples=max(1, int(getattr(args, "gemma_document_style_max_examples", DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_EXAMPLES))),
    )
    setattr(args, "_gemma_document_style_reference", document_style_summary)
    document_style_text = str(document_style_summary.get("prompt_text") or "").strip()
    print(
        (
            f"  Gemma PDF extraction completed: {len(raw_pages)} pages, {raw_extracted_chars} chars, "
            f"mode={extraction_mode}, selected_pages={len(pages)}"
        ),
        flush=True,
    )
    update_processing_metadata(
        args,
        input_mode="extracted_pdf_layout_text" if extraction_mode == "layout" else "extracted_pdf_text",
        raw_extracted_pages=len(raw_pages),
        raw_extracted_chars=raw_extracted_chars,
        extracted_pages=len(pages),
        extracted_chars=extracted_chars,
        gemma_document_style_reference=bool(document_style_summary.get("enabled")),
        gemma_document_style_reference_available=bool(document_style_summary.get("available")),
        gemma_document_style_cluster_count=document_style_summary.get("style_cluster_count", 0),
        gemma_document_style_title_like_example_count=len(document_style_summary.get("title_like_examples", []) or []),
        **extraction_metadata,
    )
    update_processing_metadata(
        args,
        gemma_level_verify=normalize_gemma_level_verify_scope(
            getattr(args, "gemma_level_verify_scope", DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE)
        ) != "none",
        gemma_level_verify_scope=normalize_gemma_level_verify_scope(
            getattr(args, "gemma_level_verify_scope", DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE)
        ),
    )
    write_parsed_pdf_text(
        pdf_path=pdf_path,
        args=args,
        pages=raw_pages,
        extraction_metadata=extraction_metadata,
    )
    write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
        stage="01_pdf_extracted",
        payload={
            "extraction": extraction_metadata,
            "raw_extracted_pages": len(raw_pages),
            "raw_extracted_chars": raw_extracted_chars,
            "extracted_pages": len(pages),
            "extracted_chars": extracted_chars,
            "parsed_pdf_file": getattr(args, "_ai_processing_metadata", {}).get("parsed_pdf_file"),
            "document_style_reference": document_style_summary,
        },
    )

    single_max_chars = max(10000, int(args.gemma_text_single_max_chars))
    chunk_chars = max(10000, int(args.gemma_text_chunk_chars))

    if extracted_chars <= single_max_chars:
        update_processing_metadata(
            args,
            chunked=False,
            chunk_count=1,
            initial_chunk_count=1,
            processed_chunk_count=1,
            gemma_text_single_max_chars=single_max_chars,
            gemma_text_chunk_chars=chunk_chars,
        )
        text_prompt = build_gemma_text_prompt(
            prompt,
            pdf_path.name,
            full_text,
            document_style_text=document_style_text,
        )
        print(f"  Gemma text TOC generation started: {model}", flush=True)
        try:
            parsed, raw_text = request_gemma_json_text(
                model=model,
                prompt_text=text_prompt,
                args=args,
                label=f"Gemma text({model})",
            )
            print(f"  Gemma text TOC generation completed: {model}", flush=True)
            toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
            add_level_reasons_to_toc(toc, chunk_text=full_text)
            write_stage_file(
                args=args,
                pdf_path=pdf_path,
                provider="gemma",
                model=model,
                stage="02_single_toc_generated",
                payload={
                    "input_chars": len(full_text),
                    "toc": toc,
                    "raw_response": raw_text,
                    "document_style_reference": document_style_summary,
                },
            )
            level_review_raw_text = ""
            level_review_status: dict[str, Any] = {}
            if normalize_gemma_level_verify_scope(
                getattr(args, "gemma_level_verify_scope", DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE)
            ) != "none":
                toc, level_review_raw_text, level_review_status = review_gemma_toc_levels(
                    model=model,
                    pdf_path=pdf_path,
                    args=args,
                    prompt=prompt,
                    toc=toc,
                    stage_label="single text",
                    source_text=full_text,
                    document_style_text=document_style_text,
                )
                write_stage_file(
                    args=args,
                    pdf_path=pdf_path,
                    provider="gemma",
                    model=model,
                    stage="03_single_level_reviewed",
                    payload={
                        "level_review": level_review_status,
                        "toc": toc,
                        "raw_response": level_review_raw_text,
                        "document_style_reference": document_style_summary,
                    },
                )
            single_level_reference_enabled = bool(getattr(args, "gemma_level_reference", DEFAULT_GEMMA_LEVEL_REFERENCE))
            single_level_reference = (
                build_toc_level_reference(toc, max_depth=args.max_depth, chunk_text=full_text)
                if single_level_reference_enabled
                else {}
            )
            update_processing_metadata(
                args,
                gemma_level_reference=single_level_reference_enabled,
                gemma_chunk_level_reference=False,
                gemma_level_reference_source="final_toc",
                gemma_level_reference_levels=sorted(single_level_reference) if single_level_reference_enabled else [],
                gemma_level_reference_examples=serialize_toc_level_reference(single_level_reference) if single_level_reference_enabled else {},
            )
            write_stage_file(
                args=args,
                pdf_path=pdf_path,
                provider="gemma",
                model=model,
                stage="04_single_final_ready",
                payload={
                    "toc": toc,
                    "level_reference": serialize_toc_level_reference(single_level_reference) if single_level_reference_enabled else {},
                    "document_style_reference": document_style_summary,
                },
            )
            level_review_change_summary_file = write_gemma_level_review_change_summary(
                args=args,
                pdf_path=pdf_path,
                provider="gemma",
                model=model,
            )
            raw_bundle = {
                "mode": "gemma_text_single",
                "model": model,
                "pdf": pdf_path.name,
                "extraction": extraction_metadata,
                "document_style_reference": document_style_summary,
                "raw_extracted_pages": len(raw_pages),
                "raw_extracted_chars": raw_extracted_chars,
                "extracted_pages": len(pages),
                "extracted_chars": extracted_chars,
                "raw_response": raw_text,
                "level_review": level_review_status,
                "level_review_raw_response": level_review_raw_text,
                "level_review_change_summary_file": str(level_review_change_summary_file) if level_review_change_summary_file is not None else None,
                "level_reference": serialize_toc_level_reference(single_level_reference) if single_level_reference_enabled else {},
            }
            return toc, json.dumps(raw_bundle, ensure_ascii=False, indent=2), model
        except Exception as error:
            if not is_cuda_out_of_memory_error(error):
                raise
            empty_torch_cuda_cache()
            print(
                f"  Gemma single text generation ran out of CUDA memory; falling back to chunks: {error}",
                file=sys.stderr,
                flush=True,
            )

    chunks = chunk_pdf_text_pages(pages, max_chars=chunk_chars)
    print(f"  Gemma text chunk processing started: {len(chunks)} chunks", flush=True)
    write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
        stage="02_chunks_created",
        payload={
            "chunk_count": len(chunks),
            "chunk_chars": chunk_chars,
            "raw_extracted_pages": len(raw_pages),
            "raw_extracted_chars": raw_extracted_chars,
            "extracted_pages": len(pages),
            "extracted_chars": extracted_chars,
            "chunks": [
                {
                    "index": chunk.get("index"),
                    "start_page": chunk.get("start_page"),
                    "end_page": chunk.get("end_page"),
                    "input_chars": len(str(chunk.get("text") or "")),
                }
                for chunk in chunks
            ],
            "document_style_reference": document_style_summary,
        },
    )
    update_processing_metadata(
        args,
        chunked=True,
        chunk_count=len(chunks),
        initial_chunk_count=len(chunks),
        processed_chunk_count=0,
        gemma_text_single_max_chars=single_max_chars,
        gemma_text_chunk_chars=chunk_chars,
    )
    partial_tocs: list[dict[str, Any]] = []
    raw_parts: list[dict[str, Any]] = []
    level_reference: dict[int, list[dict[str, Any]]] = {}
    level_reference_enabled = bool(getattr(args, "gemma_level_reference", DEFAULT_GEMMA_LEVEL_REFERENCE))
    level_verify_scope = normalize_gemma_level_verify_scope(
        getattr(args, "gemma_level_verify_scope", DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE)
    )
    chunk_level_reference_enabled = level_reference_enabled and level_verify_scope != "final"
    chunk_level_reference_source = (
        "verified_chunk_toc"
        if level_verify_scope in {"chunk", "both"}
        else "unverified_chunk_toc"
        if chunk_level_reference_enabled
        else "disabled"
    )
    setattr(args, "_gemma_chunk_level_reference_enabled", chunk_level_reference_enabled)
    update_processing_metadata(
        args,
        gemma_level_reference=level_reference_enabled,
        gemma_chunk_level_reference=chunk_level_reference_enabled,
        gemma_chunk_level_reference_source=chunk_level_reference_source,
        gemma_level_reference_source="final_toc",
    )

    for chunk in chunks:
        partial_tocs.extend(
            process_gemma_text_chunk(
                model=model,
                pdf_path=pdf_path,
                args=args,
                prompt=prompt,
                chunk=chunk,
                total_chunks=len(chunks),
                raw_parts=raw_parts,
                level_reference=level_reference,
                document_style_text=document_style_text,
            )
        )
    update_processing_metadata(args, processed_chunk_count=len(raw_parts))
    write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
        stage="03_chunks_completed",
        payload={
            "processed_chunk_count": len(raw_parts),
            "partial_count": len(partial_tocs),
            "partial_tocs": partial_tocs,
            "chunk_level_reference": serialize_toc_level_reference(level_reference) if level_reference_enabled else {},
            "chunk_level_reference_source": chunk_level_reference_source,
            "document_style_reference": document_style_summary,
        },
    )

    merge_mode = normalize_gemma_merge_mode(getattr(args, "gemma_merge_mode", DEFAULT_GEMMA_MERGE_MODE))
    update_processing_metadata(args, gemma_merge_mode=merge_mode)

    if merge_mode == "local":
        print("  Gemma chunked TOC local merge started", flush=True)
        toc = merge_partial_tocs_locally(partial_tocs, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        restore_level_reasons_from_partials(toc, partial_tocs)
        final_raw_text = json.dumps(
            {
                "mode": "local_merge",
                "title": toc.get("title"),
                "chapters": len(toc.get("chapters", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
        print("  Gemma chunked TOC local merge completed", flush=True)
    else:
        merge_strategy = normalize_gemma_merge_strategy(
            getattr(args, "gemma_merge_strategy", DEFAULT_GEMMA_MERGE_STRATEGY)
        )
        update_processing_metadata(
            args,
            gemma_merge_strategy=merge_strategy,
            gemma_merge_batch_chars=max(
                10000,
                int(getattr(args, "gemma_merge_batch_chars", DEFAULT_GEMMA_MERGE_BATCH_CHARS)),
            ),
        )
        try:
            if merge_strategy == "batched":
                toc, final_raw_text, merge_metadata = merge_gemma_partial_tocs_batched(
                    model=model,
                    pdf_path=pdf_path,
                    args=args,
                    prompt=prompt,
                    partial_tocs=partial_tocs,
                    document_style_text=document_style_text,
                )
                update_processing_metadata(args, **merge_metadata)
            else:
                print(f"  Gemma chunked TOC merge started: {model}", flush=True)
                toc, final_raw_text, merge_status = merge_gemma_partial_tocs_once(
                    model=model,
                    pdf_path=pdf_path,
                    args=args,
                    prompt=prompt,
                    partial_tocs=partial_tocs,
                    label=f"Gemma chunked TOC merge({model})",
                    document_style_text=document_style_text,
                )
                update_processing_metadata(
                    args,
                    gemma_merge_single_status=merge_status,
                )
        except Exception as merge_error:
            print(
                f"Gemma chunked TOC merge failed; falling back to local merge: {merge_error}",
                file=sys.stderr,
                flush=True,
            )
            merge_error_log = write_error_log(
                pdf_path=pdf_path,
                args=args,
                error=merge_error,
                stage="Gemma chunked TOC merge fallback",
                fatal=False,
            )
            if merge_error_log is not None:
                print(f"  Merge error log saved: {merge_error_log}", file=sys.stderr, flush=True)
            toc = merge_partial_tocs_locally(partial_tocs, fallback_title=pdf_path.stem, max_depth=args.max_depth)
            restore_level_reasons_from_partials(toc, partial_tocs)
            final_raw_text = json.dumps(
                {
                    "mode": "local_merge_after_gemma_merge_failure",
                    "error": message_of(merge_error),
                    "title": toc.get("title"),
                    "chapters": len(toc.get("chapters", [])),
                },
                ensure_ascii=False,
                indent=2,
            )

    write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
        stage="04_merge_completed",
        payload={
            "merge_mode": merge_mode,
            "merge_strategy": getattr(args, "gemma_merge_strategy", DEFAULT_GEMMA_MERGE_STRATEGY),
            "toc": toc,
            "raw_response": final_raw_text,
            "document_style_reference": document_style_summary,
        },
    )

    final_level_review_raw_text = ""
    final_level_review_status: dict[str, Any] = {}
    if gemma_level_verify_enabled(args, "final"):
        final_level_reference_text = (
            format_toc_level_reference(level_reference)
            if level_verify_scope == "both" and level_reference_enabled
            else ""
        )
        toc, final_level_review_raw_text, final_level_review_status = review_gemma_toc_levels(
            model=model,
            pdf_path=pdf_path,
            args=args,
            prompt=prompt,
            toc=toc,
            stage_label="final merged TOC",
            partial_tocs=partial_tocs,
            level_reference_text=final_level_reference_text,
            document_style_text=document_style_text,
        )
        restore_level_reasons_from_partials(toc, partial_tocs)
        toc = sort_toc_with_restored_layout(
            toc,
            partial_tocs=partial_tocs,
            fallback_title=pdf_path.stem,
            max_depth=args.max_depth,
        )
        write_stage_file(
            args=args,
            pdf_path=pdf_path,
            provider="gemma",
            model=model,
            stage="05_final_level_reviewed",
            payload={
                "level_review": final_level_review_status,
                "toc": toc,
                "raw_response": final_level_review_raw_text,
                "document_style_reference": document_style_summary,
            },
        )

    final_level_reference = (
        build_toc_level_reference(toc, max_depth=args.max_depth)
        if level_reference_enabled
        else {}
    )
    update_processing_metadata(
        args,
        gemma_level_reference_levels=sorted(final_level_reference) if level_reference_enabled else [],
        gemma_level_reference_examples=serialize_toc_level_reference(final_level_reference) if level_reference_enabled else {},
    )
    write_stage_file(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
        stage="06_final_toc_ready",
        payload={
            "toc": toc,
            "level_reference": serialize_toc_level_reference(final_level_reference) if level_reference_enabled else {},
            "final_level_review": final_level_review_status,
            "document_style_reference": document_style_summary,
        },
    )
    level_review_change_summary_file = write_gemma_level_review_change_summary(
        args=args,
        pdf_path=pdf_path,
        provider="gemma",
        model=model,
    )

    raw_bundle = {
        "mode": "gemma_text_chunk",
        "merge_mode": merge_mode,
        "merge_strategy": getattr(args, "gemma_merge_strategy", DEFAULT_GEMMA_MERGE_STRATEGY),
        "merge_batch_chars": getattr(args, "gemma_merge_batch_chars", DEFAULT_GEMMA_MERGE_BATCH_CHARS),
        "model": model,
        "pdf": pdf_path.name,
        "extraction": extraction_metadata,
        "document_style_reference": document_style_summary,
        "raw_extracted_pages": len(raw_pages),
        "raw_extracted_chars": raw_extracted_chars,
        "extracted_pages": len(pages),
        "extracted_chars": extracted_chars,
        "chunk_count": len(chunks),
        "processed_chunk_count": len(raw_parts),
        "level_verify_scope": level_verify_scope,
        "level_reference": serialize_toc_level_reference(final_level_reference) if level_reference_enabled else {},
        "chunk_level_reference": (
            serialize_toc_level_reference(level_reference)
            if chunk_level_reference_enabled
            else {}
        ),
        "chunk_level_reference_source": chunk_level_reference_source,
        "chunk_level_reference_unverified": (
            serialize_toc_level_reference(level_reference)
            if chunk_level_reference_source == "unverified_chunk_toc"
            else {}
        ),
        "input_chunk_files": getattr(args, "_ai_input_chunk_files", []),
        "chunk_raw_responses": raw_parts,
        "final_raw_response": final_raw_text,
        "final_level_review": final_level_review_status,
        "final_level_review_raw_response": final_level_review_raw_text,
        "level_review_change_summary_file": str(level_review_change_summary_file) if level_review_change_summary_file is not None else None,
    }
    return toc, json.dumps(raw_bundle, ensure_ascii=False, indent=2), model


def generate_toc_from_pdf_gemma(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    for index, model in enumerate(models):
        model = normalize_gemma_hf_model(model)
        if index > 0:
            print(f"Trying Gemma fallback model: {model}", file=sys.stderr, flush=True)

        try:
            runtime = normalize_gemma_runtime(getattr(args, "gemma_runtime", DEFAULT_GEMMA_RUNTIME))
            runtime_labels = {
                "vllm": "vLLM",
                "vllm-server": "vLLM server",
                "transformers": "Transformers",
            }
            runtime_label = runtime_labels.get(runtime, runtime)
            print(f"  Using Gemma/{runtime_label}: {model}", flush=True)
            return generate_toc_from_pdf_gemma_text(
                model=model,
                pdf_path=pdf_path,
                args=args,
                prompt=prompt,
            )
        except Exception as error:
            last_error = error
            print(f"Gemma model failed: {model} / {error}", file=sys.stderr, flush=True)

    if last_error:
        raise last_error
    raise RuntimeError("No Gemma models to try.")


# ----------------------------- Common processing -----------------------------

def generate_toc_from_pdf(
    pdf_path: Path,
    args: argparse.Namespace,
    user_prompt: str,
) -> tuple[dict[str, Any], str, str]:
    prompt = build_prompt(user_prompt=user_prompt, max_depth=args.max_depth)

    if args.provider == "gemini":
        return generate_toc_from_pdf_gemini(pdf_path=pdf_path, args=args, prompt=prompt)

    if args.provider == "openai":
        return generate_toc_from_pdf_openai(pdf_path=pdf_path, args=args, prompt=prompt)

    if args.provider == "claude":
        return generate_toc_from_pdf_claude(pdf_path=pdf_path, args=args, prompt=prompt)

    if args.provider == "gemma":
        return generate_toc_from_pdf_gemma(pdf_path=pdf_path, args=args, prompt=prompt)

    raise ValueError(f"Unsupported provider: {args.provider}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


class TeeStream:
    def __init__(self, console_stream: Any, log_stream: Any, lock: Lock, echo_console: bool = True) -> None:
        self.console_stream = console_stream
        self.log_stream = log_stream
        self.lock = lock
        self.echo_console = bool(echo_console)
        self.encoding = getattr(console_stream, "encoding", "utf-8")

    def write(self, text: str) -> int:
        with self.lock:
            if self.echo_console:
                self.console_stream.write(text)
                self.console_stream.flush()
            self.log_stream.write(text)
            self.log_stream.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            if self.echo_console:
                self.console_stream.flush()
            self.log_stream.flush()

    def isatty(self) -> bool:
        isatty = getattr(self.console_stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def fileno(self) -> int:
        if not self.echo_console:
            log_fileno = getattr(self.log_stream, "fileno", None)
            if callable(log_fileno):
                return int(log_fileno())

        fileno = getattr(self.console_stream, "fileno", None)
        if callable(fileno):
            return int(fileno())
        log_fileno = getattr(self.log_stream, "fileno", None)
        if callable(log_fileno):
            return int(log_fileno())
        raise OSError("TeeStream does not wrap a stream with fileno().")


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def timestamp_for_metadata() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def add_result_metadata(
    toc: dict[str, Any],
    pdf_path: Path,
    provider: str,
    model: str,
    elapsed_seconds: float,
    generated_timestamp: str,
    generation_started_at: str,
    generation_completed_at: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(toc)
    metadata: dict[str, Any] = {
        "source_pdf": pdf_path.name,
        "provider": provider,
        "model": model,
        "generated_timestamp": generated_timestamp,
        "generated_at": generation_completed_at,
        "generation_started_at": generation_started_at,
        "generation_completed_at": generation_completed_at,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "elapsed": format_elapsed(elapsed_seconds),
    }
    if extra_metadata:
        metadata.update({str(key): jsonable_debug_value(value) for key, value in extra_metadata.items()})

    result["_meta"] = metadata
    return result


def safe_filename_part(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "unknown_model"


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create unique output path for: {path}")


def build_result_base_path(output_dir: Path, pdf_path: Path, label: str, generated_timestamp: str) -> Path:
    label_part = safe_filename_part(label)
    pdf_part = safe_filename_part(pdf_path.stem)
    return output_dir / f"{pdf_part}_{label_part}_{generated_timestamp}"


def build_run_log_path(output_dir: Path, pdf_files: list[Path], providers: list[str], run_timestamp: str) -> Path:
    pdf_part = pdf_files[0].stem if len(pdf_files) == 1 else "batch"
    provider_part = safe_filename_part("-".join(providers))
    return output_dir / "logs" / f"{pdf_part}_{provider_part}_{run_timestamp}_log.txt"


def jsonable_debug_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [jsonable_debug_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): jsonable_debug_value(item) for key, item in value.items()}

    return str(value)


def write_error_log(
    pdf_path: Path,
    args: argparse.Namespace,
    error: Exception,
    elapsed_seconds: float | None = None,
    stage: str | None = None,
    fatal: bool = True,
) -> Path | None:
    if getattr(args, "no_error_log", False):
        return None

    debug = error_debug_info(error)
    output_dir = Path(getattr(args, "output_dir", OUTPUT_DIR))
    error_dir = output_dir / "error_logs"
    error_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_part = safe_filename_part(pdf_path.stem)
    provider_part = safe_filename_part(str(getattr(args, "provider", "unknown")))
    model_text = str(debug.get("model") or getattr(args, "model", "unknown"))
    model_part = safe_filename_part(model_text)
    stage_text = stage or str(debug.get("label") or "error")
    stage_part = safe_filename_part(stage_text)[:80]
    base_name = f"{timestamp}_{pdf_part}_{provider_part}_{model_part}_{stage_part}"

    raw_text = str(debug.get("raw_response") or "")
    raw_response_file: Path | None = None
    if raw_text:
        raw_response_file = unique_output_path(error_dir / f"{base_name}_raw_response.txt")
        raw_response_file.write_text(raw_text, encoding="utf-8")

    debug_payload = {
        str(key): jsonable_debug_value(value)
        for key, value in debug.items()
        if key != "raw_response"
    }

    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fatal": bool(fatal),
        "stage": stage_text,
        "source_pdf": str(pdf_path),
        "source_pdf_name": pdf_path.name,
        "provider": getattr(args, "provider", None),
        "model": model_text,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error": message_of(error),
        "debug": debug_payload,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }

    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
        payload["elapsed"] = format_elapsed(elapsed_seconds)

    if raw_response_file is not None:
        payload["raw_response_file"] = str(raw_response_file)
        payload["raw_response_chars"] = len(raw_text)
        payload["raw_response_preview"] = raw_text[:2000]

    log_file = unique_output_path(error_dir / f"{base_name}_error.json")
    save_json(log_file, payload)
    return log_file


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        return [path]

    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() == ".pdf")

    raise FileNotFoundError(f"Path not found: {path}")


def print_quiet_console(args: argparse.Namespace, text: str, error: bool = False) -> None:
    if not getattr(args, "quiet_console", False):
        return

    stream = getattr(args, "_ai_console_stderr", None) if error else getattr(args, "_ai_console_stdout", None)
    if stream is None:
        stream = sys.stderr if error else sys.stdout

    lock = getattr(args, "_ai_console_lock", None)
    if lock is None:
        print(text, file=stream, flush=True)
        return

    with lock:
        print(text, file=stream, flush=True)


def process_pdf(pdf_path: Path, args: argparse.Namespace, user_prompt: str) -> bool:
    generation_started_at = timestamp_for_metadata()
    processing_pdf_path, display_name = prepare_pdf_for_processing(pdf_path)
    setattr(args, "_ai_current_input_pdf_path", str(pdf_path))
    setattr(args, "_ai_current_display_name", display_name)
    setattr(args, "_ai_processing_metadata", {})
    setattr(args, "_ai_input_chunk_files", [])
    setattr(args, "_ai_stage_files", [])
    setattr(args, "_ai_stage_index", 0)
    setattr(args, "_gemma_level_review_stats", {})
    setattr(args, "_gemma_level_review_change_reports", [])
    setattr(args, "_gemma_level_review_change_summary_file", "")
    print(f"Processing started: {display_name}", flush=True)
    print(f"  provider: {args.provider}", flush=True)
    started_at = time.perf_counter()

    try:
        toc, raw_text, used_model = generate_toc_from_pdf(pdf_path=processing_pdf_path, args=args, user_prompt=user_prompt)
        elapsed_seconds = time.perf_counter() - started_at
        generation_completed_at = timestamp_for_metadata()
        generated_timestamp = timestamp_for_filename()
        result_base_path = build_result_base_path(
            output_dir=Path(args.output_dir),
            pdf_path=pdf_path,
            label=args.provider,
            generated_timestamp=generated_timestamp,
        )
        output_file = unique_output_path(result_base_path.with_name(f"{result_base_path.name}_toc.json"))
        extra_metadata = dict(getattr(args, "_ai_processing_metadata", None) or {})
        if getattr(args, "_ai_run_timestamp", None):
            extra_metadata["run_timestamp"] = getattr(args, "_ai_run_timestamp")
        if getattr(args, "_ai_run_log_file", None):
            extra_metadata["run_log_file"] = getattr(args, "_ai_run_log_file")
        result = add_result_metadata(
            toc=toc,
            pdf_path=pdf_path,
            provider=args.provider,
            model=used_model,
            elapsed_seconds=elapsed_seconds,
            generated_timestamp=generated_timestamp,
            generation_started_at=generation_started_at,
            generation_completed_at=generation_completed_at,
            extra_metadata=extra_metadata,
        )
        save_json(output_file, result)

        if args.write_raw:
            raw_file = unique_output_path(result_base_path.with_name(f"{result_base_path.name}_raw_response.txt"))
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            raw_file.write_text(raw_text, encoding="utf-8")
            print(f"Raw response saved: {raw_file}", flush=True)

        print(f"Completed: {output_file}", flush=True)
        print(f"  model: {used_model}", flush=True)
        print(f"  elapsed: {format_elapsed(elapsed_seconds)} ({elapsed_seconds:.3f}s)", flush=True)
        print(f"  title: {toc.get('title')}", flush=True)
        print(f"  chapters: {len(toc.get('chapters', []))}", flush=True)
        print_quiet_console(
            args,
            (
                f"Completed: {output_file} / "
                f"elapsed {format_elapsed(elapsed_seconds)} / "
                f"chapters {len(toc.get('chapters', []))}"
            ),
        )
        return True

    except Exception as error:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"Failed: {display_name} / {error}", file=sys.stderr, flush=True)
        print(f"  elapsed: {format_elapsed(elapsed_seconds)} ({elapsed_seconds:.3f}s)", file=sys.stderr, flush=True)
        error_log = write_error_log(
            pdf_path=pdf_path,
            args=args,
            error=error,
            elapsed_seconds=elapsed_seconds,
            stage="PDF processing failed",
            fatal=True,
        )
        if error_log is not None:
            print(f"  Error log saved: {error_log}", file=sys.stderr, flush=True)
        error_log_text = f" / error log {error_log}" if error_log is not None else ""
        print_quiet_console(
            args,
            f"Failed: {display_name} / elapsed {format_elapsed(elapsed_seconds)} / {error}{error_log_text}",
            error=True,
        )
        if args.stop_on_error:
            raise
        return False


def build_provider_args(args: argparse.Namespace, provider: str) -> argparse.Namespace:
    provider_args = argparse.Namespace(**vars(args))
    provider_args.provider = provider

    if provider_args.model is None:
        provider_args.model = default_model_for(provider)
    provider_args.model = normalize_model_for_provider(provider, provider_args.model)

    if provider_args.ai_fallback_models is None:
        provider_args.ai_fallback_models = default_fallback_models_for(provider)
    provider_args.ai_fallback_models = normalize_model_list_for_provider(provider, provider_args.ai_fallback_models)

    if provider_args.max_output_tokens is None:
        provider_args.max_output_tokens = default_max_output_tokens_for(provider)

    return provider_args


def process_pdf_with_providers(
    pdf_path: Path,
    args: argparse.Namespace,
    providers: list[str],
    user_prompt: str,
) -> dict[str, bool]:
    if len(providers) == 1:
        provider = providers[0]
        provider_args = build_provider_args(args, provider)
        return {provider: process_pdf(pdf_path=pdf_path, args=provider_args, user_prompt=user_prompt)}

    print(f"Processing started: {pdf_path.name}", flush=True)
    print(f"  Running providers concurrently: {', '.join(providers)}", flush=True)

    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(
                process_pdf,
                pdf_path,
                build_provider_args(args, provider),
                user_prompt,
            ): provider
            for provider in providers
        }

        for future in as_completed(futures):
            provider = futures[future]
            try:
                results[provider] = bool(future.result())
            except Exception:
                results[provider] = False
                if args.stop_on_error:
                    for pending in futures:
                        pending.cancel()
                    raise

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send PDF files and prompts to AI providers(gemini/openai/claude/gemma) and generate TOC JSON."
    )

    parser.add_argument("path", nargs="?", default="./data/input", help="PDF file or directory")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--prompt", default=None, help="TOC generation instruction")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 text file containing the TOC instruction")
    parser.add_argument(
        "--provider",
        default=None,
        help="AI provider to use. If omitted or set to all, gemini/openai/claude run concurrently. Use --provider gemma for Gemma.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. If omitted, provider-specific env values or defaults are used.",
    )
    parser.add_argument(
        "--ai-fallback-models",
        default=None,
        help="Comma-separated fallback models to try when the primary model fails. If omitted, provider-specific env values or defaults are used.",
    )
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES, help="Retry count for temporary errors")
    parser.add_argument(
        "--ai-retry-base-delay",
        type=float,
        default=DEFAULT_RETRY_BASE_DELAY,
        help="Base retry delay in seconds",
    )
    parser.add_argument(
        "--file-processing-timeout",
        type=int,
        default=DEFAULT_FILE_PROCESSING_TIMEOUT,
        help="Timeout in seconds while waiting for Gemini uploaded PDF processing. Gemini only.",
    )
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum TOC level")
    parser.add_argument("--temperature", type=float, default=0, help="AI temperature. Not sent to OpenAI gpt-5-family models or some newer Claude models.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Maximum output tokens")
    parser.add_argument("--no-schema", action="store_true", help="Do not use JSON schema enforcement when available; rely on the prompt only")
    parser.add_argument(
        "--gemini-thinking-budget",
        type=int,
        default=DEFAULT_GEMINI_THINKING_BUDGET,
        help="Gemini thinking token budget. 0 disables thinking; -1 uses Gemini automatic settings.",
    )
    parser.add_argument(
        "--claude-force-text",
        action="store_true",
        help="Use Claude text extraction/chunk mode directly instead of PDF document input.",
    )
    parser.add_argument(
        "--claude-text-single-max-chars",
        type=int,
        default=DEFAULT_CLAUDE_TEXT_SINGLE_MAX_CHARS,
        help="Maximum extracted text characters to send at once in Claude text mode",
    )
    parser.add_argument(
        "--claude-text-chunk-chars",
        type=int,
        default=DEFAULT_CLAUDE_TEXT_CHUNK_CHARS,
        help="Maximum extracted text characters per chunk in Claude text chunk mode",
    )
    parser.add_argument(
        "--claude-text-min-chunk-chars",
        type=int,
        default=DEFAULT_CLAUDE_TEXT_MIN_CHUNK_CHARS,
        help="Minimum chunk size when splitting failed Claude chunks into smaller chunks",
    )
    parser.add_argument(
        "--gemma-device-map",
        default=DEFAULT_GEMMA_DEVICE_MAP,
        help="Transformers device_map value used by Gemma. Meaningful only for --gemma-runtime transformers.",
    )
    parser.add_argument(
        "--gemma-torch-dtype",
        default=DEFAULT_GEMMA_TORCH_DTYPE,
        help="dtype used by Gemma. Examples: auto, bfloat16, float16",
    )
    parser.add_argument(
        "--gemma-runtime",
        default=DEFAULT_GEMMA_RUNTIME,
        help="Gemma inference runtime: transformers, vllm, or vllm-server.",
    )
    parser.add_argument(
        "--gemma-merge-mode",
        default=DEFAULT_GEMMA_MERGE_MODE,
        choices=("local", "gemma"),
        help="How to merge chunked Gemma TOCs. local uses Python only; gemma asks Gemma to merge and falls back to local on failure.",
    )
    parser.add_argument(
        "--gemma-merge-strategy",
        default=DEFAULT_GEMMA_MERGE_STRATEGY,
        choices=("single", "batched"),
        help="Gemma merge strategy. batched merges smaller groups with Gemma and stitches them locally to avoid long-output truncation.",
    )
    parser.add_argument(
        "--gemma-merge-batch-chars",
        type=int,
        default=DEFAULT_GEMMA_MERGE_BATCH_CHARS,
        help="Approximate compact partial-TOC JSON characters per Gemma merge batch.",
    )
    parser.add_argument(
        "--gemma-level-reference",
        dest="gemma_level_reference",
        action="store_true",
        default=DEFAULT_GEMMA_LEVEL_REFERENCE,
        help="Use earlier Gemma chunk TOC level examples as a reference for later chunks.",
    )
    parser.add_argument(
        "--no-gemma-level-reference",
        dest="gemma_level_reference",
        action="store_false",
        help="Disable passing earlier Gemma chunk TOC level examples to later chunks.",
    )
    parser.add_argument(
        "--gemma-level-verify",
        dest="gemma_level_verify",
        action="store_true",
        default=None,
        help="Backward-compatible shortcut for --gemma-level-verify-scope final.",
    )
    parser.add_argument(
        "--no-gemma-level-verify",
        dest="gemma_level_verify",
        action="store_false",
        help="Backward-compatible shortcut for --gemma-level-verify-scope none.",
    )
    parser.add_argument(
        "--gemma-level-verify-scope",
        default=DEFAULT_GEMMA_LEVEL_VERIFY_SCOPE,
        help="Where to run separate Gemma level verification/correction: none, chunk, final, or both. Default: final.",
    )
    parser.add_argument(
        "--gemma-document-style-reference",
        dest="gemma_document_style_reference",
        action="store_true",
        default=DEFAULT_GEMMA_DOCUMENT_STYLE_REFERENCE,
        help="Pass a compact full-document layout/style reference to Gemma chunk extraction and verification. Enabled by default.",
    )
    parser.add_argument(
        "--no-gemma-document-style-reference",
        dest="gemma_document_style_reference",
        action="store_false",
        help="Disable the full-document layout/style reference in Gemma prompts.",
    )
    parser.add_argument(
        "--gemma-document-style-max-clusters",
        type=int,
        default=DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_CLUSTERS,
        help="Maximum document-wide layout style clusters to include in Gemma prompts.",
    )
    parser.add_argument(
        "--gemma-document-style-max-examples",
        type=int,
        default=DEFAULT_GEMMA_DOCUMENT_STYLE_MAX_EXAMPLES,
        help="Maximum page-order title-like examples to include in the document-wide style reference.",
    )
    parser.add_argument(
        "--gemma-vllm-tensor-parallel-size",
        type=int,
        default=DEFAULT_GEMMA_VLLM_TENSOR_PARALLEL_SIZE,
        help="vLLM tensor_parallel_size for Gemma. Meaningful only for --gemma-runtime vllm.",
    )
    parser.add_argument(
        "--gemma-vllm-gpu-memory-utilization",
        type=float,
        default=DEFAULT_GEMMA_VLLM_GPU_MEMORY_UTILIZATION,
        help="vLLM GPU memory utilization fraction for Gemma. Meaningful only for --gemma-runtime vllm.",
    )
    parser.add_argument(
        "--gemma-vllm-max-model-len",
        default=DEFAULT_GEMMA_VLLM_MAX_MODEL_LEN,
        help="Optional vLLM max_model_len for Gemma. Leave empty to use vLLM/model default.",
    )
    parser.add_argument(
        "--gemma-vllm-trust-remote-code",
        action="store_true",
        default=DEFAULT_GEMMA_VLLM_TRUST_REMOTE_CODE,
        help="Pass trust_remote_code=True when loading Gemma with vLLM.",
    )
    parser.add_argument(
        "--gemma-vllm-server-base-url",
        default=DEFAULT_GEMMA_VLLM_SERVER_BASE_URL,
        help="OpenAI-compatible vLLM server base URL for --gemma-runtime vllm-server.",
    )
    parser.add_argument(
        "--gemma-vllm-server-api-key",
        default=DEFAULT_GEMMA_VLLM_SERVER_API_KEY,
        help="Bearer API key for --gemma-runtime vllm-server. vLLM usually accepts any value unless configured otherwise.",
    )
    parser.add_argument(
        "--gemma-vllm-server-timeout",
        type=int,
        default=DEFAULT_GEMMA_VLLM_SERVER_TIMEOUT,
        help="HTTP timeout in seconds for --gemma-runtime vllm-server requests.",
    )
    parser.add_argument(
        "--gemma-extraction-mode",
        default=DEFAULT_GEMMA_EXTRACTION_MODE,
        choices=("layout", "text"),
        help="Gemma PDF extraction mode. layout includes compact font size/style/position tags; text uses plain extracted text.",
    )
    parser.add_argument(
        "--gemma-text-single-max-chars",
        type=int,
        default=DEFAULT_GEMMA_TEXT_SINGLE_MAX_CHARS,
        help="Maximum extracted text characters to send at once in Gemma text mode",
    )
    parser.add_argument(
        "--gemma-text-chunk-chars",
        type=int,
        default=DEFAULT_GEMMA_TEXT_CHUNK_CHARS,
        help="Maximum extracted text characters per chunk in Gemma text chunk mode",
    )
    parser.add_argument(
        "--gemma-text-min-chunk-chars",
        type=int,
        default=DEFAULT_GEMMA_TEXT_MIN_CHUNK_CHARS,
        help="Minimum chunk size when splitting failed Gemma chunks into smaller chunks",
    )
    parser.add_argument("--write-raw", action="store_true", help="Also save raw response text")
    parser.add_argument(
        "--write-parsed-pdf",
        dest="write_parsed_pdf",
        action="store_true",
        default=DEFAULT_WRITE_PARSED_PDF,
        help="Save extracted PDF text/layout input under output_dir/parsed_pdfs. Enabled by default.",
    )
    parser.add_argument(
        "--no-write-parsed-pdf",
        dest="write_parsed_pdf",
        action="store_false",
        help="Do not save extracted PDF text/layout input.",
    )
    parser.add_argument(
        "--write-input-chunks",
        "--write-chunk-results",
        dest="write_input_chunks",
        action="store_true",
        default=DEFAULT_WRITE_INPUT_CHUNKS,
        help="Save each Gemma input chunk before TOC generation under output_dir/input_chunks. Enabled by default.",
    )
    parser.add_argument(
        "--no-write-input-chunks",
        "--no-write-chunk-results",
        dest="write_input_chunks",
        action="store_false",
        help="Do not save Gemma input chunk files.",
    )
    parser.add_argument(
        "--write-stage-files",
        dest="write_stage_files",
        action="store_true",
        default=DEFAULT_WRITE_STAGE_FILES,
        help="Save Gemma intermediate stage JSON files under output_dir/stages. Enabled by default.",
    )
    parser.add_argument(
        "--no-write-stage-files",
        dest="write_stage_files",
        action="store_false",
        help="Do not save Gemma intermediate stage JSON files.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to save a tee copy of console output. Defaults to output_dir/logs/name_provider_YYYYMMDD_HHMMSS_log.txt.",
    )
    parser.add_argument(
        "--quiet-console",
        action="store_true",
        help="Save detailed run logs to the log file, but print only final per-PDF results and the final summary to the terminal.",
    )
    parser.add_argument("--no-run-log", action="store_true", help="Disable writing the console output run log")
    parser.add_argument(
        "--no-error-log",
        action="store_true",
        help="Disable writing error logs under output_dir/error_logs",
    )
    parser.add_argument("--delete-uploaded-file", action="store_true", help="Delete uploaded files after processing. Meaningful only for Gemini/OpenAI.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when any PDF fails during multi-PDF processing")

    # Backward compatibility for older commands. The script always uses AI mode now.
    parser.add_argument("--use-ai", action="store_true", help=argparse.SUPPRESS)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.quiet_console and args.no_run_log:
        parser.error("--quiet-console requires run logging; remove --no-run-log.")

    try:
        providers = resolve_providers(args.provider)
    except ValueError as error:
        parser.error(str(error))

    if "gemma" in providers:
        try:
            args.gemma_runtime = normalize_gemma_runtime(args.gemma_runtime)
            args.gemma_merge_mode = normalize_gemma_merge_mode(args.gemma_merge_mode)
            args.gemma_merge_strategy = normalize_gemma_merge_strategy(args.gemma_merge_strategy)
            args.gemma_extraction_mode = normalize_gemma_extraction_mode(args.gemma_extraction_mode)
            args.gemma_level_verify_scope = normalize_gemma_level_verify_scope(args.gemma_level_verify_scope)
            if args.gemma_level_verify is False:
                args.gemma_level_verify_scope = "none"
            elif args.gemma_level_verify is True:
                args.gemma_level_verify_scope = "final"
            args.gemma_level_verify = args.gemma_level_verify_scope != "none"
        except ValueError as error:
            parser.error(str(error))

    if len(providers) > 1 and args.model is not None:
        parser.error("--model can only be used when a single provider is selected with --provider.")

    if len(providers) > 1 and args.ai_fallback_models is not None:
        parser.error("--ai-fallback-models can only be used when a single provider is selected with --provider.")

    args.max_depth = max(1, min(int(args.max_depth), 10))

    run_timestamp = timestamp_for_filename()
    setattr(args, "_ai_run_timestamp", run_timestamp)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    user_prompt = load_prompt(args)
    pdf_files = iter_pdfs(Path(args.path))

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_handle = None
    log_path: Path | None = None
    console_lock = Lock()
    setattr(args, "_ai_console_stdout", original_stdout)
    setattr(args, "_ai_console_stderr", original_stderr)
    setattr(args, "_ai_console_lock", console_lock)
    if not args.no_run_log:
        configured_log_file = clean_text(args.log_file)
        log_path = (
            Path(configured_log_file)
            if configured_log_file
            else build_run_log_path(Path(args.output_dir), pdf_files, providers, run_timestamp)
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(
            original_stdout,
            log_file_handle,
            console_lock,
            echo_console=not args.quiet_console,
        )
        sys.stderr = TeeStream(
            original_stderr,
            log_file_handle,
            console_lock,
            echo_console=not args.quiet_console,
        )
        setattr(args, "_ai_run_log_file", str(log_path))
        print(f"Run log saved: {log_path}", flush=True)

    try:
        if not pdf_files:
            print("No PDF files to process.", flush=True)
            return 0

        if "gemma" in providers:
            try:
                preflight_gemma_provider(args)
            except Exception as error:
                print(f"Gemma preflight failed: {error}", file=sys.stderr, flush=True)
                print_quiet_console(args, f"Failed before processing: Gemma preflight failed / {error}", error=True)
                return 1

        success_count = 0
        total_count = len(pdf_files) * len(providers)
        for pdf_path in pdf_files:
            results = process_pdf_with_providers(
                pdf_path=pdf_path,
                args=args,
                providers=providers,
                user_prompt=user_prompt,
            )
            success_count += sum(1 for success in results.values() if success)

        failed_count = total_count - success_count
        print(f"Processing result: success {success_count} / failed {failed_count}", flush=True)
        print_quiet_console(args, f"Processing result: success {success_count} / failed {failed_count}")

        return 0 if failed_count == 0 else 1
    finally:
        if log_file_handle is not None:
            if log_path is not None:
                print(f"Run log saved: {log_path}", flush=True)
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
