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
    "gemma": int(os.getenv("GEMMA_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "8192"))),
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
DEFAULT_WRITE_PARSED_PDF = (
    os.getenv("WRITE_PARSED_PDF", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_WRITE_INPUT_CHUNKS = (
    os.getenv("WRITE_INPUT_CHUNKS", os.getenv("WRITE_CHUNK_RESULTS", "true")).strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_GEMMA_VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("GEMMA_VLLM_TENSOR_PARALLEL_SIZE", os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
DEFAULT_GEMMA_VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("GEMMA_VLLM_GPU_MEMORY_UTILIZATION", os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")))
DEFAULT_GEMMA_VLLM_MAX_MODEL_LEN = os.getenv("GEMMA_VLLM_MAX_MODEL_LEN", os.getenv("VLLM_MAX_MODEL_LEN", "")).strip()
DEFAULT_GEMMA_VLLM_TRUST_REMOTE_CODE = (
    os.getenv("GEMMA_VLLM_TRUST_REMOTE_CODE", os.getenv("VLLM_TRUST_REMOTE_CODE", "false")).strip().lower()
    in {"1", "true", "yes", "on"}
)

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
    "_source_order": ("_source_order", "source_order"),
}

DEFAULT_USER_PROMPT = """
Read the full PDF and create a study-oriented table of contents.

TOC rules:
- Exclude covers, prefaces, existing TOC pages, indexes, references, page numbers, and repeated headers/footers.
- Use only chapter, section, and subsection titles that actually appear in the body.
- Do not invent new titles.
- Preserve original numbering when it exists. Do not add numbering when the source has none.
- Calculate page as the actual PDF viewer page order. The first page is 1.
- Use lower level numbers for larger sections and higher level numbers for subsections.
- Include as much as possible, but do not include body sentences or individual question items as TOC entries.
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
Create the table of contents using only the attached PDF.

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
- Use the document title found in the PDF. If unclear, use a title close to the file name.
- Sort chapters in the order they appear in the PDF.
- Use only levels from 1 to {max_depth}.
- page must be an integer and must use actual PDF viewer page order, where the first page is 1.
- Preserve original title numbering, but do not invent missing numbering.
- Preserve original chapter/section titles exactly, including Korean text when the source title is Korean.
- Exclude body sentences, examples, questions, table/figure captions, and references that are not suitable TOC entries.
- For every chapter object, include level_reason explaining why that entry has its level. Use numbering, hierarchy, font size/style, indentation, and surrounding context when available.
- Never output any text outside the JSON object.
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

        if target_key in {"_layout_page", "_source_order"}:
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
    source_order = int_sort_value(item.get("_source_order", item.get("_local_merge_order")), default=0)

    if y is None:
        return (page, 1, float(source_order), x if x is not None else 999999.0, source_order, level)

    return (page, 0, y, x if x is not None else 999999.0, source_order, level)


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
    }
    runtime = aliases.get(runtime, runtime)
    if runtime not in {"transformers", "vllm"}:
        raise ValueError("--gemma-runtime must be either transformers or vllm.")
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
    if "vertical_position" in layout:
        assign("_layout_y", layout["vertical_position"])
    if "left_indent" in layout:
        assign("_layout_x", layout["left_indent"])
    if "font_size" in layout:
        assign("_layout_font_size", layout["font_size"])
    if "size_ratio" in layout:
        assign("_layout_size_ratio", layout["size_ratio"])
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
        reason_parts.append("현재 chunk에서 대응 layout tag를 찾지 못해 번호 체계와 주변 문맥 기준")

    closest_level, closest_entry = closest_level_reference(layout, level_reference)
    if closest_level is not None:
        closest_title = clean_text((closest_entry or {}).get("title"))
        if closest_level == level:
            suffix = f' "{closest_title}"' if closest_title else ""
            reason_parts.append(f"이전 level {level} 예시{suffix}와 가장 가까운 style 패턴")
        else:
            reason_parts.append(
                f"이전 style 기준으로는 level {closest_level}와 가장 가까우나, 번호/문맥상 level {level}로 유지"
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


def partial_toc_level_reason_map(partial_tocs: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    reason_by_key: dict[tuple[str, int], str] = {}
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
            reason_by_key.setdefault((title, page), reason)
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
        reason = reason_by_key.get((title, page))
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

        examples = level_reference.setdefault(level, [])
        normalized = re.sub(r"\s+", "", title).lower()
        if any(normalize_toc_match_text(existing.get("title")) == normalized for existing in examples):
            continue
        if len(examples) < max_examples_per_level:
            examples.append(toc_level_reference_entry(chapter_title=chapter_title, chunk_text=chunk_text, chapter=chapter))


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
        "These examples came from earlier chunks. Style fields are from parsed layout tags: s=font size, r=size/body ratio, x=left indent, b=bold, i=italic, f=font id.",
        "Use title hierarchy and style patterns only to keep level numbers consistent.",
        "Do not copy these titles unless the same title is actually visible in the current chunk.",
        *lines,
    ])


def build_gemma_text_prompt(base_prompt: str, pdf_name: str, text: str) -> str:
    return f"""
{base_prompt}

[Gemma Text/Layout Mode]
Create the TOC using only the [Extracted PDF Text] below instead of an attached PDF.
Use each text block's [PAGE n] marker to determine page numbers.
Some lines may start with compact layout tags:
- [L s=<font size> r=<size/body ratio> x=<left indent> y=<vertical position> b=<bold 0/1> i=<italic 0/1> f=<font id>]
- Use larger font size, higher size ratio, bold/italic style, indentation, numbering, and nearby body text to infer TOC levels.
- Never copy [L ...] tags into chapter titles.
- Ignore existing table-of-contents pages, dot-leader lines, page-number-only lines, and repeated headers/footers.
- Include level_reason in every chapter object. Explain the level using numbering, hierarchy, font size/ratio, bold/italic, indentation, and context.
- For every chapter object, copy layout metadata from the actual visible body-title line you used:
  "_layout_page" from [PAGE n], "_layout_x" from x=, "_layout_y" from y=, "_layout_font_size" from s=, "_layout_size_ratio" from r=, and "_layout_text" as the exact text after the [L ...] tag.
- Use layout metadata from the same [PAGE n] as the chapter page. Do not use layout metadata from existing TOC pages, headers, footers, indexes, or repeated running titles.
- If a title spans multiple layout lines, use the main descriptive title line with the strongest title style, and keep the combined title in "chapter".
- If no reliable [L ...] line exists for an entry, omit the _layout_* fields instead of guessing.
- Output compact valid JSON only. Every object property and every array item must be separated with commas.

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
) -> str:
    level_reference_block = f"\n\n{level_reference_text}" if clean_text(level_reference_text) else ""
    return f"""
{base_prompt}

[Gemma Text/Layout Chunk Mode]
The text below is one part of the full PDF.
Include only chapter, section, and subsection titles that are actually visible within this range.
Do not infer TOC entries outside this range.
Use [PAGE n] markers to determine page numbers.
If a previous TOC level reference is provided, follow its level convention unless the current chunk visibly uses a different hierarchy.
Some lines may start with compact layout tags:
- [L s=<font size> r=<size/body ratio> x=<left indent> y=<vertical position> b=<bold 0/1> i=<italic 0/1> f=<font id>]
- Use larger font size, higher size ratio, bold/italic style, indentation, numbering, and nearby body text to infer TOC levels.
- Never copy [L ...] tags into chapter titles.
- Ignore existing table-of-contents pages, dot-leader lines, page-number-only lines, and repeated headers/footers.
- Include level_reason in every chapter object. Explain the level using numbering, hierarchy, font size/ratio, bold/italic, indentation, and context.
- For every chapter object, copy layout metadata from the actual visible body-title line you used:
  "_layout_page" from [PAGE n], "_layout_x" from x=, "_layout_y" from y=, "_layout_font_size" from s=, "_layout_size_ratio" from r=, and "_layout_text" as the exact text after the [L ...] tag.
- Use layout metadata from the same [PAGE n] as the chapter page. Do not use layout metadata from existing TOC pages, headers, footers, indexes, or repeated running titles.
- If a title spans multiple layout lines, use the main descriptive title line with the strongest title style, and keep the combined title in "chapter".
- If no reliable [L ...] line exists for an entry, omit the _layout_* fields instead of guessing.
- Output compact valid JSON only. Every object property and every array item must be separated with commas.
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
) -> str:
    partial_json = json.dumps(partial_tocs, ensure_ascii=False, indent=2)
    level_reason_instruction = (
        "- Preserve level_reason when it exists. If an entry has no level_reason, add a concise reason based only on the candidate entry and nearby level pattern."
        if include_level_reason
        else "- For this merge response, omit level_reason to keep the JSON compact. The program will restore level_reason from chunk results after merge."
    )
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
{level_reason_instruction}

[PDF File Name]
{pdf_name}

[Partial TOC Candidates]
{partial_json}
""".strip()


def build_gemma_json_retry_prompt(base_prompt: str, error: Exception, raw_text: str) -> str:
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
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    compact_partials = compact_partial_tocs_for_gemma_merge(partial_tocs)
    merge_prompt = build_gemma_merge_prompt(
        prompt,
        pdf_path.name,
        compact_partials,
        args.max_depth,
        include_level_reason=False,
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
    depth: int = 0,
) -> list[dict[str, Any]]:
    chunk_label = str(chunk["index"])
    use_level_reference = bool(getattr(args, "gemma_level_reference", DEFAULT_GEMMA_LEVEL_REFERENCE))
    level_reference_text = format_toc_level_reference(level_reference or {}) if use_level_reference else ""
    chunk_prompt = build_gemma_chunk_prompt(
        prompt,
        pdf_path.name,
        chunk,
        total_chunks,
        level_reference_text=level_reference_text,
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
        raw_parts.append({
            "chunk": chunk["index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "raw_response": raw_text,
        })
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
    pages, extraction_metadata = extract_gemma_pdf_pages(pdf_path, args)
    extracted_chars = sum(len(text) for _, text in pages)
    extraction_mode = extraction_metadata.get("gemma_extraction_mode", "text")
    print(
        f"  Gemma PDF extraction completed: {len(pages)} pages, {extracted_chars} chars, mode={extraction_mode}",
        flush=True,
    )
    update_processing_metadata(
        args,
        input_mode="extracted_pdf_layout_text" if extraction_mode == "layout" else "extracted_pdf_text",
        extracted_pages=len(pages),
        extracted_chars=extracted_chars,
        **extraction_metadata,
    )
    write_parsed_pdf_text(
        pdf_path=pdf_path,
        args=args,
        pages=pages,
        extraction_metadata=extraction_metadata,
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
        full_text = "\n\n".join(f"[PAGE {page_number}]\n{text}" for page_number, text in pages)
        text_prompt = build_gemma_text_prompt(prompt, pdf_path.name, full_text)
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
            return toc, raw_text, model
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
    update_processing_metadata(args, gemma_level_reference=level_reference_enabled)

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
            )
        )
    update_processing_metadata(args, processed_chunk_count=len(raw_parts))
    update_processing_metadata(
        args,
        gemma_level_reference_levels=sorted(level_reference) if level_reference_enabled else [],
        gemma_level_reference_examples={
            str(level): examples for level, examples in sorted(level_reference.items())
        } if level_reference_enabled else {},
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

    raw_bundle = {
        "mode": "gemma_text_chunk",
        "merge_mode": merge_mode,
        "merge_strategy": getattr(args, "gemma_merge_strategy", DEFAULT_GEMMA_MERGE_STRATEGY),
        "merge_batch_chars": getattr(args, "gemma_merge_batch_chars", DEFAULT_GEMMA_MERGE_BATCH_CHARS),
        "model": model,
        "pdf": pdf_path.name,
        "extraction": extraction_metadata,
        "extracted_pages": len(pages),
        "extracted_chars": extracted_chars,
        "chunk_count": len(chunks),
        "processed_chunk_count": len(raw_parts),
        "level_reference": {
            str(level): examples for level, examples in sorted(level_reference.items())
        } if level_reference_enabled else {},
        "input_chunk_files": getattr(args, "_ai_input_chunk_files", []),
        "chunk_raw_responses": raw_parts,
        "final_raw_response": final_raw_text,
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
            runtime_label = "vLLM" if runtime == "vllm" else "Transformers"
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
        help="Gemma inference runtime: transformers or vllm.",
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
