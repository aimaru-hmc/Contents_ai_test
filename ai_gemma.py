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
DEFAULT_GEMMA_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_GEMMA_BACKEND = os.getenv("GEMMA_BACKEND", "transformers").strip().lower() or "transformers"

DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": os.getenv("GEMINI_MODEL", os.getenv("AI_MODEL", "gemini-3.1-pro-preview")),
    "openai": os.getenv("OPENAI_MODEL", os.getenv("AI_MODEL", "gpt-5.5")),
    "claude": os.getenv("CLAUDE_MODEL", os.getenv("AI_MODEL", "claude-opus-4-8")),
    "gemma": os.getenv("GEMMA_MODEL", os.getenv("OLLAMA_MODEL", os.getenv("AI_MODEL", DEFAULT_GEMMA_HF_MODEL))),
}

DEFAULT_FALLBACK_MODELS_BY_PROVIDER = {
    "gemini": os.getenv("GEMINI_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["gemini"])),
    "openai": os.getenv("OPENAI_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["openai"])),
    "claude": os.getenv("CLAUDE_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["claude"])),
    "gemma": os.getenv(
        "GEMMA_FALLBACK_MODELS",
        os.getenv("OLLAMA_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["gemma"])),
    ),
}

DEFAULT_AI_RETRIES = int(os.getenv("AI_MAX_RETRIES", os.getenv("GEMINI_MAX_RETRIES", "5")))
DEFAULT_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", os.getenv("GEMINI_RETRY_BASE_DELAY", "2.0")))
DEFAULT_FILE_PROCESSING_TIMEOUT = int(os.getenv("GEMINI_FILE_PROCESSING_TIMEOUT", "300"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("AI_MAX_OUTPUT_TOKENS", "8192"))
DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER = {
    "gemini": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
    "openai": int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
    "claude": int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
    "gemma": int(os.getenv("GEMMA_MAX_OUTPUT_TOKENS", os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "8192")))),
}
DEFAULT_GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "256"))
DEFAULT_CLAUDE_TEXT_SINGLE_MAX_CHARS = int(os.getenv("CLAUDE_TEXT_SINGLE_MAX_CHARS", "350000"))
DEFAULT_CLAUDE_TEXT_CHUNK_CHARS = int(os.getenv("CLAUDE_TEXT_CHUNK_CHARS", "120000"))
DEFAULT_CLAUDE_TEXT_MIN_CHUNK_CHARS = int(os.getenv("CLAUDE_TEXT_MIN_CHUNK_CHARS", "30000"))
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
DEFAULT_OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))
DEFAULT_GEMMA_TEXT_SINGLE_MAX_CHARS = int(os.getenv("GEMMA_TEXT_SINGLE_MAX_CHARS", os.getenv("OLLAMA_TEXT_SINGLE_MAX_CHARS", "220000")))
DEFAULT_GEMMA_TEXT_CHUNK_CHARS = int(os.getenv("GEMMA_TEXT_CHUNK_CHARS", os.getenv("OLLAMA_TEXT_CHUNK_CHARS", "220000")))
DEFAULT_GEMMA_TEXT_MIN_CHUNK_CHARS = int(os.getenv("GEMMA_TEXT_MIN_CHUNK_CHARS", os.getenv("OLLAMA_TEXT_MIN_CHUNK_CHARS", "10000")))
DEFAULT_GEMMA_KEEP_ALIVE = os.getenv("GEMMA_KEEP_ALIVE", os.getenv("OLLAMA_KEEP_ALIVE", "5m"))
DEFAULT_GEMMA_THINK = os.getenv("GEMMA_THINK", os.getenv("OLLAMA_THINK", "")).strip()
DEFAULT_GEMMA_CONTEXT_WINDOW = int(os.getenv("GEMMA_CONTEXT_WINDOW", os.getenv("OLLAMA_CONTEXT_WINDOW", "32768")))
DEFAULT_GEMMA_DEVICE_MAP = os.getenv("GEMMA_DEVICE_MAP", "auto").strip() or "auto"
DEFAULT_GEMMA_TORCH_DTYPE = os.getenv("GEMMA_TORCH_DTYPE", os.getenv("GEMMA_DTYPE", "auto")).strip() or "auto"

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
                },
                "required": ["level", "chapter", "page"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "chapters"],
    "additionalProperties": False,
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
        "ollama": "gemma",
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
        "ollama": "gemma",
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
    {{"level": 1, "chapter": "Major section title", "page": 1}},
    {{"level": 2, "chapter": "Subsection title", "page": 3}}
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
        if last_error:
            raise last_error
        raise ValueError(f"Could not find a JSON object in the {provider} response.")

    if not isinstance(data, dict):
        raise ValueError(f"The top-level JSON value in the {provider} response must be an object.")

    return data


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
        chapters.append({
            "level": level,
            "chapter": chapter,
            "page": page,
        })

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
                chapters.append(chapter)

    chapters.sort(key=lambda item: (int(item.get("page", 1) or 1), int(item.get("level", 1) or 1)))
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


def normalize_gemma_backend(backend: str | None) -> str:
    backend = clean_text(backend).lower() or DEFAULT_GEMMA_BACKEND
    aliases = {
        "hf": "transformers",
        "huggingface": "transformers",
        "hugging-face": "transformers",
        "local": "transformers",
        "torch": "transformers",
        "api": "ollama",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"transformers", "ollama"}:
        raise ValueError("--gemma-backend must be either transformers or ollama.")
    return backend


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


def normalize_gemma_ollama_model(model: str) -> str:
    model = clean_text(model)
    if model.startswith("models/"):
        model = model.split("/", 1)[1]

    lower = model.lower()
    if lower in {"latest", "highest", "max"}:
        return DEFAULT_GEMMA_OLLAMA_MODEL

    if lower in {"e2b", "e4b", "12b", "26b", "31b"}:
        return f"gemma4:{lower}"
    if lower == "27b":
        return "gemma3:27b"

    if lower.startswith("google/"):
        lower = lower.split("/", 1)[1]

    if lower.startswith("gemma-4-"):
        if "31b" in lower:
            return "gemma4:31b"
        if "26b" in lower:
            return "gemma4:26b"
        if "12b" in lower:
            return "gemma4:12b"
        if "e4b" in lower:
            return "gemma4:e4b"
        if "e2b" in lower:
            return "gemma4:e2b"

    if lower.startswith("gemma-3-27b"):
        return "gemma3:27b"

    return model


def normalize_gemma_model_for_backend(model: str, backend: str) -> str:
    if backend == "ollama":
        return normalize_gemma_ollama_model(model)
    return normalize_gemma_hf_model(model)

def normalize_ollama_base_url(base_url: str | None) -> str:
    base_url = clean_text(base_url) or DEFAULT_OLLAMA_BASE_URL
    return base_url.rstrip("/")


def parse_ollama_think(value: str | None) -> bool | str | None:
    value = clean_text(value).lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return value


def ollama_api_get(
    base_url: str,
    path: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = normalize_ollama_base_url(base_url) + path
    request = urllib.request.Request(url=url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API error {error.code}: {error_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama API connection failed({url}): {error}") from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Failed to parse Ollama API response JSON: {body[:500]}") from error

    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama API response must be a JSON object.")

    return parsed


def ollama_installed_model_names(base_url: str, timeout_seconds: int) -> set[str]:
    response = ollama_api_get(base_url=base_url, path="/api/tags", timeout_seconds=timeout_seconds)
    names: set[str] = set()

    for model_info in response.get("models", []) or []:
        if not isinstance(model_info, dict):
            continue
        for key in ("name", "model"):
            name = clean_text(model_info.get(key))
            if name:
                names.add(name)

    return names


def ensure_ollama_model_available(model: str, args: argparse.Namespace) -> None:
    installed_models = ollama_installed_model_names(
        base_url=args.ollama_base_url,
        timeout_seconds=args.ollama_request_timeout,
    )

    if model in installed_models:
        return

    installed_text = ", ".join(sorted(installed_models)) or "none"
    raise RuntimeError(
        f"Model '{model}' is not installed in Ollama. "
        f"Run `ollama pull {model}` first. "
        f"Currently installed models: {installed_text}"
    )


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
            "Missing packages required by the Gemma transformers backend: "
            + ", ".join(missing)
            + ". Run `pip install -U transformers torch accelerate safetensors huggingface_hub` first."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Gemma transformers backend could not import the required model classes. "
            "Run `pip install -U transformers torch accelerate safetensors huggingface_hub` first."
        ) from error

    return torch, AutoModelForCausalLM, AutoTokenizer


def gemma_model_needs_gemma4_config(model: str) -> bool:
    model = clean_text(model).lower()
    return "gemma-4" in model or "gemma4" in model


def preflight_gemma_transformers(models: Iterable[str] | None = None) -> None:
    import_gemma_transformers_dependencies()

    if not models or not any(gemma_model_needs_gemma4_config(model) for model in models):
        return

    try:
        from transformers import Gemma4Config  # noqa: F401
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


def preflight_gemma_provider(args: argparse.Namespace) -> None:
    provider_args = build_provider_args(args, "gemma")
    backend = normalize_gemma_backend(getattr(provider_args, "gemma_backend", None))
    preflight_pdf_text_extraction()
    models = [
        normalize_gemma_model_for_backend(model, backend)
        for model in parse_model_list(provider_args.model, provider_args.ai_fallback_models)
    ]

    if backend == "transformers":
        preflight_gemma_transformers(models=models)
        return

    last_error: Exception | None = None

    for model in models:
        try:
            ensure_ollama_model_available(model=model, args=provider_args)
            return
        except Exception as error:
            last_error = error

    if last_error:
        raise last_error


def ollama_api_post(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    url = normalize_ollama_base_url(base_url) + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API error {error.code}: {error_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama API connection failed({url}): {error}") from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Failed to parse Ollama API response JSON: {body[:500]}") from error

    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama API response must be a JSON object.")

    return parsed


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

    bundle = {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": loaded_model,
    }
    _GEMMA_TRANSFORMERS_CACHE[cache_key] = bundle
    print(f"  Gemma/Transformers model loading completed: {model}", flush=True)
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

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at creating tables of contents for PDF textbooks and lecture materials. "
                "Output exactly one valid JSON object. Do not output Markdown or explanations."
            ),
        },
        {"role": "user", "content": prompt},
    ]

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

    return {"response": text}


def build_gemma_payloads(
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None,
    context_window: int | None,
    use_schema: bool,
    keep_alive: str | None,
    think: str | None,
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": (
            "You are an expert at creating tables of contents for PDF textbooks and lecture materials. "
            "Output exactly one valid JSON object. Do not output Markdown or explanations."
        ),
        "stream": False,
    }

    options: dict[str, Any] = {"temperature": float(temperature)}
    if max_output_tokens:
        options["num_predict"] = int(max_output_tokens)
    if context_window:
        options["num_ctx"] = int(context_window)
    base["options"] = options

    keep_alive = clean_text(keep_alive)
    if keep_alive:
        base["keep_alive"] = keep_alive

    parsed_think = parse_ollama_think(think)
    if parsed_think is not None:
        base["think"] = parsed_think

    payloads: list[dict[str, Any]] = []

    if use_schema:
        schema_payload = dict(base)
        schema_payload["format"] = TOC_SCHEMA
        payloads.append(schema_payload)

    json_payload = dict(base)
    json_payload["format"] = "json"
    payloads.append(json_payload)

    payloads.append(base)
    return payloads


def generate_gemma_once(
    model: str,
    prompt: str,
    args: argparse.Namespace,
):
    backend = normalize_gemma_backend(getattr(args, "gemma_backend", None))
    if backend == "transformers":
        return generate_gemma_once_transformers(
            model=model,
            prompt=prompt,
            args=args,
        )

    payloads = build_gemma_payloads(
        model=model,
        prompt=prompt,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        context_window=args.gemma_context_window,
        use_schema=not args.no_schema,
        keep_alive=args.gemma_keep_alive,
        think=args.gemma_think,
    )

    last_error: Exception | None = None
    for index, payload in enumerate(payloads):
        try:
            return ollama_api_post(
                base_url=args.ollama_base_url,
                path="/api/generate",
                payload=payload,
                timeout_seconds=args.ollama_request_timeout,
            )
        except Exception as error:
            last_error = error
            if index < len(payloads) - 1 and is_schema_config_error(error):
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Gemma response generation failed")


def gemma_response_text(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("response") or "")
    return str(response or "")


def build_gemma_text_prompt(base_prompt: str, pdf_name: str, text: str) -> str:
    return f"""
{base_prompt}

[Gemma Text Mode]
Create the TOC using only the [Extracted PDF Text] below instead of an attached PDF.
Use each text block's [PAGE n] marker to determine page numbers.

[PDF File Name]
{pdf_name}

[Extracted PDF Text]
{text}
""".strip()


def build_gemma_chunk_prompt(base_prompt: str, pdf_name: str, chunk: dict[str, Any], total_chunks: int) -> str:
    return f"""
{base_prompt}

[Gemma Text Chunk Mode]
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


def build_gemma_merge_prompt(
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


def process_gemma_text_chunk(
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
    chunk_prompt = build_gemma_chunk_prompt(prompt, pdf_path.name, chunk, total_chunks)
    print(
        f"  Gemma chunk {chunk_label}/{total_chunks} request: pages {chunk['start_page']}-{chunk['end_page']}",
        flush=True,
    )

    try:
        parsed, raw_text = request_gemma_json_text(
            model=model,
            prompt_text=chunk_prompt,
            args=args,
            label=f"Gemma chunk {chunk_label}/{total_chunks}",
        )
        partial_toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
        raw_parts.append({
            "chunk": chunk["index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "raw_response": raw_text,
        })
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
    print("  Gemma PDF text extraction started...", flush=True)
    pages = extract_pdf_text_pages(pdf_path)
    extracted_chars = sum(len(text) for _, text in pages)
    print(f"  Gemma PDF text extraction completed: {len(pages)} pages, {extracted_chars} chars", flush=True)

    single_max_chars = max(10000, int(args.gemma_text_single_max_chars))
    chunk_chars = max(10000, int(args.gemma_text_chunk_chars))

    if extracted_chars <= single_max_chars:
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
            return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model
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
    partial_tocs: list[dict[str, Any]] = []
    raw_parts: list[dict[str, Any]] = []

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
            )
        )

    merge_prompt = build_gemma_merge_prompt(prompt, pdf_path.name, partial_tocs, args.max_depth)
    print(f"  Gemma chunked TOC merge started: {model}", flush=True)
    try:
        parsed, final_raw_text = request_gemma_json_text(
            model=model,
            prompt_text=merge_prompt,
            args=args,
            label=f"Gemma chunked TOC merge({model})",
        )
        toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
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
        "model": model,
        "pdf": pdf_path.name,
        "extracted_pages": len(pages),
        "extracted_chars": extracted_chars,
        "chunk_count": len(chunks),
        "chunk_raw_responses": raw_parts,
        "final_raw_response": final_raw_text,
    }
    return toc, json.dumps(raw_bundle, ensure_ascii=False, indent=2), model


def generate_toc_from_pdf_gemma(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    backend = normalize_gemma_backend(getattr(args, "gemma_backend", None))
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    for index, model in enumerate(models):
        model = normalize_gemma_model_for_backend(model, backend)
        if index > 0:
            print(f"Trying Gemma fallback model: {model}", file=sys.stderr, flush=True)

        try:
            if backend == "ollama":
                print(f"  Using Gemma/Ollama: {normalize_ollama_base_url(args.ollama_base_url)} / {model}", flush=True)
            else:
                print(f"  Using Gemma/Transformers: {model}", flush=True)
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


def add_result_metadata(
    toc: dict[str, Any],
    pdf_path: Path,
    provider: str,
    model: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    result = dict(toc)
    result["_meta"] = {
        "source_pdf": pdf_path.name,
        "provider": provider,
        "model": model,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "elapsed": format_elapsed(elapsed_seconds),
    }
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


def process_pdf(pdf_path: Path, args: argparse.Namespace, user_prompt: str) -> bool:
    print(f"Processing started: {pdf_path.name}", flush=True)
    print(f"  provider: {args.provider}", flush=True)
    started_at = time.perf_counter()

    try:
        toc, raw_text, used_model = generate_toc_from_pdf(pdf_path=pdf_path, args=args, user_prompt=user_prompt)
        elapsed_seconds = time.perf_counter() - started_at
        model_name_for_file = safe_filename_part(used_model)
        output_file = Path(args.output_dir) / f"{pdf_path.stem}_{args.provider}_{model_name_for_file}_toc.json"
        result = add_result_metadata(
            toc=toc,
            pdf_path=pdf_path,
            provider=args.provider,
            model=used_model,
            elapsed_seconds=elapsed_seconds,
        )
        save_json(output_file, result)

        if args.write_raw:
            raw_file = Path(args.output_dir) / f"{pdf_path.stem}_{args.provider}_{model_name_for_file}_raw_response.txt"
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            raw_file.write_text(raw_text, encoding="utf-8")
            print(f"Raw response saved: {raw_file}", flush=True)

        print(f"Completed: {output_file}", flush=True)
        print(f"  model: {used_model}", flush=True)
        print(f"  elapsed: {format_elapsed(elapsed_seconds)} ({elapsed_seconds:.3f}s)", flush=True)
        print(f"  title: {toc.get('title')}", flush=True)
        print(f"  chapters: {len(toc.get('chapters', []))}", flush=True)
        return True

    except Exception as error:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"Failed: {pdf_path.name} / {error}", file=sys.stderr, flush=True)
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
        "--gemma",
        action="store_true",
        help="Shortcut for using the Gemma provider. Same as --provider gemma.",
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
    parser.add_argument("--temperature", type=float, default=0.1, help="AI temperature. Not sent to OpenAI gpt-5-family models or some newer Claude models.")
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
        "--gemma-backend",
        default=DEFAULT_GEMMA_BACKEND,
        help="Gemma execution backend. transformers runs directly with Hugging Face/torch; ollama uses the Ollama API.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API base URL used with --gemma-backend ollama",
    )
    parser.add_argument(
        "--ollama-request-timeout",
        type=int,
        default=DEFAULT_OLLAMA_REQUEST_TIMEOUT,
        help="Ollama request timeout in seconds used with --gemma-backend ollama",
    )
    parser.add_argument(
        "--gemma-device-map",
        default=DEFAULT_GEMMA_DEVICE_MAP,
        help="transformers device_map value used with --gemma-backend transformers",
    )
    parser.add_argument(
        "--gemma-torch-dtype",
        default=DEFAULT_GEMMA_TORCH_DTYPE,
        help="dtype used with --gemma-backend transformers. Examples: auto, bfloat16, float16",
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
    parser.add_argument(
        "--gemma-context-window",
        type=int,
        default=DEFAULT_GEMMA_CONTEXT_WINDOW,
        help="Context window(num_ctx) used for Gemma/Ollama requests",
    )
    parser.add_argument(
        "--gemma-keep-alive",
        default=DEFAULT_GEMMA_KEEP_ALIVE,
        help="Ollama model keep_alive value. If empty, it is not sent.",
    )
    parser.add_argument(
        "--gemma-think",
        default=DEFAULT_GEMMA_THINK,
        help="Ollama think option. Examples: true, false, high. If empty, it is not sent.",
    )
    parser.add_argument("--write-raw", action="store_true", help="Also save raw response text")
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

    if args.gemma:
        provider_text = clean_text(args.provider).lower()
        if provider_text and provider_text not in {"gemma", "ollama", "local"}:
            parser.error("--gemma can only be used with --provider gemma.")
        args.provider = "gemma"

    try:
        providers = resolve_providers(args.provider)
    except ValueError as error:
        parser.error(str(error))

    if "gemma" in providers:
        try:
            args.gemma_backend = normalize_gemma_backend(args.gemma_backend)
        except ValueError as error:
            parser.error(str(error))

    if len(providers) > 1 and args.model is not None:
        parser.error("--model can only be used when a single provider is selected with --provider.")

    if len(providers) > 1 and args.ai_fallback_models is not None:
        parser.error("--ai-fallback-models can only be used when a single provider is selected with --provider.")

    args.max_depth = max(1, min(int(args.max_depth), 10))

    user_prompt = load_prompt(args)
    pdf_files = iter_pdfs(Path(args.path))

    if not pdf_files:
        print("No PDF files to process.", flush=True)
        return 0

    if "gemma" in providers:
        try:
            preflight_gemma_provider(args)
        except Exception as error:
            print(f"Gemma preflight failed: {error}", file=sys.stderr, flush=True)
            return 1

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
