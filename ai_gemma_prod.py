"""Gemma-only production TOC generator.

Purpose (unchanged from ai_gemma.py): read a PDF textbook / lecture material and
produce a study-oriented table of contents as a strict JSON object
({"title": str, "chapters": [{"level", "chapter", "page"}]}).

This build is specialized for local, GPU-backed Gemma inference:

  * Backend         : Hugging Face Transformers only (no Gemini/OpenAI/Claude/Ollama).
  * Ingestion       : HYBRID - per-page rendered images + extracted text with
                      [PAGE n] markers. Sent whole when it fits the context window;
                      auto-split into context-sized page windows (processed in
                      parallel across GPUs, then merged) when it does not.
  * Multi-GPU       : one model replica per GPU. A single big document's chunks are
                      processed concurrently across replicas; use --shard to instead
                      spread one (larger) model across all GPUs.

Designed for boxes like 4x H200 (141 GB each), where a 31B Gemma fits comfortably
on a single card. On a shared box, restrict to free GPUs with CUDA_VISIBLE_DEVICES
or --gpus (or --min-free-gb) so you don't collide with other jobs.

Example:
    python ai_gemma_prod.py ./data/input --model gemma-4-31B-it
    CUDA_VISIBLE_DEVICES=4,5,6,7 python ai_gemma_prod.py book.pdf   # 4 replicas, chunks in parallel
    python ai_gemma_prod.py ./pdfs --chunk-tokens 100000 --write-raw
    python ai_gemma_prod.py huge.pdf --shard                        # one model across all GPUs
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import random
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
load_dotenv()

# The model plus a long-context KV cache leaves little headroom; reducing allocator
# fragmentation reclaims the "reserved but unallocated" memory. Set before torch imports;
# an explicit user value wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OUTPUT_DIR = Path("./data/output")
PROVIDER = "gemma"

DEFAULT_MODEL = os.getenv("GEMMA_MODEL", os.getenv("AI_MODEL", "google/gemma-4-31B-it"))
DEFAULT_FALLBACK_MODELS = os.getenv("GEMMA_FALLBACK_MODELS", "")

DEFAULT_AI_RETRIES = int(os.getenv("AI_MAX_RETRIES", "3"))
DEFAULT_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", "2.0"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("GEMMA_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "16384")))
DEFAULT_TEMPERATURE = float(os.getenv("GEMMA_TEMPERATURE", "0.0"))
DEFAULT_DTYPE = os.getenv("GEMMA_TORCH_DTYPE", os.getenv("GEMMA_DTYPE", "bfloat16")).strip() or "bfloat16"
DEFAULT_ATTN_IMPL = os.getenv("GEMMA_ATTN_IMPL", "sdpa").strip()
DEFAULT_IMAGE_DPI = int(os.getenv("GEMMA_IMAGE_DPI", "150"))
DEFAULT_IMAGE_MAX_DIM = int(os.getenv("GEMMA_IMAGE_MAX_DIM", "1536"))
DEFAULT_MAX_PAGES = int(os.getenv("GEMMA_MAX_PAGES", "0"))  # 0 = all pages
# Soft warning threshold for total input tokens. 0 = derive from model context.
DEFAULT_MAX_INPUT_TOKENS = int(os.getenv("GEMMA_MAX_INPUT_TOKENS", "0"))
# Per-request token budget. Documents above this are split into page windows and merged.
# Sized so weights + KV cache + the materialized O(n^2) attention tensor fit one card with margin.
DEFAULT_CHUNK_TOKENS = int(os.getenv("GEMMA_CHUNK_TOKENS", "60000"))
# Output cap for the final text-only merge; can exceed the per-chunk cap since merge is lighter.
DEFAULT_MERGE_MAX_NEW_TOKENS = int(os.getenv("GEMMA_MERGE_MAX_OUTPUT_TOKENS", "32768"))
# On OOM, a chunk's page window is halved and retried until it reaches this few pages.
DEFAULT_MIN_CHUNK_PAGES = int(os.getenv("GEMMA_MIN_CHUNK_PAGES", "4"))

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

SYSTEM_INSTRUCTION = (
    "You are an expert at creating tables of contents for PDF textbooks and lecture materials. "
    "Output exactly one valid JSON object. Do not output Markdown or explanations."
)

RETRYABLE_ERROR_MARKERS = (
    "TIMEOUT", "TIMED OUT", "TEMPORAR", "CONNECTION", "UNAVAILABLE", "TRY AGAIN LATER",
)

CONFIGURATION_ERROR_MARKERS = (
    "HF_TOKEN", "HUGGINGFACE", "401", "403", "UNAUTHORIZED", "GATED", "ACCESS TO MODEL",
    "PIP INSTALL", "IS NOT A LOCAL FOLDER", "REPOSITORY NOT FOUND",
)


# ----------------------------- Text / JSON utilities -----------------------------

def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_extracted_pdf_text(text: str) -> str:
    text = str(text or "").replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


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


def is_retryable_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def is_configuration_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in CONFIGURATION_ERROR_MARKERS)


def is_cuda_out_of_memory_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return "CUDA OUT OF MEMORY" in message or "OUTOFMEMORYERROR" in message


def is_fatal_cuda_error(error: Exception) -> bool:
    """Errors that corrupt the CUDA context for the whole process (not per-op recoverable)."""
    message = message_of(error).upper()
    return any(marker in message for marker in (
        "ILLEGAL MEMORY ACCESS", "DEVICE-SIDE ASSERT", "MISALIGNED ADDRESS",
        "UNSPECIFIED LAUNCH FAILURE", "UNCORRECTABLE",
    ))


def cuda_device_ctx(torch: Any, device: Any):
    """Pin the current CUDA device for a block so implicit-current-device kernels
    (KV cache, attention workspaces) target this replica's own GPU. Without this,
    multiple replica threads all default to cuda:0 and can trigger illegal memory access."""
    if device is not None and getattr(device, "type", None) == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def empty_torch_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    cuda = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:
            pass


def with_retry(label: str, func: Callable[[], Any], max_retries: int, base_delay: float) -> Any:
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
            print(f"{label} temporary error, retry {attempt}/{max_retries - 1} (after {delay:.1f}s): {error}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)

    if last_error:
        raise last_error
    raise RuntimeError(f"{label} failed")


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

    # Close any containers left open by a truncated response so a cut-off TOC
    # still parses into whatever entries completed (better than total failure).
    while stack:
        container = stack.pop()
        if output and output[-1] == ",":
            output.pop()
        if container["kind"] == "object":
            state = container["state"]
            if state == "colon":
                output.append(":null")
            elif state == "value":
                output.append("null")
            output.append("}")
        else:
            output.append("]")
        # The just-closed container was the parent's pending value; mark it complete
        # so we don't inject a spurious null for a value that was actually present.
        if stack:
            stack[-1]["state"] = "comma_or_end"

    return "".join(output)


def parse_json_response_text(text: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    if not text:
        raise ValueError("Gemma response is empty.")

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
        raise ValueError("Could not find a JSON object in the Gemma response.")

    if not isinstance(data, dict):
        raise ValueError("The top-level JSON value in the Gemma response must be an object.")
    return data


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
        chapters.append({"level": level, "chapter": chapter, "page": page})

    return {"title": title, "chapters": chapters}


# ----------------------------- Model name normalization -----------------------------

_HF_MODEL_ALIASES = {
    "latest": "google/gemma-4-31B-it",
    "highest": "google/gemma-4-31B-it",
    "max": "google/gemma-4-31B-it",
    "31b": "google/gemma-4-31B-it",
    "gemma4:31b": "google/gemma-4-31B-it",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
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


def normalize_hf_model(model: str) -> str:
    model = clean_text(model)
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    lower = model.lower()
    if lower in _HF_MODEL_ALIASES:
        return _HF_MODEL_ALIASES[lower]
    if lower.startswith("google/"):
        return model
    if lower.startswith("gemma-"):
        return f"google/{model}"
    return model


def parse_model_list(primary_model: str, fallback_models: str | Iterable[str] | None) -> list[str]:
    models: list[str] = []

    def add(value: str | None) -> None:
        value = normalize_hf_model(clean_text(value))
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


# ----------------------------- Dependencies / preflight -----------------------------

def import_model_deps() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    pip_name = {"PIL": "Pillow", "fitz": "PyMuPDF"}
    for module_name in ("torch", "transformers", "accelerate", "PIL", "fitz"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name.get(module_name, module_name))
    if missing:
        raise RuntimeError(
            "Missing packages required by the Gemma production script: "
            + ", ".join(missing)
            + ". Run `pip install -U transformers torch accelerate safetensors "
            "huggingface_hub Pillow PyMuPDF` first."
        )

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelClass
    return torch, AutoProcessor, ModelClass


def model_needs_gemma4_config(model: str) -> bool:
    model = clean_text(model).lower()
    return "gemma-4" in model or "gemma4" in model


def preflight(models: Iterable[str]) -> None:
    import_model_deps()
    models = list(models)
    if not any(model_needs_gemma4_config(model) for model in models):
        return
    try:
        from transformers import Gemma4Config  # noqa: F401
    except Exception as error:
        error_text = message_of(error)
        if "torchvision" in error_text:
            raise RuntimeError(
                "Gemma 4 config import failed because torchvision is installed but incompatible "
                "with the current torch build. Reinstall a matching torchvision wheel, or "
                "`python -m pip uninstall -y torchvision` if you do not need it."
            ) from error
        raise RuntimeError(
            "Your installed transformers package does not support Gemma 4 yet "
            "(Gemma4Config is unavailable or broken). Install the latest Transformers build: "
            "`python -m pip install --upgrade --force-reinstall "
            "git+https://github.com/huggingface/transformers.git`"
        ) from error


def hf_token_kwargs() -> dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return {"token": token} if token else {}


def torch_dtype_from_text(torch_module: Any, dtype_text: str) -> Any:
    dtype_text = clean_text(dtype_text).lower()
    if not dtype_text or dtype_text == "auto":
        return "auto"
    aliases = {
        "bf16": "bfloat16", "bfloat16": "bfloat16",
        "fp16": "float16", "float16": "float16", "half": "float16",
        "fp32": "float32", "float32": "float32",
    }
    torch_dtype_name = aliases.get(dtype_text, dtype_text)
    return getattr(torch_module, torch_dtype_name, dtype_text)


# ----------------------------- GPU / replica management -----------------------------

def resolve_gpu_indices(args: argparse.Namespace) -> list[int]:
    import torch
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = clean_text(getattr(args, "gpus", ""))
    if requested:
        indices = []
        for part in requested.split(","):
            part = part.strip()
            if part == "":
                continue
            indices.append(int(part))
        if not indices:
            raise RuntimeError("No valid GPU indices parsed from --gpus.")
        return indices
    if available <= 0:
        return []  # CPU fallback (very slow for large models)

    min_free = max(0.0, float(getattr(args, "min_free_gb", 0.0))) * (1024 ** 3)
    if not min_free:
        return list(range(available))

    # On a shared box, auto-skip GPUs that are already busy (e.g. another job's vLLM).
    indices: list[int] = []
    for index in range(available):
        try:
            free, _ = torch.cuda.mem_get_info(index)
        except Exception:
            free = None
        if free is None or free >= min_free:
            indices.append(index)
        else:
            print(f"  Skipping GPU {index}: only {free / (1024 ** 3):.1f} GiB free "
                  f"(< --min-free-gb {getattr(args, 'min_free_gb', 0)})", flush=True)
    if not indices:
        print("  No GPU meets --min-free-gb; falling back to all visible GPUs.", file=sys.stderr, flush=True)
        return list(range(available))
    return indices


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


def model_input_device(model: Any) -> Any:
    try:
        embeddings = model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None and str(device) != "meta":
            return device
    except Exception:
        pass
    return first_transformers_device(model)


def model_max_context(model: Any) -> int | None:
    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings",):
        value = getattr(config, attr, None)
        if value:
            return int(value)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "max_position_embeddings", None)
    return int(value) if value else None


def load_replica(model_name: str, args: argparse.Namespace, device_map: Any, label: str) -> dict[str, Any]:
    torch, AutoProcessor, ModelClass = import_model_deps()
    token_kwargs = hf_token_kwargs()

    print(f"  [{label}] loading processor: {model_name}", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(model_name, **token_kwargs)
    except Exception as processor_error:
        # The fast (torchvision-backed) image processor can fail to import even when the
        # text config loads fine. Retry with the slow, Pillow-only image processor.
        print(f"  [{label}] fast processor load failed ({processor_error}); retrying with use_fast=False",
              file=sys.stderr, flush=True)
        try:
            processor = AutoProcessor.from_pretrained(model_name, use_fast=False, **token_kwargs)
        except Exception:
            raise RuntimeError(
                f"Failed to load the Gemma processor for '{model_name}'. Original error: {processor_error}. "
                "Likely fixes: (1) upgrade transformers - "
                "`pip install -U --force-reinstall git+https://github.com/huggingface/transformers.git`; "
                "(2) install a torchvision wheel matching your torch/CUDA (the fast image processor needs it); "
                "(3) for a gated repo, set HF_TOKEN and accept the license on the model page."
            ) from processor_error

    load_kwargs: dict[str, Any] = dict(token_kwargs)
    load_kwargs["device_map"] = device_map
    dtype_value = torch_dtype_from_text(torch, args.dtype)
    if dtype_value is not None:
        load_kwargs["dtype"] = dtype_value
    attn_impl = clean_text(getattr(args, "attn_impl", ""))
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    print(f"  [{label}] loading model onto {device_map}: {model_name}", flush=True)

    def attempt(kwargs: dict[str, Any]) -> Any:
        return ModelClass.from_pretrained(model_name, **kwargs)

    try:
        loaded_model = attempt(load_kwargs)
    except TypeError as error:
        # Older transformers use `torch_dtype` instead of `dtype`.
        if "dtype" in str(error) and "dtype" in load_kwargs:
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            loaded_model = attempt(fallback_kwargs)
        else:
            raise
    except Exception as error:
        # A bad attn kernel (e.g. flash-attn not installed) should not be fatal.
        message = message_of(error).lower()
        if "attn_implementation" in load_kwargs and ("attn" in message or "flash" in message):
            print(f"  [{label}] attn_implementation={attn_impl} failed; retrying with default: {error}",
                  file=sys.stderr, flush=True)
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            loaded_model = attempt(fallback_kwargs)
        else:
            raise RuntimeError(
                f"Failed to load Gemma model '{model_name}'. Check the transformers version, "
                f"Hugging Face access/HF_TOKEN, and GPU memory. Cause: {error}"
            ) from error

    eval_method = getattr(loaded_model, "eval", None)
    if callable(eval_method):
        eval_method()

    device = model_input_device(loaded_model)
    context = model_max_context(loaded_model)
    print(f"  [{label}] ready (input device {device}, context {context})", flush=True)
    return {
        "torch": torch,
        "processor": processor,
        "model": loaded_model,
        "device": device,
        "context": context,
        "label": label,
        "model_name": model_name,
        "lock": threading.Lock(),
    }


def load_replicas(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    """Load one replica per GPU (data parallel), or a single sharded replica with --shard."""
    models = parse_model_list(args.model, args.fallback_models)
    gpu_indices = resolve_gpu_indices(args)

    last_error: Exception | None = None
    for model_name in models:
        try:
            if args.shard or not gpu_indices:
                # One model spread across all visible GPUs (or CPU if none).
                device_map = "auto" if gpu_indices else None
                replica = load_replica(model_name, args, device_map, label="shard")
                return model_name, [replica]

            replicas: list[dict[str, Any]] = []
            for index in gpu_indices:
                replica = load_replica(model_name, args, {"": index}, label=f"gpu{index}")
                replicas.append(replica)
            return model_name, replicas
        except Exception as error:
            last_error = error
            print(f"Model failed to load: {model_name} / {error}", file=sys.stderr, flush=True)
            empty_torch_cuda_cache()

    if last_error:
        raise last_error
    raise RuntimeError("No Gemma models could be loaded.")


# ----------------------------- PDF ingestion (hybrid) -----------------------------

def load_pdf_pages(pdf_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return per-page dicts: {"page": int, "text": str, "image": PIL.Image | None}.

    Uses PyMuPDF (fitz) for both text extraction and image rendering so page numbers
    stay aligned. Rendered images are downscaled to --image-max-dim on the long edge.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    include_images = not args.no_images
    include_text = not args.no_text
    dpi = max(48, int(args.image_dpi))
    max_dim = max(256, int(args.image_max_dim))
    max_pages = max(0, int(args.max_pages))
    zoom = dpi / 72.0

    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        total = doc.page_count
        limit = min(total, max_pages) if max_pages else total
        for page_index in range(limit):
            page = doc.load_page(page_index)
            page_number = page_index + 1

            text = ""
            if include_text:
                text = clean_extracted_pdf_text(page.get_text("text") or "")

            image = None
            if include_images:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                longest = max(image.size)
                if longest > max_dim:
                    scale = max_dim / float(longest)
                    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                    image = image.resize(new_size, Image.LANCZOS)

            pages.append({"page": page_number, "text": text, "image": image})

    if not pages:
        raise RuntimeError(f"No pages found in PDF: {pdf_path.name}")

    if include_text and not include_images and not any(p["text"] for p in pages):
        raise RuntimeError(
            "No extractable text was found and images are disabled. "
            "Scanned PDFs need --no-text off with images enabled (default), or OCR."
        )
    return pages


# ----------------------------- Prompt building -----------------------------

def build_instruction_prompt(user_prompt: str, max_depth: int) -> str:
    return f"""
You are an expert at creating tables of contents for PDF textbooks and lecture materials.
Create the table of contents using only the document provided below.

[User Prompt]
{user_prompt.strip()}

[How the document is delivered]
The full document follows as an ordered sequence of pages. Each page is introduced by a
"[PAGE n]" marker, then (when available) the rendered page image, then that page's
extracted text. Use the images for layout, structure, and correct page numbers; use the
extracted text for exact title spelling (including Korean). "n" is the actual PDF viewer
page order and the first page is 1.

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
- Use the document title found in the document. If unclear, use a title close to the file name.
- Sort chapters in the order they appear in the document.
- Use only levels from 1 to {max_depth}.
- page must be an integer and must use actual PDF viewer page order, where the first page is 1.
- Preserve original title numbering, but do not invent missing numbering.
- Preserve original chapter/section titles exactly, including Korean text when the source title is Korean.
- Exclude body sentences, examples, questions, table/figure captions, and references that are not suitable TOC entries.
- Never output any text outside the JSON object.
""".strip()


CLOSING_INSTRUCTION = (
    "Now read the entire document above and output the single JSON table-of-contents object. "
    "Output only the JSON object."
)

RETRY_INSTRUCTION = (
    "[Important Retry Instruction] The previous response failed JSON parsing. "
    "This time, output exactly one valid JSON object that Python json.loads() can parse directly. "
    "Do not output Markdown, explanations, or code fences. Never return an empty response. "
    'If there are no entries, output {"title": "Document title", "chapters": []}.'
)

CHUNK_NOTE = (
    "[Chunk {index}/{total}, pages {start}-{end}] The pages below are ONE PART of a larger PDF. "
    "Extract only chapter/section/subsection titles that actually appear within THIS page range, "
    "using the [PAGE n] markers for page numbers. Do not infer entries outside this range."
)


def build_multimodal_content(
    instruction_prompt: str,
    pdf_name: str,
    pages: list[dict[str, Any]],
    retry: bool = False,
    chunk_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    header = instruction_prompt + f"\n\n[PDF File Name]\n{pdf_name}"
    if chunk_info:
        header += "\n\n" + CHUNK_NOTE.format(
            index=chunk_info["index"], total=chunk_info["total"],
            start=chunk_info["start_page"], end=chunk_info["end_page"],
        )
    if retry:
        header = RETRY_INSTRUCTION + "\n\n" + header
    content.append({"type": "text", "text": header})

    for page in pages:
        content.append({"type": "text", "text": f"[PAGE {page['page']}]"})
        if page.get("image") is not None:
            content.append({"type": "image", "image": page["image"]})
        page_text = page.get("text") or ""
        if page_text:
            content.append({"type": "text", "text": page_text})

    content.append({"type": "text", "text": CLOSING_INSTRUCTION})
    return content


def content_image_count(content: list[dict[str, Any]]) -> int:
    return sum(1 for block in content if block.get("type") == "image")


# ----------------------------- Generation -----------------------------

def run_generate(replica: dict[str, Any], content: list[dict[str, Any]], args: argparse.Namespace,
                 log_prefix: str, max_new_tokens_override: int | None = None) -> str:
    torch = replica["torch"]
    processor = replica["processor"]
    model = replica["model"]
    device = replica["device"]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
        {"role": "user", "content": content},
    ]

    template_kwargs: dict[str, Any] = dict(
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    if args.pan_and_scan:
        template_kwargs["do_pan_and_scan"] = True

    def build_inputs(msgs: list[dict[str, Any]]) -> Any:
        kwargs = dict(template_kwargs)
        try:
            return processor.apply_chat_template(msgs, **kwargs)
        except TypeError:
            kwargs.pop("do_pan_and_scan", None)
            return processor.apply_chat_template(msgs, **kwargs)

    try:
        inputs = build_inputs(messages)
    except Exception as error:
        # Some Gemma chat templates reject a separate system role; fold it into the user turn.
        merged = [{"role": "user", "content": (
            [{"type": "text", "text": SYSTEM_INSTRUCTION}] + content
        )}]
        try:
            inputs = build_inputs(merged)
        except Exception:
            raise RuntimeError(f"Failed to build Gemma inputs: {error}") from error

    if device is not None:
        inputs = inputs.to(device)

    # pixel_values must match the model's compute dtype; input ids must stay integer.
    model_dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype
    if isinstance(inputs, dict) or hasattr(inputs, "keys"):
        if "pixel_values" in inputs and model_dtype is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    input_ids = inputs["input_ids"] if "input_ids" in inputs else getattr(inputs, "input_ids", None)
    input_length = int(input_ids.shape[-1]) if input_ids is not None else 0

    max_new_tokens = max(1, int(max_new_tokens_override or args.max_new_tokens))
    warn_threshold = int(args.max_input_tokens) or (replica.get("context") or 0)
    image_count = content_image_count(content)
    print(f"  {log_prefix} input tokens: {input_length} (+{image_count} page images), "
          f"max_new_tokens {max_new_tokens}", flush=True)
    if warn_threshold and input_length + max_new_tokens > warn_threshold:
        print(f"  {log_prefix} WARNING: input({input_length}) + output({max_new_tokens}) exceeds "
              f"context budget ({warn_threshold}). Consider --image-dpi/--image-max-dim lower, "
              f"--max-pages, or drop --no-text. Output may be truncated.",
              file=sys.stderr, flush=True)

    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    temperature = float(args.temperature)
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    try:
        with cuda_device_ctx(torch, device), torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
    except Exception as error:
        if is_cuda_out_of_memory_error(error):
            empty_torch_cuda_cache()
            raise RuntimeError(
                f"CUDA out of memory during generation. Lower --chunk-tokens/--image-max-dim "
                f"or --max-new-tokens. Cause: {error}"
            ) from error
        raise RuntimeError(f"Gemma generation failed: {error}") from error

    generated_ids = outputs[0][input_length:] if input_length else outputs[0]
    decode = getattr(processor, "decode", None) or getattr(tokenizer, "decode")
    try:
        text = decode(generated_ids, skip_special_tokens=True)
    except TypeError:
        text = decode(generated_ids)
    return str(text)


def image_tokens_for(processor: Any, args: argparse.Namespace) -> int:
    """Tokens a single page image expands to; used only for chunk sizing."""
    for attr in ("image_seq_length", "image_seq_len", "num_image_tokens"):
        value = getattr(processor, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return max(1, int(getattr(args, "image_tokens", 256)))


def estimate_page_token_counts(
    pages: list[dict[str, Any]], tokenizer: Any, image_tokens: int,
    include_text: bool, include_images: bool,
) -> list[int]:
    counts: list[int] = []
    for page in pages:
        total = 8  # [PAGE n] marker + separators
        text = page.get("text") or ""
        if include_text and text:
            try:
                total += len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                total += max(1, len(text) // 2)  # conservative fallback if encode fails
        if include_images and page.get("image") is not None:
            total += image_tokens
        counts.append(total)
    return counts


def plan_chunks(
    pages: list[dict[str, Any]], tokenizer: Any, processor: Any, args: argparse.Namespace,
    max_pages_per_chunk: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Pack whole pages into windows that each fit the per-request token budget.

    A document that fits the budget becomes a single window (i.e. one-shot, no merge).
    Page boundaries are never split, so page numbers stay intact.
    """
    include_text = not args.no_text
    include_images = not args.no_images
    image_tokens = image_tokens_for(processor, args)
    per_page = estimate_page_token_counts(pages, tokenizer, image_tokens, include_text, include_images)

    budget = max(20000, int(args.chunk_tokens))
    overhead = 2000  # reserve for instruction/scaffolding tokens per request

    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = overhead
    for page, page_tokens in zip(pages, per_page):
        over_tokens = current_tokens + page_tokens > budget
        over_pages = max_pages_per_chunk and len(current) >= max_pages_per_chunk
        if current and (over_tokens or over_pages):
            windows.append(current)
            current = []
            current_tokens = overhead
        current.append(page)
        current_tokens += page_tokens
    if current:
        windows.append(current)

    total = len(windows)
    chunks: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        chunks.append({
            "index": index,
            "total": total,
            "start_page": window[0]["page"],
            "end_page": window[-1]["page"],
            "pages": window,
        })
    return chunks, sum(per_page)


def generate_json_toc(
    replica: dict[str, Any],
    content_builder: Callable[[bool], list[dict[str, Any]]],
    args: argparse.Namespace,
    fallback_title: str,
    log_prefix: str,
    max_new_tokens: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Generate once and parse, retrying with a stronger JSON instruction on parse failure.

    content_builder(retry: bool) returns the message content for the attempt.
    The replica lock serializes access to that GPU model object.
    """
    retry_count = max(1, int(args.ai_retries))
    last_error: Exception | None = None

    with replica["lock"]:
        for attempt in range(1, retry_count + 1):
            content = content_builder(attempt > 1)
            if attempt > 1:
                print(f"  {log_prefix} JSON parse retry {attempt}/{retry_count}: {last_error}",
                      file=sys.stderr, flush=True)

            raw_text = with_retry(
                label=f"{log_prefix} generation",
                func=lambda content=content: run_generate(replica, content, args, log_prefix, max_new_tokens),
                max_retries=args.ai_retries,
                base_delay=args.ai_retry_base_delay,
            )

            try:
                parsed = parse_json_response_text(raw_text)
                return validate_toc(parsed, fallback_title, args.max_depth), raw_text
            except Exception as error:
                attach_error_debug(
                    error, provider=PROVIDER, model=replica.get("model_name"),
                    attempt=attempt, retry_count=retry_count,
                    raw_response=raw_text, raw_response_chars=len(raw_text),
                )
                last_error = error
                if attempt >= retry_count:
                    raise

    if last_error:
        raise last_error
    raise RuntimeError(f"{log_prefix} produced no result")


def generate_chunk_partial(
    replica: dict[str, Any], pdf_path: Path, chunk: dict[str, Any],
    args: argparse.Namespace, instruction_prompt: str, log_prefix: str,
) -> tuple[dict[str, Any], str]:
    chunk_info = None
    if chunk["total"] > 1:
        chunk_info = {
            "index": chunk["index"], "total": chunk["total"],
            "start_page": chunk["start_page"], "end_page": chunk["end_page"],
        }

    def builder(retry: bool) -> list[dict[str, Any]]:
        return build_multimodal_content(
            instruction_prompt, pdf_path.name, chunk["pages"], retry=retry, chunk_info=chunk_info,
        )

    return generate_json_toc(replica, builder, args, pdf_path.stem, log_prefix)


def generate_chunk_with_splitting(
    replica: dict[str, Any], pdf_path: Path, chunk: dict[str, Any],
    args: argparse.Namespace, instruction_prompt: str, log_prefix: str, depth: int = 0,
) -> list[dict[str, Any]]:
    """Generate a chunk's partial TOC; on CUDA OOM, halve the page window and retry each half.

    Returns a list of partials ({"toc","raw","start_page","end_page"}) - one when the window
    fits, more when it had to be split down to fit memory. Page boundaries are never split.
    """
    try:
        toc, raw = generate_chunk_partial(replica, pdf_path, chunk, args, instruction_prompt, log_prefix)
        return [{"toc": toc, "raw": raw, "start_page": chunk["start_page"], "end_page": chunk["end_page"]}]
    except Exception as error:
        oom = is_cuda_out_of_memory_error(error)
        if oom:
            empty_torch_cuda_cache()
        pages = chunk["pages"]
        min_pages = max(1, int(getattr(args, "min_chunk_pages", DEFAULT_MIN_CHUNK_PAGES)))
        if not oom or len(pages) <= min_pages or depth >= 5:
            raise
        mid = len(pages) // 2
        if mid < 1:
            raise
        left, right = pages[:mid], pages[mid:]
        print(f"  {log_prefix} OOM at {len(pages)} pages ({chunk['start_page']}-{chunk['end_page']}); "
              f"retrying as {left[0]['page']}-{left[-1]['page']} + {right[0]['page']}-{right[-1]['page']}",
              file=sys.stderr, flush=True)
        partials: list[dict[str, Any]] = []
        for sub in (left, right):
            sub_chunk = {
                "index": chunk["index"], "total": chunk["total"],
                "start_page": sub[0]["page"], "end_page": sub[-1]["page"], "pages": sub,
            }
            sub_prefix = f"[{replica['label']} {pdf_path.name} pages {sub[0]['page']}-{sub[-1]['page']}]"
            partials.extend(generate_chunk_with_splitting(
                replica, pdf_path, sub_chunk, args, instruction_prompt, sub_prefix, depth + 1,
            ))
        return partials


def build_merge_prompt(
    user_prompt: str, pdf_name: str, partial_tocs: list[dict[str, Any]], max_depth: int,
) -> str:
    partial_json = json.dumps(partial_tocs, ensure_ascii=False, indent=2)
    return f"""
You are an expert at creating tables of contents for PDF textbooks and lecture materials.

[User Prompt]
{user_prompt.strip()}

[Chunked TOC Merge Instruction]
Create one final TOC JSON object from the [Partial TOC Candidates] below (each was extracted
from a different page range of the same PDF).
- Keep only one copy of duplicate entries.
- Preserve ascending page order and the original appearance order.
- Remove covers, prefaces, existing TOC pages, indexes, references, and repeated headers/footers.
- Use only levels from 1 to {max_depth}.
- Do not invent chapter titles.
- Preserve original chapter/section titles exactly, including Korean text when the source title is Korean.
- Output exactly one JSON object: {{"title": "...", "chapters": [{{"level": 1, "chapter": "...", "page": 1}}]}}.
  Do not output Markdown, explanations, or code fences.

[PDF File Name]
{pdf_name}

[Partial TOC Candidates]
{partial_json}
""".strip()


def merge_partial_tocs_locally(
    partial_tocs: list[dict[str, Any]], fallback_title: str, max_depth: int,
) -> dict[str, Any]:
    title = ""
    chapters: list[dict[str, Any]] = []
    for toc in partial_tocs:
        if not title:
            title = clean_text(toc.get("title"))
        for chapter in toc.get("chapters", []) or []:
            chapters.append(chapter)
    merged = validate_toc({"title": title or fallback_title, "chapters": chapters}, fallback_title, max_depth)
    merged["chapters"].sort(key=lambda entry: entry.get("page", 1))  # stable ascending page order
    return merged


def merge_chunk_tocs(
    replica: dict[str, Any], pdf_path: Path, partial_tocs: list[dict[str, Any]],
    args: argparse.Namespace, user_prompt: str, log_prefix: str,
) -> tuple[dict[str, Any], str]:
    if getattr(args, "local_merge_only", False):
        return merge_partial_tocs_locally(partial_tocs, pdf_path.stem, args.max_depth), "<local merge>"

    def builder(retry: bool) -> list[dict[str, Any]]:
        prompt = build_merge_prompt(user_prompt, pdf_path.name, partial_tocs, args.max_depth)
        if retry:
            prompt = RETRY_INSTRUCTION + "\n\n" + prompt
        return [{"type": "text", "text": prompt}]

    merge_cap = int(getattr(args, "merge_max_new_tokens", args.max_new_tokens))
    try:
        return generate_json_toc(replica, builder, args, pdf_path.stem, f"{log_prefix} merge",
                                 max_new_tokens=merge_cap)
    except Exception as error:
        print(f"  {log_prefix} model merge failed; using local merge: {error}", file=sys.stderr, flush=True)
        return merge_partial_tocs_locally(partial_tocs, pdf_path.stem, args.max_depth), "<local merge fallback>"


# ----------------------------- Output / logging -----------------------------

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def add_result_metadata(toc: dict[str, Any], pdf_path: Path, model: str, elapsed_seconds: float) -> dict[str, Any]:
    result = dict(toc)
    result["_meta"] = {
        "source_pdf": pdf_path.name,
        "provider": PROVIDER,
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


def write_error_log(pdf_path: Path, args: argparse.Namespace, error: Exception,
                    model: str, elapsed_seconds: float | None = None, stage: str | None = None,
                    fatal: bool = True) -> Path | None:
    if getattr(args, "no_error_log", False):
        return None

    debug = error_debug_info(error)
    output_dir = Path(getattr(args, "output_dir", OUTPUT_DIR))
    error_dir = output_dir / "error_logs"
    error_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_part = safe_filename_part(pdf_path.stem)
    model_text = str(debug.get("model") or model)
    model_part = safe_filename_part(model_text)
    stage_text = stage or "error"
    stage_part = safe_filename_part(stage_text)[:80]
    base_name = f"{timestamp}_{pdf_part}_{PROVIDER}_{model_part}_{stage_part}"

    raw_text = str(debug.get("raw_response") or "")
    raw_response_file: Path | None = None
    if raw_text:
        raw_response_file = unique_output_path(error_dir / f"{base_name}_raw_response.txt")
        raw_response_file.write_text(raw_text, encoding="utf-8")

    debug_payload = {
        str(key): jsonable_debug_value(value)
        for key, value in debug.items() if key != "raw_response"
    }

    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fatal": bool(fatal),
        "stage": stage_text,
        "source_pdf": str(pdf_path),
        "source_pdf_name": pdf_path.name,
        "provider": PROVIDER,
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


# ----------------------------- Per-PDF + orchestration -----------------------------

def run_chunks(
    chunks: list[dict[str, Any]], replicas: list[dict[str, Any]], args: argparse.Namespace,
    pdf_path: Path, instruction_prompt: str,
) -> dict[int, dict[str, Any]]:
    """Generate a partial TOC per chunk, distributing chunks across GPU replicas.

    A failed chunk is recorded (not raised) so one bad window doesn't lose the rest.
    """
    results: dict[int, dict[str, Any]] = {}
    results_lock = threading.Lock()
    stop_event = threading.Event()

    def handle(replica: dict[str, Any], chunk: dict[str, Any]) -> None:
        lp = f"[{replica['label']} {pdf_path.name} chunk {chunk['index']}/{chunk['total']}]"
        print(f"  {lp} pages {chunk['start_page']}-{chunk['end_page']}", flush=True)
        try:
            partials = generate_chunk_with_splitting(replica, pdf_path, chunk, args, instruction_prompt, lp)
            with results_lock:
                results[chunk["index"]] = {"partials": partials}
            chapters = sum(len(p["toc"].get("chapters", [])) for p in partials)
            suffix = "" if len(partials) == 1 else f" across {len(partials)} sub-parts"
            print(f"  {lp} done: {chapters} chapters{suffix}", flush=True)
        except Exception as error:
            if is_cuda_out_of_memory_error(error):
                empty_torch_cuda_cache()
            print(f"  {lp} FAILED: {error}", file=sys.stderr, flush=True)
            with results_lock:
                results[chunk["index"]] = {
                    "error": message_of(error),
                    "start_page": chunk["start_page"], "end_page": chunk["end_page"],
                }
            if is_fatal_cuda_error(error):
                print(f"  {lp} FATAL CUDA error - the process CUDA context is corrupted; aborting "
                      f"remaining chunks and merging what completed. Re-run to reprocess.",
                      file=sys.stderr, flush=True)
                stop_event.set()

    def pin_device(replica: dict[str, Any]) -> None:
        torch = replica.get("torch")
        device = replica.get("device")
        if torch is not None and device is not None and getattr(device, "type", None) == "cuda":
            try:
                torch.cuda.set_device(device)
            except Exception:
                pass

    if len(replicas) == 1 or len(chunks) == 1:
        pin_device(replicas[0])
        for chunk in chunks:
            if stop_event.is_set():
                break
            handle(replicas[0], chunk)
        return results

    chunk_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    for chunk in chunks:
        chunk_queue.put(chunk)

    def worker(replica: dict[str, Any]) -> None:
        pin_device(replica)  # pin this thread's current CUDA device to its replica
        while not stop_event.is_set():
            try:
                chunk = chunk_queue.get_nowait()
            except queue.Empty:
                return
            try:
                handle(replica, chunk)
            finally:
                chunk_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
        for replica in replicas:
            executor.submit(worker, replica)
    return results


def process_pdf(pdf_path: Path, replicas: list[dict[str, Any]], args: argparse.Namespace,
                instruction_prompt: str, user_prompt: str) -> bool:
    log_prefix = f"[{pdf_path.name}]"
    print(f"Processing started: {log_prefix}", flush=True)
    started_at = time.perf_counter()
    model_name = replicas[0].get("model_name", args.model)
    processor = replicas[0]["processor"]
    tokenizer = getattr(processor, "tokenizer", processor)

    try:
        pages = load_pdf_pages(pdf_path, args)
        image_pages = sum(1 for p in pages if p.get("image") is not None)
        text_pages = sum(1 for p in pages if p.get("text"))
        chunks, est_tokens = plan_chunks(pages, tokenizer, processor, args)
        mode = "one-shot" if len(chunks) == 1 else f"{len(chunks)} chunks x {len(replicas)} replica(s)"
        print(f"  {log_prefix} ingested {len(pages)} pages ({image_pages} images, {text_pages} with text), "
              f"~{est_tokens} tokens -> {mode} (budget {args.chunk_tokens})", flush=True)

        chunk_results = run_chunks(chunks, replicas, args, pdf_path, instruction_prompt)
        ordered = [chunk_results[i] for i in sorted(chunk_results)]
        partial_entries: list[dict[str, Any]] = []
        for entry in ordered:
            partial_entries.extend(entry.get("partials", []))
        partial_entries.sort(key=lambda p: p.get("start_page", 1))
        partial_tocs = [p["toc"] for p in partial_entries]
        failed = [entry for entry in ordered if "error" in entry]

        if not partial_tocs:
            first = failed[0]["error"] if failed else "unknown error"
            raise RuntimeError(f"All {len(chunks)} chunk(s) failed. First error: {first}")
        if failed:
            print(f"  {log_prefix} WARNING: {len(failed)}/{len(chunks)} chunk(s) failed; "
                  f"merging the {len(partial_tocs)} partial TOC(s) that succeeded", file=sys.stderr, flush=True)

        if len(partial_tocs) == 1:
            toc = partial_tocs[0]
            raw_text = partial_entries[0].get("raw", "")
        else:
            toc, merge_raw = merge_chunk_tocs(replicas[0], pdf_path, partial_tocs, args, user_prompt, log_prefix)
            raw_text = json.dumps({
                "mode": "chunked",
                "model": model_name,
                "pdf": pdf_path.name,
                "pages": len(pages),
                "chunks": len(chunks),
                "partials": len(partial_tocs),
                "failed_chunks": [
                    {"start_page": f["start_page"], "end_page": f["end_page"], "error": f["error"]}
                    for f in failed
                ],
                "chunk_raw_responses": [
                    {"start_page": p.get("start_page"), "end_page": p.get("end_page"), "raw_response": p.get("raw", "")}
                    for p in partial_entries
                ],
                "final_raw_response": merge_raw,
            }, ensure_ascii=False, indent=2)

        elapsed_seconds = time.perf_counter() - started_at
        model_name_for_file = safe_filename_part(model_name)
        output_file = Path(args.output_dir) / f"{pdf_path.stem}_{PROVIDER}_{model_name_for_file}_toc.json"
        result = add_result_metadata(toc, pdf_path, model_name, elapsed_seconds)
        result["_meta"]["pages"] = len(pages)
        result["_meta"]["chunks"] = len(chunks)
        if failed:
            result["_meta"]["failed_chunks"] = len(failed)
        save_json(output_file, result)

        if args.write_raw:
            raw_file = Path(args.output_dir) / f"{pdf_path.stem}_{PROVIDER}_{model_name_for_file}_raw_response.txt"
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            raw_file.write_text(raw_text, encoding="utf-8")
            print(f"Raw response saved: {raw_file}", flush=True)

        print(f"Completed: {output_file}", flush=True)
        print(f"  {log_prefix} model {model_name}, elapsed {format_elapsed(elapsed_seconds)}, "
              f"title {toc.get('title')!r}, chapters {len(toc.get('chapters', []))}", flush=True)
        return True
    except Exception as error:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"Failed: {log_prefix} / {error}", file=sys.stderr, flush=True)
        error_log = write_error_log(pdf_path, args, error, model=model_name,
                                    elapsed_seconds=elapsed_seconds, stage="PDF processing failed", fatal=True)
        if error_log is not None:
            print(f"  Error log saved: {error_log}", file=sys.stderr, flush=True)
        if args.stop_on_error:
            raise
        return False


def process_all(pdf_files: list[Path], replicas: list[dict[str, Any]], args: argparse.Namespace,
                instruction_prompt: str, user_prompt: str) -> dict[str, bool]:
    """Process PDFs one at a time; within each PDF, chunks run in parallel across replicas."""
    results: dict[str, bool] = {}
    for pdf_path in pdf_files:
        ok = process_pdf(pdf_path, replicas, args, instruction_prompt, user_prompt)
        results[str(pdf_path)] = ok
        if not ok and args.stop_on_error:
            break
    return results


# ----------------------------- vLLM backend -----------------------------

def preflight_vllm() -> None:
    missing: list[str] = []
    pip_name = {"fitz": "PyMuPDF", "PIL": "Pillow"}
    for module_name in ("vllm", "torch", "fitz", "PIL"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name.get(module_name, module_name))
    if missing:
        raise RuntimeError(
            "vLLM backend requires: " + ", ".join(missing)
            + ". Install with `pip install vllm PyMuPDF Pillow` (torch comes with vllm)."
        )


def pil_to_data_uri(image: Any) -> str:
    import base64
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_vllm_conversation(
    instruction_prompt: str, pdf_name: str, pages: list[dict[str, Any]],
    chunk_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    header = instruction_prompt + f"\n\n[PDF File Name]\n{pdf_name}"
    if chunk_info:
        header += "\n\n" + CHUNK_NOTE.format(
            index=chunk_info["index"], total=chunk_info["total"],
            start=chunk_info["start_page"], end=chunk_info["end_page"],
        )
    content.append({"type": "text", "text": header})
    for page in pages:
        content.append({"type": "text", "text": f"[PAGE {page['page']}]"})
        if page.get("image") is not None:
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_uri(page["image"])}})
        page_text = page.get("text") or ""
        if page_text:
            content.append({"type": "text", "text": page_text})
    content.append({"type": "text", "text": CLOSING_INSTRUCTION})
    return [{"role": "user", "content": content}]


def make_vllm_sampling_params(args: argparse.Namespace, max_tokens: int, use_schema: bool):
    from vllm import SamplingParams
    base = dict(max_tokens=max(1, int(max_tokens)), temperature=max(0.0, float(args.temperature)))
    if use_schema and not args.no_schema:
        # API name changed across vLLM versions; try newest first, then fall back to prompt-only.
        for module_name, class_name, kwarg in (
            ("vllm.sampling_params", "StructuredOutputsParams", "structured_outputs"),
            ("vllm.sampling_params", "GuidedDecodingParams", "guided_decoding"),
        ):
            try:
                module = __import__(module_name, fromlist=[class_name])
                params_cls = getattr(module, class_name)
                return SamplingParams(**base, **{kwarg: params_cls(json=TOC_SCHEMA)})
            except Exception:
                continue
    return SamplingParams(**base)


def build_vllm_engine(args: argparse.Namespace) -> tuple[Any, str]:
    try:
        from vllm import LLM
    except ImportError as error:
        raise RuntimeError("vLLM backend requires `pip install vllm`.") from error
    import torch

    model_name = normalize_hf_model(parse_model_list(args.model, args.fallback_models)[0])
    tp = int(args.tensor_parallel_size) or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    tp = max(1, tp)

    if int(args.vllm_max_model_len) > 0:
        max_model_len = int(args.vllm_max_model_len)
    else:
        max_model_len = int(args.chunk_tokens) + int(args.max_new_tokens) + 8192

    engine_kwargs: dict[str, Any] = dict(
        model=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": int(args.vllm_max_images)},
        dtype=(clean_text(args.dtype) or "auto"),
    )
    if clean_text(getattr(args, "vllm_model_impl", "")):
        engine_kwargs["model_impl"] = args.vllm_model_impl
    if getattr(args, "vllm_trust_remote_code", False):
        engine_kwargs["trust_remote_code"] = True
    if getattr(args, "vllm_enforce_eager", False):
        engine_kwargs["enforce_eager"] = True

    print(f"  Building vLLM engine: {model_name} (TP={tp}, max_model_len={max_model_len}, "
          f"max_images/prompt={args.vllm_max_images})", flush=True)
    llm = LLM(**engine_kwargs)
    return llm, model_name


def vllm_generate_tocs(
    llm: Any, conversations: list[list[dict[str, Any]]], args: argparse.Namespace,
    max_tokens: int, use_schema: bool, fallback_title: str,
) -> list[tuple[dict[str, Any] | None, str]]:
    """Batch-generate over conversations; parse each into a validated TOC (None on parse failure)."""
    sampling = make_vllm_sampling_params(args, max_tokens, use_schema)
    outputs = llm.chat(conversations, sampling_params=sampling)
    results: list[tuple[dict[str, Any] | None, str]] = []
    for output in outputs:
        text = output.outputs[0].text if getattr(output, "outputs", None) else ""
        try:
            results.append((validate_toc(parse_json_response_text(text), fallback_title, args.max_depth), text))
        except Exception as error:
            attach_error_debug(error, provider=PROVIDER, raw_response=text, raw_response_chars=len(text))
            results.append((None, text))
    return results


def process_pdf_vllm(llm: Any, model_name: str, pdf_path: Path, args: argparse.Namespace,
                     instruction_prompt: str, user_prompt: str) -> bool:
    log_prefix = f"[{pdf_path.name}]"
    print(f"Processing started: {log_prefix}", flush=True)
    started_at = time.perf_counter()
    tokenizer = llm.get_tokenizer()

    try:
        pages = load_pdf_pages(pdf_path, args)
        chunks, est_tokens = plan_chunks(pages, tokenizer, None, args,
                                         max_pages_per_chunk=int(args.vllm_max_images))
        print(f"  {log_prefix} ingested {len(pages)} pages, ~{est_tokens} tokens -> {len(chunks)} chunk(s) "
              f"(budget {args.chunk_tokens}); batching through vLLM", flush=True)

        conversations: list[list[dict[str, Any]]] = []
        for chunk in chunks:
            chunk_info = None
            if chunk["total"] > 1:
                chunk_info = {"index": chunk["index"], "total": chunk["total"],
                              "start_page": chunk["start_page"], "end_page": chunk["end_page"]}
            conversations.append(build_vllm_conversation(instruction_prompt, pdf_path.name, chunk["pages"], chunk_info))

        chunk_out = vllm_generate_tocs(llm, conversations, args, args.max_new_tokens,
                                       use_schema=True, fallback_title=pdf_path.stem)
        partials: list[dict[str, Any]] = []
        raws: list[dict[str, Any]] = []
        failed = 0
        for chunk, (toc, raw) in zip(chunks, chunk_out):
            if toc is None:
                failed += 1
                print(f"  {log_prefix} chunk {chunk['index']}/{chunk['total']} "
                      f"(pages {chunk['start_page']}-{chunk['end_page']}) parse failed", file=sys.stderr, flush=True)
                continue
            partials.append(toc)
            raws.append({"start_page": chunk["start_page"], "end_page": chunk["end_page"], "raw_response": raw})
            print(f"  {log_prefix} chunk {chunk['index']}/{chunk['total']}: "
                  f"{len(toc.get('chapters', []))} chapters", flush=True)

        if not partials:
            raise RuntimeError(f"All {len(chunks)} chunk(s) failed to produce valid JSON.")

        if len(partials) == 1:
            toc = partials[0]
            raw_text = raws[0]["raw_response"]
        elif args.local_merge_only:
            toc = merge_partial_tocs_locally(partials, pdf_path.stem, args.max_depth)
            raw_text = "<local merge>"
        else:
            merge_prompt = build_merge_prompt(user_prompt, pdf_path.name, partials, args.max_depth)
            merge_conv = [[{"role": "user", "content": [{"type": "text", "text": merge_prompt}]}]]
            merged = vllm_generate_tocs(llm, merge_conv, args, args.merge_max_new_tokens,
                                        use_schema=True, fallback_title=pdf_path.stem)
            toc, merge_raw = merged[0]
            if toc is None:
                print(f"  {log_prefix} model merge parse failed; using local merge", file=sys.stderr, flush=True)
                toc = merge_partial_tocs_locally(partials, pdf_path.stem, args.max_depth)
                merge_raw = "<local merge fallback>"
            raw_text = json.dumps({
                "mode": "vllm_chunked", "model": model_name, "pdf": pdf_path.name,
                "pages": len(pages), "chunks": len(chunks), "failed_chunks": failed,
                "chunk_raw_responses": raws, "final_raw_response": merge_raw,
            }, ensure_ascii=False, indent=2)

        elapsed = time.perf_counter() - started_at
        model_part = safe_filename_part(model_name)
        output_file = Path(args.output_dir) / f"{pdf_path.stem}_{PROVIDER}_{model_part}_toc.json"
        result = add_result_metadata(toc, pdf_path, model_name, elapsed)
        result["_meta"]["pages"] = len(pages)
        result["_meta"]["chunks"] = len(chunks)
        if failed:
            result["_meta"]["failed_chunks"] = failed
        save_json(output_file, result)

        if args.write_raw:
            raw_file = Path(args.output_dir) / f"{pdf_path.stem}_{PROVIDER}_{model_part}_raw_response.txt"
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            raw_file.write_text(raw_text, encoding="utf-8")
            print(f"Raw response saved: {raw_file}", flush=True)

        print(f"Completed: {output_file}", flush=True)
        print(f"  {log_prefix} elapsed {format_elapsed(elapsed)}, title {toc.get('title')!r}, "
              f"chapters {len(toc.get('chapters', []))}"
              + (f", {failed} chunk(s) failed" if failed else ""), flush=True)
        return True
    except Exception as error:
        elapsed = time.perf_counter() - started_at
        print(f"Failed: {log_prefix} / {error}", file=sys.stderr, flush=True)
        log = write_error_log(pdf_path, args, error, model=model_name,
                              elapsed_seconds=elapsed, stage="vLLM processing failed", fatal=True)
        if log is not None:
            print(f"  Error log saved: {log}", file=sys.stderr, flush=True)
        if args.stop_on_error:
            raise
        return False


def run_vllm(pdf_files: list[Path], args: argparse.Namespace,
             instruction_prompt: str, user_prompt: str) -> dict[str, bool]:
    llm, model_name = build_vllm_engine(args)
    print(f"vLLM engine ready: {model_name}", flush=True)
    results: dict[str, bool] = {}
    for pdf_path in pdf_files:
        ok = process_pdf_vllm(llm, model_name, pdf_path, args, instruction_prompt, user_prompt)
        results[str(pdf_path)] = ok
        if not ok and args.stop_on_error:
            break
    return results


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return DEFAULT_USER_PROMPT


# ----------------------------- CLI -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemma-only production TOC generator (local Transformers, hybrid text+image, whole-PDF)."
    )
    parser.add_argument("path", nargs="?", default="./data/input", help="PDF file or directory")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--prompt", default=None, help="TOC generation instruction")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 text file containing the TOC instruction")

    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemma model id or alias (e.g. gemma-4-31B-it, 12b, e4b)")
    parser.add_argument("--fallback-models", default=DEFAULT_FALLBACK_MODELS,
                        help="Comma-separated models to try if the primary fails to load")

    parser.add_argument("--max-depth", type=int, default=6, help="Maximum TOC level (1-10)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature; 0 = greedy/deterministic (recommended for TOC)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Maximum generated tokens (cap; generation stops at EOS)")
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES,
                        help="Retries for JSON parse failures / transient errors")
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)

    # Ingestion (hybrid text + image).
    parser.add_argument("--image-dpi", type=int, default=DEFAULT_IMAGE_DPI, help="Render DPI for page images")
    parser.add_argument("--image-max-dim", type=int, default=DEFAULT_IMAGE_MAX_DIM,
                        help="Downscale rendered pages so the long edge <= this many pixels")
    parser.add_argument("--pan-and-scan", action="store_true",
                        help="Enable Gemma pan-and-scan (better on dense pages, more image tokens)")
    parser.add_argument("--no-images", action="store_true", help="Text-only: do not render/send page images")
    parser.add_argument("--no-text", action="store_true", help="Images-only: do not send extracted text")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Cap pages sent (0 = all)")
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS,
                        help="Warn if input+output exceeds this (0 = use model context length)")

    # Chunking (auto: whole-PDF when it fits the budget, else split into page windows + merge).
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS,
                        help="Per-request token budget; documents over this split into page windows and merge")
    parser.add_argument("--image-tokens", type=int, default=256,
                        help="Assumed tokens per page image for chunk sizing when the processor doesn't report it")
    parser.add_argument("--local-merge-only", action="store_true",
                        help="Merge chunk TOCs with local dedup/sort instead of an extra model call")
    parser.add_argument("--merge-max-new-tokens", type=int, default=DEFAULT_MERGE_MAX_NEW_TOKENS,
                        help="Output token cap for the final merge step (text-only, can exceed per-chunk cap)")
    parser.add_argument("--min-chunk-pages", type=int, default=DEFAULT_MIN_CHUNK_PAGES,
                        help="On OOM, keep halving a chunk's page window until it has this few pages")

    # Backend / GPU.
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="auto, bfloat16 (recommended on H200), float16, float32")
    parser.add_argument("--attn-impl", default=DEFAULT_ATTN_IMPL,
                        help="attn_implementation: sdpa, flash_attention_2, eager (blank = library default)")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU indices (default: all visible)")
    parser.add_argument("--min-free-gb", type=float, default=0.0,
                        help="When --gpus is unset, only use GPUs with at least this many GiB free "
                             "(auto-skips cards busy with other jobs; 0 = use all)")
    parser.add_argument("--shard", action="store_true",
                        help="[transformers] Spread ONE model across all GPUs instead of one replica per GPU")

    # Backend selection + vLLM engine options.
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"],
                        help="Inference backend. 'vllm' = high-throughput engine (recommended for large jobs).")
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="[vllm] GPUs to shard one engine across (0 = all visible)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="[vllm] Fraction of each GPU vLLM may use for weights + KV cache")
    parser.add_argument("--vllm-max-model-len", type=int, default=0,
                        help="[vllm] Max sequence length (0 = chunk-tokens + max-new-tokens + margin)")
    parser.add_argument("--vllm-max-images", type=int, default=128,
                        help="[vllm] Max images per prompt; also caps pages per chunk")
    parser.add_argument("--vllm-model-impl", default="",
                        help="[vllm] model_impl: blank=auto, or 'transformers' if the native arch is unsupported")
    parser.add_argument("--vllm-trust-remote-code", action="store_true",
                        help="[vllm] pass trust_remote_code=True to the engine")
    parser.add_argument("--vllm-enforce-eager", action="store_true",
                        help="[vllm] disable CUDA graphs (slower; for debugging)")

    parser.add_argument("--write-raw", action="store_true", help="Also save raw model response text")
    parser.add_argument("--no-error-log", action="store_true", help="Disable error logs under output_dir/error_logs")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop as soon as any PDF fails")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.no_images and args.no_text:
        parser.error("--no-images and --no-text cannot be used together (no content would be sent).")

    args.max_depth = max(1, min(int(args.max_depth), 10))

    models = parse_model_list(args.model, args.fallback_models)
    try:
        if args.backend == "vllm":
            preflight_vllm()
        else:
            preflight(models)
    except Exception as error:
        print(f"Preflight failed: {error}", file=sys.stderr, flush=True)
        return 1

    user_prompt = load_prompt(args)
    instruction_prompt = build_instruction_prompt(user_prompt, args.max_depth)

    try:
        pdf_files = iter_pdfs(Path(args.path))
    except Exception as error:
        print(f"Input error: {error}", file=sys.stderr, flush=True)
        return 1
    if not pdf_files:
        print("No PDF files to process.", flush=True)
        return 0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    try:
        if args.backend == "vllm":
            results = run_vllm(pdf_files, args, instruction_prompt, user_prompt)
        else:
            print(f"Loading Gemma replicas (mode: {'shard' if args.shard else 'one-per-GPU'})...", flush=True)
            model_name, replicas = load_replicas(args)
            print(f"Loaded model {model_name} on {len(replicas)} replica(s).", flush=True)
            results = process_all(pdf_files, replicas, args, instruction_prompt, user_prompt)
    except Exception as error:
        print(f"Aborted: {error}", file=sys.stderr, flush=True)
        return 1

    elapsed = time.perf_counter() - started_at
    success_count = sum(1 for ok in results.values() if ok)
    failed_count = len(pdf_files) - success_count
    print(f"Processing result: success {success_count} / failed {failed_count} "
          f"(total {format_elapsed(elapsed)})", flush=True)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
