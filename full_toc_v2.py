from __future__ import annotations

import argparse
import math
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parent


def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_dotenv_files() -> None:
    candidates = [Path.cwd() / ".env", ROOT / ".env", ROOT.parent / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_dotenv_file(resolved)


load_dotenv_files()
DEFAULT_INPUT_DIR = ROOT / "data/input"
DEFAULT_DATA_DIR = ROOT / "data/full_toc"
DEFAULT_PARSED_DIR = DEFAULT_DATA_DIR / "parsed"
DEFAULT_LAYOUT_DIR = DEFAULT_DATA_DIR / "layout"
DEFAULT_FULL_JSON_DIR = DEFAULT_DATA_DIR / "full_json"
DEFAULT_LOG_DIR = DEFAULT_DATA_DIR / "log"
DEFAULT_OPENAI_LAYOUT_MODEL = os.getenv("OPENAI_LAYOUT_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.5"))
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "3600"))
DEFAULT_OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_LAYOUT_MAX_TOKENS", "12000"))
LOGGER: "RunLogger | None" = None



DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-4-31B-it")
DEFAULT_FALLBACK_MODELS = os.getenv("GEMMA_FALLBACK_MODELS", "")
DEFAULT_AI_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 3600

PAGE_RE = re.compile(r"^\[PAGE\s+(\d+)\]$")
LINE_RE = re.compile(
    r"^\[L s=(?P<s>[\d.]+) r=(?P<r>[\d.]+) x=(?P<x>-?[\d.]+) "
    r"y=(?P<y>-?[\d.]+) b=(?P<b>[01]) i=(?P<i>[01]) "
    r"f=(?P<f>[^]]+)\]\s*(?P<t>.*)$"
)
RETRYABLE_ERROR_MARKERS = (
    "429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "500", "INTERNAL", "502", "503",
    "UNAVAILABLE", "HIGH DEMAND", "OVERLOADED", "TRY AGAIN LATER", "504",
    "DEADLINE_EXCEEDED", "TIMEOUT", "TIMED OUT", "TEMPORAR", "CONNECTION",
)
CONFIGURATION_ERROR_MARKERS = (
    "API KEY", "API_KEY", "UNAUTHENTICATED", "PERMISSION_DENIED", "UNAUTHORIZED", "401", "403",
)


@dataclass(frozen=True)
class Line:
    page: int
    order: int
    s: float
    r: float
    x: float
    y: float
    b: int
    i: int
    f: str
    text: str


class GemmaOutputTruncatedError(ValueError):
    pass


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._stream.close()

    def _write(self, level: str, message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self._stream.write(f"[{timestamp}] {level}: {message}\n")
        self._stream.flush()

    def detail(self, message: str) -> None:
        self._write("INFO", message)

    def info(self, message: str) -> None:
        print(message, flush=True)
        self._write("INFO", message)

    def error(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        self._write("ERROR", message)


def log_detail(message: str) -> None:
    if LOGGER is not None:
        LOGGER.detail(message)


def log_info(message: str) -> None:
    if LOGGER is not None:
        LOGGER.info(message)
    else:
        print(message, flush=True)


def log_error(message: str) -> None:
    if LOGGER is not None:
        LOGGER.error(message)
    else:
        print(message, file=sys.stderr, flush=True)


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_extracted_pdf_text(text: str) -> str:
    text = str(text or "").replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def norm(value: Any) -> str:
    return re.sub(r"\s+", "", clean_text(value)).replace("–", "-").replace("—", "-")


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
    return max(order, key=lambda item: counts[item]) if counts else ""


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


def format_layout_line(line: dict[str, Any], body_size: float, font_ids: dict[str, str]) -> str:
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
    return pages, {
        "extraction_mode": "layout",
        "layout_line_count": line_count,
        "layout_font_count": len(font_ids),
        "plain_extracted_chars": plain_chars,
    }


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


def parse_layout(text: str) -> list[Line]:
    page = order = 0
    result: list[Line] = []
    for raw in text.splitlines():
        page_match = PAGE_RE.match(raw.strip())
        if page_match:
            page, order = int(page_match.group(1)), 0
            continue
        match = LINE_RE.match(raw.strip())
        if not match or not page:
            continue
        order += 1
        values = match.groupdict()
        title = clean_text(values["t"])
        if title:
            result.append(Line(
                page, order, float(values["s"]), float(values["r"]),
                float(values["x"]), float(values["y"]), int(values["b"]),
                int(values["i"]), values["f"], title,
            ))
    if not result:
        raise ValueError("No layout lines found in parsed text.")
    return result


def annotate_source_orders(text: str) -> str:
    page = order = 0
    annotated: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        page_match = PAGE_RE.match(stripped)
        if page_match:
            page, order = int(page_match.group(1)), 0
            annotated.append(raw)
            continue
        if page and LINE_RE.match(stripped):
            order += 1
            leading = raw[:len(raw) - len(raw.lstrip())]
            raw = leading + f"[L source_order={order} {stripped[3:]}"
        annotated.append(raw)
    return "\n".join(annotated)


def compact_layout_json(layout: dict[str, Any]) -> dict[str, Any]:
    rules = layout.get("rules")
    if not isinstance(rules, list):
        rules = []
    compact_rules: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        compact_rules.append({
            "level": rule.get("level"),
            "s": rule.get("s"),
            "r": rule.get("r"),
            "b": rule.get("b"),
            "i": rule.get("i"),
            "f": rule.get("f"),
            "x_min": rule.get("x_min"),
            "x_max": rule.get("x_max"),
            "y_min": rule.get("y_min"),
            "y_max": rule.get("y_max"),
            "start_shapes": rule.get("start_shapes", []),
            "reference_count": rule.get("reference_count"),
            "examples": rule.get("examples", [])[:5],
        })
    candidates = layout.get("contextual_heading_candidates")
    if not isinstance(candidates, list):
        candidates = []
    compact_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        compact_candidates.append({
            "level": candidate.get("level"),
            "chapter": candidate.get("chapter"),
            "page": candidate.get("page"),
            "source_order": candidate.get("source_order"),
            "evidence": candidate.get("evidence", []),
        })
    return {
        "source_reference": layout.get("source_reference"),
        "source_chunk": layout.get("source_chunk"),
        "matching_priority": layout.get("matching_priority", []),
        "rules": compact_rules,
        "contextual_heading_candidates": compact_candidates,
        "unmatched_reference_titles": layout.get("unmatched_reference_titles", []),
    }


def build_prompt(layout_json: dict[str, Any], parsed_text: str, title: str, max_depth: int, user_prompt: str) -> str:
    layout_text = json.dumps(compact_layout_json(layout_json), ensure_ascii=False, indent=2)
    return f"""
You are Gemma 31B. Create the full table-of-contents JSON from the parsed PDF layout text.
The layout JSON was generated from the full parsed PDF by OpenAI and provides heading-level rules.

[Task]
{user_prompt.strip()}

[Document Title]
{title}

[Layout JSON]
{layout_text}

[Rules]
- First apply all exclusion rules. Only then use the Layout JSON rules to classify the remaining headings.
- Exclusion rules have higher priority than every layout, style, position, and numbering rule.
- Treat the Layout JSON rules as strong examples, not as an exhaustive whitelist of heading styles or levels.
- For styles represented in Layout JSON, match headings by S/R/B/I/F first, then start_shapes, X/Y position, and numbering.
- Before returning JSON, audit the current parsed lines against every Layout JSON rule and include every valid matching heading.
- Treat Layout JSON contextual_heading_candidates as strong heading candidates, not final truth. Include them when surrounding parsed text supports a subsection role.
- Do not reject a short standalone candidate solely because its font size, ratio, or font matches body text.
- For body-like typography, use context: parent heading, line brevity, standalone placement, following explanatory paragraphs, and neighboring heading sequence.
- Never reject a heading only because its S/R/B/I/F style or target level is absent from Layout JSON.
- For an unseen style, infer heading candidacy and level from numbering hierarchy, indentation and position, typographic contrast, neighboring headings, and parent-child sequence.
- Use only text that appears in the parsed layout text. Do not invent, rewrite, or summarize titles.
- Preserve original numbering and title text exactly.
- Exclude covers, prefaces, existing TOC listing rows, indexes, page numbers, repeated headers/footers, captions, and body sentences.
- Existing TOC/Contents pages may help you understand structure, but do not output listing rows unless the same title appears as a real body heading.
- Preserve PDF page order and source order within each page.
- Every parsed layout line has a page-local source_order. Copy the exact source_order of the selected heading line.
- Use integer levels from 1 to {max_depth}; lower numbers are higher-level headings.
- Include level_reason for every chapter, explaining the layout or numbering evidence briefly.

[Output Format]
Output exactly one valid JSON object. Do not output Markdown, code fences, comments, or explanations outside JSON.

{{
  "title": "{title}",
  "chapters": [
    {{"level": 1, "chapter": "Source heading", "page": 1, "source_order": 3, "level_reason": "Matched layout rule and numbering."}}
  ]
}}

[Parsed PDF Layout Text]
{parsed_text}
""".strip()


def parsed_page_blocks(parsed_text: str) -> list[tuple[int, list[str]]]:
    blocks: list[tuple[int, list[str]]] = []
    current_page: int | None = None
    current_lines: list[str] = []
    preamble: list[str] = []
    for raw in parsed_text.splitlines():
        page_match = PAGE_RE.match(raw.strip())
        if page_match:
            if current_page is not None:
                blocks.append((current_page, current_lines))
            elif preamble:
                preamble = []
            current_page = int(page_match.group(1))
            current_lines = [raw]
            continue
        if current_page is None:
            preamble.append(raw)
        else:
            current_lines.append(raw)
    if current_page is not None:
        blocks.append((current_page, current_lines))
    return blocks


TOKEN_ENCODER: Any | None = None


def estimate_text_tokens(text: str) -> int:
    global TOKEN_ENCODER
    text = str(text or "")
    if not text:
        return 0
    if TOKEN_ENCODER is None:
        try:
            import tiktoken  # type: ignore
            TOKEN_ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:
            TOKEN_ENCODER = False
    if TOKEN_ENCODER:
        return len(TOKEN_ENCODER.encode(text))
    # Korean and JSON are usually denser than English token estimates; keep this conservative.
    return max(1, int(len(text) / 2.2))


def finalize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        chunk["index"] = index
        chunk["total"] = total
    return chunks


def page_chunk_text(blocks: list[tuple[int, list[str]]], start: int, end: int, overlap_pages: int) -> tuple[int, int, str]:
    context_start = max(0, start - overlap_pages)
    context_end = min(len(blocks), end + overlap_pages)
    selected_blocks = blocks[context_start:context_end]
    chunk_text = "\n".join("\n".join(lines) for _, lines in selected_blocks)
    return context_start, context_end, chunk_text


def build_page_count_chunks(blocks: list[tuple[int, list[str]]], chunk_pages: int, overlap_pages: int) -> list[dict[str, Any]]:
    chunk_pages = max(1, int(chunk_pages))
    overlap_pages = max(0, min(int(overlap_pages), chunk_pages - 1))
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(blocks):
        end = min(len(blocks), start + chunk_pages)
        context_start, context_end, chunk_text = page_chunk_text(blocks, start, end, overlap_pages)
        chunks.append({
            "index": 0,
            "total": 0,
            "mode": "pages",
            "target_start_page": blocks[start][0],
            "target_end_page": blocks[end - 1][0],
            "context_start_page": blocks[context_start][0],
            "context_end_page": blocks[context_end - 1][0],
            "estimated_tokens": estimate_text_tokens(chunk_text),
            "text": chunk_text,
        })
        start = end
    return finalize_chunks(chunks)


def sum_range(values: list[int], start: int, end: int) -> int:
    return sum(values[max(0, start):max(0, end)])


def build_token_limit_chunks(blocks: list[tuple[int, list[str]]], token_limit: int, overlap_pages: int) -> list[dict[str, Any]]:
    token_limit = max(1, int(token_limit))
    overlap_pages = max(0, int(overlap_pages))
    page_texts = ["\n".join(lines) for _, lines in blocks]
    page_tokens = [estimate_text_tokens(text) for text in page_texts]
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(blocks):
        end = start + 1
        best_end = end
        while end <= len(blocks):
            context_start = max(0, start - overlap_pages)
            context_end = min(len(blocks), end + overlap_pages)
            context_tokens = sum_range(page_tokens, context_start, context_end)
            if context_tokens > token_limit and end > start + 1:
                break
            best_end = end
            if context_tokens >= token_limit:
                break
            end += 1
        end = best_end
        context_start, context_end, chunk_text = page_chunk_text(blocks, start, end, overlap_pages)
        target_tokens = sum_range(page_tokens, start, end)
        context_tokens = sum_range(page_tokens, context_start, context_end)
        if context_tokens > token_limit:
            log_error(
                f"Single Gemma chunk exceeds token target: pages {blocks[start][0]}-{blocks[end - 1][0]} "
                f"estimated={context_tokens}, limit={token_limit}. Continuing with smallest possible chunk."
            )
        chunks.append({
            "index": 0,
            "total": 0,
            "mode": "tokens",
            "target_start_page": blocks[start][0],
            "target_end_page": blocks[end - 1][0],
            "context_start_page": blocks[context_start][0],
            "context_end_page": blocks[context_end - 1][0],
            "estimated_tokens": estimate_text_tokens(chunk_text),
            "target_estimated_tokens": target_tokens,
            "context_estimated_tokens": context_tokens,
            "text": chunk_text,
        })
        start = end
    return finalize_chunks(chunks)


def build_gemma_chunks(
    parsed_text: str,
    *,
    chunk_mode: str,
    chunk_pages: int,
    token_limit: int,
    overlap_pages: int,
) -> list[dict[str, Any]]:
    blocks = parsed_page_blocks(parsed_text)
    if not blocks:
        return [{
            "index": 1,
            "total": 1,
            "mode": "single",
            "target_start_page": None,
            "target_end_page": None,
            "context_start_page": None,
            "context_end_page": None,
            "estimated_tokens": estimate_text_tokens(parsed_text),
            "text": parsed_text,
        }]

    if clean_text(chunk_mode).lower() == "pages":
        return build_page_count_chunks(blocks, chunk_pages=chunk_pages, overlap_pages=overlap_pages)
    return build_token_limit_chunks(blocks, token_limit=token_limit, overlap_pages=overlap_pages)


def resolve_gemma_chunk_token_limit(
    *,
    layout_json: dict[str, Any],
    title: str,
    args: argparse.Namespace,
) -> int:
    requested = int(args.gemma_chunk_token_limit or 0)
    if requested > 0:
        return requested
    overhead_prompt = build_prompt(
        layout_json=layout_json,
        parsed_text="",
        title=title,
        max_depth=args.max_depth,
        user_prompt=build_chunk_user_prompt(args.prompt, {
            "index": 1,
            "total": 1,
            "target_start_page": 1,
            "target_end_page": 1,
            "context_start_page": 1,
            "context_end_page": 1,
        }),
    )
    overhead_tokens = estimate_text_tokens(overhead_prompt)
    max_context = max(1, int(args.gemma_max_context_tokens))
    max_output = max(1, int(args.max_output_tokens))
    safety = max(0, int(args.gemma_chunk_safety_tokens))
    return max(1000, max_context - max_output - overhead_tokens - safety)


def build_chunk_user_prompt(base_prompt: str, chunk: dict[str, Any]) -> str:
    target_start = chunk.get("target_start_page")
    target_end = chunk.get("target_end_page")
    context_start = chunk.get("context_start_page")
    context_end = chunk.get("context_end_page")
    if target_start is None or target_end is None:
        return base_prompt
    return f"""
{base_prompt.strip()}

This is chunk {chunk.get('index')} of {chunk.get('total')}.
Target pages: {target_start}-{target_end}.
Context pages included: {context_start}-{context_end}.
Output headings only when the selected heading line is inside the target page range.
Use context pages only to infer hierarchy and avoid boundary mistakes.
""".strip()


def aggregate_api_usages(provider: str, model: str, usages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(usage_prompt_tokens(usage.get("usage", usage)) for usage in usages if isinstance(usage, dict))
    completion_tokens = sum(usage_completion_tokens(usage.get("usage", usage)) for usage in usages if isinstance(usage, dict))
    total_tokens = sum(usage_total_tokens(usage.get("usage", usage)) for usage in usages if isinstance(usage, dict))
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "chunks": usages,
    }


def chat_url(base_url: str) -> str:
    url = clean_text(base_url).rstrip("/")
    if not url:
        raise ValueError("--base-url is empty.")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def extract_chat_response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Model returned no choices: {str(data)[:1000]}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            return "\n".join(
                clean_text(part.get("text") if isinstance(part, dict) else part)
                for part in content
                if clean_text(part.get("text") if isinstance(part, dict) else part)
            )
        return str(content or "")
    return str(first.get("text") or "")


def call_gemma_openai_compatible(
    *,
    model: str,
    prompt: str,
    base_url: str,
    api_key: str,
    timeout: int,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output exactly one valid JSON object. Do not output Markdown or explanations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": max(0.0, float(temperature)),
        "max_tokens": max(1, int(max_output_tokens)),
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if clean_text(api_key):
        headers["Authorization"] = f"Bearer {clean_text(api_key)}"
    url = chat_url(base_url)
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout))) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemma request failed: HTTP {error.code} {error.reason}. URL={url}. Body: {error_body[:1200]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Gemma server is not reachable at {url}: {error}") from error
    try:
        data = json.loads(response_text)
    except Exception as error:
        raise RuntimeError(f"Gemma server returned non-JSON response: {response_text[:1000]}") from error
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    if clean_text(first.get("finish_reason")).lower() == "length":
        raise GemmaOutputTruncatedError(f"Gemma output reached the {max_output_tokens}-token limit before JSON completion.")
    return {"text": extract_chat_response_text(data), "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {}}


def message_of(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def is_retryable_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def is_configuration_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in CONFIGURATION_ERROR_MARKERS)


def is_non_retryable_request_error(error: Exception) -> bool:
    message = message_of(error).upper()
    markers = ("HTTP 400", "HTTP 404", "MAXIMUM CONTEXT LENGTH", "INPUT_TOKENS", "DOES NOT EXIST", "NOTFOUNDERROR")
    return any(marker in message for marker in markers)


def with_retry(label: str, func: Callable[[], Any], max_retries: int, base_delay: float) -> Any:
    max_retries = max(1, int(max_retries))
    base_delay = max(0.1, float(base_delay))
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as error:
            last_error = error
            if is_configuration_error(error) or is_non_retryable_request_error(error):
                raise
            if not is_retryable_error(error) or attempt >= max_retries:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
            delay += random.uniform(0, min(1.0, base_delay))
            log_error(f"{label} temporary error, retry {attempt}/{max_retries - 1} (after {delay:.1f}s): {error}")
            time.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError(f"{label} failed")


def strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object_text(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
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
    return ""


def parse_json_response_text(text: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    if not text:
        raise ValueError("Response is empty.")
    candidates = [text]
    extracted = extract_json_object_text(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON value must be an object.")
            return data
        except Exception as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Could not parse JSON response.")


def validate_toc(data: dict[str, Any], fallback_title: str, max_depth: int) -> dict[str, Any]:
    title = clean_text(data.get("title")) or clean_text(fallback_title) or "Document title"
    raw_chapters = data.get("chapters", [])
    if not isinstance(raw_chapters, list):
        raw_chapters = []
    chapters: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
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
        level = max(1, min(level, max(1, int(max_depth))))
        page = max(1, page)
        key = (re.sub(r"\s+", "", chapter).lower(), page)
        if key in seen:
            continue
        seen.add(key)
        row = {"level": level, "chapter": chapter, "page": page}
        try:
            source_order = int(item.get("source_order"))
        except (TypeError, ValueError):
            source_order = None
        if source_order is not None and source_order > 0:
            row["source_order"] = source_order
        reason = clean_text(item.get("level_reason"))
        if reason:
            row["level_reason"] = reason
        chapters.append(row)
    return {"title": title, "chapters": chapters}


def match_toc_source_line(title: str, page_lines: list[Line]) -> Line | None:
    wanted = norm(title)
    if not wanted:
        return None
    ranked: list[tuple[int, int, int, Line]] = []
    for line in page_lines:
        actual = norm(line.text)
        if actual == wanted:
            score = 5
        elif actual.startswith(wanted):
            score = 4
        elif wanted in actual:
            score = 3
        elif actual in wanted and len(actual) >= 4:
            score = 2
        else:
            continue
        ranked.append((score, -abs(len(actual) - len(wanted)), -line.order, line))
    return max(ranked, key=lambda item: item[:3])[3] if ranked else None


def attach_layout_metadata_and_sort(toc: dict[str, Any], parsed_lines: list[Line]) -> dict[str, Any]:
    pages: dict[int, list[Line]] = defaultdict(list)
    lines_by_id: dict[tuple[int, int], Line] = {}
    for line in parsed_lines:
        pages[line.page].append(line)
        lines_by_id[(line.page, line.order)] = line
    chapters: list[dict[str, Any]] = []
    for merge_index, item in enumerate(toc.get("chapters", [])):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        page = int(row.get("page", 1))
        try:
            requested_source_order = int(row.pop("source_order"))
        except (KeyError, TypeError, ValueError):
            requested_source_order = None
        line = lines_by_id.get((page, requested_source_order)) if requested_source_order is not None else None
        if line is None:
            line = match_toc_source_line(clean_text(row.get("chapter")), pages.get(page, []))
        if line is None:
            row["metadata"] = {"matched_parsed_layout": False, "source_order": None}
            sort_order = math.inf
            sort_y = math.inf
        else:
            row["metadata"] = {
                "matched_parsed_layout": True,
                "source_order": line.order,
                "s": line.s,
                "r": line.r,
                "x": line.x,
                "y": line.y,
                "b": line.b,
                "i": line.i,
                "f": line.f,
                "source_text": line.text,
            }
            sort_order = line.order
            sort_y = line.y
        row["_sort_key"] = (page, sort_order, sort_y, merge_index)
        chapters.append(row)
    chapters.sort(key=lambda item: item["_sort_key"])
    for item in chapters:
        item.pop("_sort_key", None)
    result = dict(toc)
    result["chapters"] = chapters
    return result


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


def normalize_gemma_model(model: str) -> str:
    model = clean_text(model) or DEFAULT_MODEL
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    if model in {"latest", "highest", "max"}:
        return "gemma4:31b"
    if model in {"e2b", "e4b", "12b", "26b", "31b"}:
        return f"gemma4:{model}"
    if model == "27b":
        return "gemma3:27b"
    return model


def request_gemma_toc(prompt: str, args: argparse.Namespace, fallback_title: str) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    models = parse_model_list(args.model, args.ai_fallback_models) or [DEFAULT_MODEL]
    last_error: Exception | None = None
    for model_index, raw_model in enumerate(models):
        model = normalize_gemma_model(raw_model)
        if model_index > 0:
            log_detail(f"Trying Gemma fallback model: {model}")
        for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
            request_prompt = prompt
            if parse_attempt > 1:
                request_prompt = prompt + "\n\n[Retry Instruction]\nThe previous response was not valid JSON. Return only one JSON object parseable by Python json.loads. Do not use Markdown."
                log_detail(f"Retrying Gemma JSON generation {parse_attempt}/{args.ai_retries}: {model}")
            try:
                log_detail(f"Gemma full JSON generation started: {model}")
                response = with_retry(
                    label=f"Gemma full JSON generation({model})",
                    func=lambda model=model, request_prompt=request_prompt: call_gemma_openai_compatible(
                        model=model,
                        prompt=request_prompt,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        timeout=args.timeout,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    ),
                    max_retries=args.ai_retries,
                    base_delay=args.ai_retry_base_delay,
                )
                log_detail(f"Gemma full JSON generation completed: {model}")
                raw_text = str(response.get("text") or "")
                usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
                parsed = parse_json_response_text(raw_text)
                api_usage = build_token_usage("gemma_vllm", model, usage)
                return validate_toc(parsed, fallback_title=fallback_title, max_depth=args.max_depth), raw_text, model, api_usage
            except Exception as error:
                last_error = error
                if isinstance(error, GemmaOutputTruncatedError):
                    raise
                if is_non_retryable_request_error(error):
                    log_error(f"Gemma model failed: {model} / {error}")
                    break
                if parse_attempt >= max(1, int(args.ai_retries)):
                    log_error(f"Gemma model failed: {model} / {error}")
                    break
    if last_error:
        raise last_error
    raise RuntimeError("No Gemma model was available.")


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def safe_filename_part(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "document"


def iter_input_pdfs(input_dir: Path, input_files: list[Path] | None) -> list[Path]:
    if input_files:
        pdfs: list[Path] = []
        for item in input_files:
            path = item.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Input file not found: {path}")
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"Input file is not a PDF: {path}")
            pdfs.append(path)
        return sorted(dict.fromkeys(pdfs))

    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    pdfs = sorted(path.resolve() for path in input_dir.rglob("*.pdf") if path.is_file())
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under: {input_dir}")
    return pdfs


def parsed_path_for_pdf(pdf_path: Path, parsed_dir: Path) -> Path:
    return parsed_dir / f"{safe_filename_part(pdf_path.stem)}_parsed.txt"


def parse_pdf_if_needed(pdf_path: Path, parsed_dir: Path, force_parse: bool = False) -> tuple[Path, str, bool]:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = parsed_path_for_pdf(pdf_path, parsed_dir)
    if parsed_path.is_file() and not force_parse:
        return parsed_path, parsed_path.read_text(encoding="utf-8"), False

    log_detail(f"PDF parsing started: {pdf_path.name}")
    pages, metadata = extract_pdf_layout_pages(pdf_path)
    parsed_text = parsed_pdf_text_for_file(
        pdf_path=pdf_path,
        pages=pages,
        extraction_metadata=metadata,
        source_pdf_path=pdf_path,
    )
    parsed_path.write_text(parsed_text, encoding="utf-8")
    log_detail(f"Parsed text saved: {parsed_path}")
    return parsed_path, parsed_text, True


def usage_total_tokens(usage: dict[str, Any]) -> int:
    for key in ("total_tokens", "total_tokens_used"):
        try:
            return int(usage.get(key) or 0)
        except (TypeError, ValueError):
            pass
    return usage_prompt_tokens(usage) + usage_completion_tokens(usage)


def usage_prompt_tokens(usage: dict[str, Any]) -> int:
    for key in ("prompt_tokens", "input_tokens"):
        try:
            return int(usage.get(key) or 0)
        except (TypeError, ValueError):
            pass
    return 0


def usage_completion_tokens(usage: dict[str, Any]) -> int:
    for key in ("completion_tokens", "output_tokens"):
        try:
            return int(usage.get(key) or 0)
        except (TypeError, ValueError):
            pass
    return 0


def build_token_usage(provider: str, model: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "usage": usage,
        "prompt_tokens": usage_prompt_tokens(usage),
        "completion_tokens": usage_completion_tokens(usage),
        "total_tokens": usage_total_tokens(usage),
    }


def openai_chat_url(base_url: str) -> str:
    url = clean_text(base_url).rstrip("/")
    if not url:
        raise ValueError("--openai-base-url is empty.")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenAI returned no choices: {str(data)[:1000]}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else first.get("text")
    if isinstance(content, list):
        return "\n".join(
            clean_text(part.get("text") if isinstance(part, dict) else part)
            for part in content
            if clean_text(part.get("text") if isinstance(part, dict) else part)
        )
    return str(content or "")


def call_openai_layout(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
    max_tokens: int,
    temperature: float | None,
) -> dict[str, Any]:
    if not clean_text(api_key):
        raise RuntimeError("OpenAI API key is empty. Set OPENAI_API_KEY or pass --openai-api-key.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You analyze parsed PDF layout text and output exactly one JSON object. "
                    "Do not output Markdown or explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    if max_tokens:
        payload["max_completion_tokens"] = max(1, int(max_tokens))
    if temperature is not None:
        payload["temperature"] = max(0.0, float(temperature))

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {clean_text(api_key)}",
    }
    url = openai_chat_url(base_url)
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout))) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI layout request failed: HTTP {error.code} {error.reason}. "
            f"URL={url}. Body: {error_body[:1200]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OpenAI layout API is not reachable at {url}: {error}") from error

    try:
        data = json.loads(response_text)
    except Exception as error:
        raise RuntimeError(f"OpenAI returned non-JSON response: {response_text[:1000]}") from error
    return {"text": extract_chat_content(data), "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {}}


def build_openai_layout_prompt(parsed_text: str, title: str, max_depth: int) -> str:
    return f"""
Create a compact Layout JSON for table-of-contents extraction from the parsed PDF layout text.

[Document Title]
{title}

[Task]
- Inspect all parsed layout lines.
- Infer reusable heading-level rules from typography, numbering hierarchy, indentation, page order, examples, and surrounding context.
- The output will be passed to Gemma to produce the final full TOC JSON.
- Include only rules that help identify real body headings.
- Use surrounding context, not only typography. Some true subheadings may share the same font size/style as body text.
- Preserve short standalone lines that introduce the following paragraph or subsection as contextual heading candidates, even if their typography matches body text.
- A contextual candidate is stronger when it appears under a parent heading, is followed by explanatory body paragraphs, is shorter than surrounding body lines, and fits the neighboring heading sequence.
- Do not discard a candidate solely because its font size, ratio, or font matches body text.
- Exclude covers, prefaces, existing TOC listing rows, repeated headers/footers, page numbers, captions, references, problem sets, and body sentences.
- Use integer levels from 1 to {max_depth}; lower numbers are higher-level headings.
- Keep examples as exact source text from parsed lines.

[Output Format]
Output exactly one valid JSON object with this shape:
{{
  "source_reference": "openai_layout_from_full_pdf",
  "source_chunk": "full_pdf",
  "matching_priority": ["S/R/B/I/F", "numbering", "X", "Y", "examples"],
  "rules": [
    {{
      "level": 1,
      "s": 31.0,
      "r": 3.27,
      "b": 0,
      "i": 0,
      "f": "f1",
      "x_min": 70.0,
      "x_max": 90.0,
      "y_min": 80.0,
      "y_max": 760.0,
      "start_shapes": ["A", "#."],
      "reference_count": 3,
      "examples": ["Exact heading text"]
    }}
  ],
  "contextual_heading_candidates": [
    {{
      "level": 6,
      "chapter": "Exact candidate text",
      "page": 1,
      "source_order": 12,
      "evidence": [
        "short standalone line",
        "under a parent heading",
        "followed by explanatory body paragraph",
        "body-like typography but subsection role"
      ]
    }}
  ],
  "unmatched_reference_titles": []
}}

[Parsed PDF Layout Text]
{parsed_text}
""".strip()


def normalize_layout_json(layout: dict[str, Any], parsed_path: Path, pdf_path: Path) -> dict[str, Any]:
    rules = layout.get("rules")
    if not isinstance(rules, list):
        rules = []

    normalized_rules: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        try:
            level = int(rule.get("level"))
        except (TypeError, ValueError):
            continue
        row = {
            "level": level,
            "s": rule.get("s"),
            "r": rule.get("r"),
            "b": rule.get("b"),
            "i": rule.get("i"),
            "f": rule.get("f"),
            "x_min": rule.get("x_min"),
            "x_max": rule.get("x_max"),
            "y_min": rule.get("y_min"),
            "y_max": rule.get("y_max"),
            "start_shapes": rule.get("start_shapes") if isinstance(rule.get("start_shapes"), list) else [],
            "reference_count": rule.get("reference_count"),
            "examples": rule.get("examples") if isinstance(rule.get("examples"), list) else [],
        }
        normalized_rules.append(row)

    raw_candidates = layout.get("contextual_heading_candidates")
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    normalized_candidates: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        chapter = clean_text(candidate.get("chapter"))
        if not chapter:
            continue
        try:
            level = int(candidate.get("level"))
        except (TypeError, ValueError):
            continue
        try:
            page = int(candidate.get("page"))
        except (TypeError, ValueError):
            page = None
        try:
            source_order = int(candidate.get("source_order"))
        except (TypeError, ValueError):
            source_order = None
        row: dict[str, Any] = {"level": level, "chapter": chapter}
        if page is not None and page > 0:
            row["page"] = page
        if source_order is not None and source_order > 0:
            row["source_order"] = source_order
        evidence = candidate.get("evidence")
        row["evidence"] = evidence if isinstance(evidence, list) else []
        normalized_candidates.append(row)

    return {
        "source_reference": str(pdf_path),
        "source_chunk": "full_pdf",
        "source_parsed": str(parsed_path),
        "matching_priority": layout.get("matching_priority")
        if isinstance(layout.get("matching_priority"), list)
        else ["S/R/B/I/F", "numbering", "X", "Y", "examples"],
        "rules": normalized_rules,
        "contextual_heading_candidates": normalized_candidates,
        "unmatched_reference_titles": layout.get("unmatched_reference_titles")
        if isinstance(layout.get("unmatched_reference_titles"), list)
        else [],
    }


def generate_layout_json(parsed_text: str, parsed_path: Path, pdf_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any]]:
    title = clean_text(args.title) or pdf_path.stem
    prompt = build_openai_layout_prompt(parsed_text=parsed_text, title=title, max_depth=args.max_depth)
    log_detail(f"OpenAI layout generation started: {args.openai_model}")
    response = with_retry(
        label=f"OpenAI layout generation({args.openai_model})",
        func=lambda: call_openai_layout(
            prompt=prompt,
            model=args.openai_model,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            timeout=args.openai_timeout,
            max_tokens=args.openai_max_tokens,
            temperature=args.openai_temperature,
        ),
        max_retries=args.openai_retries,
        base_delay=args.openai_retry_base_delay,
    )
    raw_text = str(response.get("text") or "")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    parsed = parse_json_response_text(raw_text)
    layout_json = normalize_layout_json(parsed, parsed_path=parsed_path, pdf_path=pdf_path)
    layout_json["_api_usage"] = build_token_usage("openai", args.openai_model, usage)
    log_detail(f"OpenAI layout generation completed: rules={len(layout_json.get('rules', []))}, tokens={usage_total_tokens(usage)}")
    return layout_json, raw_text, layout_json["_api_usage"]


def result_path_for_pdf(pdf_path: Path, full_json_dir: Path, timestamp: str) -> Path:
    stem = safe_filename_part(pdf_path.stem)
    return full_json_dir / f"{stem}_{timestamp}_full_foc.json"


def add_metadata(
    toc: dict[str, Any],
    *,
    pdf_path: Path,
    parsed_path: Path,
    layout_path: Path,
    started_at: datetime,
    completed_at: datetime,
    elapsed_seconds: float,
    parsed_created: bool,
    args: argparse.Namespace,
    gemma_model: str,
    api_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(toc)
    result["_meta"] = {
        "source_pdf": str(pdf_path),
        "source_parsed": str(parsed_path),
        "source_layout": str(layout_path),
        "parsed_created": parsed_created,
        "provider_layout": "openai",
        "openai_layout_model": args.openai_model,
        "provider_toc": "gemma_vllm",
        "gemma_model": gemma_model,
        "generated_at": completed_at.isoformat(timespec="seconds"),
        "generation_started_at": started_at.isoformat(timespec="seconds"),
        "generation_completed_at": completed_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed": format_elapsed(elapsed_seconds),
        "api_usage": api_usage or {},
    }
    return result


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"



def latest_layout_path_for_pdf(pdf_path: Path, layout_dir: Path) -> Path:
    stem = safe_filename_part(pdf_path.stem)
    candidates = sorted(
        layout_dir.glob(f"{stem}_*_layout.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No layout JSON found for {pdf_path.name} under {layout_dir}. "
            "Run --stage layout first or pass --layout-file."
        )
    return candidates[-1]


def layout_path_for_args(pdf_path: Path, args: argparse.Namespace) -> Path:
    if args.layout_file:
        return args.layout_file.expanduser().resolve()
    return latest_layout_path_for_pdf(pdf_path, args.layout_dir.resolve())


def run_layout_stage(
    pdf_path: Path,
    args: argparse.Namespace,
    timestamp: str,
) -> tuple[Path, Path, str, bool]:
    layout_started_at = datetime.now().astimezone()
    layout_started = time.perf_counter()
    parsed_path, parsed_text, parsed_created = parse_pdf_if_needed(
        pdf_path=pdf_path,
        parsed_dir=args.parsed_dir.resolve(),
        force_parse=args.force_parse,
    )
    if not parsed_created:
        log_detail(f"Existing parsed text found, skipping PDF parsing: {parsed_path}")

    layout_json, layout_raw_text, layout_api_usage = generate_layout_json(
        parsed_text=parsed_text,
        parsed_path=parsed_path,
        pdf_path=pdf_path,
        args=args,
    )
    layout_completed_at = datetime.now().astimezone()
    layout_elapsed_seconds = time.perf_counter() - layout_started
    layout_json["_meta"] = {
        "source_pdf": str(pdf_path),
        "source_parsed": str(parsed_path),
        "parsed_created": parsed_created,
        "provider_layout": "openai",
        "openai_layout_model": args.openai_model,
        "generated_at": layout_completed_at.isoformat(timespec="seconds"),
        "generation_started_at": layout_started_at.isoformat(timespec="seconds"),
        "generation_completed_at": layout_completed_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(layout_elapsed_seconds, 3),
        "elapsed": format_elapsed(layout_elapsed_seconds),
        "api_usage": {"layout_generation": layout_api_usage},
    }
    layout_dir = args.layout_dir.resolve()
    layout_dir.mkdir(parents=True, exist_ok=True)
    layout_path = layout_dir / f"{safe_filename_part(pdf_path.stem)}_{timestamp}_layout.json"
    layout_path.write_text(json.dumps(layout_json, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    setattr(args, "_last_layout_api_usage", layout_api_usage)
    if args.write_raw:
        (layout_dir / f"{safe_filename_part(pdf_path.stem)}_{timestamp}_layout_raw.txt").write_text(
            layout_raw_text,
            encoding="utf-8",
        )
    log_info(f"layout saved: {layout_path} | elapsed={format_elapsed(layout_elapsed_seconds)}")
    return parsed_path, layout_path, parsed_text, parsed_created


def run_gemma_stage(
    pdf_path: Path,
    args: argparse.Namespace,
    timestamp: str,
    *,
    parsed_path: Path | None = None,
    parsed_text: str | None = None,
    parsed_created: bool = False,
    layout_path: Path | None = None,
    started_at: datetime | None = None,
    started: float | None = None,
) -> Path:
    if started_at is None:
        started_at = datetime.now().astimezone()
    if started is None:
        started = time.perf_counter()

    if parsed_path is None or parsed_text is None:
        parsed_path, parsed_text, parsed_created = parse_pdf_if_needed(
            pdf_path=pdf_path,
            parsed_dir=args.parsed_dir.resolve(),
            force_parse=args.force_parse,
        )
        if not parsed_created:
            log_detail(f"Existing parsed text found, skipping PDF parsing: {parsed_path}")

    if layout_path is None:
        layout_path = layout_path_for_args(pdf_path, args)
    if not layout_path.is_file():
        raise FileNotFoundError(f"Layout JSON not found: {layout_path}")
    layout_json = json.loads(layout_path.read_text(encoding="utf-8"))
    layout_api_usage = layout_json.get("_api_usage") if isinstance(layout_json, dict) and isinstance(layout_json.get("_api_usage"), dict) else getattr(args, "_last_layout_api_usage", {})
    if not isinstance(layout_json, dict):
        raise ValueError(f"Layout JSON top-level value must be an object: {layout_path}")
    log_detail(f"Using layout JSON: {layout_path}")

    title = clean_text(args.title) or pdf_path.stem
    parsed_lines = parse_layout(parsed_text)
    chunk_token_limit = resolve_gemma_chunk_token_limit(layout_json=layout_json, title=title, args=args)
    chunks = build_gemma_chunks(
        parsed_text=parsed_text,
        chunk_mode=args.gemma_chunk_mode,
        chunk_pages=args.gemma_chunk_pages,
        token_limit=chunk_token_limit,
        overlap_pages=args.gemma_chunk_overlap_pages,
    )
    log_info(
        f"gemma chunks: {len(chunks)} | mode={args.gemma_chunk_mode} | token_limit={chunk_token_limit} | pages_per_chunk={args.gemma_chunk_pages} | overlap={args.gemma_chunk_overlap_pages}"
    )

    all_chapters: list[dict[str, Any]] = []
    raw_text_parts: list[str] = []
    chunk_usages: list[dict[str, Any]] = []
    used_models: list[str] = []
    for chunk in chunks:
        chunk_prompt = build_prompt(
            layout_json=layout_json,
            parsed_text=annotate_source_orders(str(chunk["text"])),
            title=title,
            max_depth=args.max_depth,
            user_prompt=build_chunk_user_prompt(args.prompt, chunk),
        )
        log_info(
            f"gemma chunk {chunk['index']}/{chunk['total']}: target pages {chunk['target_start_page']}-{chunk['target_end_page']} | est_tokens={chunk.get('estimated_tokens', 0)}"
        )
        chunk_toc, chunk_raw_text, used_model, chunk_api_usage = request_gemma_toc(
            prompt=chunk_prompt,
            args=args,
            fallback_title=title,
        )
        used_models.append(used_model)
        raw_text_parts.append(f"[CHUNK {chunk['index']}/{chunk['total']} pages {chunk['target_start_page']}-{chunk['target_end_page']}]\n{chunk_raw_text}")
        chunk_api_usage = dict(chunk_api_usage)
        chunk_api_usage["chunk_index"] = chunk["index"]
        chunk_api_usage["chunk_total"] = chunk["total"]
        chunk_api_usage["target_start_page"] = chunk["target_start_page"]
        chunk_api_usage["target_end_page"] = chunk["target_end_page"]
        chunk_api_usage["context_start_page"] = chunk["context_start_page"]
        chunk_api_usage["context_end_page"] = chunk["context_end_page"]
        chunk_api_usage["chunk_mode"] = chunk.get("mode")
        chunk_api_usage["estimated_input_text_tokens"] = chunk.get("estimated_tokens", 0)
        chunk_usages.append(chunk_api_usage)
        all_chapters.extend(chunk_toc.get("chapters", []))

    used_model = used_models[0] if used_models else normalize_gemma_model(args.model)
    if any(model != used_model for model in used_models):
        used_model = ",".join(dict.fromkeys(used_models))
    gemma_api_usage = aggregate_api_usages("gemma_vllm", used_model, chunk_usages)
    toc = validate_toc({"title": title, "chapters": all_chapters}, fallback_title=title, max_depth=args.max_depth)
    toc = attach_layout_metadata_and_sort(toc, parsed_lines)
    raw_text = "\n\n".join(raw_text_parts)

    completed_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started
    result = add_metadata(
        toc,
        pdf_path=pdf_path,
        parsed_path=parsed_path,
        layout_path=layout_path,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
        parsed_created=parsed_created,
        args=args,
        gemma_model=used_model,
        api_usage={"layout_generation": layout_api_usage, "toc_generation": gemma_api_usage},
    )
    full_json_dir = args.full_json_dir.resolve()
    full_json_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_path_for_pdf(pdf_path, full_json_dir=full_json_dir, timestamp=timestamp)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    if args.write_raw:
        (full_json_dir / f"{safe_filename_part(pdf_path.stem)}_{timestamp}_gemma_raw.txt").write_text(
            raw_text,
            encoding="utf-8",
        )
    layout_tokens = layout_api_usage.get("total_tokens", 0) if isinstance(layout_api_usage, dict) else 0
    gemma_tokens = gemma_api_usage.get("total_tokens", 0) if isinstance(gemma_api_usage, dict) else 0
    log_info(f"toc saved: {output_path} | chapters={len(toc.get('chapters', []))} | tokens(openai={layout_tokens}, gemma={gemma_tokens}) | elapsed={format_elapsed(elapsed_seconds)}")
    return output_path


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> Path | None:
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    timestamp = timestamp_for_filename()

    log_info(f"processing: {pdf_path.name} | stage={args.stage}")
    if args.stage == "layout":
        run_layout_stage(pdf_path, args, timestamp)
        return None

    if args.stage == "gemma":
        return run_gemma_stage(
            pdf_path,
            args,
            timestamp,
            started_at=started_at,
            started=started,
        )

    parsed_path, layout_path, parsed_text, parsed_created = run_layout_stage(pdf_path, args, timestamp)
    return run_gemma_stage(
        pdf_path,
        args,
        timestamp,
        parsed_path=parsed_path,
        parsed_text=parsed_text,
        parsed_created=parsed_created,
        layout_path=layout_path,
        started_at=started_at,
        started=started,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse input PDFs, generate OpenAI layout JSON, then generate full TOC JSON with Gemma.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="기본 PDF 입력 폴더")
    parser.add_argument(
        "--input-file",
        "-f",
        action="append",
        type=Path,
        help="처리할 PDF 파일. 여러 번 지정 가능. 미지정 시 input-dir 아래 모든 PDF를 처리합니다.",
    )
    parser.add_argument("--stage", choices=("all", "layout", "gemma"), default="all", help="실행 단계: all=layout+gemma, layout=layout JSON까지만, gemma=기존 layout JSON으로 TOC만")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="full_toc 산출물 기본 폴더")
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR, help="layout JSON 저장/검색 폴더")
    parser.add_argument("--full-json-dir", type=Path, default=DEFAULT_FULL_JSON_DIR, help="최종 full TOC JSON 저장 폴더")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="실행 로그 저장 폴더")
    parser.add_argument("--layout-file", type=Path, help="--stage gemma에서 사용할 layout JSON. 미지정 시 layout-dir에서 최신 layout을 자동 선택합니다.")
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR, help="PDF 파싱 txt 캐시 저장 폴더")
    parser.add_argument("--force-parse", action="store_true", help="기존 parsed txt가 있어도 PDF를 다시 파싱합니다.")
    parser.add_argument("--title", help="TOC title. 기본값은 PDF 파일명 stem입니다.")
    parser.add_argument("--prompt", default="Create the complete table of contents JSON for the whole document from the layout text.")
    parser.add_argument("--max-depth", type=int, default=7)

    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_LAYOUT_MODEL)
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-base-url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--openai-timeout", type=int, default=DEFAULT_OPENAI_TIMEOUT)
    parser.add_argument("--openai-max-tokens", type=int, default=DEFAULT_OPENAI_MAX_TOKENS)
    parser.add_argument("--openai-temperature", type=float, default=None)
    parser.add_argument("--openai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--openai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)

    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemma/vLLM served model name. '31b' normalizes to gemma4:31b.")
    parser.add_argument("--ai-fallback-models", default=DEFAULT_FALLBACK_MODELS, help="Comma-separated fallback model names. 기본값은 없음.")
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--gemma-chunk-mode", choices=("tokens", "pages"), default="tokens", help="Gemma TOC 청크 방식. 기본값은 tokens")
    parser.add_argument("--gemma-max-context-tokens", type=int, default=262144, help="vLLM 서버 --max-model-len 값")
    parser.add_argument("--gemma-chunk-token-limit", type=int, default=0, help="청크별 parsed text 목표 토큰 수. 0이면 context 한도 기준 최대값 자동 계산")
    parser.add_argument("--gemma-chunk-safety-tokens", type=int, default=81920, help="토큰 청크 자동 계산 시 남겨둘 안전 여유 토큰")
    parser.add_argument("--gemma-chunk-pages", type=int, default=80, help="--gemma-chunk-mode pages일 때 한 번에 처리할 target 페이지 수")
    parser.add_argument("--gemma-chunk-overlap-pages", type=int, default=2, help="청크 경계 문맥용 overlap 페이지 수")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible Gemma/vLLM base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer API key for the Gemma OpenAI-compatible server")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--write-raw", action="store_true", help="OpenAI/Gemma raw responses도 data-dir에 저장합니다.")
    return parser.parse_args()


def main() -> None:
    global LOGGER
    args = parse_args()
    if args.parsed_dir == DEFAULT_PARSED_DIR:
        args.parsed_dir = args.data_dir / "parsed"
    if args.layout_dir == DEFAULT_LAYOUT_DIR:
        args.layout_dir = args.data_dir / "layout"
    if args.full_json_dir == DEFAULT_FULL_JSON_DIR:
        args.full_json_dir = args.data_dir / "full_json"
    if args.log_dir == DEFAULT_LOG_DIR:
        args.log_dir = args.data_dir / "log"
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.parsed_dir.mkdir(parents=True, exist_ok=True)
    args.layout_dir.mkdir(parents=True, exist_ok=True)
    args.full_json_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = timestamp_for_filename()
    LOGGER = RunLogger(args.log_dir / f"full_toc_{run_timestamp}.log")
    log_info(f"log: {LOGGER.path}")
    pdfs = iter_input_pdfs(args.input_dir, args.input_file)
    if args.layout_file and len(pdfs) != 1:
        raise ValueError("--layout-file can be used only when exactly one PDF is selected with --input-file.")
    log_info(f"PDF files: {len(pdfs)}")

    created: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for pdf_path in pdfs:
        try:
            output_path = process_pdf(pdf_path, args)
            if output_path is not None:
                created.append(output_path)
        except Exception as error:
            failures.append((pdf_path, error))
            log_error(f"Failed: {pdf_path} / {error}")

    log_info(f"Created full TOC files: {len(created)}")
    for path in created:
        log_detail(f"created: {path}")

    if failures:
        log_error(f"Failed files: {len(failures)}")
        for pdf_path, error in failures:
            log_error(f"{pdf_path}: {error}")
        if LOGGER is not None:
            LOGGER.close()
        raise SystemExit(1)
    if LOGGER is not None:
        LOGGER.close()


if __name__ == "__main__":
    main()
