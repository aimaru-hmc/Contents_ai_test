from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data/input/chunk1"
OUTPUT_ROOT = ROOT / "data/output_chunk1"
INPUT_CHUNKS_DIR = ROOT / "data/output/input_chunks"
PARSED_PDFS_DIR = ROOT / "data/output/parsed_pdfs"

DEFAULT_MODEL = "gemma4:31b"
DEFAULT_FALLBACK_MODELS = DEFAULT_MODEL
DEFAULT_AI_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 3600
DEFAULT_PARSED_CHUNK_CHARS = 80000
DEFAULT_MAX_CHUNK_SPLIT_DEPTH = 3

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
    """Raised when vLLM stops generation at the configured output-token limit."""


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def norm(value: Any) -> str:
    return re.sub(r"\s+", "", clean(value)).replace("–", "-").replace("—", "-")


def normalized_document_name(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", value)


def token_shape(token: str) -> str:
    result: list[str] = []
    previous = None
    for char in clean(token):
        category = unicodedata.category(char)
        if category.startswith("N"):
            value = "#"
        elif category.startswith("L"):
            value = "A"
        else:
            value = char
        if value != previous or value not in ("#", "A"):
            result.append(value)
        previous = value
    return "".join(result)


def tokens_with_offsets(text: str) -> list[tuple[str, int]]:
    return [(m.group(), m.start()) for m in re.finditer(r"\S+", text)]


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
        title = clean(values["t"])
        if title:
            result.append(Line(
                page, order, float(values["s"]), float(values["r"]),
                float(values["x"]), float(values["y"]), int(values["b"]),
                int(values["i"]), values["f"], title,
            ))
    if not result:
        raise ValueError("레이아웃 줄을 찾지 못했습니다.")
    return result


def match_reference_title(title: str, page_lines: list[Line]) -> Line | None:
    wanted = norm(title)
    ranked: list[tuple[int, int, Line]] = []
    for line in page_lines:
        actual = norm(line.text)
        if actual == wanted:
            score = 4
        elif actual.startswith(wanted) or wanted in actual:
            score = 3
        elif wanted.endswith(actual) and len(actual) >= 4:
            score = 2
        elif actual.endswith(wanted) and len(wanted) >= 4:
            score = 1
        else:
            continue
        ranked.append((score, len(actual), line))
    return max(ranked, key=lambda item: (item[0], item[1], item[2].s))[2] if ranked else None


def learned_start_shape(reference_title: str, matched_line: Line) -> str:
    wanted = norm(reference_title)
    for token, _ in tokens_with_offsets(matched_line.text):
        if norm(token) and norm(token) in wanted:
            return token_shape(token)
    tokens = tokens_with_offsets(reference_title)
    return token_shape(tokens[0][0]) if tokens else ""


def learn_layout_levels(reference: dict[str, Any], chunk_lines: list[Line], reference_path: Path, chunk_path: Path) -> dict[str, Any]:
    pages: dict[int, list[Line]] = defaultdict(list)
    for line in chunk_lines:
        pages[line.page].append(line)

    observations: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for item in reference.get("chapters", []):
        page = int(item["page"])
        line = match_reference_title(item["chapter"], pages[page])
        if line is None:
            unmatched.append({"level": int(item["level"]), "chapter": item["chapter"], "page": page})
            continue
        observations.append({
            "level": int(item["level"]),
            "layout": {"s": line.s, "r": line.r, "x": line.x, "y": line.y, "b": line.b, "i": line.i, "f": line.f},
            "start_shape": learned_start_shape(item["chapter"], line),
            "example": item["chapter"],
        })
    if not observations:
        raise ValueError("기준 TOC와 청크에서 일치하는 제목을 찾지 못했습니다.")

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for obs in observations:
        layout = obs["layout"]
        key = (obs["level"], layout["s"], layout["r"], layout["b"], layout["i"], layout["f"])
        group = grouped.setdefault(key, {
            "level": obs["level"], "s": layout["s"], "r": layout["r"], "b": layout["b"],
            "i": layout["i"], "f": layout["f"], "xs": [], "ys": [],
            "start_shapes": Counter(), "examples": [], "reference_count": 0,
        })
        group["xs"].append(layout["x"])
        group["ys"].append(layout["y"])
        group["start_shapes"][obs["start_shape"]] += 1
        group["examples"].append(obs["example"])
        group["reference_count"] += 1

    rules: list[dict[str, Any]] = []
    for group in grouped.values():
        rules.append({
            "level": group["level"], "s": group["s"], "r": group["r"], "b": group["b"],
            "i": group["i"], "f": group["f"], "x_min": min(group["xs"]), "x_max": max(group["xs"]),
            "y_min": min(group["ys"]), "y_max": max(group["ys"]),
            "start_shapes": sorted(group["start_shapes"]),
            "reference_count": group["reference_count"], "examples": group["examples"][:8],
        })
    rules.sort(key=lambda rule: (rule["level"], -rule["s"], rule["f"]))
    return {
        "source_reference": str(reference_path),
        "source_chunk": str(chunk_path),
        "matching_priority": ["S/R/B/I/F", "learned start shape", "X", "Y"],
        "rules": rules,
        "unmatched_reference_titles": unmatched,
    }


def timestamped_output(explicit: Path | None, default_dir: Path, default_stem: str, label: str, timestamp: str, suffix: str = ".json") -> Path:
    if explicit:
        explicit = Path(explicit)
        if explicit.suffix:
            directory = explicit.parent
            stem = explicit.stem
            suffix = explicit.suffix
        else:
            directory = explicit
            stem = f"{default_stem}_{label}"
    else:
        directory = default_dir
        stem = f"{default_stem}_{label}"
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}_{timestamp}{suffix}"
    sequence = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{timestamp}_{sequence}{suffix}"
        sequence += 1
    return candidate


def discover_input_file(input_dir: Path, kind: str, source_hint: str | None = None) -> Path:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(path for path in input_dir.iterdir() if path.is_file())
    if kind == "parsed":
        candidates = [path for path in files if path.suffix.lower() == ".txt" and "parsed" in path.stem.lower()]
        fallback = [path for path in files if path.suffix.lower() == ".txt"]
    elif kind == "chunk":
        candidates = [
            path for path in files
            if path.suffix.lower() == ".json" and "chunk" in path.stem.lower()
            and "toc" not in path.stem.lower() and "reference" not in path.stem.lower()
        ]
        fallback = []
    elif kind == "reference":
        candidates = [
            path for path in files
            if path.suffix.lower() == ".json"
            and ("reference" in path.stem.lower() or "toc" in path.stem.lower())
        ]
        fallback = []
    else:
        raise ValueError(f"Unknown input kind: {kind}")

    matches = candidates or fallback
    if kind == "reference":
        chunk1_references = [path for path in matches if path.stem.lower().endswith("_chunk1")]
        if chunk1_references:
            if source_hint:
                wanted = normalized_document_name(Path(source_hint).stem)
                matching_references = [
                    path
                    for path in chunk1_references
                    if normalized_document_name(path.stem).startswith(wanted)
                ]
                if matching_references:
                    chunk1_references = matching_references
                else:
                    raise FileNotFoundError(
                        f"No chunk1 reference matching source PDF {source_hint!r} found in: {input_dir}"
                    )
            matches = [max(chunk1_references, key=lambda path: (path.stat().st_mtime_ns, path.name))]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find the {kind} input in {input_dir}. "
            "Expected one parsed .txt, one chunk .json, and one reference/toc .json file."
        )
    names = ", ".join(path.name for path in matches)
    raise ValueError(f"Multiple {kind} inputs found in {input_dir}: {names}. Specify --{kind} explicitly.")


def document_stem(parsed_path: Path) -> str:
    stem = parsed_path.stem
    for suffix in ("_parsed_pdf", "_parsed"):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem or parsed_path.stem


def create_run_output_dir(output_root: Path, stem: str, timestamp: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / f"{stem}_{timestamp}"
    sequence = 2
    while candidate.exists():
        candidate = output_root / f"{stem}_{timestamp}_{sequence}"
        sequence += 1
    candidate.mkdir(parents=True)
    return candidate


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file: {path} / {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object: {path}")
    return data


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
    return {
        "source_reference": layout.get("source_reference"),
        "source_chunk": layout.get("source_chunk"),
        "matching_priority": layout.get("matching_priority", []),
        "rules": compact_rules,
        "unmatched_reference_titles": layout.get("unmatched_reference_titles", []),
    }


def build_prompt(layout_json: dict[str, Any], parsed_text: str, title: str, max_depth: int, user_prompt: str) -> str:
    layout_text = json.dumps(compact_layout_json(layout_json), ensure_ascii=False, indent=2)
    return f"""
You are Gemma 31B. Create the full table-of-contents JSON from the parsed PDF layout text.
This replaces only the code-based full-document matching step from toc_txt_test.py.
The layout JSON was generated from CHUNK and REFERENCE and is the primary evidence for heading levels.

[Task]
{user_prompt.strip()}

[Document Title]
{title}

[Layout JSON]
{layout_text}

[Rules]
- Behave like the generate_toc step, but do the matching and filtering with Gemma instead of Python rule matching.
- Use the Layout JSON rules as the primary style reference learned from the verified chunk.
- Match headings by S/R/B/I/F first, then start_shapes, X/Y position, and numbering.
- Use only text that appears in the parsed layout text. Do not invent, rewrite, or summarize titles.
- Preserve original numbering and title text exactly.
- Exclude covers, prefaces, existing TOC listing rows, indexes, references, page numbers, repeated headers/footers, captions, questions, and body sentences.
- Existing TOC/Contents pages may help you understand structure, but do not output listing rows unless the same title appears as a real body heading.
- Preserve PDF page order and source order within each page.
- Use integer levels from 1 to {max_depth}; lower numbers are higher-level headings.
- Include level_reason for every chapter, explaining the layout or numbering evidence briefly.

[Output Format]
Output exactly one valid JSON object. Do not output Markdown, code fences, comments, or explanations outside JSON.

{{
  "title": "{title}",
  "chapters": [
    {{"level": 1, "chapter": "Source heading", "page": 1, "level_reason": "Matched layout rule and numbering."}}
  ]
}}

[Parsed PDF Layout Text]
{parsed_text}
""".strip()


def build_chunk_prompt(
    layout_json: dict[str, Any],
    chunk: dict[str, Any],
    title: str,
    max_depth: int,
    user_prompt: str,
    total_chunks: int,
) -> str:
    layout_text = json.dumps(compact_layout_json(layout_json), ensure_ascii=False, indent=2)
    return f"""
You are Gemma 31B. Create a partial table-of-contents JSON from one parsed PDF layout text chunk.
The layout JSON was generated from CHUNK and REFERENCE and is the primary evidence for heading levels.

[Task]
{user_prompt.strip()}

[Document Title]
{title}

[Chunk]
{chunk["index"]}/{total_chunks}, pages {chunk["start_page"]}-{chunk["end_page"]}

[Layout JSON]
{layout_text}

[Rules]
- Output only TOC entries whose real body heading appears inside this chunk.
- Use the Layout JSON rules as the primary style reference learned from the verified chunk.
- Match headings by S/R/B/I/F first, then start_shapes, X/Y position, and numbering.
- Use only text that appears in the parsed layout text. Do not invent, rewrite, or summarize titles.
- Preserve original numbering and title text exactly.
- Exclude covers, prefaces, existing TOC listing rows, indexes, references, page numbers, repeated headers/footers, captions, questions, and body sentences.
- Existing TOC/Contents pages may help you understand structure, but do not output listing rows unless the same title appears as a real body heading.
- Preserve PDF page order and source order within the chunk.
- Use integer levels from 1 to {max_depth}; lower numbers are higher-level headings.
- Include level_reason for every chapter, explaining the layout or numbering evidence briefly.

[Output Format]
Output exactly one valid JSON object. Do not output Markdown, code fences, comments, or explanations outside JSON.

{{
  "title": "{title}",
  "chapters": [
    {{"level": 1, "chapter": "Source heading", "page": {chunk["start_page"]}, "level_reason": "Matched layout rule and numbering."}}
  ]
}}

[Parsed PDF Layout Text Chunk]
{chunk["text"]}
""".strip()


def page_number_from_segment(segment: str) -> int | None:
    match = re.search(r"(?m)^\[PAGE\s+(\d+)\]$", segment)
    return int(match.group(1)) if match else None


def split_parsed_text_chunks(text: str, max_chars: int) -> list[dict[str, Any]]:
    max_chars = max(10000, int(max_chars))
    starts = [match.start() for match in re.finditer(r"(?m)^\[PAGE\s+\d+\]$", text)]
    if not starts:
        return [{"index": 1, "start_page": 1, "end_page": 1, "text": text.strip()}]

    starts.append(len(text))
    page_segments: list[tuple[int, str]] = []
    for index in range(len(starts) - 1):
        segment = text[starts[index]:starts[index + 1]].strip()
        page = page_number_from_segment(segment)
        if page is not None and segment:
            page_segments.append((page, segment))

    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_start: int | None = None
    current_end: int | None = None
    current_chars = 0

    def flush() -> None:
        nonlocal current_parts, current_start, current_end, current_chars
        if not current_parts:
            return
        chunks.append({
            "index": len(chunks) + 1,
            "start_page": current_start or 1,
            "end_page": current_end or current_start or 1,
            "text": "\n\n".join(current_parts),
        })
        current_parts = []
        current_start = None
        current_end = None
        current_chars = 0

    for page, segment in page_segments:
        segment_len = len(segment) + 2
        if current_parts and current_chars + segment_len > max_chars:
            flush()
        if current_start is None:
            current_start = page
        current_end = page
        current_parts.append(segment)
        current_chars += segment_len

    flush()
    return chunks


def split_failed_parsed_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(chunk.get("text") or "").strip()
    if not text:
        return []

    subchunks = split_parsed_text_chunks(text, max(10000, len(text) // 2))
    if len(subchunks) <= 1:
        return []

    parent_index = str(chunk.get("index", ""))
    for index, subchunk in enumerate(subchunks, start=1):
        subchunk["index"] = f"{parent_index}.{index}"
    return subchunks


def merge_partial_tocs(title: str, partial_tocs: list[dict[str, Any]]) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for toc in partial_tocs:
        chapters = toc.get("chapters", [])
        if not isinstance(chapters, list):
            continue
        for item in chapters:
            if not isinstance(item, dict):
                continue
            chapter = clean_text(item.get("chapter"))
            if not chapter:
                continue
            try:
                page = int(item.get("page", 1))
            except Exception:
                page = 1
            key = (page, norm(chapter))
            if key in seen:
                continue
            seen.add(key)
            row = {
                "level": int(item.get("level", 1) or 1),
                "chapter": chapter,
                "page": max(1, page),
            }
            reason = clean_text(item.get("level_reason"))
            if reason:
                row["level_reason"] = reason
            merged.append(row)
    return {"title": title, "chapters": merged}


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
    for line in parsed_lines:
        pages[line.page].append(line)

    chapters: list[dict[str, Any]] = []
    for merge_index, item in enumerate(toc.get("chapters", [])):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        page = int(row.get("page", 1))
        line = match_toc_source_line(clean_text(row.get("chapter")), pages.get(page, []))
        if line is None:
            row["metadata"] = {
                "matched_parsed_layout": False,
                "source_order": None,
            }
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


def chat_url(base_url: str) -> str:
    url = clean_text(base_url).rstrip("/")
    if not url:
        raise ValueError("--base-url is empty.")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def call_gemma_openai_compatible(
    *,
    model: str,
    prompt: str,
    base_url: str,
    api_key: str,
    timeout: int,
    temperature: float,
    max_output_tokens: int,
) -> str:
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
        hint = ""
        if "maximum context length" in error_body.lower() or "max context" in error_body.lower():
            hint = " Hint: reduce --parsed-chunk-chars or --max-output-tokens."
        raise RuntimeError(
            f"Gemma request failed: HTTP {error.code} {error.reason}. URL={url}. "
            f"Body: {error_body[:1000]}{hint}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Gemma server is not reachable at {url}: {error}") from error

    try:
        data = json.loads(response_text)
    except Exception as error:
        raise RuntimeError(f"Gemma server returned non-JSON response: {response_text[:1000]}") from error

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Gemma server returned no choices: {response_text[:1000]}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            output_text = "\n".join(
                clean_text(part.get("text") if isinstance(part, dict) else part)
                for part in content
                if clean_text(part.get("text") if isinstance(part, dict) else part)
            )
        else:
            output_text = str(content or "")
    else:
        output_text = str(first.get("text") or "")

    if clean_text(first.get("finish_reason")).lower() == "length":
        raise GemmaOutputTruncatedError(
            f"Gemma output reached the {max_output_tokens}-token limit before JSON completion."
        )
    return output_text


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
    markers = (
        "HTTP 400",
        "HTTP 404",
        "MAXIMUM CONTEXT LENGTH",
        "INPUT_TOKENS",
        "DOES NOT EXIST",
        "NOTFOUNDERROR",
    )
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
            print(f"{label} temporary error, retry {attempt}/{max_retries - 1} (after {delay:.1f}s): {error}", file=sys.stderr, flush=True)
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
        raise ValueError("Gemma response is empty.")
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
    raise ValueError("Could not parse Gemma JSON response.")


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
        reason = clean_text(item.get("level_reason"))
        if reason:
            row["level_reason"] = reason
        chapters.append(row)
    return {"title": title, "chapters": chapters}


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def request_gemma_toc(prompt: str, args: argparse.Namespace, fallback_title: str) -> tuple[dict[str, Any], str, str]:
    models = parse_model_list(args.model, args.ai_fallback_models) or [DEFAULT_MODEL]
    last_error: Exception | None = None
    for model_index, raw_model in enumerate(models):
        model = normalize_gemma_model(raw_model)
        if model_index > 0:
            print(f"Trying Gemma fallback model: {model}", flush=True)
        for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
            request_prompt = prompt
            if parse_attempt > 1:
                request_prompt = (
                    prompt
                    + "\n\n[Retry Instruction]\n"
                    + "The previous response was not valid JSON. Return only one JSON object parseable by Python json.loads. "
                    + "Escape quotes inside strings and do not use Markdown."
                )
                print(f"  Retrying Gemma JSON generation {parse_attempt}/{args.ai_retries}: {model}", flush=True)
            try:
                print(f"  Gemma full JSON generation started: {model}", flush=True)
                raw_text = with_retry(
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
                print(f"  Gemma full JSON generation completed: {model}", flush=True)
                parsed = parse_json_response_text(raw_text)
                return validate_toc(parsed, fallback_title=fallback_title, max_depth=args.max_depth), raw_text, model
            except Exception as error:
                last_error = error
                if isinstance(error, GemmaOutputTruncatedError):
                    raise
                if is_non_retryable_request_error(error):
                    print(f"Gemma model failed: {model} / {error}", file=sys.stderr, flush=True)
                    break
                if parse_attempt >= max(1, int(args.ai_retries)):
                    print(f"Gemma model failed: {model} / {error}", file=sys.stderr, flush=True)
                    break
    if last_error:
        raise last_error
    raise RuntimeError("No Gemma model was available.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="toc_txt_test.py와 같은 흐름으로 layout JSON을 만든 뒤, 코드 매칭 단계만 Gemma 31B로 수행합니다.",
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="입력 파일 폴더")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="실행별 결과 폴더를 생성할 상위 폴더")
    parser.add_argument("--chunk", type=Path, help="학습 구간 레이아웃 청크 JSON 파일(미지정 시 input-dir에서 자동 탐색)")
    parser.add_argument("--reference", type=Path, help="학습 기준 TOC JSON 파일(미지정 시 input-dir에서 자동 탐색)")
    parser.add_argument("--parsed", type=Path, help="전체 parsed PDF 텍스트 파일(미지정 시 input-dir에서 자동 탐색)")
    parser.add_argument("--layout-level-output", type=Path, help="레이아웃 결과 파일명 또는 저장 폴더(생성시간 자동 추가)")
    parser.add_argument("--output", type=Path, help="TOC 결과 파일명 또는 저장 폴더(생성시간 자동 추가)")
    parser.add_argument("--raw-output", type=Path, help="Gemma 원본 응답 txt 파일명 또는 저장 폴더(생성시간 자동 추가)")
    parser.add_argument("--title", help="TOC title. Defaults to reference title or parsed filename stem.")
    parser.add_argument("--prompt", default="Create the complete table of contents JSON for the whole document from the layout text.")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemma model. '31b' normalizes to gemma4:31b.")
    parser.add_argument("--ai-fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible Gemma/vLLM base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer API key for the OpenAI-compatible server")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--parsed-chunk-chars", type=int, default=DEFAULT_PARSED_CHUNK_CHARS, help="청크별 parsed text 최대 문자 수")
    parser.add_argument(
        "--max-chunk-split-depth",
        type=int,
        default=DEFAULT_MAX_CHUNK_SPLIT_DEPTH,
        help="JSON 출력 실패 시 해당 parsed 청크를 자동 재분할할 최대 단계",
    )
    parser.add_argument("--single-request", action="store_true", help="청크 분할 없이 전체 parsed text를 한 번에 요청")
    return parser.parse_args()


def reference_source_name(reference_path: Path) -> str:
    reference = read_json(reference_path)
    metadata = reference.get("_meta")
    if isinstance(metadata, dict):
        source_input = clean_text(metadata.get("chunk1_source_input"))
        if source_input:
            return Path(source_input).stem
    return clean_text(reference.get("title")) or reference_path.stem


def input_chunk_matches_source(path: Path, source_name: str) -> bool:
    try:
        chunk = read_json(path)
    except (OSError, ValueError):
        return False
    source_pdf = clean_text(chunk.get("source_pdf"))
    if not source_pdf:
        return False
    return normalized_document_name(Path(source_pdf).stem) == normalized_document_name(source_name)


def find_input_chunk_for_reference(reference_path: Path) -> Path:
    reference = read_json(reference_path)
    metadata = reference.get("_meta")
    source_name = reference_source_name(reference_path)

    if isinstance(metadata, dict):
        source_run = clean_text(metadata.get("chunk1_source_input_chunks_dir"))
        if source_run:
            local_run = INPUT_CHUNKS_DIR / Path(source_run).name
            local_candidates = sorted(local_run.glob("chunk_1_pages_*.json"))
            if local_candidates:
                return max(local_candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

    candidates = [
        path
        for path in INPUT_CHUNKS_DIR.glob("*/chunk_1_pages_*.json")
        if input_chunk_matches_source(path, source_name)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No input chunk matching reference source {source_name!r} found under: {INPUT_CHUNKS_DIR}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def find_parsed_for_chunk(chunk_path: Path) -> Path:
    exact = PARSED_PDFS_DIR / f"{chunk_path.parent.name}_parsed_pdf.txt"
    if exact.is_file():
        return exact

    chunk = read_json(chunk_path)
    source_pdf = clean_text(chunk.get("source_pdf"))
    source_name = Path(source_pdf).stem if source_pdf else chunk_path.parent.name
    candidates = [
        path
        for path in PARSED_PDFS_DIR.glob("*_parsed_pdf.txt")
        if normalized_document_name(path.stem).startswith(normalized_document_name(source_name))
    ]
    if not candidates:
        raise FileNotFoundError(f"No parsed PDF text matching chunk found under: {PARSED_PDFS_DIR}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def resolve_document_jobs(args: argparse.Namespace) -> list[tuple[Path, Path, Path]]:
    if args.reference:
        references = [args.reference.resolve()]
    else:
        references = sorted(
            path.resolve()
            for path in args.input_dir.glob("*_chunk1.json")
            if path.is_file()
        )
    if not references:
        raise FileNotFoundError(f"No *_chunk1.json reference files found in: {args.input_dir}")
    if len(references) > 1 and (args.chunk or args.parsed):
        raise ValueError("--chunk and --parsed can only be used with one explicit --reference.")

    jobs: list[tuple[Path, Path, Path]] = []
    for reference_path in references:
        chunk_path = args.chunk.resolve() if args.chunk else find_input_chunk_for_reference(reference_path)
        parsed_path = args.parsed.resolve() if args.parsed else find_parsed_for_chunk(chunk_path)
        jobs.append((reference_path, chunk_path, parsed_path))
    return jobs


def process_document(
    base_args: argparse.Namespace,
    reference_path: Path,
    chunk_path: Path,
    parsed_path: Path,
) -> None:
    args = argparse.Namespace(**vars(base_args))
    args.reference = reference_path
    args.chunk = chunk_path
    args.parsed = parsed_path
    chunk = read_json(args.chunk)

    reference = read_json(args.reference)
    if not isinstance(chunk.get("text"), str):
        raise ValueError(f"Chunk JSON must contain a text string: {args.chunk}")

    learned = learn_layout_levels(reference, parse_layout(chunk["text"]), args.reference, args.chunk)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = document_stem(args.parsed)
    run_output_dir = create_run_output_dir(args.output_root, output_stem, timestamp)
    layout_output = timestamped_output(
        args.layout_level_output,
        run_output_dir / "layout",
        output_stem,
        "layout_levels",
        timestamp,
    )
    layout_output.write_text(json.dumps(learned, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    layout_json = read_json(layout_output)
    parsed_text = args.parsed.read_text(encoding="utf-8")
    parsed_lines = parse_layout(parsed_text)
    title = clean_text(args.title) or clean_text(reference.get("title")) or args.parsed.stem

    generation_started_at = datetime.now().astimezone()
    started = time.perf_counter()
    raw_records: list[dict[str, Any]] = []
    used_model = ""
    parsed_chunk_count = 1

    if args.single_request:
        prompt = build_prompt(layout_json, parsed_text, title, max(1, int(args.max_depth)), args.prompt)
        toc, raw_text, used_model = request_gemma_toc(prompt, args, fallback_title=title)
        raw_records.append({"mode": "single", "raw_text": raw_text})
    else:
        chunks = split_parsed_text_chunks(parsed_text, args.parsed_chunk_chars)
        partial_tocs: list[dict[str, Any]] = []
        total_chunks = len(chunks)
        parsed_chunk_count = total_chunks
        print(f"Parsed chunks: {total_chunks} (max_chars={args.parsed_chunk_chars})", flush=True)

        def process_parsed_chunk(chunk_item: dict[str, Any], split_depth: int = 0) -> None:
            nonlocal parsed_chunk_count, used_model
            prompt = build_chunk_prompt(
                layout_json=layout_json,
                chunk=chunk_item,
                title=title,
                max_depth=max(1, int(args.max_depth)),
                user_prompt=args.prompt,
                total_chunks=parsed_chunk_count,
            )
            print(
                f"Chunk {chunk_item['index']}/{parsed_chunk_count}: "
                f"pages {chunk_item['start_page']}-{chunk_item['end_page']}",
                flush=True,
            )
            try:
                partial_toc, raw_text, used_model = request_gemma_toc(prompt, args, fallback_title=title)
            except ValueError as error:
                subchunks = split_failed_parsed_chunk(chunk_item)
                if split_depth >= max(0, int(args.max_chunk_split_depth)) or not subchunks:
                    raise
                parsed_chunk_count += len(subchunks) - 1
                print(
                    f"  Invalid or truncated JSON for pages "
                    f"{chunk_item['start_page']}-{chunk_item['end_page']}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"  Splitting this chunk into {len(subchunks)} smaller page ranges.",
                    file=sys.stderr,
                    flush=True,
                )
                for subchunk in subchunks:
                    process_parsed_chunk(subchunk, split_depth + 1)
                return

            partial_tocs.append(partial_toc)
            raw_records.append({
                "chunk": chunk_item["index"],
                "start_page": chunk_item["start_page"],
                "end_page": chunk_item["end_page"],
                "split_depth": split_depth,
                "toc_entries": len(partial_toc.get("chapters", [])),
                "raw_text": raw_text,
            })

        for chunk_item in chunks:
            process_parsed_chunk(chunk_item)
        toc = validate_toc(merge_partial_tocs(title, partial_tocs), fallback_title=title, max_depth=args.max_depth)

    toc = attach_layout_metadata_and_sort(toc, parsed_lines)
    elapsed = time.perf_counter() - started
    generation_completed_at = datetime.now().astimezone()
    matched_layout_entries = sum(
        1
        for item in toc.get("chapters", [])
        if isinstance(item.get("metadata"), dict) and item["metadata"].get("matched_parsed_layout")
    )

    output = timestamped_output(args.output, run_output_dir / "toc", output_stem, "toc", timestamp)
    raw_path = timestamped_output(
        args.raw_output,
        run_output_dir / "raw",
        output_stem,
        "gemma_raw",
        timestamp,
        suffix=".json",
    )
    toc["_meta"] = {
        "source_input_directory": str(args.input_dir),
        "source_chunk": str(args.chunk),
        "source_reference": str(args.reference),
        "source_parsed": str(args.parsed),
        "provider": "gemma_vllm",
        "model": used_model,
        "generated_at": generation_completed_at.isoformat(timespec="seconds"),
        "generation_started_at": generation_started_at.isoformat(timespec="seconds"),
        "generation_completed_at": generation_completed_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "elapsed": format_elapsed(elapsed),
        "generation_mode": "single_request" if args.single_request else "parsed_chunks",
        "parsed_chunk_count": parsed_chunk_count,
        "parsed_chunk_chars": None if args.single_request else args.parsed_chunk_chars,
        "max_chunk_split_depth": None if args.single_request else args.max_chunk_split_depth,
        "chapter_sort": "page_then_parsed_source_order",
        "matched_layout_entries": matched_layout_entries,
        "unmatched_layout_entries": len(toc.get("chapters", [])) - matched_layout_entries,
        "layout_metadata_fields": ["source_order", "s", "r", "x", "y", "b", "i", "f", "source_text"],
        "output_directory": str(run_output_dir),
    }
    output.write_text(json.dumps(toc, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    raw_path.write_text(json.dumps(raw_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Input directory: {args.input_dir}", flush=True)
    print(f"Output directory: {run_output_dir}", flush=True)
    print(f"Layout levels: {layout_output}", flush=True)
    print(f'Unmatched reference titles: {len(layout_json["unmatched_reference_titles"])}', flush=True)
    print(f"Created: {output}", flush=True)
    print(f"Raw responses: {raw_path}", flush=True)
    print(f"Gemma model: {used_model}", flush=True)
    print(f"TOC entries: {len(toc.get('chapters', []))}", flush=True)
    print(f"Elapsed: {format_elapsed(elapsed)}", flush=True)


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_root = args.output_root.resolve()
    jobs = resolve_document_jobs(args)
    print(f"Documents: {len(jobs)}", flush=True)
    failures: list[tuple[Path, Exception]] = []
    for index, (reference_path, chunk_path, parsed_path) in enumerate(jobs, start=1):
        print(f"\nDocument {index}/{len(jobs)}: {reference_path.name}", flush=True)
        print(f"  Input chunk: {chunk_path}", flush=True)
        print(f"  Parsed text: {parsed_path}", flush=True)
        try:
            process_document(args, reference_path, chunk_path, parsed_path)
        except Exception as error:
            failures.append((reference_path, error))
            print(f"Document failed: {reference_path.name} / {error}", file=sys.stderr, flush=True)

    if failures:
        print(f"\nFailed documents: {len(failures)}/{len(jobs)}", file=sys.stderr, flush=True)
        for reference_path, error in failures:
            print(f"  {reference_path.name}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
