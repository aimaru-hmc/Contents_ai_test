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
CHUNK = ROOT / "data/output/input_chunks/2_5-8_2021_gemma_20260708_153404/chunk_1_pages_1-39.json"
REFERENCE = ROOT / "data/output/2편5-8장_2021_openai_gpt-5.5_20260710_144043_toc_39.json"
PARSED = ROOT / "data/output/parsed_pdfs/2_5-8_2021_gemma_20260708_153404_parsed_pdf.txt"
TOC_OUTPUT_DIR = ROOT / "data/output/toc_txt/toc"
LAYOUT_OUTPUT_DIR = ROOT / "data/output/toc_txt/layout"
RAW_OUTPUT_DIR = ROOT / "data/output/toc_txt/raw"

DEFAULT_MODEL = "gemma4:31b"
DEFAULT_FALLBACK_MODELS = DEFAULT_MODEL
DEFAULT_AI_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_MAX_OUTPUT_TOKENS = 8192
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
            hint = " Hint: reduce --max-output-tokens, for example --max-output-tokens 8192 or 4096."
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
            return "\n".join(
                clean_text(part.get("text") if isinstance(part, dict) else part)
                for part in content
                if clean_text(part.get("text") if isinstance(part, dict) else part)
            )
        return str(content or "")
    return str(first.get("text") or "")


def message_of(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def is_retryable_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def is_configuration_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in CONFIGURATION_ERROR_MARKERS)


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
    parser.add_argument("--chunk", type=Path, default=CHUNK, help="학습 구간 레이아웃 청크 JSON 파일")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="학습 기준 TOC JSON 파일")
    parser.add_argument("--parsed", type=Path, default=PARSED, help="전체 parsed PDF 텍스트 파일")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = read_json(args.reference)
    chunk = read_json(args.chunk)
    if not isinstance(chunk.get("text"), str):
        raise ValueError(f"Chunk JSON must contain a text string: {args.chunk}")

    learned = learn_layout_levels(reference, parse_layout(chunk["text"]), args.reference, args.chunk)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layout_output = timestamped_output(args.layout_level_output, LAYOUT_OUTPUT_DIR, args.parsed.stem, "layout_levels", timestamp)
    layout_output.write_text(json.dumps(learned, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    layout_json = read_json(layout_output)
    parsed_text = args.parsed.read_text(encoding="utf-8")
    title = clean_text(args.title) or clean_text(reference.get("title")) or args.parsed.stem
    prompt = build_prompt(layout_json, parsed_text, title, max(1, int(args.max_depth)), args.prompt)

    started = time.perf_counter()
    toc, raw_text, used_model = request_gemma_toc(prompt, args, fallback_title=title)
    elapsed = time.perf_counter() - started

    output = timestamped_output(args.output, TOC_OUTPUT_DIR, args.parsed.stem, "toc", timestamp)
    output.write_text(json.dumps(toc, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    raw_path = timestamped_output(args.raw_output, RAW_OUTPUT_DIR, args.parsed.stem, "gemma_raw", timestamp, suffix=".txt")
    raw_path.write_text(raw_text.rstrip() + "\n", encoding="utf-8")

    print(f"Layout levels: {layout_output}", flush=True)
    print(f'Unmatched reference titles: {len(layout_json["unmatched_reference_titles"])}', flush=True)
    print(f"Created: {output}", flush=True)
    print(f"Raw response: {raw_path}", flush=True)
    print(f"Gemma model: {used_model}", flush=True)
    print(f"TOC entries: {len(toc.get('chapters', []))}", flush=True)
    print(f"Elapsed: {format_elapsed(elapsed)}", flush=True)


if __name__ == "__main__":
    main()
