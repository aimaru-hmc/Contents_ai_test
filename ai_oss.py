from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
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
DEFAULT_MODEL = os.getenv("OSS_MODEL", os.getenv("AI_MODEL", "gpt-oss-120b"))
DEFAULT_BASE_URL = os.getenv("OSS_BASE_URL", os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:31892/v1"))
DEFAULT_API_KEY = os.getenv("OSS_API_KEY", os.getenv("OPENAI_COMPATIBLE_API_KEY", ""))
DEFAULT_FALLBACK_MODELS = os.getenv("OSS_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL))
DEFAULT_AI_RETRIES = int(os.getenv("AI_MAX_RETRIES", "5"))
DEFAULT_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", "2.0"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("OSS_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "4096")))
DEFAULT_MAX_INPUT_CHARS = int(os.getenv("OSS_MAX_INPUT_CHARS", "30000"))
DEFAULT_TIMEOUT = int(os.getenv("OSS_TIMEOUT", "3600"))


DEFAULT_USER_PROMPT = """
Create a table of contents from real document hierarchy titles.
Exclude covers, prefaces, indexes, references, standalone page numbers, repeated headers/footers, captions, questions, and body sentences.
Keep exact source titles/numbering and actual PDF viewer page numbers.
""".strip()

RETRYABLE_ERROR_MARKERS = (
    "429",
    "RATE LIMIT",
    "500",
    "INTERNAL",
    "502",
    "503",
    "UNAVAILABLE",
    "OVERLOADED",
    "TRY AGAIN LATER",
    "504",
    "DEADLINE_EXCEEDED",
    "TIMEOUT",
    "TIMED OUT",
    "TEMPORAR",
    "CONNECTION",
)


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return DEFAULT_USER_PROMPT


def median_number(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return float(default)
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2.0


def compact_font_name(value: Any) -> str:
    font = clean_text(value)
    if "+" in font:
        font = font.split("+", 1)[1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "", font)[:40] or "unknown"


def font_style_flags(font_name: str) -> tuple[int, int]:
    name = clean_text(font_name).lower()
    bold = int(any(token in name for token in ("bold", "black", "heavy", "demi", "semibold", "semi-bold")))
    italic = int(any(token in name for token in ("italic", "oblique", "slant")))
    return bold, italic


def extract_pdf_layout_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError("Layout extraction requires `pip install pdfplumber`.") from error

    output = [
        "# Exact PDF line layout metadata",
        "# P=viewer page, O=source line order, S=font size, R=size/body ratio, X/Y=position, B=bold, I=italic, F=font name",
        "# Use O to preserve order for lines sharing the same page.",
    ]
    total_lines = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_chars = [
                char for char in (getattr(page, "chars", None) or [])
                if isinstance(char, dict) and clean_text(char.get("text")) and char.get("size") is not None
            ]
            body_size = median_number((char.get("size") for char in page_chars), default=10.0) or 10.0
            output.append(f"[PAGE P={page_number} W={float(page.width):.1f} H={float(page.height):.1f} BODY={body_size:.2f}]")
            try:
                lines = page.extract_text_lines(layout=False, return_chars=True) or []
            except Exception:
                lines = []

            if not lines:
                fallback = str(page.extract_text() or "").strip()
                for order, text in enumerate(fallback.splitlines(), start=1):
                    text = clean_text(text)
                    if text:
                        output.append(f"[L P={page_number} O={order} S={body_size:.2f} R=1.000 X=0.0 Y=0.0 B=0 I=0 F=unknown] {text}")
                        total_lines += 1
                continue

            order = 0
            for line in lines:
                if not isinstance(line, dict):
                    continue
                chars = [char for char in (line.get("chars") or []) if isinstance(char, dict)]
                text = clean_text(line.get("text") or "".join(str(char.get("text") or "") for char in chars))
                if not text:
                    continue
                order += 1
                size = median_number((char.get("size") for char in chars), default=body_size) or body_size
                x = min((float(char.get("x0") or 0.0) for char in chars), default=float(line.get("x0") or 0.0))
                y = min((float(char.get("top") or 0.0) for char in chars), default=float(line.get("top") or 0.0))
                font_counts: dict[str, int] = {}
                for char in chars:
                    font = compact_font_name(char.get("fontname"))
                    font_counts[font] = font_counts.get(font, 0) + 1
                font = max(font_counts, key=font_counts.get) if font_counts else compact_font_name(line.get("fontname"))
                bold, italic = font_style_flags(font)
                ratio = size / body_size if body_size else 1.0
                output.append(
                    f"[L P={page_number} O={order} S={size:.2f} R={ratio:.3f} "
                    f"X={x:.1f} Y={y:.1f} B={bold} I={italic} F={font}] {text}"
                )
                total_lines += 1

    if total_lines == 0:
        raise RuntimeError("No extractable PDF text/layout was found. OCR is required for scanned PDFs.")
    return "\n".join(output) + "\n"


def build_prompt(user_prompt: str, pdf_name: str, layout_text: str, max_depth: int) -> str:
    pdf_title = Path(pdf_name).stem
    return f"""
You are an expert at creating tables of contents for PDF textbooks and lecture materials.
Create the TOC using only the parsed PDF layout text below.

[User Prompt]
{user_prompt.strip()}

[Output Format]
Output exactly one compact valid JSON object. Do not output Markdown, code fences, or explanations outside JSON.

{{
  "title": "{pdf_title}",
  "chapters": [
    {{"level": 1, "chapter": "Major section title", "page": 1, "level_reason": "Concise evidence from layout/numbering."}},
    {{"level": 2, "chapter": "Subsection title", "page": 3, "level_reason": "Concise evidence from layout/numbering."}}
  ]
}}

[Required Rules]
- Set the top-level JSON title exactly to "{pdf_title}".
- Include real document hierarchy titles, sorted in PDF page order.
- Exclude covers, prefaces, indexes, references, standalone page numbers, repeated headers/footers, captions, questions, and body sentences.
- Use exact source titles and existing numbering only. Do not invent titles or numbering.
- Use layout metadata P(page), O(source order), S(font size), R(size/body ratio), X/Y(position), B(bold), I(italic), and F(font name).
- Highest-priority body heading evidence: S/R/B/I/F style signature.
- Use explicit numbering and X/Y as secondary evidence.
- Preserve O source order for entries on the same page.
- Use levels 1 to {max_depth}.
- Include a concise level_reason for every entry.

[Parsed PDF Layout Text]
{layout_text}
""".strip()


def maybe_limit_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED: input limited by --max-input-chars]\n"


def openai_compatible_chat_url(base_url: str) -> str:
    url = clean_text(base_url).rstrip("/")
    if not url:
        raise ValueError("OpenAI-compatible base URL is empty.")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def call_oss_chat(
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

    url = openai_compatible_chat_url(base_url)
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout))) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OSS request failed: HTTP {error.code} {error.reason}. URL={url}. Body: {error_body[:1000]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OSS server is not reachable at {url}: {error}") from error

    data = json.loads(response_text)
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OSS server returned no choices: {response_text[:1000]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(choices[0].get("text") or "")


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
    return None


def parse_json_response_text(text: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    candidates = [text]
    extracted = extract_json_object_text(text)
    if extracted and extracted != text:
        candidates.append(extracted)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("AI response JSON must be an object.")
            return data
        except Exception as error:
            last_error = error

    if last_error:
        raise last_error
    raise ValueError("No JSON object found in OSS response.")


def validate_toc(data: dict[str, Any], fallback_title: str, max_depth: int) -> dict[str, Any]:
    title = clean_text(data.get("title")) or fallback_title
    chapters: list[dict[str, Any]] = []
    raw_chapters = data.get("chapters", [])
    if not isinstance(raw_chapters, list):
        raw_chapters = []

    seen: set[tuple[str, int]] = set()
    for raw in raw_chapters:
        if not isinstance(raw, dict):
            continue
        chapter = clean_text(raw.get("chapter"))
        if not chapter:
            continue
        try:
            level = int(raw.get("level", 1))
        except Exception:
            level = 1
        try:
            page = int(raw.get("page", 1))
        except Exception:
            page = 1
        level = max(1, min(level, max_depth))
        page = max(1, page)
        key = (chapter.lower(), page)
        if key in seen:
            continue
        seen.add(key)
        item: dict[str, Any] = {"level": level, "chapter": chapter, "page": page}
        level_reason = clean_text(raw.get("level_reason"))
        if level_reason:
            item["level_reason"] = level_reason
        chapters.append(item)

    chapters.sort(key=lambda item: int(item.get("page", 1)))
    return {"title": title, "chapters": chapters}


def message_of(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def is_retryable_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def with_retry(label: str, func: Callable[[], Any], max_retries: int, base_delay: float) -> Any:
    max_retries = max(1, int(max_retries))
    base_delay = max(0.1, float(base_delay))
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as error:
            last_error = error
            if not is_retryable_error(error) or attempt >= max_retries:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
            delay += random.uniform(0, min(1.0, base_delay))
            print(f"{label} temporary error, retry {attempt}/{max_retries - 1} (after {delay:.1f}s): {error}", file=sys.stderr, flush=True)
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


def generate_toc_from_pdf_oss(pdf_path: Path, args: argparse.Namespace, user_prompt: str) -> tuple[dict[str, Any], str, str, str]:
    print("  PDF layout extraction started...", flush=True)
    layout_text = extract_pdf_layout_text(pdf_path)
    limited_layout_text = maybe_limit_text(layout_text, args.max_input_chars)
    print(
        f"  PDF layout extraction completed: {len(layout_text.encode('utf-8')):,} bytes"
        f" / used chars {len(limited_layout_text):,}",
        flush=True,
    )

    base_prompt = build_prompt(
        user_prompt=user_prompt,
        pdf_name=pdf_path.name,
        layout_text=limited_layout_text,
        max_depth=args.max_depth,
    )
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    for index, model in enumerate(models):
        if index > 0:
            print(f"Trying OSS fallback model: {model}", file=sys.stderr, flush=True)

        for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
            prompt = base_prompt
            if parse_attempt > 1:
                print(
                    f"  Retrying OSS generation after JSON parse failure {parse_attempt}/{args.ai_retries}: {model}",
                    file=sys.stderr,
                    flush=True,
                )
                prompt = (
                    base_prompt
                    + "\n\n[Important Retry Instruction]\n"
                    + "The previous response failed JSON parsing. Output exactly one valid JSON object "
                    + "that Python json.loads() can parse directly. Do not output Markdown or explanations."
                )
            try:
                print(f"  OSS TOC generation started: {model}", flush=True)
                raw_text = with_retry(
                    label=f"OSS generation({model})",
                    func=lambda model=model, prompt=prompt: call_oss_chat(
                        model=model,
                        prompt=prompt,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        timeout=args.timeout,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    ),
                    max_retries=args.ai_retries,
                    base_delay=args.ai_retry_base_delay,
                )
                print(f"  OSS TOC generation completed: {model}", flush=True)
                parsed = parse_json_response_text(raw_text)
                toc = validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth)
                return toc, raw_text, model, limited_layout_text
            except Exception as error:
                last_error = error
                if parse_attempt < max(1, int(args.ai_retries)) and isinstance(error, (json.JSONDecodeError, ValueError)):
                    continue
                print(f"OSS model failed: {model} / {error}", file=sys.stderr, flush=True)
                break

    if last_error:
        raise last_error
    raise RuntimeError("No OSS models to try.")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def timestamp_for_metadata() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {sec:.1f}s"
    if minutes:
        return f"{minutes}m {sec:.1f}s"
    return f"{sec:.1f}s"


def safe_filename_part(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^\w가-힣.-]+", "_", value, flags=re.UNICODE)
    value = value.strip("._-")
    return value or "file"


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique output path for: {path}")


def add_result_metadata(
    toc: dict[str, Any],
    pdf_path: Path,
    model: str,
    base_url: str,
    elapsed_seconds: float,
    generated_timestamp: str,
    generation_started_at: str,
    generation_completed_at: str,
    max_input_chars: int,
) -> dict[str, Any]:
    result = dict(toc)
    result["_meta"] = {
        "source_pdf": str(pdf_path),
        "provider": "oss",
        "base_url": base_url,
        "model": model,
        "max_input_chars": max_input_chars,
        "elapsed_seconds": elapsed_seconds,
        "elapsed": format_elapsed(elapsed_seconds),
        "generated_timestamp": generated_timestamp,
        "generation_started_at": generation_started_at,
        "generation_completed_at": generation_completed_at,
    }
    return result


def write_error_log(pdf_path: Path, args: argparse.Namespace, error: Exception, elapsed_seconds: float) -> Path | None:
    if args.no_error_log:
        return None
    error_dir = Path(args.output_dir) / "error_logs"
    error_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{safe_filename_part(pdf_path.stem)}_oss_{timestamp_for_filename()}"
    log_file = unique_output_path(error_dir / f"{base_name}_error.json")
    payload = {
        "timestamp": timestamp_for_metadata(),
        "source_pdf": str(pdf_path),
        "provider": "oss",
        "base_url": args.base_url,
        "elapsed_seconds": elapsed_seconds,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    save_json(log_file, payload)
    return log_file


def print_quiet_console(args: argparse.Namespace, text: str, error: bool = False) -> None:
    if not getattr(args, "quiet_console", False):
        return
    stream = getattr(args, "_console_stderr" if error else "_console_stdout", None)
    print(text, file=stream or (sys.stderr if error else sys.stdout), flush=True)


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() == ".pdf")
    raise FileNotFoundError(f"Path not found: {path}")


def process_pdf(pdf_path: Path, args: argparse.Namespace, user_prompt: str) -> bool:
    generation_started_at = timestamp_for_metadata()
    print(f"Processing started: {pdf_path}", flush=True)
    started_at = time.perf_counter()

    try:
        toc, raw_text, used_model, layout_text = generate_toc_from_pdf_oss(pdf_path=pdf_path, args=args, user_prompt=user_prompt)
        elapsed_seconds = time.perf_counter() - started_at
        generation_completed_at = timestamp_for_metadata()
        generated_timestamp = timestamp_for_filename()
        base_name = f"{safe_filename_part(pdf_path.stem)}_oss_{safe_filename_part(used_model)}_{generated_timestamp}"
        output_file = unique_output_path(Path(args.output_dir) / f"{base_name}_toc.json")
        result = add_result_metadata(
            toc=toc,
            pdf_path=pdf_path,
            model=used_model,
            base_url=args.base_url,
            elapsed_seconds=elapsed_seconds,
            generated_timestamp=generated_timestamp,
            generation_started_at=generation_started_at,
            generation_completed_at=generation_completed_at,
            max_input_chars=args.max_input_chars,
        )
        save_json(output_file, result)

        if args.write_raw:
            raw_file = unique_output_path(Path(args.output_dir) / f"{base_name}_raw_response.txt")
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            raw_file.write_text(raw_text, encoding="utf-8")
            print(f"Raw response saved: {raw_file}", flush=True)

        if args.write_layout:
            layout_file = unique_output_path(Path(args.output_dir) / f"{base_name}_layout.txt")
            layout_file.parent.mkdir(parents=True, exist_ok=True)
            layout_file.write_text(layout_text, encoding="utf-8")
            print(f"Layout input saved: {layout_file}", flush=True)

        print(f"Completed: {output_file}", flush=True)
        print(f"  model: {used_model}", flush=True)
        print(f"  elapsed: {format_elapsed(elapsed_seconds)} ({elapsed_seconds:.3f}s)", flush=True)
        print(f"  title: {toc.get('title')}", flush=True)
        print(f"  chapters: {len(toc.get('chapters', []))}", flush=True)
        print_quiet_console(args, f"Completed: {output_file} / elapsed {format_elapsed(elapsed_seconds)} / chapters {len(toc.get('chapters', []))}")
        return True

    except Exception as error:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"Failed: {pdf_path} / {error}", file=sys.stderr, flush=True)
        print(f"  elapsed: {format_elapsed(elapsed_seconds)} ({elapsed_seconds:.3f}s)", file=sys.stderr, flush=True)
        error_log = write_error_log(pdf_path=pdf_path, args=args, error=error, elapsed_seconds=elapsed_seconds)
        if error_log is not None:
            print(f"  Error log saved: {error_log}", file=sys.stderr, flush=True)
        error_log_text = f" / error log {error_log}" if error_log is not None else ""
        print_quiet_console(args, f"Failed: {pdf_path} / elapsed {format_elapsed(elapsed_seconds)} / {error}{error_log_text}", error=True)
        if args.stop_on_error:
            raise
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gpt-oss-120b PDF TOC JSON generator with ai_openai_v2-compatible CLI.")
    parser.add_argument("path", nargs="?", default="./data/input", help="PDF file or directory")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--prompt", default=None, help="TOC generation instruction")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 text file containing the TOC instruction")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer API key if server requires it")
    parser.add_argument("--ai-fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS, help="Limit parsed layout chars sent to OSS; 0 disables limit")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--no-schema", action="store_true", help="Accepted for ai_openai_v2 compatibility; OSS chat mode always prompts for JSON")
    parser.add_argument("--write-raw", action="store_true", help="Save raw OSS response text")
    parser.add_argument("--write-layout", action="store_true", help="Save parsed layout text sent to OSS")
    parser.add_argument("--delete-uploaded-file", action="store_true", help="Accepted for ai_openai_v2 compatibility; OSS mode does not upload files")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when any PDF fails")
    parser.add_argument("--no-error-log", action="store_true", help="Disable writing error logs under output_dir/error_logs")
    parser.add_argument(
        "--quiet-console",
        action="store_true",
        help="Write detailed output to a log file and show only per-PDF results and the final summary in the terminal",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Detailed log path used with --quiet-console (default: output_dir/logs/batch_oss_TIMESTAMP_log.txt)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.max_depth = max(1, min(int(args.max_depth), 10))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    user_prompt = load_prompt(args)
    pdf_files = iter_pdfs(Path(args.path))
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None
    setattr(args, "_console_stdout", original_stdout)
    setattr(args, "_console_stderr", original_stderr)

    if args.quiet_console:
        log_path = (
            Path(args.log_file)
            if args.log_file
            else Path(args.output_dir) / "logs" / f"batch_oss_{timestamp_for_filename()}_log.txt"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file

    try:
        if not pdf_files:
            print("No PDF files to process.", flush=True)
            print_quiet_console(args, "No PDF files to process.")
            return 0

        success_count = 0
        for pdf_path in pdf_files:
            if process_pdf(pdf_path=pdf_path, args=args, user_prompt=user_prompt):
                success_count += 1

        failed_count = len(pdf_files) - success_count
        summary = f"Processing result: success {success_count} / failed {failed_count}"
        print(summary, flush=True)
        print_quiet_console(args, summary)
        return 0 if failed_count == 0 else 1
    finally:
        if log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
