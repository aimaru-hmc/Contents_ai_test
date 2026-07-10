from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
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
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", os.getenv("AI_MODEL", "gpt-5.5"))
DEFAULT_FALLBACK_MODELS = os.getenv("OPENAI_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL))
DEFAULT_AI_RETRIES = int(os.getenv("AI_MAX_RETRIES", "5"))
DEFAULT_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", "2.0"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768")))

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

DEFAULT_USER_PROMPT = """
Create a table of contents from real document hierarchy titles, including titles visible on existing TOC/Contents pages when present.
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

CONFIGURATION_ERROR_MARKERS = (
    "API KEY",
    "API_KEY",
    "UNAUTHENTICATED",
    "PERMISSION_DENIED",
    "UNAUTHORIZED",
    "401",
    "403",
    "NO API KEY",
    "MISSING API KEY",
)

SCHEMA_CONFIG_ERROR_MARKERS = (
    "RESPONSE_SCHEMA",
    "RESPONSE_JSON_SCHEMA",
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
  "title": "PDF file name without extension",
  "chapters": [
    {{"level": 1, "chapter": "Major section title", "page": 1, "level_reason": "Concise reason this is a top-level section."}},
    {{"level": 2, "chapter": "Subsection title", "page": 3, "level_reason": "Concise reason this is a subsection under the previous level 1 entry."}}
  ]
}}

[Required Rules]
- Include real document hierarchy titles, including entries visible on existing TOC/Contents pages, sorted in PDF page order.
- Exclude body sentences, examples, questions, captions, references, and repeated headers/footers.
- Use exact source titles and existing numbering only. Do not invent titles or numbering.
- Level priority for verified BODY headings: when S, R, B, I, and F are the same (allowing only minor numeric rounding differences), prefer the same level unless secondary layout or explicit structural evidence clearly proves otherwise.
- For verified BODY headings, use evidence in this order: (1) S/R/B/I/F text-format signature, (2) explicit structural numbering, (3) X/Y position and indentation, (4) semantic containment or wording.
- X/Y must not be ignored. Use X/Y as secondary layout evidence and as a tie-breaker when format signatures alone are ambiguous, while avoiding semantic-only nesting such as making "Linear regression" a child of "Regression" when their format signatures and layout roles match.
- Exception: on existing TOC/Contents listing pages, indentation/X is valid hierarchy evidence because listing rows often share the same font format. Use TOC indentation as a reference, then verify levels against BODY heading formats.
- Explicit structural numbering such as Part/Chapter or 1, 1.1, 1.1.1 may override a matching format only when the source clearly uses that numbering as hierarchy.
- Do not put a chapter/section title only in the top-level title field; include hierarchy titles in chapters.
- Use levels 1 to {max_depth}; page is the actual PDF viewer page number.
- Include a concise level_reason for every entry.
- Output JSON only.
""".strip()


def build_attached_pdf_prompt(base_prompt: str, pdf_name: str) -> str:
    pdf_title = Path(pdf_name).stem
    return f"""
{base_prompt}

[PDF Attachment Mode]
Create the TOC using only the attached PDF.
Use actual PDF viewer page numbers.
Rules:
- Set the top-level JSON title exactly to "{pdf_title}".
- If a chapter/section title looks like the document title, still include it as a chapters entry.
- Output only body hierarchy titles: chapters, sections, subsections.
- Ignore existing TOC/Contents pages and listings, repeated headers/footers, body sentences, captions, questions, and references.
- A separate exact layout metadata file is attached with P(page), O(source order), S(font size), R(size/body ratio), X/Y(position), B(bold), I(italic), and F(font name).
- Use the numeric layout metadata together with PDF visual layout and source hierarchy for level inference.
- Highest-priority BODY style rule: headings with the same S/R/B/I/F signature should normally receive the same level, unless explicit numbering or secondary X/Y layout evidence clearly distinguishes their structural roles.
- Do not discard X/Y. Evaluate it after S/R/B/I/F as secondary evidence for BODY headings and use it to resolve ambiguous style matches.
- Apply X indentation as primary hierarchy evidence on TOC/Contents listing pages, where identical fonts are commonly used across multiple indentation levels.
- Preserve O source order for entries on the same page.
- Include level_reason for every entry when possible.
- Return compact valid JSON only.

[PDF File Name]
{pdf_name}
""".strip()


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Add OPENAI_API_KEY=... to your .env file.")
    return api_key


def message_of(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def is_retryable_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def is_configuration_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in CONFIGURATION_ERROR_MARKERS)


def is_schema_config_error(error: Exception) -> bool:
    message = message_of(error).upper()
    return any(marker in message for marker in SCHEMA_CONFIG_ERROR_MARKERS)


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
    if not pdf_path.exists():
        return pdf_path, pdf_path.name
    ascii_name = make_ascii_upload_name(pdf_path)
    if ascii_name == pdf_path.name and pdf_path.suffix.lower() == ".pdf":
        return pdf_path, pdf_path.name

    temp_dir = Path(tempfile.gettempdir()) / "ai_openai_ascii_pdfs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / ascii_name
    if not target_path.exists() or target_path.stat().st_size != pdf_path.stat().st_size:
        shutil.copy2(pdf_path, target_path)
    return target_path, pdf_path.name


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
    raise ValueError("No JSON object found in OpenAI response.")


def parse_json_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, dict):
        return parsed

    text = str(getattr(response, "output_text", "") or "")
    if not text:
        text = str(response)
    return parse_json_response_text(text)


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
        level = max(1, min(level, max_depth))

        try:
            page = int(raw.get("page", 1))
        except Exception:
            page = 1
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

    # Stable page-only sorting preserves the model/source order for entries on the same page.
    chapters.sort(key=lambda item: int(item.get("page", 1)))
    return {"title": title, "chapters": chapters}


def create_openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI API requires `pip install openai`.") from error
    return OpenAI(api_key=api_key)


def build_openai_response_payload(
    model: str,
    prompt: str,
    file_ids: list[str],
    max_output_tokens: int | None,
    use_schema: bool,
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    *[{"type": "input_file", "file_id": file_id} for file_id in file_ids],
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "store": False,
    }
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
    client: Any,
    model: str,
    prompt: str,
    file_ids: list[str],
    max_output_tokens: int | None,
    use_schema: bool,
) -> Any:
    payloads = build_openai_response_payload(
        model=model,
        prompt=prompt,
        file_ids=file_ids,
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
    raise RuntimeError("OpenAI response generation failed.")


def generate_toc_from_pdf_openai(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    client = create_openai_client(api_key=get_api_key())
    pdf_prompt = build_attached_pdf_prompt(prompt, pdf_path.name)
    uploaded_files: list[Any] = []
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    print("  PDF layout extraction started...", flush=True)
    layout_text = extract_pdf_layout_text(pdf_path)
    print(f"  PDF layout extraction completed: {len(layout_text.encode('utf-8')):,} bytes", flush=True)

    try:
        print("  OpenAI PDF + layout upload started...", flush=True)
        with tempfile.TemporaryDirectory(prefix="openai_pdf_layout_upload_") as temp_dir:
            temp_path = Path(temp_dir)
            pdf_upload = temp_path / make_ascii_upload_name(pdf_path)
            shutil.copy2(pdf_path, pdf_upload)
            layout_upload = temp_path / f"{safe_filename_part(pdf_path.stem)}_exact_layout.txt"
            layout_upload.write_text(layout_text, encoding="utf-8")

            for upload_path in (pdf_upload, layout_upload):
                def do_upload(upload_path: Path = upload_path) -> Any:
                    with upload_path.open("rb") as file_obj:
                        return client.files.create(file=file_obj, purpose="user_data")

                uploaded_files.append(with_retry(
                    label=f"OpenAI upload({upload_path.name})",
                    func=do_upload,
                    max_retries=args.ai_retries,
                    base_delay=args.ai_retry_base_delay,
                ))
        print("  OpenAI PDF + layout upload completed", flush=True)
        file_ids = [uploaded.id for uploaded in uploaded_files]

        for index, model in enumerate(models):
            if index > 0:
                print(f"Trying OpenAI fallback model: {model}", file=sys.stderr, flush=True)

            for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
                parse_retry_prompt = pdf_prompt
                if parse_attempt > 1:
                    print(
                        f"  Retrying OpenAI generation after JSON parse failure {parse_attempt}/{args.ai_retries}: {model}",
                        file=sys.stderr,
                        flush=True,
                    )
                    parse_retry_prompt = (
                        pdf_prompt
                        + "\n\n[Important Retry Instruction]\n"
                        + "The previous response failed JSON parsing. Output exactly one valid JSON object "
                        + "that Python json.loads() can parse directly. Do not output Markdown or explanations."
                    )

                try:
                    print(f"  OpenAI TOC generation started: {model}", flush=True)
                    response = with_retry(
                        label=f"OpenAI generation({model})",
                        func=lambda model=model, parse_retry_prompt=parse_retry_prompt: generate_openai_once(
                            client=client,
                            model=model,
                            prompt=parse_retry_prompt,
                            file_ids=file_ids,
                            max_output_tokens=args.max_output_tokens,
                            use_schema=not args.no_schema,
                        ),
                        max_retries=args.ai_retries,
                        base_delay=args.ai_retry_base_delay,
                    )
                    print(f"  OpenAI TOC generation completed: {model}", flush=True)
                    raw_text = str(getattr(response, "output_text", "") or "")
                    parsed = parse_json_response(response)
                    return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model
                except Exception as error:
                    last_error = error
                    if is_configuration_error(error):
                        raise
                    if parse_attempt < max(1, int(args.ai_retries)) and isinstance(error, (json.JSONDecodeError, ValueError)):
                        continue
                    print(f"OpenAI model failed: {model} / {error}", file=sys.stderr, flush=True)
                    break

        if last_error:
            raise last_error
        raise RuntimeError("No OpenAI models to try.")
    finally:
        if args.delete_uploaded_file:
            for uploaded in uploaded_files:
                try:
                    client.files.delete(uploaded.id)
                    print(f"OpenAI uploaded file deleted: {uploaded.id}", file=sys.stderr, flush=True)
                except Exception as error:
                    print(f"Failed to delete OpenAI uploaded file {uploaded.id}: {error}", file=sys.stderr, flush=True)


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
    elapsed_seconds: float,
    generated_timestamp: str,
    generation_started_at: str,
    generation_completed_at: str,
) -> dict[str, Any]:
    result = dict(toc)
    result["_meta"] = {
        "source_pdf": str(pdf_path),
        "provider": "openai",
        "model": model,
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
    base_name = f"{safe_filename_part(pdf_path.stem)}_openai_{timestamp_for_filename()}"
    log_file = unique_output_path(error_dir / f"{base_name}_error.json")
    payload = {
        "timestamp": timestamp_for_metadata(),
        "source_pdf": str(pdf_path),
        "provider": "openai",
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
    processing_pdf_path, display_name = prepare_pdf_for_processing(pdf_path)
    print(f"Processing started: {display_name}", flush=True)
    started_at = time.perf_counter()

    try:
        prompt = build_prompt(user_prompt=user_prompt, max_depth=args.max_depth)
        toc, raw_text, used_model = generate_toc_from_pdf_openai(
            pdf_path=processing_pdf_path,
            args=args,
            prompt=prompt,
        )
        elapsed_seconds = time.perf_counter() - started_at
        generation_completed_at = timestamp_for_metadata()
        generated_timestamp = timestamp_for_filename()
        base_name = f"{safe_filename_part(pdf_path.stem)}_openai_{safe_filename_part(used_model)}_{generated_timestamp}"
        output_file = unique_output_path(Path(args.output_dir) / f"{base_name}_toc.json")
        result = add_result_metadata(
            toc=toc,
            pdf_path=pdf_path,
            model=used_model,
            elapsed_seconds=elapsed_seconds,
            generated_timestamp=generated_timestamp,
            generation_started_at=generation_started_at,
            generation_completed_at=generation_completed_at,
        )
        save_json(output_file, result)

        if args.write_raw:
            raw_file = unique_output_path(Path(args.output_dir) / f"{base_name}_raw_response.txt")
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
            f"Completed: {output_file} / elapsed {format_elapsed(elapsed_seconds)} / "
            f"chapters {len(toc.get('chapters', []))}",
        )
        return True

    except Exception as error:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"Failed: {display_name} / {error}", file=sys.stderr, flush=True)
        print(f"  elapsed: {format_elapsed(elapsed_seconds)} ({elapsed_seconds:.3f}s)", file=sys.stderr, flush=True)
        error_log = write_error_log(pdf_path=pdf_path, args=args, error=error, elapsed_seconds=elapsed_seconds)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-only PDF TOC JSON generator.")
    parser.add_argument("path", nargs="?", default="./data/input", help="PDF file or directory")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--prompt", default=None, help="TOC generation instruction")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 text file containing the TOC instruction")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ai-fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--no-schema", action="store_true", help="Disable OpenAI JSON schema response format")
    parser.add_argument("--write-raw", action="store_true", help="Save raw OpenAI response text")
    parser.add_argument("--delete-uploaded-file", action="store_true", help="Delete uploaded OpenAI file after processing")
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
        help="Detailed log path used with --quiet-console (default: output_dir/logs/batch_openai_TIMESTAMP_log.txt)",
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
            else Path(args.output_dir) / "logs" / f"batch_openai_{timestamp_for_filename()}_log.txt"
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
