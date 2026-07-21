from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_openai_v3 import (
    DEFAULT_AI_RETRIES,
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRY_BASE_DELAY,
    clean_text,
    create_openai_client,
    extract_pdf_layout_text,
    format_elapsed,
    get_api_key,
    is_configuration_error,
    is_schema_config_error,
    make_ascii_upload_name,
    parse_json_response,
    parse_model_list,
    prepare_pdf_for_processing,
    safe_filename_part,
    timestamp_for_filename,
    timestamp_for_metadata,
    unique_output_path,
    with_retry,
)


INPUT_DIR = Path("./data/input")
OUTPUT_DIR = Path("./data/output_pdf")
API_INPUT_MODES = ("both", "pdf", "txt")

HEADINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer"},
                    "text": {"type": "string"},
                    "page": {"type": "integer"},
                    "order": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["level", "text", "page", "order", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["headings"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    page: int
    order: int
    reason: str = ""


@dataclass(frozen=True)
class ExtractionPlan:
    source_pdf: str
    model: str
    api_input_mode: str
    page_count: int
    max_level: int
    start_page: int
    end_page: int
    target: dict[str, Any]
    range_parent: dict[str, Any]
    path: list[dict[str, Any]]
    output_pdf: str
    headings_json: str | None
    generated_at: str


def pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PyMuPDF is required. Install it with `pip install pymupdf`.") from error
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def write_pdf_range(source_pdf: Path, output_pdf: Path, start_page: int, end_page: int) -> None:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PyMuPDF is required. Install it with `pip install pymupdf`.") from error

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source_pdf) as src:
        page_count = src.page_count
        start = max(1, min(start_page, page_count)) - 1
        end = max(1, min(end_page, page_count)) - 1
        with fitz.open() as out:
            out.insert_pdf(src, from_page=start, to_page=end)
            out.save(output_pdf)


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() == ".pdf")
    raise FileNotFoundError(f"Path not found: {path}")


def build_ai_prompt(pdf_name: str, max_depth: int) -> str:
    return f"""
You are analyzing a PDF document hierarchy. Your job is not to create a nice TOC; your job is to identify the real hierarchy headings needed to cut the PDF range.

[Goal]
Return the document heading lines with levels. After this, code will find the first deepest-level heading and extract PDF pages from its level 1 ancestor through the page where the next heading higher than that deepest level appears. PDF extraction is page-based, so this includes deepest-level siblings that appear earlier on that boundary page.

[Important Rules]
- Use only evidence from the attached source.
- Do not assume any fixed document format, numbering style, language, textbook template, or section naming convention.
- Do not rely on hard-coded patterns. Infer hierarchy from the source itself.
- Decide whether a line is a heading by combining visual/layout evidence, source order, repeated role, indentation, spacing, and document context.
- If parsed layout metadata is attached, use P(page), O(line order), S(font size), R(size/body ratio), X/Y(position), B/I(style), and F(font) as evidence.
- If the PDF is attached, use the PDF visual layout as evidence too.
- Exclude body sentences, running headers/footers, page numbers, captions, figure/table labels, examples, exercises/questions, references, and incidental lists unless they clearly function as hierarchy headings in this document.
- Include headings deep enough to reach the deepest real hierarchy level visible in the document, up to level {max_depth}.
- Do not skip parent headings between level 1 and the deepest real heading when those parent headings are visible in the source.
- Level 1 is the highest document hierarchy role found in the body.
- Use actual PDF viewer page numbers.
- Use order as the source line order within the page.
- Keep exact source heading text.
- Return valid JSON only.

[Output JSON]
{{
  "headings": [
    {{"level": 1, "text": "exact heading text", "page": 1, "order": 1, "reason": "short evidence"}},
    {{"level": 2, "text": "exact heading text", "page": 2, "order": 3, "reason": "short evidence"}}
  ]
}}

[Source File]
{pdf_name}
""".strip()


def build_response_payloads(
    model: str,
    prompt: str,
    file_ids: list[str],
    max_output_tokens: int | None,
    use_schema: bool,
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": model,
        "input": [{
            "role": "user",
            "content": [
                *[{"type": "input_file", "file_id": file_id} for file_id in file_ids],
                {"type": "input_text", "text": prompt},
            ],
        }],
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
                "name": "level_pdf_headings_schema",
                "schema": HEADINGS_SCHEMA,
                "strict": True,
            }
        }
        payloads.append(schema_payload)

        json_payload = dict(base)
        json_payload["text"] = {"format": {"type": "json_object"}}
        payloads.append(json_payload)

    payloads.append(base)
    return payloads


def create_ai_response(
    client: Any,
    model: str,
    prompt: str,
    file_ids: list[str],
    max_output_tokens: int | None,
    use_schema: bool,
) -> Any:
    last_error: Exception | None = None
    for payload in build_response_payloads(model, prompt, file_ids, max_output_tokens, use_schema):
        try:
            return client.responses.create(**payload)
        except Exception as error:
            last_error = error
            if is_schema_config_error(error):
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("OpenAI response generation failed.")


def validate_headings(data: dict[str, Any], max_depth: int, page_count: int) -> list[Heading]:
    raw_headings = data.get("headings", [])
    if not isinstance(raw_headings, list):
        raise ValueError("AI response must contain a headings array.")

    headings: list[Heading] = []
    seen: set[tuple[int, int, str]] = set()
    for raw in raw_headings:
        if not isinstance(raw, dict):
            continue
        text = clean_text(raw.get("text"))
        if not text:
            continue
        try:
            level = max(1, min(int(raw.get("level", 1)), int(max_depth)))
            page = max(1, min(int(raw.get("page", 1)), int(page_count)))
            order = max(1, int(raw.get("order", 1)))
        except Exception:
            continue
        key = (page, order, text.casefold())
        if key in seen:
            continue
        seen.add(key)
        headings.append(Heading(
            level=level,
            text=text,
            page=page,
            order=order,
            reason=clean_text(raw.get("reason")),
        ))

    headings.sort(key=lambda item: (item.page, item.order))
    if not headings:
        raise RuntimeError("AI returned no headings.")
    return normalize_levels(headings)


def normalize_levels(headings: list[Heading]) -> list[Heading]:
    used = sorted({heading.level for heading in headings})
    mapping = {level: index + 1 for index, level in enumerate(used)}
    return [Heading(**{**asdict(heading), "level": mapping[heading.level]}) for heading in headings]


def generate_ai_headings(pdf_path: Path, args: argparse.Namespace, page_count: int) -> tuple[list[Heading], str, str]:
    client = create_openai_client(api_key=get_api_key())
    mode = clean_text(args.api_input_mode).lower()
    if mode not in API_INPUT_MODES:
        raise ValueError(f"Unsupported api input mode: {mode}")

    processing_pdf, display_name = prepare_pdf_for_processing(pdf_path)
    models = parse_model_list(args.model, args.ai_fallback_models)
    uploaded_files: list[Any] = []
    raw_text = ""

    try:
        with tempfile.TemporaryDirectory(prefix="level_pdf_ai_upload_") as temp_name:
            temp_dir = Path(temp_name)
            upload_paths: list[Path] = []

            if mode in ("both", "pdf"):
                pdf_upload = temp_dir / make_ascii_upload_name(pdf_path)
                shutil.copy2(processing_pdf, pdf_upload)
                upload_paths.append(pdf_upload)

            if mode in ("both", "txt"):
                print("  PDF layout extraction started...", flush=True)
                try:
                    layout_text = extract_pdf_layout_text(pdf_path)
                except Exception as error:
                    if mode == "txt":
                        raise
                    print(f"  PDF layout extraction failed; continuing with PDF only: {error}", file=sys.stderr, flush=True)
                    mode = "pdf"
                else:
                    print(f"  PDF layout extraction completed: {len(layout_text.encode('utf-8')):,} bytes", flush=True)
                    layout_upload = temp_dir / f"{safe_filename_part(pdf_path.stem)}_layout.txt"
                    layout_upload.write_text(layout_text, encoding="utf-8")
                    upload_paths.append(layout_upload)

            print(f"  OpenAI upload started: {mode}", flush=True)
            for upload_path in upload_paths:
                def do_upload(upload_path: Path = upload_path) -> Any:
                    with upload_path.open("rb") as file_obj:
                        return client.files.create(file=file_obj, purpose="user_data")

                uploaded = with_retry(
                    label=f"OpenAI upload({upload_path.name})",
                    func=do_upload,
                    max_retries=args.ai_retries,
                    base_delay=args.ai_retry_base_delay,
                )
                uploaded_files.append(uploaded)
            print("  OpenAI upload completed", flush=True)

        setattr(args, "_effective_api_input_mode", mode)
        file_ids = [uploaded.id for uploaded in uploaded_files]
        prompt = build_ai_prompt(display_name, args.max_depth)
        last_error: Exception | None = None

        for model in models:
            for attempt in range(1, max(1, int(args.ai_retries)) + 1):
                retry_prompt = prompt
                if attempt > 1:
                    retry_prompt += "\n\nReturn exactly one valid JSON object. Do not include Markdown."
                try:
                    print(f"  OpenAI heading detection started: {model}", flush=True)
                    response = with_retry(
                        label=f"OpenAI heading detection({model})",
                        func=lambda model=model, retry_prompt=retry_prompt: create_ai_response(
                            client=client,
                            model=model,
                            prompt=retry_prompt,
                            file_ids=file_ids,
                            max_output_tokens=args.max_output_tokens,
                            use_schema=not args.no_schema,
                        ),
                        max_retries=args.ai_retries,
                        base_delay=args.ai_retry_base_delay,
                    )
                    print(f"  OpenAI heading detection completed: {model}", flush=True)
                    raw_text = str(getattr(response, "output_text", "") or "")
                    headings = validate_headings(parse_json_response(response), args.max_depth, page_count)
                    return headings, raw_text, model
                except Exception as error:
                    last_error = error
                    if is_configuration_error(error):
                        raise
                    if attempt < max(1, int(args.ai_retries)) and isinstance(error, (json.JSONDecodeError, ValueError)):
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
                except Exception as error:
                    print(f"Failed to delete OpenAI uploaded file {uploaded.id}: {error}", file=sys.stderr, flush=True)


def find_first_deepest_path(headings: list[Heading]) -> tuple[Heading, list[Heading]]:
    max_level = max(heading.level for heading in headings)
    target = next(heading for heading in headings if heading.level == max_level)
    target_pos = headings.index(target)

    ancestors: list[Heading] = []
    current_level = target.level
    for heading in reversed(headings[:target_pos]):
        if heading.level < current_level:
            ancestors.append(heading)
            current_level = heading.level
            if current_level == 1:
                break

    ancestors.reverse()
    return target, [*ancestors, target]


def next_upper_heading_end_page(headings: list[Heading], target: Heading, page_count: int) -> int:
    started = False
    for heading in headings:
        if heading is target:
            started = True
            continue
        if not started:
            continue
        if heading.level < target.level:
            return max(target.page, heading.page)
    return page_count


def compute_extract_range(headings: list[Heading], path: list[Heading], page_count: int) -> tuple[int, int, Heading]:
    if not path:
        raise RuntimeError("Could not build a heading path.")
    target = path[-1]
    parent = path[-2] if len(path) >= 2 else target
    start_page = path[0].page
    end_page = next_upper_heading_end_page(headings, target, page_count)
    start_page = max(1, min(start_page, page_count))
    end_page = max(start_page, min(end_page, page_count))
    return start_page, end_page, parent


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> ExtractionPlan:
    started = time.perf_counter()
    page_count = pdf_page_count(pdf_path)
    headings, raw_text, model = generate_ai_headings(pdf_path, args, page_count)
    target, path = find_first_deepest_path(headings)
    start_page, end_page, parent = compute_extract_range(headings, path, page_count)

    stamp = timestamp_for_filename()
    base = f"{safe_filename_part(pdf_path.stem)}_ai_{safe_filename_part(model)}_L{target.level}_p{start_page}-{end_page}_{stamp}"
    output_pdf = unique_output_path(Path(args.output_dir) / f"{base}.pdf")
    write_pdf_range(pdf_path, output_pdf, start_page, end_page)

    generated_at = timestamp_for_metadata()
    if args.write_raw:
        raw_path = unique_output_path(Path(args.output_dir) / f"{base}_raw_response.txt")
        raw_path.write_text(raw_text, encoding="utf-8")

    return ExtractionPlan(
        source_pdf=str(pdf_path),
        model=model,
        api_input_mode=args.api_input_mode,
        page_count=page_count,
        max_level=target.level,
        start_page=start_page,
        end_page=end_page,
        target=asdict(target),
        range_parent=asdict(parent),
        path=[asdict(item) for item in path],
        output_pdf=str(output_pdf),
        headings_json=None,
        generated_at=generated_at,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-only PDF range extractor for the first deepest heading group.")
    parser.add_argument("path", nargs="?", default=str(INPUT_DIR), help="PDF file or directory (default: ./data/input)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory (default: ./data/output_pdf)")
    parser.add_argument("--api-input-mode", choices=API_INPUT_MODES, default="both", help="both=PDF + parsed layout TXT, pdf=PDF only, txt=parsed layout TXT only")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ai-fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--no-schema", action="store_true", help="Disable JSON schema response format")
    parser.add_argument("--write-raw", action="store_true", help="Save raw OpenAI response text")
    parser.add_argument("--delete-uploaded-file", action="store_true", help="Delete uploaded OpenAI files after processing")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one PDF fails")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.max_depth = max(1, min(int(args.max_depth), 10))
    args.api_input_mode = clean_text(args.api_input_mode).lower()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pdf_files = iter_pdfs(Path(args.path))
    if not pdf_files:
        print(f"No PDF files to process: {args.path}", flush=True)
        return 0

    total_started = time.perf_counter()
    success_count = 0
    failed_count = 0
    print(f"PDF files found: {len(pdf_files)}", flush=True)

    for index, pdf_path in enumerate(pdf_files, start=1):
        started = time.perf_counter()
        print(f"Processing started ({index}/{len(pdf_files)}): {pdf_path}", flush=True)
        try:
            plan = process_pdf(pdf_path, args)
            elapsed = time.perf_counter() - started
            success_count += 1
            print(f"Completed: {plan.output_pdf}", flush=True)
            print(f"  model: {plan.model}", flush=True)
            print(f"  api_input_mode: {plan.api_input_mode}", flush=True)
            print(f"  range: pages {plan.start_page}-{plan.end_page} / {plan.page_count}", flush=True)
            print(f"  max_level: {plan.max_level}", flush=True)
            print(f"  range_parent: {plan.range_parent.get('text')}", flush=True)
            print(f"  first_deepest: {plan.target.get('text')}", flush=True)
            print(f"  elapsed: {format_elapsed(elapsed)} ({elapsed:.3f}s)", flush=True)
        except Exception as error:
            elapsed = time.perf_counter() - started
            failed_count += 1
            print(f"Failed: {pdf_path} / {error}", file=sys.stderr, flush=True)
            print(f"  elapsed: {format_elapsed(elapsed)} ({elapsed:.3f}s)", file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise

    total_elapsed = time.perf_counter() - total_started
    print(
        f"Processing result: success {success_count} / failed {failed_count} / elapsed {format_elapsed(total_elapsed)}",
        flush=True,
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
