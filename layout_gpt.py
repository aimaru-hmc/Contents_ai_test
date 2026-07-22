from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from ai_openai_v3 import (
    DEFAULT_AI_RETRIES,
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRY_BASE_DELAY,
    create_openai_client,
    get_api_key,
    is_configuration_error,
    parse_json_response,
    parse_model_list,
    with_retry,
)


INPUT_DIR = Path("./data/output_pdf")
OUTPUT_DIR = Path("./data/output/layout_gpt")


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_filename_part(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^\w가-힣.-]+", "_", value, flags=re.UNICODE)
    value = value.strip("._-")
    return value or "file"


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


def extract_pdf_layout_json(pdf_path: Path) -> dict[str, Any]:
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError("Layout extraction requires `pip install pdfplumber`.") from error

    pages: list[dict[str, Any]] = []
    total_lines = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_chars = [
                char for char in (getattr(page, "chars", None) or [])
                if isinstance(char, dict) and clean_text(char.get("text")) and char.get("size") is not None
            ]
            body_size = median_number((char.get("size") for char in page_chars), default=10.0) or 10.0
            page_record: dict[str, Any] = {
                "page": page_number,
                "width": float(page.width),
                "height": float(page.height),
                "body_size": round(body_size, 3),
                "lines": [],
            }

            try:
                lines = page.extract_text_lines(layout=False, return_chars=True) or []
            except Exception:
                lines = []

            if not lines:
                fallback = str(page.extract_text() or "").strip()
                for order, text in enumerate(fallback.splitlines(), start=1):
                    text = clean_text(text)
                    if not text:
                        continue
                    page_record["lines"].append({
                        "page": page_number,
                        "order": order,
                        "s": round(body_size, 3),
                        "r": 1.0,
                        "x": 0.0,
                        "y": 0.0,
                        "b": 0,
                        "i": 0,
                        "f": "unknown",
                        "text": text,
                    })
                    total_lines += 1
                pages.append(page_record)
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
                page_record["lines"].append({
                    "page": page_number,
                    "order": order,
                    "s": round(size, 3),
                    "r": round(ratio, 3),
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "b": bold,
                    "i": italic,
                    "f": font,
                    "text": text,
                })
                total_lines += 1
            pages.append(page_record)

    if total_lines == 0:
        raise RuntimeError("No extractable PDF text/layout was found. OCR is required for scanned PDFs.")

    return {
        "source_pdf": str(pdf_path),
        "generated_at": timestamp_for_metadata(),
        "schema": layout_schema(),
        "page_count": len(pages),
        "line_count": total_lines,
        "pages": pages,
    }


def layout_schema() -> dict[str, str]:
    return {
        "page": "PDF viewer page number",
        "order": "source line order within page",
        "s": "font size",
        "r": "font size / page body font size ratio",
        "x": "left x position",
        "y": "top y position",
        "b": "bold flag inferred from font name",
        "i": "italic flag inferred from font name",
        "f": "compact dominant font name",
        "text": "source line text",
    }


def layout_json_to_text(data: dict[str, Any]) -> str:
    output = [
        "# Exact PDF line layout metadata",
        "# P=viewer page, O=source line order, S=font size, R=size/body ratio, X/Y=position, B=bold, I=italic, F=font name",
        "# Use O to preserve order for lines sharing the same page.",
    ]
    for page in data.get("pages", []):
        output.append(
            f"[PAGE P={page['page']} W={float(page['width']):.1f} H={float(page['height']):.1f} BODY={float(page['body_size']):.2f}]"
        )
        for line in page.get("lines", []):
            output.append(
                f"[L P={line['page']} O={line['order']} S={float(line['s']):.2f} R={float(line['r']):.3f} "
                f"X={float(line['x']):.1f} Y={float(line['y']):.1f} B={int(line['b'])} I={int(line['i'])} F={line['f']}] {line['text']}"
            )
    return "\n".join(output) + "\n"


def build_layout_prompt(pdf_path: Path) -> str:
    example = {
        "source_pdf": str(pdf_path),
        "generated_by": "gpt",
        "page_count": 0,
        "line_count": 0,
        "schema": layout_schema(),
        "pages": [
            {
                "page": 1,
                "width": 595.0,
                "height": 842.0,
                "body_size": 10.0,
                "lines": [
                    {
                        "page": 1,
                        "order": 1,
                        "s": 14.0,
                        "r": 1.4,
                        "x": 72.0,
                        "y": 90.0,
                        "b": 1,
                        "i": 0,
                        "f": "FontName",
                        "text": "source text",
                    }
                ],
            }
        ],
    }
    return f"""
You convert an attached parsed PDF layout text file into layout JSON.
No PDF file is attached. Use only the attached parsed text file.

[Input Format]
The attached .txt contains lines like:
- [PAGE P=1 W=595.0 H=842.0 BODY=10.00]
- [L P=1 O=1 S=14.00 R=1.400 X=72.0 Y=90.0 B=1 I=0 F=FontName] source text

[Task]
Create one JSON object that preserves every parsed page and every parsed line from the attached text file.
Do not infer missing visual information. Do not summarize, translate, merge, drop, or rewrite text.

[Output JSON]
Return exactly this structure:
{json.dumps(example, ensure_ascii=False, indent=2)}

[Rules]
- Output valid JSON only. No Markdown, comments, or code fences.
- page_count must equal the number of PAGE records.
- line_count must equal the number of L records.
- Preserve numeric fields as numbers and text/font fields as strings.
- Preserve page order and line order exactly.
""".strip()


def call_openai_layout(client: Any, model: str, prompt: str, file_id: str, max_output_tokens: int) -> Any:
    return client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
        max_output_tokens=max(1, int(max_output_tokens)),
        store=False,
    )


def validate_layout_json(data: dict[str, Any], fallback_source_pdf: Path) -> dict[str, Any]:
    pages = data.get("pages")
    if not isinstance(pages, list):
        raise ValueError("GPT layout JSON must contain a pages array.")

    line_count = 0
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Each page entry must be an object.")
        lines = page.get("lines", [])
        if not isinstance(lines, list):
            raise ValueError("Each page entry must contain a lines array.")
        line_count += len([line for line in lines if isinstance(line, dict)])

    result = dict(data)
    result["source_pdf"] = clean_text(result.get("source_pdf")) or str(fallback_source_pdf)
    result["page_count"] = len(pages)
    result["line_count"] = line_count
    if "schema" not in result or not isinstance(result.get("schema"), dict):
        result["schema"] = layout_schema()
    return result


def generate_layout_json_from_parsed_text(pdf_path: Path, parsed_text: str, args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    client = create_openai_client(api_key=get_api_key())
    models = parse_model_list(args.model, args.ai_fallback_models)
    prompt = build_layout_prompt(pdf_path)
    last_error: Exception | None = None
    uploaded_file: Any | None = None

    with tempfile.TemporaryDirectory(prefix="layout_gpt_upload_") as temp_dir:
        upload_path = Path(temp_dir) / f"{safe_filename_part(pdf_path.stem)}_parsed_layout.txt"
        upload_path.write_text(parsed_text, encoding="utf-8")

        def do_upload() -> Any:
            with upload_path.open("rb") as file_obj:
                return client.files.create(file=file_obj, purpose="user_data")

        print(f"  OpenAI parsed layout txt upload started: {upload_path.name}", flush=True)
        uploaded_file = with_retry(
            label=f"OpenAI upload({upload_path.name})",
            func=do_upload,
            max_retries=args.ai_retries,
            base_delay=args.ai_retry_base_delay,
        )
        print("  OpenAI parsed layout txt upload completed", flush=True)

        try:
            for index, model in enumerate(models):
                if index > 0:
                    print(f"Trying OpenAI fallback model: {model}", file=sys.stderr, flush=True)
                for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
                    request_prompt = prompt
                    if parse_attempt > 1:
                        request_prompt += (
                            "\n\n[Retry Instruction]\n"
                            "The previous response failed JSON parsing. Return exactly one valid JSON object "
                            "that Python json.loads() can parse. Do not output Markdown."
                        )
                    try:
                        print(f"  OpenAI layout JSON generation started: {model}", flush=True)
                        response = with_retry(
                            label=f"OpenAI layout generation({model})",
                            func=lambda model=model, request_prompt=request_prompt: call_openai_layout(
                                client=client,
                                model=model,
                                prompt=request_prompt,
                                file_id=uploaded_file.id,
                                max_output_tokens=args.max_output_tokens,
                            ),
                            max_retries=args.ai_retries,
                            base_delay=args.ai_retry_base_delay,
                        )
                        print(f"  OpenAI layout JSON generation completed: {model}", flush=True)
                        raw_text = str(getattr(response, "output_text", "") or "")
                        parsed = validate_layout_json(parse_json_response(response), fallback_source_pdf=pdf_path)
                        return parsed, raw_text, model
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
            if args.delete_uploaded_file and uploaded_file is not None:
                try:
                    client.files.delete(uploaded_file.id)
                    print(f"  OpenAI uploaded file deleted: {uploaded_file.id}", flush=True)
                except Exception as error:
                    print(f"Failed to delete OpenAI uploaded file {uploaded_file.id}: {error}", file=sys.stderr, flush=True)


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() == ".pdf")
    raise FileNotFoundError(f"Path not found: {path}")


def output_subdir_for_pdf(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_dir():
        try:
            relative_parent = pdf_path.parent.relative_to(input_root)
        except ValueError:
            relative_parent = Path()
        return output_root / relative_parent
    return output_root


def process_pdf(pdf_path: Path, args: argparse.Namespace, input_root: Path) -> Path:
    local_layout = extract_pdf_layout_json(pdf_path)
    parsed_text = layout_json_to_text(local_layout)
    print(f"  Parsed layout text created: {len(parsed_text.encode('utf-8')):,} bytes", flush=True)

    data, raw_text, used_model = generate_layout_json_from_parsed_text(pdf_path, parsed_text, args)
    data.setdefault("_meta", {})
    if isinstance(data["_meta"], dict):
        data["_meta"].update({
            "model": used_model,
            "api_input": "parsed_layout_txt_only",
            "local_parser": "pdfplumber",
        })

    stamp = timestamp_for_filename()
    base = f"{safe_filename_part(pdf_path.stem)}_layout_{stamp}"
    output_dir = output_subdir_for_pdf(pdf_path, input_root, Path(args.output_dir))
    output_json = unique_output_path(output_dir / f"{base}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.write_txt:
        output_txt = unique_output_path(output_dir / f"{base}.txt")
        output_txt.write_text(parsed_text, encoding="utf-8")

    if args.write_raw:
        raw_path = unique_output_path(output_dir / f"{base}_raw_response.txt")
        raw_path.write_text(raw_text, encoding="utf-8")

    return output_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send parsed layout text from level_pdf.py output PDFs to GPT and save layout JSON.")
    parser.add_argument("path", nargs="?", default=str(INPUT_DIR), help="PDF file or directory (default: ./data/output_pdf)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory (default: ./data/output/layout_gpt)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model (default: gpt-5.5 or OPENAI_MODEL)")
    parser.add_argument("--ai-fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--write-txt", action="store_true", help="Also save the parsed layout .txt sent to OpenAI")
    parser.add_argument("--write-raw", action="store_true", help="Save raw GPT response text")
    parser.add_argument("--delete-uploaded-file", action="store_true", default=True, help="Delete uploaded parsed txt file after processing")
    parser.add_argument("--keep-uploaded-file", action="store_false", dest="delete_uploaded_file", help="Keep uploaded parsed txt file")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one PDF fails")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_root = Path(args.path)
    pdf_files = iter_pdfs(input_root)
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
            output_json = process_pdf(pdf_path, args, input_root)
            elapsed = time.perf_counter() - started
            success_count += 1
            print(f"Completed: {output_json}", flush=True)
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
