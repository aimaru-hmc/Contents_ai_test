from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable


INPUT_DIR = Path("./data/output_pdf")
OUTPUT_DIR = Path("./data/output/layout_openai")


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
        "schema": {
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
        },
        "page_count": len(pages),
        "line_count": total_lines,
        "pages": pages,
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


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*.pdf") if item.is_file())
    raise FileNotFoundError(f"Path not found: {path}")


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> Path:
    data = extract_pdf_layout_json(pdf_path)
    stamp = timestamp_for_filename()
    base = f"{safe_filename_part(pdf_path.stem)}_layout_{stamp}"
    output_dir = Path(args.output_dir)
    output_json = unique_output_path(output_dir / f"{base}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.write_txt:
        output_txt = unique_output_path(output_dir / f"{base}.txt")
        output_txt.write_text(layout_json_to_text(data), encoding="utf-8")

    return output_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract layout JSON from level_pdf.py output PDFs.")
    parser.add_argument("path", nargs="?", default=str(INPUT_DIR), help="PDF file or directory (default: ./data/output_pdf)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory (default: ./data/output/layout_openai)")
    parser.add_argument("--write-txt", action="store_true", help="Also save ai_openai_v3-compatible parsed layout .txt")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one PDF fails")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
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
            output_json = process_pdf(pdf_path, args)
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
