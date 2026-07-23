from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data/input"
DEFAULT_OUTPUT_DIR = ROOT / "data/test_del"


HEADING_PATTERNS = (
    re.compile(r"^CHAPTER(?:\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^SECTION(?:\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^\d{1,3}\s+[가-힣A-Za-z]"),
    re.compile(r"^\d{1,3}\.\s*[가-힣A-Za-z]"),
    re.compile(r"^\d{1,3}\)\s*[가-힣A-Za-z]"),
    re.compile(r"^\(\d{1,3}\)\s*[가-힣A-Za-z]"),
    re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s*[가-힣A-Za-z]"),
    re.compile(r"^[가-힣]\.\s*[가-힣A-Za-z]"),
    re.compile(r"^[A-Z]\.\s+[A-Za-z가-힣]"),
)

AUTHOR_LINE_RE = re.compile(r"^[가-힣]{2,4}(?:,\s*[가-힣]{2,4})+(?:\s*\(.+\))?$")


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "").replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_filename_part(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "document"


def median_float(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return float(default)
    mid = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[mid]
    return (numbers[mid - 1] + numbers[mid]) / 2.0


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
    bold = int(any(marker in fontname for marker in ("bold", "black", "heavy", "demi", "semibold", "medium")))
    italic = int(any(marker in fontname for marker in ("italic", "oblique", "slant")))
    return bold, italic


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
    text = clean_text(fallback_text)
    if text:
        return text
    return clean_text("".join(str(char.get("text") or "") for char in chars))


def page_has_two_columns(page: Any, split_x: float, min_chars_per_side: int = 80) -> bool:
    chars = [char for char in getattr(page, "chars", []) or [] if clean_text(char.get("text"))]
    if len(chars) < min_chars_per_side * 2:
        return False
    left = sum(1 for char in chars if float(char.get("x0", 0.0) or 0.0) < split_x - 20)
    right = sum(1 for char in chars if float(char.get("x0", 0.0) or 0.0) > split_x + 20)
    total = max(1, left + right)
    return left >= min_chars_per_side and right >= min_chars_per_side and left / total >= 0.25 and right / total >= 0.25


def is_heading_like(
    text: str,
    *,
    size: float,
    ratio: float,
    bold: int,
    body_size: float,
    max_heading_chars: int,
    drop_author_lines: bool,
) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if drop_author_lines and AUTHOR_LINE_RE.match(text):
        return False
    if any(pattern.match(text) for pattern in HEADING_PATTERNS):
        return len(text) <= max_heading_chars
    if len(text) > max_heading_chars:
        return False
    if ratio >= 1.25:
        return True
    if bold and ratio >= 0.95 and len(text) <= 80:
        return True
    if body_size and size >= body_size + 2.0 and len(text) <= 120:
        return True
    return False


def format_heading_line(
    line: dict[str, Any],
    *,
    body_size: float,
    font_ids: dict[str, str],
    column: str | None,
    max_heading_chars: int,
    drop_author_lines: bool,
) -> str:
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
    if not is_heading_like(
        text,
        size=size,
        ratio=ratio,
        bold=bold,
        body_size=body_size,
        max_heading_chars=max_heading_chars,
        drop_author_lines=drop_author_lines,
    ):
        return ""
    font_id = font_id_for(fontname, font_ids)
    column_part = f" c={column}" if column else ""
    return f"[L s={size:.1f} r={ratio:.2f} x={x0:.0f} y={top:.0f} b={bold} i={italic} f={font_id}{column_part}] {text}"


def extract_region_headings(
    page: Any,
    bbox: tuple[float, float, float, float] | None,
    *,
    body_size: float,
    font_ids: dict[str, str],
    column: str | None,
    max_heading_chars: int,
    drop_author_lines: bool,
) -> list[str]:
    region = page.crop(bbox) if bbox else page
    try:
        raw_lines = region.extract_text_lines(layout=False, return_chars=True)
    except Exception:
        raw_lines = []
    result: list[str] = []
    for line in raw_lines:
        if not isinstance(line, dict):
            continue
        formatted = format_heading_line(
            line,
            body_size=body_size,
            font_ids=font_ids,
            column=column,
            max_heading_chars=max_heading_chars,
            drop_author_lines=drop_author_lines,
        )
        if formatted:
            result.append(formatted)
    return result


def parse_pdf_headings_only(pdf_path: Path, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError("PDF parsing requires `pip install pdfplumber`.") from error

    font_ids: dict[str, str] = {}
    output: list[str] = []
    kept_lines = 0
    two_column_pages = 0
    mode = clean_text(args.pdf_column_mode).lower() or "auto"

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            body_size = page_body_font_size(page)
            width = float(getattr(page, "width", 0.0) or 0.0)
            height = float(getattr(page, "height", 0.0) or 0.0)
            split_x = float(args.pdf_column_split_x or 0.0) or (width / 2.0 if width else 0.0)
            use_two_columns = False
            if mode == "two":
                use_two_columns = bool(width and height and split_x > 0)
            elif mode == "auto":
                use_two_columns = bool(width and height and split_x > 0 and page_has_two_columns(page, split_x))

            page_lines: list[str] = []
            if use_two_columns:
                two_column_pages += 1
                gap = max(0.0, float(args.pdf_column_gap))
                left_bbox = (0.0, 0.0, max(0.0, split_x - gap / 2.0), height)
                right_bbox = (min(width, split_x + gap / 2.0), 0.0, width, height)
                page_lines.extend(extract_region_headings(
                    page, left_bbox, body_size=body_size, font_ids=font_ids, column="L",
                    max_heading_chars=args.max_heading_chars, drop_author_lines=args.drop_author_lines,
                ))
                page_lines.extend(extract_region_headings(
                    page, right_bbox, body_size=body_size, font_ids=font_ids, column="R",
                    max_heading_chars=args.max_heading_chars, drop_author_lines=args.drop_author_lines,
                ))
            else:
                page_lines.extend(extract_region_headings(
                    page, None, body_size=body_size, font_ids=font_ids, column=None,
                    max_heading_chars=args.max_heading_chars, drop_author_lines=args.drop_author_lines,
                ))

            if page_lines:
                output.append(f"[PAGE {page_index}]")
                output.extend(page_lines)
                output.append("")
                kept_lines += len(page_lines)

    metadata = {
        "source_pdf": str(pdf_path),
        "mode": "headings_only",
        "pdf_column_mode": args.pdf_column_mode,
        "pdf_column_split_x": args.pdf_column_split_x,
        "pdf_column_gap": args.pdf_column_gap,
        "two_column_pages": two_column_pages,
        "kept_lines": kept_lines,
        "font_count": len(font_ids),
        "max_heading_chars": args.max_heading_chars,
        "drop_author_lines": args.drop_author_lines,
    }
    header = [
        "# Parsed PDF Text - Body Removed",
        f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}",
        "",
    ]
    return "\n".join(header + output).rstrip() + "\n", metadata


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
    return sorted(path.resolve() for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDFs and save heading-like lines only, removing body text.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("-f", "--input-file", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-column-mode", choices=("auto", "none", "two"), default="auto")
    parser.add_argument("--pdf-column-split-x", type=float, default=0.0)
    parser.add_argument("--pdf-column-gap", type=float, default=8.0)
    parser.add_argument("--max-heading-chars", type=int, default=140)
    parser.add_argument("--drop-author-lines", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = iter_input_pdfs(args.input_dir, args.input_file)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under: {args.input_dir}")

    print(f"PDF files: {len(pdfs)}", flush=True)
    for pdf_path in pdfs:
        print(f"processing: {pdf_path}", flush=True)
        text, metadata = parse_pdf_headings_only(pdf_path, args)
        output_path = args.output_dir / f"{safe_filename_part(pdf_path.stem)}_body_removed.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"saved: {output_path} | kept_lines={metadata['kept_lines']} | two_column_pages={metadata['two_column_pages']}", flush=True)


if __name__ == "__main__":
    main()
