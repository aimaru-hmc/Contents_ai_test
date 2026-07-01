from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("data/input")
DEFAULT_OUTPUT_DIR = Path("data/output")


@dataclass(frozen=True)
class TimeRecord:
    source_pdf: str
    provider: str
    model: str
    elapsed_seconds: float
    path: Path


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}초"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}분 {remaining_seconds:.1f}초"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}시간 {int(remaining_minutes)}분 {remaining_seconds:.1f}초"


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_time_record(path: Path) -> TimeRecord | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"스킵: {path.name} / JSON 읽기 실패: {error}")
        return None

    meta = data.get("_meta")
    if not isinstance(meta, dict):
        print(f"스킵: {path.name} / _meta 없음")
        return None

    source_pdf = clean_text(meta.get("source_pdf"))
    provider = clean_text(meta.get("provider"))
    model = clean_text(meta.get("model"))

    try:
        elapsed_seconds = float(meta.get("elapsed_seconds"))
    except Exception:
        print(f"스킵: {path.name} / elapsed_seconds 없음")
        return None

    if not source_pdf or not provider or not model:
        print(f"스킵: {path.name} / source_pdf, provider, model 중 누락")
        return None

    return TimeRecord(
        source_pdf=source_pdf,
        provider=provider,
        model=model,
        elapsed_seconds=elapsed_seconds,
        path=path,
    )


def load_records(output_dir: Path) -> list[TimeRecord]:
    records: list[TimeRecord] = []
    for path in sorted(output_dir.glob("*_toc.json")):
        record = load_time_record(path)
        if record is not None:
            records.append(record)
    return records


def input_pdf_names(input_dir: Path) -> set[str]:
    if not input_dir.exists():
        return set()
    return {path.name.lower() for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"}


def filter_records(records: list[TimeRecord], input_dir: Path, include_all: bool) -> tuple[list[TimeRecord], set[str]]:
    names = input_pdf_names(input_dir)
    if include_all or not names:
        return records, names

    filtered = [record for record in records if record.source_pdf.lower() in names]
    return filtered, names


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def summarize_by_model(records: list[TimeRecord]) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[TimeRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.provider, record.model)].append(record)

    rows: list[list[str]] = []
    for (provider, model), items in sorted(grouped.items()):
        total = sum(item.elapsed_seconds for item in items)
        average = total / len(items)
        rows.append([
            provider,
            model,
            str(len(items)),
            f"{total:.3f}",
            format_elapsed(total),
            format_elapsed(average),
        ])
    return rows


def summarize_by_file(records: list[TimeRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in sorted(records, key=lambda item: (item.source_pdf.lower(), item.provider, item.model)):
        rows.append([
            record.source_pdf,
            record.provider,
            record.model,
            f"{record.elapsed_seconds:.3f}",
            format_elapsed(record.elapsed_seconds),
        ])
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 목차 생성 결과 JSON의 모델별 소요 시간을 합산합니다.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="합산 대상 PDF가 있는 폴더")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="*_toc.json 결과가 있는 폴더")
    parser.add_argument("--all", action="store_true", help="input-dir 기준 필터 없이 output-dir의 모든 결과를 합산")
    parser.add_argument("--detail", action="store_true", help="파일별 상세 시간을 같이 출력")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        parser.error(f"output-dir을 찾을 수 없습니다: {output_dir}")

    records = load_records(output_dir)
    records, input_names = filter_records(records, input_dir=input_dir, include_all=args.all)

    if not records:
        print("합산할 elapsed_seconds 기록이 없습니다.")
        return 0

    if args.all:
        print(f"대상: {output_dir} 전체 결과 {len(records)}개")
    elif input_names:
        print(f"대상 input PDF: {len(input_names)}개 / 결과 기록: {len(records)}개")
    else:
        print(f"input-dir PDF가 없어 전체 결과를 사용합니다: {len(records)}개")

    print("\n모델별 합계")
    print_table(
        ["provider", "model", "files", "total_seconds", "total", "average"],
        summarize_by_model(records),
    )

    if args.detail:
        print("\n파일별 상세")
        print_table(
            ["source_pdf", "provider", "model", "seconds", "elapsed"],
            summarize_by_file(records),
        )

    if input_names and not args.all:
        seen = {record.source_pdf.lower() for record in records}
        missing = sorted(input_names - seen)
        if missing:
            print("\n결과가 없는 input PDF")
            for name in missing:
                print(f"- {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
