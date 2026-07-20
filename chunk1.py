from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data/input"
DEFAULT_OUTPUT_DIR = ROOT / "data/output"
DEFAULT_INPUT_CHUNKS_DIR = DEFAULT_OUTPUT_DIR / "input_chunks"
DEFAULT_DESTINATION_DIR = DEFAULT_INPUT_DIR / "chunk1_v2"

SOURCE_TOC_RE = re.compile(r"^.+_openai_gpt-5\.5_toc\.json$", re.IGNORECASE)
CHUNK1_INPUT_RE = re.compile(
    r"^chunk_1_pages_(?P<start>\d+)-(?P<end>\d+)\.json$",
    re.IGNORECASE,
)


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


def normalized_document_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", value)


def find_input_documents(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    documents = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not documents:
        raise FileNotFoundError(f"No PDF input files found in: {input_dir}")
    return documents


def match_input_document(toc_path: Path, toc: dict[str, Any], documents: list[Path]) -> Path | None:
    source_values = [toc_path.stem, str(toc.get("title", ""))]
    metadata = toc.get("_meta")
    if isinstance(metadata, dict):
        source_values.extend([
            str(metadata.get("source_pdf", "")),
            str(metadata.get("source_file", "")),
        ])
    normalized_sources = [normalized_document_name(Path(value).stem) for value in source_values if value]

    matches: list[tuple[int, Path]] = []
    for document in documents:
        wanted = normalized_document_name(document.stem)
        if wanted and any(source.startswith(wanted) for source in normalized_sources):
            matches.append((len(wanted), document))
    return max(matches, key=lambda item: (item[0], str(item[1])))[1] if matches else None


def is_complete_toc(path: Path) -> bool:
    return bool(SOURCE_TOC_RE.fullmatch(path.name))


def toc_max_page(toc: dict[str, Any]) -> int:
    pages: list[int] = []
    for chapter in toc.get("chapters", []):
        if not isinstance(chapter, dict):
            continue
        try:
            pages.append(int(chapter.get("page")))
        except (TypeError, ValueError):
            continue
    return max(pages, default=0)


def collect_toc_candidates(
    output_dir: Path,
    documents: list[Path],
) -> dict[Path, list[tuple[Path, dict[str, Any], int]]]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    candidates: dict[Path, list[tuple[Path, dict[str, Any], int]]] = {}
    for path in output_dir.iterdir():
        if not path.is_file() or not is_complete_toc(path):
            continue
        try:
            toc = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(toc.get("chapters"), list):
            continue
        input_document = match_input_document(path, toc, documents)
        if input_document is not None:
            candidates.setdefault(input_document, []).append((path, toc, toc_max_page(toc)))

    if not candidates:
        raise FileNotFoundError(
            f"No complete TOC JSON matching a PDF filename under {DEFAULT_INPUT_DIR} was found in: {output_dir}"
        )
    return candidates


def choose_source_toc(
    candidates: list[tuple[Path, dict[str, Any], int]],
    required_start_page: int,
    required_end_page: int,
) -> tuple[Path, dict[str, Any], int] | None:
    with_entries = [
        item
        for item in candidates
        if any(
            isinstance(chapter, dict)
            and required_start_page <= int(chapter.get("page", 0)) <= required_end_page
            for chapter in item[1].get("chapters", [])
        )
    ]
    if not with_entries:
        return None
    covering = [item for item in with_entries if item[2] >= required_end_page]
    if covering:
        return max(covering, key=lambda item: (item[0].stat().st_mtime_ns, item[0].name))
    return max(
        with_entries,
        key=lambda item: (item[2], item[0].stat().st_mtime_ns, item[0].name),
    )


def chunk1_input_files(run_dir: Path) -> list[Path]:
    return sorted(
        path for path in run_dir.iterdir()
        if path.is_file() and CHUNK1_INPUT_RE.match(path.name)
    )


def input_chunk_matches_document(path: Path, input_document: Path) -> bool:
    try:
        source_pdf = read_json(path).get("source_pdf")
    except (OSError, ValueError):
        return False
    if not source_pdf:
        return False
    return normalized_document_name(Path(str(source_pdf)).stem) == normalized_document_name(input_document.stem)


def find_input_chunks_run(input_chunks_dir: Path, input_document: Path) -> tuple[Path, list[Path]]:
    if not input_chunks_dir.is_dir():
        raise FileNotFoundError(f"Input chunks directory not found: {input_chunks_dir}")

    candidates: list[tuple[Path, list[Path]]] = []
    for run_dir in input_chunks_dir.iterdir():
        if not run_dir.is_dir():
            continue
        files = chunk1_input_files(run_dir)
        if files and any(input_chunk_matches_document(path, input_document) for path in files):
            candidates.append((run_dir, files))

    if not candidates:
        raise FileNotFoundError(
            f"No chunk 1 input matching input file {input_document.name!r} found in: {input_chunks_dir}"
        )
    return max(
        candidates,
        key=lambda item: (max(path.stat().st_mtime_ns for path in item[1]), item[0].name),
    )


def chunk1_page_range(paths: list[Path]) -> tuple[int, int]:
    if not paths:
        raise ValueError("Chunk 1 result files are empty.")

    ranges: list[tuple[int, int]] = []
    for path in paths:
        match = CHUNK1_INPUT_RE.match(path.name)
        if not match:
            continue
        data = read_json(path)
        try:
            start = int(data.get("start_page", match.group("start")))
            end = int(data.get("end_page", match.group("end")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid page range in input chunk: {path}") from error
        if start < 1 or end < start:
            raise ValueError(f"Invalid chunk page range {start}-{end}: {path}")
        ranges.append((start, end))

    if not ranges:
        raise ValueError("No valid chunk 1 page ranges were found.")
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def trim_toc_to_page_range(toc: dict[str, Any], start_page: int, end_page: int) -> dict[str, Any]:
    chapters = toc.get("chapters")
    if not isinstance(chapters, list):
        raise ValueError("TOC JSON must contain a chapters list.")

    filtered: list[dict[str, Any]] = []
    for index, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, dict):
            raise ValueError(f"chapters[{index}] must be an object.")
        try:
            page = int(chapter.get("page"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"chapters[{index}].page must be an integer.") from error
        if start_page <= page <= end_page:
            filtered.append(dict(chapter))

    result = dict(toc)
    result["chapters"] = filtered
    metadata = dict(result.get("_meta")) if isinstance(result.get("_meta"), dict) else {}
    metadata.update({
        "chunk1_created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chunk1_start_page": start_page,
        "chunk1_end_page": end_page,
        "source_chapter_count": len(chapters),
        "chunk1_chapter_count": len(filtered),
    })
    result["_meta"] = metadata
    return result


def create_chunk1_output(
    *,
    source: Path,
    source_toc: dict[str, Any],
    source_max_page: int,
    input_document: Path,
    chunk_run_dir: Path,
    chunk_files: list[Path],
    start_page: int,
    end_page: int,
    destination_dir: Path,
) -> Path:
    chunk1_toc = trim_toc_to_page_range(source_toc, start_page, end_page)
    destination = destination_dir / f"{source.stem}_chunk1.json"
    chunk1_toc["_meta"].update({
        "chunk1_source_toc": str(source),
        "chunk1_source_toc_max_page": source_max_page,
        "chunk1_source_input": str(input_document),
        "chunk1_source_input_chunks_dir": str(chunk_run_dir),
        "chunk1_source_input_chunks": [str(path) for path in chunk_files],
    })
    destination.write_text(json.dumps(chunk1_toc, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    print(f"Input file: {input_document}")
    print(f"Source TOC: {source}")
    print(f"Source TOC max page: {source_max_page}")
    print(f"Input chunks: {chunk_run_dir}")
    print(f"Chunk 1 file: {chunk_files[0]}")
    print(f"Page range: {start_page}-{end_page}")
    print(f"Chapters: {len(source_toc['chapters'])} -> {len(chunk1_toc['chapters'])}")
    print(f"Created: {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="data/input의 PDF와 대응하는 *_openai_gpt-5.5_toc.json을 1번 청크 범위로 잘라 저장합니다.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-chunks-dir", type=Path, default=DEFAULT_INPUT_CHUNKS_DIR)
    parser.add_argument("--destination-dir", type=Path, default=DEFAULT_DESTINATION_DIR)
    parser.add_argument("--source", type=Path, help="자동 탐색 대신 사용할 원본 TOC JSON")
    parser.add_argument("--input-file", type=Path, help="자동 매칭 대신 사용할 원본 PDF")
    parser.add_argument("--input-chunks-run", type=Path, help="자동 탐색 대신 사용할 input_chunks 실행 폴더")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    input_chunks_dir = args.input_chunks_dir.resolve()
    destination_dir = args.destination_dir.resolve()
    documents = find_input_documents(input_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if args.source:
        source = args.source.resolve()
        source_toc = read_json(source)
        input_document = args.input_file.resolve() if args.input_file else match_input_document(source, source_toc, documents)
        if input_document is None:
            raise ValueError(f"Could not match source TOC to a PDF under {input_dir}: {source}")
        if args.input_chunks_run:
            chunk_run_dir = args.input_chunks_run.resolve()
            chunk_files = chunk1_input_files(chunk_run_dir)
        else:
            chunk_run_dir, chunk_files = find_input_chunks_run(input_chunks_dir, input_document)
        start_page, end_page = chunk1_page_range(chunk_files)
        create_chunk1_output(
            source=source,
            source_toc=source_toc,
            source_max_page=toc_max_page(source_toc),
            input_document=input_document,
            chunk_run_dir=chunk_run_dir,
            chunk_files=chunk_files,
            start_page=start_page,
            end_page=end_page,
            destination_dir=destination_dir,
        )
        return

    toc_candidates = collect_toc_candidates(output_dir, documents)
    created: list[Path] = []
    skipped: list[str] = []
    for input_document in documents:
        candidates = toc_candidates.get(input_document)
        if not candidates:
            skipped.append(f"{input_document.name}: matching TOC not found")
            continue
        try:
            chunk_run_dir, chunk_files = find_input_chunks_run(input_chunks_dir, input_document)
        except FileNotFoundError as error:
            skipped.append(f"{input_document.name}: {error}")
            continue
        start_page, end_page = chunk1_page_range(chunk_files)
        selected = choose_source_toc(candidates, start_page, end_page)
        if selected is None:
            skipped.append(
                f"{input_document.name}: no TOC contains chapters in chunk 1 pages {start_page}-{end_page}"
            )
            continue
        source, source_toc, source_max_page = selected
        print()
        created.append(create_chunk1_output(
            source=source,
            source_toc=source_toc,
            source_max_page=source_max_page,
            input_document=input_document,
            chunk_run_dir=chunk_run_dir,
            chunk_files=chunk_files,
            start_page=start_page,
            end_page=end_page,
            destination_dir=destination_dir,
        ))

    print(f"\nCreated files: {len(created)}")
    if skipped:
        print(f"Skipped inputs: {len(skipped)}")
        for reason in skipped:
            print(f"  - {reason}")
    if not created:
        raise RuntimeError("No chunk1 JSON files were created.")


if __name__ == "__main__":
    main()
