from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_gemma_overlap import (
    DEFAULT_AI_RETRIES,
    DEFAULT_GEMMA_BACKEND,
    DEFAULT_GEMMA_CONTEXT_WINDOW,
    DEFAULT_GEMMA_KEEP_ALIVE,
    DEFAULT_GEMMA_OLLAMA_MODEL,
    DEFAULT_GEMMA_OPENAI_BASE_URL,
    DEFAULT_GEMMA_TEXT_CHUNK_CHARS,
    DEFAULT_GEMMA_TEXT_CHUNK_OVERLAP_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_REQUEST_TIMEOUT,
    DEFAULT_RETRY_BASE_DELAY,
    clean_text,
    chunk_pdf_text_pages,
    extract_gemma_pdf_pages,
    format_elapsed,
    gemma_response_text,
    generate_gemma_once,
    is_configuration_error,
    normalize_gemma_backend,
    normalize_gemma_model_for_backend,
    parse_json_response_text,
    safe_filename_part,
    unique_output_path,
    with_retry,
)


INPUT_DIR = Path("./data/input")
OUTPUT_DIR = Path("./data/output_pdf")
DEFAULT_MODEL = DEFAULT_GEMMA_OLLAMA_MODEL
DEFAULT_LEVEL_GEMMA_BACKEND = "openai"
DEFAULT_CHUNK_CHARS = max(int(DEFAULT_GEMMA_TEXT_CHUNK_CHARS), 220000)
DEFAULT_CHUNK_OVERLAP_CHARS = max(int(DEFAULT_GEMMA_TEXT_CHUNK_OVERLAP_CHARS), 8000)
DEFAULT_LEVEL_MAX_OUTPUT_TOKENS = max(int(DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER["gemma"]), 32768)


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    page: int
    order: int
    reason: str = ""


@dataclass(frozen=True)
class OutputPart:
    level_start: int
    level_end: int
    start_page: int
    end_page: int
    output_pdf: str


@dataclass(frozen=True)
class ExtractionPlan:
    source_pdf: str
    model: str
    page_count: int
    max_level: int
    target: dict[str, Any]
    path: list[dict[str, Any]]
    outputs: list[dict[str, Any]]


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


def build_heading_prompt(pdf_name: str, max_depth: int) -> str:
    return f"""
You are analyzing PDF document hierarchy from extracted text/layout metadata.
Return only real hierarchy headings as JSON. Do not create a pretty table of contents.

[Task]
Identify every real heading line needed to understand the document hierarchy.
The result will be used to cut PDFs by hierarchy levels.

[Rules]
- Use only evidence in the extracted PDF text/layout.
- First apply all exclusion rules. Only then classify the remaining real body headings.
- Exclusion rules have higher priority than every layout, style, position, and numbering rule.
- Do not assume any fixed document format, numbering style, language, textbook template, or section naming convention.
- Infer hierarchy from the source itself: S/R/B/I/F text-format signature, font size, size ratio, indentation, X/Y position, spacing, page order, explicit numbering, local context, and structural containment.
- For body headings, headings with the same S/R/B/I/F signature should normally receive the same level unless explicit numbering or secondary X/Y layout evidence clearly distinguishes their structural roles.
- Do not discard X/Y. Use X/Y as secondary layout evidence and as a tie-breaker when format signatures alone are ambiguous.
- Existing TOC/Contents pages may help you understand structure, but do not output listing rows unless the same title appears as a real body heading.
- Exclude covers, prefaces, existing TOC listing rows, indexes, standalone page numbers, repeated headers/footers, captions, figure/table labels, references, body sentences, examples, questions/exercises, and incidental lists.
- Before finalizing, perform a second review of unmatched lines with heading-like numbering or typography and include valid unseen-style headings, while still excluding numbered body lists and sentences.
- Include headings deep enough to reach the deepest real hierarchy level visible in the document, up to level {max_depth}.
- Do not skip visible parent headings between level 1 and the deepest heading.
- Never reject a heading only because its S/R/B/I/F style or target level was absent from another part of the document.
- Level 1 is the highest document hierarchy role found in the body.
- Use actual PDF viewer page numbers from [PAGE] markers or P metadata.
- Use order as the source line order within the page. Never calculate or invent order.
- Keep exact source heading text and original numbering. Do not invent, rewrite, summarize, or include layout tags in text.
- Output exactly one valid compact JSON object. Do not output Markdown, comments, code fences, trailing commas, or partial JSON.

[Output JSON]
{{
  "headings": [
    {{"level": 1, "text": "exact heading text", "page": 1, "order": 1, "reason": "short evidence"}},
    {{"level": 2, "text": "exact heading text", "page": 2, "order": 3, "reason": "short evidence"}}
  ]
}}

[PDF File]
{pdf_name}
""".strip()


def build_chunk_prompt(base_prompt: str, pdf_name: str, chunk: dict[str, Any], total_chunks: int) -> str:
    return f"""
{base_prompt}

[Chunk Instruction]
This is one large chunk of the PDF. Include only headings visible in this chunk.
Do not infer headings outside this chunk. Adjacent chunks may overlap; keep duplicated headings identical if they appear.

[Chunk]
{chunk['index']}/{total_chunks}, pages {chunk['start_page']}-{chunk['end_page']}

[Extracted PDF Text/Layout]
{chunk['text']}
""".strip()


def build_merge_prompt(base_prompt: str, pdf_name: str, partials: list[dict[str, Any]], max_depth: int) -> str:
    partial_json = json.dumps(partials, ensure_ascii=False, indent=2)
    return f"""
{base_prompt}

[Merge Instruction]
Merge the partial heading candidates into one final heading list.
- Remove duplicates caused by chunk overlap.
- Preserve page order and source order.
- Keep only real hierarchy headings.
- Keep levels 1 to {max_depth}.
- Do not invent headings.
- Return exactly one compact valid JSON object with a headings array. Do not output trailing commas or partial JSON.

[PDF File]
{pdf_name}

[Partial Heading Candidates]
{partial_json}
""".strip()


def request_gemma_json(model: str, prompt: str, args: argparse.Namespace, label: str) -> tuple[dict[str, Any], str]:
    retry_count = max(1, int(args.ai_retries))
    last_error: Exception | None = None
    last_raw = ""

    for attempt in range(1, retry_count + 1):
        request_prompt = prompt
        if attempt > 1:
            preview = clean_text(last_raw)[:500]
            request_prompt += (
                "\n\n[Retry Instruction]\n"
                f"Previous JSON parse failed: {type(last_error).__name__}: {last_error}\n"
                f"Previous response preview: {preview}\n"
                "Return exactly one valid compact JSON object. Do not include Markdown, comments, code fences, trailing commas, or partial JSON."
            )

        response = with_retry(
            label=f"{label} generation",
            func=lambda request_prompt=request_prompt: generate_gemma_once(
                model=model,
                prompt=request_prompt,
                args=args,
            ),
            max_retries=args.ai_retries,
            base_delay=args.ai_retry_base_delay,
        )
        raw = gemma_response_text(response)
        try:
            return parse_json_response_text(raw, provider="gemma"), raw
        except Exception as error:
            last_error = error
            last_raw = raw
            if attempt >= retry_count:
                raise

    if last_error:
        raise last_error
    raise RuntimeError(f"{label} failed")


def validate_headings(data: dict[str, Any], max_depth: int, page_count: int) -> list[Heading]:
    raw_headings = data.get("headings", [])
    if not isinstance(raw_headings, list):
        raise ValueError("Gemma response must contain a headings array.")

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
        raise RuntimeError("Gemma returned no headings.")
    return normalize_levels(headings)


def normalize_levels(headings: list[Heading]) -> list[Heading]:
    used = sorted({heading.level for heading in headings})
    mapping = {level: index + 1 for index, level in enumerate(used)}
    return [Heading(**{**asdict(heading), "level": mapping[heading.level]}) for heading in headings]


def extract_chunks(pdf_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    print("  Gemma PDF text/layout extraction started...", flush=True)
    pages, metadata = extract_gemma_pdf_pages(pdf_path, args)
    page_count = pdf_page_count(pdf_path)
    extracted_chars = sum(len(text) for _, text in pages)
    print(
        f"  Gemma PDF extraction completed: {len(pages)} pages, {extracted_chars:,} chars, mode={metadata.get('gemma_extraction_mode')}",
        flush=True,
    )
    chunk_chars = max(10000, int(args.gemma_chunk_chars))
    overlap_chars = max(0, min(int(args.gemma_chunk_overlap_chars), chunk_chars // 2))
    chunks = chunk_pdf_text_pages(pages, max_chars=chunk_chars, overlap_chars=overlap_chars)
    print(f"  Gemma chunking completed: {len(chunks)} chunks, chunk_chars={chunk_chars:,}, overlap={overlap_chars:,}", flush=True)
    return chunks, page_count


def generate_gemma_headings(pdf_path: Path, args: argparse.Namespace, page_count: int) -> tuple[list[Heading], str, str]:
    model = normalize_gemma_model_for_backend(args.model, args.gemma_backend)
    base_prompt = build_heading_prompt(pdf_path.name, args.max_depth)
    chunks, _ = extract_chunks(pdf_path, args)
    raw_parts: list[dict[str, Any]] = []
    partials: list[dict[str, Any]] = []

    for chunk in chunks:
        print(f"  Gemma chunk {chunk['index']}/{len(chunks)} request: pages {chunk['start_page']}-{chunk['end_page']}", flush=True)
        parsed, raw = request_gemma_json(
            model=model,
            prompt=build_chunk_prompt(base_prompt, pdf_path.name, chunk, len(chunks)),
            args=args,
            label=f"Gemma chunk {chunk['index']}/{len(chunks)}",
        )
        headings = validate_headings(parsed, args.max_depth, page_count)
        partials.append({
            "chunk": chunk["index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "headings": [asdict(item) for item in headings],
        })
        raw_parts.append({"chunk": chunk["index"], "raw_response": raw})
        print(f"  Gemma chunk {chunk['index']} completed: {len(headings)} headings", flush=True)

    if len(partials) == 1:
        final_data = {"headings": partials[0]["headings"]}
        final_raw = raw_parts[0]["raw_response"]
    else:
        print(f"  Gemma chunk merge started: {model}", flush=True)
        final_data, final_raw = request_gemma_json(
            model=model,
            prompt=build_merge_prompt(base_prompt, pdf_path.name, partials, args.max_depth),
            args=args,
            label="Gemma chunk merge",
        )
        print("  Gemma chunk merge completed", flush=True)

    headings = validate_headings(final_data, args.max_depth, page_count)
    raw_bundle = json.dumps(
        {
            "model": model,
            "chunk_count": len(chunks),
            "raw_parts": raw_parts if args.write_raw else [],
            "final_raw_response": final_raw if args.write_raw else "",
        },
        ensure_ascii=False,
        indent=2,
    )
    return headings, raw_bundle, model


def first_deepest_path(headings: list[Heading]) -> tuple[Heading, list[Heading]]:
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


def next_heading_page_for_level(headings: list[Heading], current: Heading, level: int, page_count: int) -> int:
    started = False
    for heading in headings:
        if heading is current:
            started = True
            continue
        if not started:
            continue
        if heading.level <= level:
            return max(current.page, heading.page)
    return page_count


def deepest_level_section_end_page(headings: list[Heading], current: Heading, page_count: int) -> int:
    deepest_level = current.level
    started = False
    for heading in headings:
        if heading is current:
            started = True
            continue
        if not started:
            continue
        if heading.level < deepest_level:
            if heading.page <= current.page:
                return current.page
            return max(current.page, heading.page)
    return page_count


def heading_for_level(path: list[Heading], level: int) -> Heading:
    eligible = [heading for heading in path if heading.level <= level]
    if not eligible:
        return path[0]
    return eligible[-1]


def containing_heading_for_level(headings: list[Heading], target: Heading, level: int) -> Heading:
    target_pos = headings.index(target)
    for heading in reversed(headings[: target_pos + 1]):
        if heading.level == level:
            return heading
    for heading in reversed(headings[: target_pos + 1]):
        if heading.level <= level:
            return heading
    return headings[0]


def first_heading_at_level(headings: list[Heading], level: int) -> Heading | None:
    for heading in headings:
        if heading.level == level:
            return heading
    return None


def level_groups(max_level: int, group_size: int) -> list[tuple[int, int]]:
    group_size = max(2, int(group_size))
    groups: list[tuple[int, int]] = []
    start = 1
    while start <= max_level:
        end = min(max_level, start + group_size - 1)
        groups.append((start, end))
        if end == max_level:
            break
        start = end
    return groups


def compute_level_group_ranges(
    headings: list[Heading],
    path: list[Heading],
    page_count: int,
    group_size: int,
) -> list[tuple[int, int, int, int, Heading]]:
    max_level = path[-1].level
    ranges: list[tuple[int, int, int, int, Heading]] = []
    for group_index, (start_level, end_level) in enumerate(level_groups(max_level, group_size)):
        if group_index == 0:
            end_heading = first_heading_at_level(headings, end_level) or heading_for_level(path, end_level)
            start_heading = containing_heading_for_level(headings, end_heading, start_level)
        else:
            end_heading = heading_for_level(path, end_level)
            start_heading = containing_heading_for_level(headings, end_heading, start_level)

        # Non-final PDFs end when the target level first appears inside that group.
        # The final PDF extends through the next higher-level heading page.
        start_page = max(1, min(start_heading.page, page_count))
        if end_level >= max_level:
            end_page = deepest_level_section_end_page(headings, path[-1], page_count)
        else:
            end_page = max(1, min(end_heading.page, page_count))
        end_page = max(start_page, min(end_page, page_count))
        ranges.append((start_level, end_level, start_page, end_page, end_heading))
    return ranges


def level_examples(headings: list[Heading]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_levels: set[int] = set()
    for heading in headings:
        if heading.level in seen_levels:
            continue
        seen_levels.add(heading.level)
        examples.append(asdict(heading))
    return examples


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> ExtractionPlan:
    page_count = pdf_page_count(pdf_path)
    headings, raw_text, model = generate_gemma_headings(pdf_path, args, page_count)
    target, path = first_deepest_path(headings)
    ranges = compute_level_group_ranges(headings, path, page_count, args.level_group_size)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    file_stem = safe_filename_part(pdf_path.stem)
    model_stem = safe_filename_part(model)
    run_dir = unique_output_path(Path(args.output_dir) / f"{file_stem}_gemma_{model_stem}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[OutputPart] = []
    for start_level, end_level, start_page, end_page, end_heading in ranges:
        base = f"L{start_level}-{end_level}_p{start_page}-{end_page}"
        output_pdf = unique_output_path(run_dir / f"{base}.pdf")
        write_pdf_range(pdf_path, output_pdf, start_page, end_page)
        outputs.append(OutputPart(
            level_start=start_level,
            level_end=end_level,
            start_page=start_page,
            end_page=end_page,
            output_pdf=str(output_pdf),
        ))

    examples_path = unique_output_path(run_dir / "level_examples.json")
    examples_path.write_text(
        json.dumps(
            {
                "source_pdf": str(pdf_path),
                "model": model,
                "max_level": target.level,
                "examples": level_examples(headings),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    if args.write_raw:
        raw_path = unique_output_path(run_dir / "raw_response.txt")
        raw_path.write_text(raw_text, encoding="utf-8")

    return ExtractionPlan(
        source_pdf=str(pdf_path),
        model=model,
        page_count=page_count,
        max_level=target.level,
        target=asdict(target),
        path=[asdict(item) for item in path],
        outputs=[asdict(item) for item in outputs],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma/vLLM PDF range extractor split by hierarchy level groups.")
    parser.add_argument("path", nargs="?", default=str(INPUT_DIR), help="PDF file or directory (default: ./data/input)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory (default: ./data/output_pdf)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemma model alias or server model name (default: gemma4:31b)")
    parser.add_argument("--ai-fallback-models", default="", help="Reserved for compatibility; currently unused")
    parser.add_argument(
        "--gemma-backend",
        default=DEFAULT_LEVEL_GEMMA_BACKEND,
        choices=("ollama", "transformers", "openai", "vllm", "server"),
        help="Gemma runtime backend. Default uses the local OpenAI-compatible vLLM server.",
    )
    parser.add_argument(
        "--gemma-runtime",
        default=None,
        choices=("server", "vllm", "openai", "ollama", "transformers"),
        help="Compatibility alias for --gemma-backend. Default backend is server/openai.",
    )
    parser.add_argument("--ollama-base-url", default=DEFAULT_GEMMA_OPENAI_BASE_URL, help="Base URL for Ollama or OpenAI-compatible/vLLM server")
    parser.add_argument("--vllm-base-url", dest="ollama_base_url", help="OpenAI-compatible/vLLM server base URL")
    parser.add_argument("--gemma-vllm-server-base-url", dest="ollama_base_url", help="OpenAI-compatible/vLLM server base URL")
    parser.add_argument("--ollama-request-timeout", type=int, default=DEFAULT_OLLAMA_REQUEST_TIMEOUT)
    parser.add_argument("--api-key", default="", help="API key for OpenAI-compatible/vLLM server")
    parser.add_argument("--gemma-context-window", type=int, default=DEFAULT_GEMMA_CONTEXT_WINDOW)
    parser.add_argument("--gemma-keep-alive", default=DEFAULT_GEMMA_KEEP_ALIVE)
    parser.add_argument("--gemma-think", default="")
    parser.add_argument("--gemma-device-map", default="auto")
    parser.add_argument("--gemma-torch-dtype", default="auto")
    parser.add_argument("--gemma-extraction-mode", default="layout", choices=("layout", "text"))
    parser.add_argument("--gemma-chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS, help="Maximum extracted text/layout chars per Gemma chunk")
    parser.add_argument("--gemma-chunk-overlap-chars", type=int, default=DEFAULT_CHUNK_OVERLAP_CHARS)
    parser.add_argument("--level-group-size", type=int, default=3, help="Number of levels per output PDF; adjacent groups overlap by one level")
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_LEVEL_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--no-schema", action="store_true", help="Compatibility option; headings mode always disables the TOC schema")
    parser.add_argument("--write-raw", action="store_true", help="Save Gemma raw response bundle")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one PDF fails")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.gemma_runtime:
        args.gemma_backend = args.gemma_runtime
    args.gemma_backend = normalize_gemma_backend(args.gemma_backend)
    args.max_depth = max(1, min(int(args.max_depth), 7))
    args.level_group_size = max(2, int(args.level_group_size))
    args.no_schema = True
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
            print(f"Completed: {len(plan.outputs)} PDFs", flush=True)
            print(f"  model: {plan.model}", flush=True)
            print(f"  max_level: {plan.max_level}", flush=True)
            print(f"  first_deepest: {plan.target.get('text')}", flush=True)
            for output in plan.outputs:
                print(
                    f"  L{output['level_start']}-{output['level_end']}: "
                    f"pages {output['start_page']}-{output['end_page']} -> {output['output_pdf']}",
                    flush=True,
                )
            print(f"  elapsed: {format_elapsed(elapsed)} ({elapsed:.3f}s)", flush=True)
        except Exception as error:
            elapsed = time.perf_counter() - started
            failed_count += 1
            print(f"Failed: {pdf_path} / {error}", file=sys.stderr, flush=True)
            print(f"  elapsed: {format_elapsed(elapsed)} ({elapsed:.3f}s)", file=sys.stderr, flush=True)
            if is_configuration_error(error) or args.stop_on_error:
                raise

    total_elapsed = time.perf_counter() - total_started
    print(
        f"Processing result: success {success_count} / failed {failed_count} / elapsed {format_elapsed(total_elapsed)}",
        flush=True,
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
