from __future__ import annotations

import argparse
import atexit
import json
import math
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ai_gemma import (
    DEFAULT_AI_RETRIES,
    DEFAULT_FALLBACK_MODELS_BY_PROVIDER,
    DEFAULT_GEMMA_DEVICE_MAP,
    DEFAULT_GEMMA_HF_MODEL,
    DEFAULT_GEMMA_RUNTIME,
    DEFAULT_GEMMA_TORCH_DTYPE,
    DEFAULT_GEMMA_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_GEMMA_VLLM_MAX_MODEL_LEN,
    DEFAULT_GEMMA_VLLM_SERVER_API_KEY,
    DEFAULT_GEMMA_VLLM_SERVER_BASE_URL,
    DEFAULT_GEMMA_VLLM_SERVER_TIMEOUT,
    DEFAULT_GEMMA_VLLM_TENSOR_PARALLEL_SIZE,
    DEFAULT_GEMMA_VLLM_TRUST_REMOTE_CODE,
    DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_GEMMA_TEXT_CHUNK_CHARS,
    chunk_pdf_text_pages,
    extract_pdf_layout_pages,
    parsed_pdf_text_for_file,
    generate_gemma_once,
    normalize_model_for_provider,
    parse_json_response_text,
    parse_model_list,
    validate_toc,
    with_retry,
)


ROOT = Path(__file__).resolve().parent
CHUNK = ROOT / "data/output/input_chunks/2_5-8_2021_gemma_20260708_153404/chunk_1_pages_1-39.json"
REFERENCE = ROOT / "data/output/2편5-8장_2021_openai_gpt-5.5_20260710_144043_toc_39.json"
PARSED = ROOT / "data/output/parsed_pdfs/2_5-8_2021_gemma_20260708_153404_parsed_pdf.txt"
INPUT_DIR = ROOT / "data/input"
TOC_OUTPUT_DIR = ROOT / "data/output/toc_txt/toc"
LAYOUT_OUTPUT_DIR = ROOT / "data/output/toc_txt/layout"
RAW_OUTPUT_DIR = ROOT / "data/output/toc_txt/raw"
REFERENCE_OUTPUT_DIR = ROOT / "data/output/toc_txt/reference"
DEBUG_OUTPUT_DIR = ROOT / "data/output/toc_txt/debug"
GENERATED_INPUT_DIR = ROOT / "data/output/toc_txt/generated_input"

DEFAULT_REFERENCE_MODEL = "gpt-oss-120b"
DEFAULT_REFERENCE_BASE_URL = DEFAULT_GEMMA_VLLM_SERVER_BASE_URL
DEFAULT_REFERENCE_API_KEY = DEFAULT_GEMMA_VLLM_SERVER_API_KEY
DEFAULT_REFERENCE_TIMEOUT = DEFAULT_GEMMA_VLLM_SERVER_TIMEOUT

PAGE_OLD_RE = re.compile(r"^\[PAGE\s+(\d+)\]$")
PAGE_NEW_RE = re.compile(r"^\[PAGE\s+P=(\d+)[^]]*\]$")
LINE_OLD_RE = re.compile(
    r"^\[L s=(?P<s>[\d.]+) r=(?P<r>[\d.]+) x=(?P<x>-?[\d.]+) "
    r"y=(?P<y>-?[\d.]+) b=(?P<b>[01]) i=(?P<i>[01]) "
    r"f=(?P<f>[^]]+)\]\s*(?P<t>.*)$"
)
LINE_NEW_RE = re.compile(
    r"^\[L\s+P=(?P<p>\d+)\s+O=(?P<o>\d+)\s+S=(?P<s>[\d.]+)\s+"
    r"R=(?P<r>[\d.]+)\s+X=(?P<x>-?[\d.]+)\s+Y=(?P<y>-?[\d.]+)\s+"
    r"B=(?P<b>[01])\s+I=(?P<i>[01])\s+F=(?P<f>[^]]+)\]\s*(?P<t>.*)$"
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
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def norm(value: Any) -> str:
    return re.sub(r"\s+", "", clean(value)).replace("–", "-").replace("—", "-")


def token_shape(token: Any) -> str:
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
    return [(match.group(), match.start()) for match in re.finditer(r"\S+", text)]


def parse_layout(text: str) -> list[Line]:
    page = 0
    order = 0
    result: list[Line] = []

    for raw in str(text or "").splitlines():
        stripped = raw.strip()
        page_match = PAGE_OLD_RE.match(stripped) or PAGE_NEW_RE.match(stripped)
        if page_match:
            page = int(page_match.group(1))
            order = 0
            continue

        match = LINE_NEW_RE.match(stripped)
        if match:
            values = match.groupdict()
            title = clean(values["t"])
            if title:
                result.append(
                    Line(
                        page=int(values["p"]),
                        order=int(values["o"]),
                        s=float(values["s"]),
                        r=float(values["r"]),
                        x=float(values["x"]),
                        y=float(values["y"]),
                        b=int(values["b"]),
                        i=int(values["i"]),
                        f=values["f"],
                        text=title,
                    )
                )
            continue

        match = LINE_OLD_RE.match(stripped)
        if match and page:
            order += 1
            values = match.groupdict()
            title = clean(values["t"])
            if title:
                result.append(
                    Line(
                        page=page,
                        order=order,
                        s=float(values["s"]),
                        r=float(values["r"]),
                        x=float(values["x"]),
                        y=float(values["y"]),
                        b=int(values["b"]),
                        i=int(values["i"]),
                        f=values["f"],
                        text=title,
                    )
                )

    if not result:
        raise ValueError("레이아웃 줄을 찾지 못했습니다.")
    return result


def match_reference_title(title: Any, page_lines: Iterable[Line]) -> Line | None:
    wanted = norm(title)
    ranked: list[tuple[int, int, float, Line]] = []
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
        ranked.append((score, len(actual), line.s, line))
    return max(ranked, key=lambda item: (item[0], item[1], item[2]))[3] if ranked else None


def learned_start_shape(reference_title: Any, matched_line: Line) -> str:
    wanted = norm(reference_title)
    for token, _ in tokens_with_offsets(matched_line.text):
        if norm(token) and norm(token) in wanted:
            return token_shape(token)
    tokens = tokens_with_offsets(clean(reference_title))
    return token_shape(tokens[0][0]) if tokens else ""


def learn_layout_levels(
    reference: dict[str, Any],
    chunk_lines: list[Line],
    reference_path: Path,
    chunk_path: Path,
) -> dict[str, Any]:
    pages: dict[int, list[Line]] = defaultdict(list)
    for line in chunk_lines:
        pages[line.page].append(line)

    observations: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for item in reference.get("chapters", []) or []:
        if not isinstance(item, dict):
            continue
        page = int(item.get("page") or 1)
        chapter = clean(item.get("chapter"))
        if not chapter:
            continue
        line = match_reference_title(chapter, pages[page])
        if line is None:
            unmatched.append({"level": int(item.get("level") or 1), "chapter": chapter, "page": page})
            continue
        observations.append(
            {
                "level": int(item.get("level") or 1),
                "chapter": chapter,
                "page": page,
                "source_order": line.order,
                "source_text": line.text,
                "layout": {
                    "s": line.s,
                    "r": line.r,
                    "x": line.x,
                    "y": line.y,
                    "b": line.b,
                    "i": line.i,
                    "f": line.f,
                },
                "start_shape": learned_start_shape(chapter, line),
                "level_reason": clean(item.get("level_reason")),
            }
        )

    if not observations:
        raise ValueError("기준 TOC와 청크에서 일치하는 제목을 찾지 못했습니다.")

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for obs in observations:
        layout = obs["layout"]
        key = (obs["level"], layout["s"], layout["r"], layout["b"], layout["i"], layout["f"])
        group = grouped.setdefault(
            key,
            {
                "level": obs["level"],
                "s": layout["s"],
                "r": layout["r"],
                "b": layout["b"],
                "i": layout["i"],
                "f": layout["f"],
                "xs": [],
                "ys": [],
                "start_shapes": Counter(),
                "examples": [],
                "reference_count": 0,
            },
        )
        group["xs"].append(layout["x"])
        group["ys"].append(layout["y"])
        group["start_shapes"][obs["start_shape"]] += 1
        group["examples"].append(obs["chapter"])
        group["reference_count"] += 1

    rules: list[dict[str, Any]] = []
    for group in grouped.values():
        rules.append(
            {
                "level": group["level"],
                "s": group["s"],
                "r": group["r"],
                "b": group["b"],
                "i": group["i"],
                "f": group["f"],
                "x_min": min(group["xs"]),
                "x_max": max(group["xs"]),
                "y_min": min(group["ys"]),
                "y_max": max(group["ys"]),
                "start_shapes": sorted(group["start_shapes"]),
                "reference_count": group["reference_count"],
                "examples": group["examples"][:8],
            }
        )
    rules.sort(key=lambda rule: (int(rule["level"]), -float(rule["s"]), str(rule["f"])))

    return {
        "source_reference": str(reference_path),
        "source_chunk": str(chunk_path),
        "matching_priority": ["S/R/B/I/F", "learned start shape", "X", "Y", "explicit numbering"],
        "rules": rules,
        "matched_reference_titles": observations,
        "unmatched_reference_titles": unmatched,
    }


def timestamped_output(
    explicit: Path | None,
    default_dir: Path,
    default_stem: str,
    label: str,
    timestamp: str,
    suffix: str = ".json",
) -> Path:
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


def make_debug_dir(args: argparse.Namespace, default_stem: str, timestamp: str) -> Path | None:
    if getattr(args, "no_debug", False):
        return None
    base = Path(args.debug_dir) if getattr(args, "debug_dir", None) else DEBUG_OUTPUT_DIR
    path = base / f"{default_stem}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class TeeStream:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_simple_log(args: argparse.Namespace, timestamp: str) -> Path | None:
    if not getattr(args, "log_simple", False):
        return None
    log_path = Path(args.log_simple_file) if getattr(args, "log_simple_file", None) else RAW_OUTPUT_DIR / f"toc_chunk_level_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    print(f"Simple log: {log_path}", flush=True)
    return log_path


def parse_host_port_from_base_url(base_url: str) -> tuple[str, int]:
    match = re.search(r"^https?://([^/:]+)(?::(\d+))?", clean(base_url))
    if not match:
        return "127.0.0.1", 8000
    host = match.group(1)
    port = int(match.group(2) or 8000)
    if host in {"0.0.0.0", "localhost"}:
        host = "127.0.0.1"
    return host, port


def server_models_url(base_url: str) -> str:
    url = clean(base_url).rstrip("/")
    if url.endswith("/chat/completions"):
        url = url.rsplit("/chat/completions", 1)[0]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/models"


def wait_for_openai_server(base_url: str, timeout: int) -> bool:
    deadline = time.time() + max(1, int(timeout))
    url = server_models_url(base_url)
    while time.time() < deadline:
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= int(getattr(response, "status", 200)) < 500:
                    return True
        except Exception:
            time.sleep(2)
    return False


def build_server_command(args: argparse.Namespace) -> list[str]:
    if clean(getattr(args, "server_command", "")):
        return shlex.split(args.server_command)
    host, port = parse_host_port_from_base_url(args.reference_base_url)
    serve_host = "0.0.0.0" if host == "127.0.0.1" else host
    cmd = [
        "vllm",
        "serve",
        clean(args.reference_model) or DEFAULT_REFERENCE_MODEL,
        "--host",
        serve_host,
        "--port",
        str(port),
    ]
    if clean(getattr(args, "server_extra_args", "")):
        cmd.extend(shlex.split(args.server_extra_args))
    return cmd


def start_openai_server(args: argparse.Namespace) -> tuple[subprocess.Popen[Any], Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.server_log) if getattr(args, "server_log", None) else RAW_OUTPUT_DIR / f"server_{safe_file_part(args.reference_model)}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    cmd = build_server_command(args)
    print("Server command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    print(f"Server log: {log_path}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return proc, log_path


def stop_server_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    print("Stopping server...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def run_server_only(args: argparse.Namespace) -> None:
    proc, log_path = start_openai_server(args)
    print(f"Server PID: {proc.pid}", flush=True)
    print(f"Server log: {log_path}", flush=True)
    if wait_for_openai_server(args.reference_base_url, args.server_wait_timeout):
        print(f"Server ready: {args.reference_base_url}", flush=True)
    else:
        print(f"Server did not become ready within {args.server_wait_timeout}s. Check log: {log_path}", flush=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=30)


def write_debug_file(debug_dir: Path | None, name: str, payload: Any, suffix: str = ".json") -> Path | None:
    if debug_dir is None:
        return None
    path = debug_dir / f"{name}{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")
    return path


def args_debug_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    hidden = {"reference_api_key", "gemma_vllm_server_api_key"}
    result: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key in hidden and clean(value):
            result[key] = "***"
        else:
            result[key] = jsonable(value)
    return result


def chunk_debug_payload(chunk: dict[str, Any], chunk_text: str, chunk_path: Path) -> dict[str, Any]:
    return {
        "chunk_path": str(chunk_path),
        "text_chars": len(chunk_text),
        "text_bytes_utf8": len(chunk_text.encode("utf-8")),
        "keys": sorted(str(key) for key in chunk.keys()),
        "start_page": chunk.get("start_page"),
        "end_page": chunk.get("end_page"),
        "chunk_id": chunk.get("chunk_id") or chunk.get("id") or chunk.get("label"),
    }


def compact_rules_for_prompt(learned: dict[str, Any]) -> dict[str, Any]:
    return {
        "matching_priority": learned.get("matching_priority", []),
        "rules": learned.get("rules", []),
        "matched_reference_titles": learned.get("matched_reference_titles", []),
        "unmatched_reference_titles": learned.get("unmatched_reference_titles", []),
    }


def openai_compatible_chat_url(base_url: str) -> str:
    url = clean(base_url).rstrip("/")
    if not url:
        raise ValueError("OpenAI-compatible base URL is empty.")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def build_chunk_reference_prompt(chunk_text: str, title: str, max_depth: int) -> str:
    return f"""
You are an expert at creating tables of contents for PDF textbooks and lecture materials.
Create a verified reference TOC only for this one parsed PDF chunk.

[Document Title]
{title}

[Required Rules]
- Use only real body hierarchy titles visible in this chunk.
- Exclude covers, prefaces, existing TOC/Contents listing rows, indexes, references, page numbers, repeated headers/footers, captions, questions, and body sentences.
- Use exact source title text and existing numbering. Do not invent titles or numbering.
- Use layout metadata P(page), O(source order), S(font size), R(size/body ratio), X/Y(position), B(bold), I(italic), and F(font name).
- Highest priority for body heading levels: S/R/B/I/F style signature.
- Use explicit numbering and X/Y as secondary evidence.
- Preserve PDF page order and source order on the same page.
- Use levels 1 to {max_depth}.
- Include a concise level_reason for every entry.

[Output Format]
Output exactly one compact valid JSON object. Do not output Markdown, code fences, or explanations outside JSON.

{{
  "title": "{title}",
  "chapters": [
    {{"level": 1, "chapter": "Major section title", "page": 1, "level_reason": "Concise evidence from layout/numbering."}},
    {{"level": 2, "chapter": "Subsection title", "page": 3, "level_reason": "Concise evidence from layout/numbering."}}
  ]
}}

[Parsed Chunk Layout Text]
{chunk_text}
""".strip()


def call_openai_compatible_chat(
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
    if clean(api_key):
        headers["Authorization"] = f"Bearer {clean(api_key)}"

    url = openai_compatible_chat_url(base_url)
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout))) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compatible request failed: HTTP {error.code} {error.reason}. "
            f"URL={url}. Body: {error_body[:1000]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OpenAI-compatible server is not reachable at {url}: {error}") from error

    try:
        data = json.loads(response_text)
    except Exception as error:
        raise RuntimeError(f"OpenAI-compatible server returned non-JSON response: {response_text[:1000]}") from error

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenAI-compatible server returned no choices: {response_text[:1000]}")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            return "\n".join(
                clean(part.get("text") if isinstance(part, dict) else part)
                for part in content
                if clean(part.get("text") if isinstance(part, dict) else part)
            )
        return str(content or "")
    return str(first.get("text") or "")


def generate_reference_with_gpt_oss(
    chunk_text: str,
    args: argparse.Namespace,
    title: str,
) -> tuple[dict[str, Any], str, str]:
    model = clean(args.reference_model) or DEFAULT_REFERENCE_MODEL
    prompt = build_chunk_reference_prompt(chunk_text=chunk_text, title=title, max_depth=args.max_depth)
    last_error: Exception | None = None

    for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
        parse_prompt = prompt
        if parse_attempt > 1:
            parse_prompt = (
                prompt
                + "\n\n[Important Retry Instruction]\n"
                + "The previous response failed JSON parsing. Output exactly one valid JSON object that Python json.loads can parse."
            )
            print(
                f"  Retrying chunk reference generation after JSON parse failure {parse_attempt}/{args.ai_retries}: {model}",
                flush=True,
            )
        try:
            print(f"  Chunk reference generation started: {model}", flush=True)
            raw_text = with_retry(
                label=f"Chunk reference generation({model})",
                func=lambda parse_prompt=parse_prompt: call_openai_compatible_chat(
                    model=model,
                    prompt=parse_prompt,
                    base_url=args.reference_base_url,
                    api_key=args.reference_api_key,
                    timeout=args.reference_timeout,
                    temperature=args.reference_temperature,
                    max_output_tokens=args.reference_max_output_tokens,
                ),
                max_retries=args.ai_retries,
                base_delay=args.ai_retry_base_delay,
            )
            print(f"  Chunk reference generation completed: {model}", flush=True)
            parsed = parse_json_response_text(raw_text, provider=model)
            return validate_toc(parsed, fallback_title=title, max_depth=args.max_depth), raw_text, model
        except Exception as error:
            last_error = error
            if parse_attempt >= max(1, int(args.ai_retries)):
                raise

    if last_error:
        raise last_error
    raise RuntimeError("Chunk reference generation failed.")


def build_gemma_prompt(
    parsed_text: str,
    learned: dict[str, Any],
    title: str,
    max_depth: int,
    user_prompt: str,
) -> str:
    rules_json = json.dumps(compact_rules_for_prompt(learned), ensure_ascii=False, indent=2)
    return f"""
You are an expert at creating tables of contents for PDF textbooks and lecture materials.
Use the code-generated layout rules from one verified chunk as the main level reference, then create the full-document TOC from the parsed PDF layout text.

[User Prompt]
{user_prompt.strip()}

[Document Title]
{title}

[Code-Generated Layout Rules From Verified Chunk]
{rules_json}

[How To Use The Rules]
- The rules were generated by code from a verified chunk/reference TOC pair.
- Treat S/R/B/I/F as the strongest style signal for body heading levels.
- Treat start_shapes, X, Y, and explicit numbering as secondary evidence.
- When a full-document line matches a rule's S/R/B/I/F and start shape, normally assign that rule's level.
- If numbering such as 1, 1.1, 1.1.1 or Part/Chapter clearly proves a different hierarchy, adjust the level and explain it in level_reason.
- Do not invent headings. Use exact source text from the parsed layout.
- Exclude body sentences, examples, questions, captions, references, standalone page numbers, and repeated headers/footers.
- Exclude existing TOC/Contents listing rows unless they are also verified body headings.
- Preserve PDF page order and source order on the same page.
- Use levels 1 to {max_depth}.

[Output Format]
Output exactly one compact valid JSON object. Do not output Markdown, code fences, or explanations outside JSON.

{{
  "title": "{title}",
  "chapters": [
    {{"level": 1, "chapter": "Major section title", "page": 1, "level_reason": "Matched rule S/R/B/I/F..."}},
    {{"level": 2, "chapter": "Subsection title", "page": 3, "level_reason": "Matched rule and numbering..."}}
  ]
}}

[Parsed PDF Layout Text]
{parsed_text}
""".strip()


def generate_toc_with_gemma(prompt: str, args: argparse.Namespace, fallback_title: str) -> tuple[dict[str, Any], str, str]:
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    for index, model in enumerate(models):
        model = normalize_model_for_provider("gemma", model)
        if index > 0:
            print(f"Trying Gemma fallback model: {model}", flush=True)

        parse_prompt = prompt
        for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
            if parse_attempt > 1:
                parse_prompt = (
                    prompt
                    + "\n\n[Important Retry Instruction]\n"
                    + "The previous response failed JSON parsing. Output exactly one valid JSON object that Python json.loads can parse. "
                    + "Do not omit commas. Escape double quotes inside strings. Do not output Markdown or code fences."
                )
                print(
                    f"  Retrying Gemma generation after JSON parse failure {parse_attempt}/{args.ai_retries}: {model}",
                    flush=True,
                )

            try:
                print(f"  Gemma TOC generation started: {model}", flush=True)
                response = with_retry(
                    label=f"Gemma chunk-level generation({model})",
                    func=lambda model=model, parse_prompt=parse_prompt: generate_gemma_once(
                        model=model,
                        prompt=parse_prompt,
                        args=args,
                    ),
                    max_retries=args.ai_retries,
                    base_delay=args.ai_retry_base_delay,
                )
                print(f"  Gemma TOC generation completed: {model}", flush=True)

                raw_text = str(response.get("response") if isinstance(response, dict) else response)
                parsed = parse_json_response_text(raw_text, provider="gemma")
                toc = validate_toc(parsed, fallback_title=fallback_title, max_depth=args.max_depth)
                return toc, raw_text, model
            except Exception as error:
                last_error = error
                if parse_attempt >= max(1, int(args.ai_retries)):
                    print(f"Gemma model failed: {model} / {error}", flush=True)
                    break

    if last_error:
        raise last_error
    raise RuntimeError("No Gemma models to try.")


def prompt_file_text(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return (
        "Create a full table of contents from real body hierarchy titles. "
        "Use the chunk-derived layout rules as the primary level reference."
    )


def maybe_limit_parsed_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    raise ValueError(
        f"parsed text is {len(text):,} characters, larger than --parsed-max-chars={max_chars:,}. "
        "Increase --parsed-max-chars or pass 0 to disable this guard."
    )



def safe_file_part(value: Any) -> str:
    text = clean(value)
    text = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", text)
    return text.strip("._-") or "input"


def discover_pdf_files(input_dir: Path, explicit_pdfs: list[Path] | None = None) -> list[Path]:
    if explicit_pdfs:
        pdfs = [Path(path) for path in explicit_pdfs]
    else:
        root = Path(input_dir)
        pdfs = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"]
    pdfs = sorted(path.resolve() for path in pdfs)
    if not pdfs:
        raise FileNotFoundError(f"PDF files were not found under: {input_dir}")
    return pdfs


def prepare_pdf_inputs(pdf_path: Path, args: argparse.Namespace, timestamp: str) -> tuple[dict[str, Any], str, Path, str, Path, str]:
    pages, metadata = extract_pdf_layout_pages(pdf_path)
    parsed_text = parsed_pdf_text_for_file(
        pdf_path=pdf_path,
        pages=pages,
        extraction_metadata=metadata,
        source_pdf_path=pdf_path,
    )
    chunks = chunk_pdf_text_pages(pages, max_chars=args.chunk_chars)
    if not chunks:
        raise RuntimeError(f"No chunks were created from PDF: {pdf_path}")

    chunk_index = max(1, int(args.chunk_index))
    if chunk_index > len(chunks):
        raise ValueError(f"--chunk-index {chunk_index} is out of range for {pdf_path.name}; total chunks={len(chunks)}")
    chunk = chunks[chunk_index - 1]

    default_stem = safe_file_part(pdf_path.stem)
    generated_base = Path(args.generated_input_output) if args.generated_input_output else GENERATED_INPUT_DIR
    generated_dir = generated_base / f"{default_stem}_{timestamp}"
    generated_dir.mkdir(parents=True, exist_ok=True)

    parsed_path = generated_dir / f"{default_stem}_parsed_pdf.txt"
    parsed_path.write_text(parsed_text, encoding="utf-8")

    chunk_path = generated_dir / f"chunk_{chunk.get('index', chunk_index)}_pages_{chunk.get('start_page')}-{chunk.get('end_page')}.json"
    chunk_payload = {
        "timestamp": timestamp,
        "stage": "toc_chunk_level_pdf_input_prepared",
        "source_pdf": str(pdf_path),
        "chunk": chunk.get("index"),
        "total_chunks": len(chunks),
        "start_page": chunk.get("start_page"),
        "end_page": chunk.get("end_page"),
        "input_chars": len(str(chunk.get("text") or "")),
        "text": str(chunk.get("text") or ""),
    }
    chunk_path.write_text(json.dumps(chunk_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return chunk_payload, str(chunk.get("text") or ""), chunk_path, parsed_text, parsed_path, default_stem


def load_explicit_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], str, Path, str, Path, str]:
    if not args.chunk or not args.parsed:
        raise ValueError("Use both --chunk and --parsed for explicit chunk mode, or omit both to process PDFs from --input-dir.")
    chunk_path = Path(args.chunk)
    parsed_path = Path(args.parsed)
    chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
    chunk_text = str(chunk.get("text") or "")
    if not chunk_text:
        raise ValueError(f"chunk text is empty: {chunk_path}")
    parsed_text = parsed_path.read_text(encoding="utf-8")
    default_stem = parsed_path.stem
    return chunk, chunk_text, chunk_path, parsed_text, parsed_path, default_stem


def process_prepared_input(
    args: argparse.Namespace,
    *,
    chunk: dict[str, Any],
    chunk_text: str,
    chunk_path: Path,
    parsed_text: str,
    parsed_path: Path,
    default_stem: str,
    source_pdf: Path | None = None,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = clean(args.title) or (source_pdf.stem if source_pdf else default_stem)
    debug_dir = make_debug_dir(args, default_stem, timestamp)
    write_debug_file(debug_dir, "001_run_config", {"timestamp": timestamp, "source_pdf": str(source_pdf) if source_pdf else None, "args": args_debug_snapshot(args)})
    write_debug_file(debug_dir, "002_chunk_loaded", chunk_debug_payload(chunk, chunk_text, chunk_path))
    write_debug_file(debug_dir, "003_chunk_text", chunk_text, suffix=".txt")
    write_debug_file(debug_dir, "003b_parsed_text", parsed_text, suffix=".txt")

    if args.test:
        write_debug_file(debug_dir, "004_reference_prompt", build_chunk_reference_prompt(chunk_text, title, args.max_depth), suffix=".txt")
        reference, reference_raw_text, reference_model = generate_reference_with_gpt_oss(
            chunk_text=chunk_text,
            args=args,
            title=title,
        )
        reference_path = timestamped_output(
            args.reference_output,
            REFERENCE_OUTPUT_DIR,
            default_stem,
            f"reference_{reference_model}",
            timestamp,
        )
        reference_path.write_text(json.dumps(reference, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        reference_raw_output = timestamped_output(
            args.reference_raw_output,
            RAW_OUTPUT_DIR,
            default_stem,
            f"reference_{reference_model}_raw",
            timestamp,
            suffix=".txt",
        )
        reference_raw_output.write_text(reference_raw_text, encoding="utf-8")
        write_debug_file(debug_dir, "005_reference_toc", reference)
        write_debug_file(debug_dir, "006_reference_raw", reference_raw_text, suffix=".txt")
        print(f"Reference TOC: {reference_path}")
        print(f"Reference raw response: {reference_raw_output}")
        print(f"Reference model: {reference_model}")
        if debug_dir is not None:
            print(f"Debug dir: {debug_dir}")
        print(f"Reference TOC entries: {len(reference.get('chapters', []))}")
        return

    if args.reference:
        reference_path = Path(args.reference)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        title = clean(args.title) or clean(reference.get("title")) or title
        write_debug_file(debug_dir, "004_reference_loaded", {"reference_path": str(reference_path), "reference": reference})
    else:
        write_debug_file(debug_dir, "004_reference_prompt", build_chunk_reference_prompt(chunk_text, title, args.max_depth), suffix=".txt")
        reference, reference_raw_text, reference_model = generate_reference_with_gpt_oss(
            chunk_text=chunk_text,
            args=args,
            title=title,
        )
        reference_path = timestamped_output(
            args.reference_output,
            REFERENCE_OUTPUT_DIR,
            default_stem,
            f"reference_{reference_model}",
            timestamp,
        )
        reference_path.write_text(json.dumps(reference, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        reference_raw_output = timestamped_output(
            args.reference_raw_output,
            RAW_OUTPUT_DIR,
            default_stem,
            f"reference_{reference_model}_raw",
            timestamp,
            suffix=".txt",
        )
        reference_raw_output.write_text(reference_raw_text, encoding="utf-8")
        write_debug_file(debug_dir, "005_reference_toc", reference)
        write_debug_file(debug_dir, "006_reference_raw", reference_raw_text, suffix=".txt")
        print(f"Reference TOC: {reference_path}")
        print(f"Reference raw response: {reference_raw_output}")

    learned = learn_layout_levels(reference, parse_layout(chunk_text), reference_path, chunk_path)
    layout_output = timestamped_output(
        args.layout_level_output,
        LAYOUT_OUTPUT_DIR,
        default_stem,
        "layout_levels",
        timestamp,
    )
    layout_output.write_text(json.dumps(learned, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    write_debug_file(debug_dir, "007_layout_levels", learned)
    write_debug_file(debug_dir, "008_layout_levels_path", {"layout_output": str(layout_output), "parsed_path": str(parsed_path), "chunk_path": str(chunk_path)})

    if args.dry_run_layout:
        print(f"Layout levels: {layout_output}")
        print("Rules:", len(learned["rules"]))
        print("Matched reference titles:", len(learned["matched_reference_titles"]))
        print("Unmatched reference titles:", len(learned["unmatched_reference_titles"]))
        if debug_dir is not None:
            print(f"Debug dir: {debug_dir}")
        return

    parsed_text = maybe_limit_parsed_text(parsed_text, args.parsed_max_chars)
    write_debug_file(debug_dir, "009_parsed_input", {"parsed_path": str(parsed_path), "parsed_chars": len(parsed_text), "parsed_bytes_utf8": len(parsed_text.encode("utf-8"))})
    prompt = build_gemma_prompt(
        parsed_text=parsed_text,
        learned=learned,
        title=title,
        max_depth=max(1, int(args.max_depth)),
        user_prompt=prompt_file_text(args),
    )
    write_debug_file(debug_dir, "010_gemma_prompt", prompt, suffix=".txt")
    toc, raw_text, used_model = generate_toc_with_gemma(prompt, args, fallback_title=title)

    output = timestamped_output(args.output, TOC_OUTPUT_DIR, default_stem, "toc", timestamp)
    output.write_text(json.dumps(toc, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    raw_output = timestamped_output(args.raw_output, RAW_OUTPUT_DIR, default_stem, "gemma_raw", timestamp, suffix=".txt")
    raw_output.write_text(raw_text, encoding="utf-8")
    write_debug_file(debug_dir, "011_final_toc", toc)
    write_debug_file(debug_dir, "012_gemma_raw", raw_text, suffix=".txt")
    write_debug_file(debug_dir, "013_outputs", {"layout_output": str(layout_output), "toc_output": str(output), "raw_output": str(raw_output), "model": used_model})

    print(f"Layout levels: {layout_output}")
    print(f"Unmatched reference titles: {len(learned['unmatched_reference_titles'])}")
    print(f"Raw response: {raw_output}")
    print(f"Created: {output}")
    print(f"Model: {used_model}")
    print(f"TOC entries: {len(toc.get('chapters', []))}")
    if debug_dir is not None:
        print(f"Debug dir: {debug_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="청크 기준 TOC에서 코드가 layout rule을 만들고, Gemma가 그 rule로 전체 TOC 레벨을 생성합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="기본 PDF 입력 폴더. 기본은 data/input")
    parser.add_argument("--pdf", type=Path, action="append", help="처리할 PDF 파일. 여러 번 지정 가능; 생략하면 --input-dir 아래 PDF 전체")
    parser.add_argument("--chunk", type=Path, help="이미 생성된 input chunk JSON. --parsed와 같이 쓰면 PDF 자동 입력 대신 이 파일 사용")
    parser.add_argument("--parsed", type=Path, help="이미 생성된 전체 parsed PDF layout text. --chunk와 같이 사용")
    parser.add_argument("--chunk-index", type=int, default=1, help="PDF 자동 입력에서 기준으로 사용할 청크 번호")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_GEMMA_TEXT_CHUNK_CHARS, help="PDF 자동 입력에서 청크 하나의 최대 문자 수")
    parser.add_argument("--generated-input-output", type=Path, help="PDF에서 생성한 parsed/chunk 입력 파일 저장 폴더")
    parser.add_argument("--reference", type=Path, help="이미 생성된 chunk 기준 TOC JSON. 생략하면 gpt-oss-120b가 생성")
    parser.add_argument("--reference-output", type=Path, help="gpt-oss-120b 기준 TOC 저장 파일명 또는 저장 폴더")
    parser.add_argument("--reference-raw-output", type=Path, help="gpt-oss-120b raw response 저장 파일명 또는 저장 폴더")
    parser.add_argument("--reference-model", default=DEFAULT_REFERENCE_MODEL, help="chunk 기준 TOC 생성 모델")
    parser.add_argument("--reference-base-url", default=DEFAULT_REFERENCE_BASE_URL, help="chunk 기준 TOC 생성용 OpenAI-compatible base URL")
    parser.add_argument("--reference-api-key", default=DEFAULT_REFERENCE_API_KEY, help="chunk 기준 TOC 생성용 bearer API key")
    parser.add_argument("--reference-timeout", type=int, default=DEFAULT_REFERENCE_TIMEOUT)
    parser.add_argument("--reference-temperature", type=float, default=0)
    parser.add_argument("--reference-max-output-tokens", type=int, default=8192)
    parser.add_argument("--layout-level-output", type=Path, help="layout rule 결과 파일명 또는 저장 폴더")
    parser.add_argument("--output", type=Path, help="최종 TOC 결과 파일명 또는 저장 폴더")
    parser.add_argument("--raw-output", type=Path, help="Gemma raw response 파일명 또는 저장 폴더")
    parser.add_argument("--title", help="최종 TOC title. 없으면 PDF/reference/parsed stem 사용")
    parser.add_argument("--prompt", help="Gemma에 추가로 전달할 사용자 지시문")
    parser.add_argument("--prompt-file", type=Path, help="Gemma 사용자 지시문 파일")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--parsed-max-chars", type=int, default=0, help="0이면 제한 없음")
    parser.add_argument("--dry-run-layout", action="store_true", help="layout rule만 생성/저장하고 Gemma 호출은 하지 않음")
    parser.add_argument("--test", action="store_true", help="gpt-oss-120b로 chunk 기준 TOC만 생성/저장하고 종료")
    parser.add_argument("--debug-dir", type=Path, help="단계별 디버그 파일 저장 폴더. 기본은 data/output/toc_txt/debug/{run_id}")
    parser.add_argument("--no-debug", action="store_true", help="단계별 디버그 파일 저장을 끔")
    parser.add_argument("--server", action="store_true", help="OpenAI-compatible vLLM 서버만 띄우고 대기")
    parser.add_argument("--server_ver", action="store_true", help="이미 떠 있는 OpenAI-compatible 서버 준비 확인 후 나머지 작업 실행")
    parser.add_argument("--server-command", help="서버 실행 명령 직접 지정. 예: vllm serve gpt-oss-120b --host 0.0.0.0 --port 8000")
    parser.add_argument("--server-extra-args", default="", help="기본 vllm serve 명령 뒤에 붙일 추가 인자")
    parser.add_argument("--server-log", type=Path, help="서버 stdout/stderr 로그 파일")
    parser.add_argument("--server-wait-timeout", type=int, default=600, help="--server_ver에서 기존 서버 준비 확인 대기 시간(초)")
    parser.add_argument("--log_simple", action="store_true", help="콘솔 로그를 별도 simple log 파일에도 저장")
    parser.add_argument("--log-simple-file", type=Path, help="--log_simple 로그 파일 경로")

    parser.add_argument("--model", default=DEFAULT_GEMMA_HF_MODEL)
    parser.add_argument("--ai-fallback-models", default=DEFAULT_FALLBACK_MODELS_BY_PROVIDER["gemma"])
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER["gemma"])
    parser.add_argument("--gemma-runtime", default=DEFAULT_GEMMA_RUNTIME, choices=("transformers", "vllm", "vllm-server"))
    parser.add_argument("--gemma-device-map", default=DEFAULT_GEMMA_DEVICE_MAP)
    parser.add_argument("--gemma-torch-dtype", default=DEFAULT_GEMMA_TORCH_DTYPE)
    parser.add_argument("--gemma-vllm-tensor-parallel-size", type=int, default=DEFAULT_GEMMA_VLLM_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gemma-vllm-gpu-memory-utilization", type=float, default=DEFAULT_GEMMA_VLLM_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--gemma-vllm-max-model-len", default=DEFAULT_GEMMA_VLLM_MAX_MODEL_LEN)
    parser.add_argument("--gemma-vllm-trust-remote-code", action="store_true", default=DEFAULT_GEMMA_VLLM_TRUST_REMOTE_CODE)
    parser.add_argument("--gemma-vllm-server-base-url", default=DEFAULT_GEMMA_VLLM_SERVER_BASE_URL)
    parser.add_argument("--gemma-vllm-server-api-key", default=DEFAULT_GEMMA_VLLM_SERVER_API_KEY)
    parser.add_argument("--gemma-vllm-server-timeout", type=int, default=DEFAULT_GEMMA_VLLM_SERVER_TIMEOUT)
    args = parser.parse_args()
    main_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_simple_log(args, main_timestamp)

    server_proc = None
    if args.server:
        run_server_only(args)
        return
    if args.server_ver:
        print(f"Checking existing server: {args.reference_base_url}", flush=True)
        if not wait_for_openai_server(args.reference_base_url, args.server_wait_timeout):
            raise RuntimeError(
                f"Existing server is not ready within {args.server_wait_timeout}s: {args.reference_base_url}. "
                "Run --server first in another terminal, or check --reference-base-url."
            )
        print(f"Server ready: {args.reference_base_url}", flush=True)

    if args.chunk or args.parsed:
        chunk, chunk_text, chunk_path, parsed_text, parsed_path, default_stem = load_explicit_inputs(args)
        process_prepared_input(
            args,
            chunk=chunk,
            chunk_text=chunk_text,
            chunk_path=chunk_path,
            parsed_text=parsed_text,
            parsed_path=parsed_path,
            default_stem=default_stem,
        )
        stop_server_process(server_proc)
        return

    pdfs = discover_pdf_files(args.input_dir, args.pdf)
    print(f"PDF inputs: {len(pdfs)}")
    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"\n[{index}/{len(pdfs)}] PDF: {pdf_path}", flush=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk, chunk_text, chunk_path, parsed_text, parsed_path, default_stem = prepare_pdf_inputs(pdf_path, args, timestamp)
        print(f"Prepared parsed input: {parsed_path}")
        print(f"Prepared chunk input: {chunk_path}")
        process_prepared_input(
            args,
            chunk=chunk,
            chunk_text=chunk_text,
            chunk_path=chunk_path,
            parsed_text=parsed_text,
            parsed_path=parsed_path,
            default_stem=default_stem,
            source_pdf=pdf_path,
        )

    stop_server_process(server_proc)


if __name__ == "__main__":
    main()
