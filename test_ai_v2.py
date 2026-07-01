from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

SUPPORTED_PROVIDERS = ("gemini", "openai", "claude")
ALL_PROVIDERS_VALUE = "all"
DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", ALL_PROVIDERS_VALUE).strip().lower() or ALL_PROVIDERS_VALUE

DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": os.getenv("GEMINI_MODEL", os.getenv("AI_MODEL", "gemini-3.1-pro-preview")),
    "openai": os.getenv("OPENAI_MODEL", os.getenv("AI_MODEL", "gpt-5.5")),
    "claude": os.getenv("CLAUDE_MODEL", os.getenv("AI_MODEL", "claude-opus-4-8")),
}

DEFAULT_FALLBACK_MODELS_BY_PROVIDER = {
    "gemini": os.getenv("GEMINI_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["gemini"])),
    "openai": os.getenv("OPENAI_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["openai"])),
    "claude": os.getenv("CLAUDE_FALLBACK_MODELS", os.getenv("AI_FALLBACK_MODELS", DEFAULT_MODEL_BY_PROVIDER["claude"])),
}

DEFAULT_AI_RETRIES = int(os.getenv("AI_MAX_RETRIES", os.getenv("GEMINI_MAX_RETRIES", "5")))
DEFAULT_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", os.getenv("GEMINI_RETRY_BASE_DELAY", "2.0")))
DEFAULT_FILE_PROCESSING_TIMEOUT = int(os.getenv("GEMINI_FILE_PROCESSING_TIMEOUT", "300"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("AI_MAX_OUTPUT_TOKENS", "8192"))
DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER = {
    "gemini": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "32768"))),
    "openai": int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "8192"))),
    "claude": int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", os.getenv("AI_MAX_OUTPUT_TOKENS", "8192"))),
}
DEFAULT_GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "256"))

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
PDF 전체를 읽고 학습용 목차를 작성해줘.

목차 작성 기준:
- 표지, 머리말, 차례/목차 페이지, 찾아보기/색인, 참고문헌, 쪽번호, 반복 머리말/꼬리말은 제외한다.
- 본문에 실제로 존재하는 장/절/소제목만 사용한다.
- 제목을 새로 지어내지 않는다.
- 원문에 번호가 있으면 번호를 그대로 보존한다. 원문에 번호가 없으면 임의로 번호를 붙이지 않는다.
- page는 PDF 뷰어 기준 실제 페이지 순서로 계산한다. 첫 페이지는 1이다.
- level은 큰 단원일수록 1에 가깝게, 하위 단원일수록 숫자를 크게 쓴다.
- 가능한 한 누락 없이 작성하되, 본문 문장이나 문제 문항은 목차에 넣지 않는다.
""".strip()

RETRYABLE_ERROR_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "RATE LIMIT",
    "500",
    "INTERNAL",
    "502",
    "503",
    "UNAVAILABLE",
    "HIGH DEMAND",
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
    "PIP INSTALL",
    "NO API KEY",
    "MISSING API KEY",
)

SCHEMA_CONFIG_ERROR_MARKERS = (
    "RESPONSE_SCHEMA",
    "RESPONSE_JSON_SCHEMA",
    "THINKING",
    "BUDGET",
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


def normalize_provider(provider: str | None) -> str:
    provider = clean_text(provider).lower()
    aliases = {
        "anthropic": "claude",
        "open-ai": "openai",
        "google": "gemini",
    }
    provider = aliases.get(provider, provider)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"지원하지 않는 provider입니다: {provider} / 가능: {', '.join(SUPPORTED_PROVIDERS)}")
    return provider


def resolve_providers(provider_option: str | None) -> list[str]:
    provider_text = clean_text(provider_option).lower() or DEFAULT_PROVIDER
    aliases = {
        "*": ALL_PROVIDERS_VALUE,
        "multi": ALL_PROVIDERS_VALUE,
        "all": ALL_PROVIDERS_VALUE,
        "anthropic": "claude",
        "open-ai": "openai",
        "google": "gemini",
    }
    provider_text = aliases.get(provider_text, provider_text)

    if provider_text == ALL_PROVIDERS_VALUE:
        return list(SUPPORTED_PROVIDERS)

    providers: list[str] = []
    for item in provider_text.split(","):
        provider = normalize_provider(item)
        if provider not in providers:
            providers.append(provider)

    if not providers:
        raise ValueError("실행할 provider가 없습니다.")

    return providers


def default_model_for(provider: str) -> str:
    return DEFAULT_MODEL_BY_PROVIDER[provider]


def default_fallback_models_for(provider: str) -> str:
    return DEFAULT_FALLBACK_MODELS_BY_PROVIDER[provider]


def default_max_output_tokens_for(provider: str) -> int:
    return DEFAULT_MAX_OUTPUT_TOKENS_BY_PROVIDER[provider]


def normalize_model_for_provider(provider: str, model: str) -> str:
    """사용자가 축약형으로 넣은 모델명을 provider별 실제 API 모델명에 가깝게 보정합니다."""
    model = clean_text(model)

    if provider == "gemini" and model:
        if model.startswith("models/"):
            model = model.split("/", 1)[1]
        if not model.startswith("gemini-"):
            known_suffixes = (
                "3.1-pro-preview",
                "3.1-pro",
                "2.5-pro",
                "2.5-flash",
                "2.0-flash",
                "1.5-pro",
                "1.5-flash",
            )
            if model in known_suffixes:
                model = f"gemini-{model}"

    return model


def normalize_model_list_for_provider(provider: str, models_text: str | Iterable[str] | None) -> str:
    models = parse_model_list("", models_text)
    return ",".join(normalize_model_for_provider(provider, model) for model in models)


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()

    if args.prompt:
        return args.prompt.strip()

    return DEFAULT_USER_PROMPT


def build_prompt(user_prompt: str, max_depth: int) -> str:
    return f"""
너는 PDF 교재/강의자료의 목차를 만드는 전문가다.
첨부된 PDF 파일만 근거로 목차를 작성한다.

[사용자 프롬프트]
{user_prompt.strip()}

[출력 형식]
반드시 아래 JSON 객체 하나만 출력한다. 설명, 마크다운, 코드블록은 출력하지 않는다.

{{
  "title": "문서 제목",
  "chapters": [
    {{"level": 1, "chapter": "큰 단원 제목", "page": 1}},
    {{"level": 2, "chapter": "하위 단원 제목", "page": 3}}
  ]
}}

[필수 규칙]
- title은 PDF에서 확인되는 문서 제목으로 작성한다. 불명확하면 파일명에 가까운 제목을 사용한다.
- chapters는 PDF에 등장하는 순서대로 정렬한다.
- level은 1부터 {max_depth}까지만 사용한다.
- page는 정수이며, PDF 뷰어 기준 첫 페이지를 1로 세는 실제 페이지 순서다.
- 원문 제목의 번호는 보존하되, 없는 번호를 새로 만들지 않는다.
- 목차에 넣기 애매한 본문 문장, 예제, 문제, 표/그림 캡션, 참고문헌 항목은 제외한다.
- JSON 외 텍스트는 절대 출력하지 않는다.
""".strip()


def get_api_key(provider: str) -> str:
    env_name_by_provider = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_name = env_name_by_provider[provider]
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(f"{env_name} 환경 변수가 필요합니다. .env 파일에 {env_name}=... 형식으로 넣어주세요.")
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


def with_retry(
    label: str,
    func: Callable[[], Any],
    max_retries: int,
    base_delay: float,
) -> Any:
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
                f"{label} 일시 오류, 재시도 {attempt}/{max_retries - 1} "
                f"({delay:.1f}초 후): {error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    if last_error:
        raise last_error

    raise RuntimeError(f"{label} 실패")


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


def tokenize_json_like(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    index = 0

    while index < len(text):
        char = text[index]

        if char.isspace():
            index += 1
            continue

        if char in "{}[]:,":
            tokens.append((char, char))
            index += 1
            continue

        if char == '"':
            start = index
            index += 1
            escape = False
            while index < len(text):
                current = text[index]
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    index += 1
                    break
                index += 1
            tokens.append(("string", text[start:index]))
            continue

        start = index
        while index < len(text) and not text[index].isspace() and text[index] not in '{}[]:,"':
            index += 1
        tokens.append(("atom", text[start:index]))

    return tokens


def repair_json_text(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text.strip())
    tokens = tokenize_json_like(text)
    output: list[str] = []
    stack: list[dict[str, str]] = []

    def mark_value_complete() -> None:
        if stack:
            stack[-1]["state"] = "comma_or_end"

    def append_value(token_kind: str, token_value: str) -> None:
        output.append(token_value)
        if token_kind == "{":
            stack.append({"kind": "object", "state": "key_or_end"})
        elif token_kind == "[":
            stack.append({"kind": "array", "state": "value_or_end"})
        else:
            mark_value_complete()

    index = 0
    while index < len(tokens):
        token_kind, token_value = tokens[index]

        if not stack:
            append_value(token_kind, token_value)
            index += 1
            continue

        container = stack[-1]
        state = container["state"]
        kind = container["kind"]

        if state == "comma_or_end":
            closing = "}" if kind == "object" else "]"
            next_state = "key_or_end" if kind == "object" else "value_or_end"

            if token_kind == ",":
                output.append(token_value)
                container["state"] = next_state
                index += 1
                continue

            if token_kind == closing:
                output.append(token_value)
                stack.pop()
                mark_value_complete()
                index += 1
                continue

            output.append(",")
            container["state"] = next_state
            continue

        if kind == "object" and state == "key_or_end":
            if token_kind == "}":
                output.append(token_value)
                stack.pop()
                mark_value_complete()
                index += 1
                continue

            if token_kind == ",":
                index += 1
                continue

            if token_kind == "atom":
                output.append(json.dumps(token_value, ensure_ascii=False))
            else:
                output.append(token_value)
            container["state"] = "colon"
            index += 1
            continue

        if kind == "object" and state == "colon":
            if token_kind == ":":
                output.append(token_value)
                container["state"] = "value"
                index += 1
                continue

            output.append(":")
            container["state"] = "value"
            continue

        if kind == "object" and state == "value":
            if token_kind == ",":
                index += 1
                continue

            append_value(token_kind, token_value)
            index += 1
            continue

        if kind == "array" and state == "value_or_end":
            if token_kind == "]":
                output.append(token_value)
                stack.pop()
                mark_value_complete()
                index += 1
                continue

            if token_kind == ",":
                index += 1
                continue

            append_value(token_kind, token_value)
            index += 1
            continue

        output.append(token_value)
        index += 1

    return "".join(output)


def parse_json_response_text(text: str, provider: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    if not text:
        raise ValueError(f"{provider} 응답이 비어 있습니다.")

    candidates: list[str] = [text]
    extracted = extract_json_object_text(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in list(candidates):
        repaired = repair_json_text(candidate)
        if repaired and repaired not in candidates:
            candidates.append(repaired)

    last_error: Exception | None = None
    data: Any = None

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as error:
            last_error = error
    else:
        if last_error:
            raise last_error
        raise ValueError(f"{provider} 응답에서 JSON 객체를 찾지 못했습니다.")

    if not isinstance(data, dict):
        raise ValueError(f"{provider} 응답 JSON의 최상위 값은 object여야 합니다.")

    return data


def parsed_response_to_dict(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, str):
        return parse_json_response_text(parsed, provider="parsed")

    for method_name in ("model_dump", "dict"):
        method = getattr(parsed, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return value

    return None


def extract_gemini_text(response: Any) -> str:
    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))

    if parts:
        return "\n".join(parts)

    return str(getattr(response, "text", "") or "")


def parse_json_response(response: Any, provider: str) -> dict[str, Any]:
    if provider == "gemini":
        return parse_json_response_text(extract_gemini_text(response), provider)

    parsed = getattr(response, "parsed", None)
    parsed_dict = parsed_response_to_dict(parsed)
    if parsed_dict is not None:
        return parsed_dict

    if provider == "openai":
        text = str(getattr(response, "output_text", "") or "")
        if text:
            return parse_json_response_text(text, provider)

    if provider == "claude":
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "") or ""))
        return parse_json_response_text("\n".join(parts), provider)

    text = extract_gemini_text(response)
    return parse_json_response_text(text, provider)


def validate_toc(data: dict[str, Any], fallback_title: str, max_depth: int) -> dict[str, Any]:
    title = clean_text(data.get("title")) or fallback_title
    chapters: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    raw_chapters = data.get("chapters", [])
    if not isinstance(raw_chapters, list):
        raw_chapters = []

    for item in raw_chapters:
        if not isinstance(item, dict):
            continue

        chapter = clean_text(item.get("chapter"))
        if not chapter:
            continue

        try:
            level = int(item.get("level", 1))
        except Exception:
            level = 1

        try:
            page = int(item.get("page", 1))
        except Exception:
            page = 1

        level = max(1, min(level, max_depth))
        page = max(1, page)
        key = (re.sub(r"\s+", "", chapter).lower(), page)

        if key in seen:
            continue

        seen.add(key)
        chapters.append({
            "level": level,
            "chapter": chapter,
            "page": page,
        })

    return {
        "title": title,
        "chapters": chapters,
    }


# ----------------------------- Gemini -----------------------------

def import_gemini():
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("Gemini API를 쓰려면 `pip install google-genai`가 필요합니다.") from error

    return genai, types


def create_gemini_client(api_key: str):
    genai, types = import_gemini()
    return genai.Client(api_key=api_key), types


def upload_pdf_to_gemini(client, pdf_path: Path, retries: int, base_delay: float):
    safe_name = make_ascii_upload_name(pdf_path)

    def do_upload():
        with tempfile.TemporaryDirectory(prefix="gemini_pdf_upload_") as temp_dir:
            upload_path = Path(temp_dir) / safe_name
            shutil.copy2(pdf_path, upload_path)

            try:
                return client.files.upload(
                    file=str(upload_path),
                    config={
                        "mime_type": "application/pdf",
                        "display_name": safe_name,
                    },
                )
            except TypeError:
                return client.files.upload(file=str(upload_path))

    print("  Gemini PDF 업로드 시작...", flush=True)
    uploaded = with_retry(
        label=f"Gemini PDF 업로드({pdf_path.name})",
        func=do_upload,
        max_retries=retries,
        base_delay=base_delay,
    )
    print("  Gemini PDF 업로드 완료", flush=True)
    return uploaded


def file_state_name(uploaded_file: Any) -> str:
    state = getattr(uploaded_file, "state", None)
    if state is None:
        return ""

    name = getattr(state, "name", None)
    if name:
        return str(name).upper()

    text = str(state).upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip()


def wait_for_gemini_file_ready(client, uploaded_file: Any, timeout_seconds: int) -> Any:
    started = time.time()
    current = uploaded_file

    while True:
        state = file_state_name(current)

        if not state or state == "ACTIVE":
            return current

        if "FAILED" in state:
            raise RuntimeError(f"Gemini 파일 처리 실패: {state}")

        if time.time() - started > timeout_seconds:
            raise TimeoutError(
                f"Gemini 파일 처리가 {timeout_seconds}초 안에 끝나지 않았습니다. 마지막 상태: {state}"
            )

        print(f"  Gemini 파일 처리 중: {state}", file=sys.stderr, flush=True)
        time.sleep(5)
        current = client.files.get(name=current.name)


def build_gemini_generation_configs(
    types,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
    thinking_budget: int | None,
) -> list[Any]:
    base: dict[str, Any] = {
        "response_mime_type": "application/json",
        "temperature": float(temperature),
    }

    if max_output_tokens:
        base["max_output_tokens"] = int(max_output_tokens)

    configs: list[Any] = []
    base_variants: list[dict[str, Any]] = []

    if thinking_budget is not None:
        thinking_base = dict(base)
        try:
            thinking_base["thinking_config"] = types.ThinkingConfig(
                thinking_budget=int(thinking_budget),
                include_thoughts=False,
            )
        except Exception:
            thinking_base["thinking_config"] = {
                "thinking_budget": int(thinking_budget),
                "include_thoughts": False,
            }
        base_variants.append(thinking_base)

    base_variants.append(base)

    for base_config in base_variants:
        if use_schema:
            try:
                kwargs = dict(base_config)
                kwargs["response_schema"] = TOC_SCHEMA
                configs.append(types.GenerateContentConfig(**kwargs))
            except Exception:
                pass

            schema_config = dict(base_config)
            schema_config["response_schema"] = TOC_SCHEMA
            configs.append(schema_config)

            json_schema_config = dict(base_config)
            json_schema_config["response_json_schema"] = TOC_SCHEMA
            configs.append(json_schema_config)

        configs.append(base_config)
    return configs


def generate_gemini_once(
    client,
    model: str,
    uploaded_file: Any,
    prompt: str,
    types,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
    thinking_budget: int | None,
):
    configs = build_gemini_generation_configs(
        types=types,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        use_schema=use_schema,
        thinking_budget=thinking_budget,
    )

    last_error: Exception | None = None

    for index, config in enumerate(configs):
        try:
            return client.models.generate_content(
                model=model,
                contents=[prompt, uploaded_file],
                config=config,
            )
        except Exception as error:
            last_error = error

            if index < len(configs) - 1 and is_schema_config_error(error):
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("Gemini 응답 생성 실패")


def generate_toc_from_pdf_gemini(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    client, types = create_gemini_client(api_key=get_api_key("gemini"))
    uploaded_file = upload_pdf_to_gemini(
        client=client,
        pdf_path=pdf_path,
        retries=args.ai_retries,
        base_delay=args.ai_retry_base_delay,
    )

    uploaded_file = wait_for_gemini_file_ready(
        client=client,
        uploaded_file=uploaded_file,
        timeout_seconds=args.file_processing_timeout,
    )

    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    try:
        for index, model in enumerate(models):
            model = normalize_model_for_provider("gemini", model)

            if index > 0:
                print(f"Gemini fallback 모델 시도: {model}", file=sys.stderr, flush=True)

            parse_retry_prompt = prompt

            for parse_attempt in range(1, max(1, int(args.ai_retries)) + 1):
                try:
                    if parse_attempt > 1:
                        print(
                            f"  Gemini JSON 파싱 실패 후 재생성 시도 {parse_attempt}/{args.ai_retries}: {model}",
                            file=sys.stderr,
                            flush=True,
                        )
                        parse_retry_prompt = (
                            prompt
                            + "\n\n[중요 재시도 지시]\n"
                            + "이전 응답은 JSON 문법 오류로 파싱에 실패했습니다. "
                            + "이번 응답은 반드시 Python json.loads()로 바로 파싱 가능한 유효한 JSON 객체 하나만 출력하세요. "
                            + "쉼표를 절대 누락하지 말고, 마크다운/설명/코드블록을 출력하지 마세요."
                        )

                    print(f"  Gemini 목차 생성 요청 시작: {model}", flush=True)
                    response = with_retry(
                        label=f"Gemini 생성({model})",
                        func=lambda model=model, parse_retry_prompt=parse_retry_prompt: generate_gemini_once(
                            client=client,
                            model=model,
                            uploaded_file=uploaded_file,
                            prompt=parse_retry_prompt,
                            types=types,
                            temperature=args.temperature,
                            max_output_tokens=args.max_output_tokens,
                            use_schema=not args.no_schema,
                            thinking_budget=args.gemini_thinking_budget,
                        ),
                        max_retries=args.ai_retries,
                        base_delay=args.ai_retry_base_delay,
                    )
                    print(f"  Gemini 목차 생성 완료: {model}", flush=True)

                    raw_text = extract_gemini_text(response)
                    try:
                        parsed = parse_json_response_text(raw_text, provider="gemini")
                    except Exception as parse_error:
                        last_error = parse_error
                        if args.write_raw:
                            model_name_for_file = safe_filename_part(model)
                            raw_file = (
                                Path(args.output_dir)
                                / f"{pdf_path.stem}_gemini_{model_name_for_file}_raw_response_failed_attempt{parse_attempt}.txt"
                            )
                            raw_file.parent.mkdir(parents=True, exist_ok=True)
                            raw_file.write_text(raw_text, encoding="utf-8")
                            print(f"원본 응답 저장: {raw_file}", flush=True)

                        if parse_attempt >= max(1, int(args.ai_retries)):
                            raise

                        continue

                    return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model

                except Exception as error:
                    last_error = error
                    if is_configuration_error(error):
                        raise

                    # 파싱 실패는 같은 모델로 재생성하고, API/모델 실패는 다음 fallback 모델로 넘깁니다.
                    if isinstance(error, json.JSONDecodeError) and parse_attempt < max(1, int(args.ai_retries)):
                        continue

                    print(f"Gemini 모델 실패: {model} / {error}", file=sys.stderr, flush=True)
                    break

        if last_error:
            raise last_error
        raise RuntimeError("시도할 Gemini 모델이 없습니다.")

    finally:
        if args.delete_uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
                print("Gemini 업로드 파일 삭제 완료", file=sys.stderr, flush=True)
            except Exception as error:
                print(f"Gemini 업로드 파일 삭제 실패: {error}", file=sys.stderr, flush=True)


# ----------------------------- OpenAI -----------------------------

def create_openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI API를 쓰려면 `pip install openai`가 필요합니다.") from error
    return OpenAI(api_key=api_key)


def build_openai_response_payload(
    model: str,
    prompt: str,
    file_id: str,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "store": False,
    }

    # 일부 OpenAI reasoning 모델(gpt-5 계열 등)은 temperature 파라미터를 지원하지 않습니다.
    # 따라서 OpenAI는 기본적으로 temperature를 보내지 않습니다.
    # 필요하면 프롬프트/스키마로 출력 안정성을 제어합니다.

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
    client,
    model: str,
    prompt: str,
    file_id: str,
    temperature: float,
    max_output_tokens: int | None,
    use_schema: bool,
):
    payloads = build_openai_response_payload(
        model=model,
        prompt=prompt,
        file_id=file_id,
        temperature=temperature,
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
    raise RuntimeError("OpenAI 응답 생성 실패")


def generate_toc_from_pdf_openai(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    client = create_openai_client(api_key=get_api_key("openai"))
    uploaded_file = None
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    # OpenAI file/context input currently validates the filename extension case-sensitively.
    # A real PDF named "*.PDF" can fail with "supported format ... but got .PDF".
    # Upload through a temporary ASCII-safe lowercase ".pdf" filename while preserving
    # the original filename for output paths.
    safe_name = make_ascii_upload_name(pdf_path)

    try:
        print("  OpenAI PDF 업로드 시작...", flush=True)
        with tempfile.TemporaryDirectory(prefix="openai_pdf_upload_") as temp_dir:
            upload_path = Path(temp_dir) / safe_name
            shutil.copy2(pdf_path, upload_path)

            def do_upload():
                with upload_path.open("rb") as file_obj:
                    return client.files.create(file=file_obj, purpose="user_data")

            uploaded_file = with_retry(
                label=f"OpenAI PDF 업로드({pdf_path.name})",
                func=do_upload,
                max_retries=args.ai_retries,
                base_delay=args.ai_retry_base_delay,
            )
        print("  OpenAI PDF 업로드 완료", flush=True)

        for index, model in enumerate(models):
            try:
                if index > 0:
                    print(f"OpenAI fallback 모델 시도: {model}", file=sys.stderr, flush=True)

                print(f"  OpenAI 목차 생성 요청 시작: {model}", flush=True)
                response = with_retry(
                    label=f"OpenAI 생성({model})",
                    func=lambda model=model: generate_openai_once(
                        client=client,
                        model=model,
                        prompt=prompt,
                        file_id=uploaded_file.id,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        use_schema=not args.no_schema,
                    ),
                    max_retries=args.ai_retries,
                    base_delay=args.ai_retry_base_delay,
                )
                print(f"  OpenAI 목차 생성 완료: {model}", flush=True)

                raw_text = str(getattr(response, "output_text", "") or "")
                parsed = parse_json_response(response, provider="openai")
                return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model

            except Exception as error:
                last_error = error
                if is_configuration_error(error):
                    raise
                print(f"OpenAI 모델 실패: {model} / {error}", file=sys.stderr, flush=True)

        if last_error:
            raise last_error
        raise RuntimeError("시도할 OpenAI 모델이 없습니다.")

    finally:
        if args.delete_uploaded_file and uploaded_file is not None:
            try:
                client.files.delete(uploaded_file.id)
                print("OpenAI 업로드 파일 삭제 완료", file=sys.stderr, flush=True)
            except Exception as error:
                print(f"OpenAI 업로드 파일 삭제 실패: {error}", file=sys.stderr, flush=True)


# ----------------------------- Claude -----------------------------

def create_claude_client(api_key: str):
    try:
        from anthropic import Anthropic
    except ImportError as error:
        raise RuntimeError("Claude API를 쓰려면 `pip install anthropic`가 필요합니다.") from error
    return Anthropic(api_key=api_key)


def generate_claude_once(
    client,
    model: str,
    pdf_path: Path,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None,
):
    pdf_bytes = pdf_path.read_bytes()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    return client.messages.create(
        model=model,
        max_tokens=int(max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS),
        # claude-opus/sonnet 최신 계열 일부 모델은 temperature가 deprecated 처리되어
        # 전송하면 400 invalid_request_error가 발생합니다. 안정성은 프롬프트로 제어합니다.
        system=(
            "너는 PDF 교재/강의자료의 목차를 만드는 전문가다. "
            "반드시 유효한 JSON 객체 하나만 출력한다. 마크다운과 설명은 출력하지 않는다."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )


def generate_toc_from_pdf_claude(pdf_path: Path, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str, str]:
    client = create_claude_client(api_key=get_api_key("claude"))
    models = parse_model_list(args.model, args.ai_fallback_models)
    last_error: Exception | None = None

    for index, model in enumerate(models):
        try:
            if index > 0:
                print(f"Claude fallback 모델 시도: {model}", file=sys.stderr, flush=True)

            print(f"  Claude 목차 생성 요청 시작: {model}", flush=True)
            response = with_retry(
                label=f"Claude 생성({model})",
                func=lambda model=model: generate_claude_once(
                    client=client,
                    model=model,
                    pdf_path=pdf_path,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                ),
                max_retries=args.ai_retries,
                base_delay=args.ai_retry_base_delay,
            )
            print(f"  Claude 목차 생성 완료: {model}", flush=True)

            raw_text_parts: list[str] = []
            for block in getattr(response, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    raw_text_parts.append(str(getattr(block, "text", "") or ""))
            raw_text = "\n".join(raw_text_parts)
            parsed = parse_json_response(response, provider="claude")
            return validate_toc(parsed, fallback_title=pdf_path.stem, max_depth=args.max_depth), raw_text, model

        except Exception as error:
            last_error = error
            if is_configuration_error(error):
                raise
            print(f"Claude 모델 실패: {model} / {error}", file=sys.stderr, flush=True)

    if last_error:
        raise last_error
    raise RuntimeError("시도할 Claude 모델이 없습니다.")


# ----------------------------- Common processing -----------------------------

def generate_toc_from_pdf(
    pdf_path: Path,
    args: argparse.Namespace,
    user_prompt: str,
) -> tuple[dict[str, Any], str, str]:
    prompt = build_prompt(user_prompt=user_prompt, max_depth=args.max_depth)

    if args.provider == "gemini":
        return generate_toc_from_pdf_gemini(pdf_path=pdf_path, args=args, prompt=prompt)

    if args.provider == "openai":
        return generate_toc_from_pdf_openai(pdf_path=pdf_path, args=args, prompt=prompt)

    if args.provider == "claude":
        return generate_toc_from_pdf_claude(pdf_path=pdf_path, args=args, prompt=prompt)

    raise ValueError(f"지원하지 않는 provider입니다: {args.provider}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def safe_filename_part(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "unknown_model"


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {path}")
        return [path]

    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() == ".pdf")

    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")


def process_pdf(pdf_path: Path, args: argparse.Namespace, user_prompt: str) -> bool:
    print(f"처리 시작: {pdf_path.name}", flush=True)
    print(f"  provider: {args.provider}", flush=True)

    try:
        toc, raw_text, used_model = generate_toc_from_pdf(pdf_path=pdf_path, args=args, user_prompt=user_prompt)
        model_name_for_file = safe_filename_part(used_model)
        output_file = Path(args.output_dir) / f"{pdf_path.stem}_{args.provider}_{model_name_for_file}_toc.json"
        save_json(output_file, toc)

        if args.write_raw:
            raw_file = Path(args.output_dir) / f"{pdf_path.stem}_{args.provider}_{model_name_for_file}_raw_response.txt"
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            raw_file.write_text(raw_text, encoding="utf-8")
            print(f"원본 응답 저장: {raw_file}", flush=True)

        print(f"완료: {output_file}", flush=True)
        print(f"  model: {used_model}", flush=True)
        print(f"  title: {toc.get('title')}", flush=True)
        print(f"  chapters: {len(toc.get('chapters', []))}개", flush=True)
        return True

    except Exception as error:
        print(f"실패: {pdf_path.name} / {error}", file=sys.stderr, flush=True)
        if args.stop_on_error:
            raise
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF 파일과 프롬프트를 AI provider(gemini/openai/claude)에 보내 목차 JSON을 생성합니다."
    )

    provider_for_defaults = normalize_provider(DEFAULT_PROVIDER)

    parser.add_argument("path", nargs="?", default="./data/input", help="PDF 파일 또는 PDF 폴더")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--prompt", default=None, help="목차 작성 지시문")
    parser.add_argument("--prompt-file", default=None, help="목차 작성 지시문이 들어 있는 UTF-8 텍스트 파일")
    parser.add_argument(
        "--provider",
        default=provider_for_defaults,
        choices=SUPPORTED_PROVIDERS,
        help="사용할 AI provider: gemini, openai, claude",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="모델명. 생략하면 provider별 env 또는 기본값을 사용합니다.",
    )
    parser.add_argument(
        "--ai-fallback-models",
        default=None,
        help="기본 모델 실패 시 시도할 모델 목록, 쉼표로 구분. 생략하면 provider별 env 또는 기본값을 사용합니다.",
    )
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES, help="일시 오류 재시도 횟수")
    parser.add_argument(
        "--ai-retry-base-delay",
        type=float,
        default=DEFAULT_RETRY_BASE_DELAY,
        help="재시도 기본 대기 시간(초)",
    )
    parser.add_argument(
        "--file-processing-timeout",
        type=int,
        default=DEFAULT_FILE_PROCESSING_TIMEOUT,
        help="Gemini 업로드 PDF 처리 대기 제한(초). Gemini에서만 사용합니다.",
    )
    parser.add_argument("--max-depth", type=int, default=6, help="목차 level 최대값")
    parser.add_argument("--temperature", type=float, default=0.1, help="AI temperature. OpenAI gpt-5 계열 및 일부 Claude 최신 모델에는 전송하지 않습니다.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="최대 출력 토큰 수")
    parser.add_argument("--no-schema", action="store_true", help="가능한 JSON schema 강제 옵션을 쓰지 않고 프롬프트만 사용")
    parser.add_argument(
        "--gemini-thinking-budget",
        type=int,
        default=DEFAULT_GEMINI_THINKING_BUDGET,
        help="Gemini thinking token budget. 0은 thinking 비활성화, -1은 Gemini 자동 설정입니다.",
    )
    parser.add_argument("--write-raw", action="store_true", help="원본 응답 텍스트도 저장")
    parser.add_argument("--delete-uploaded-file", action="store_true", help="처리 후 업로드 파일 삭제. Gemini/OpenAI에서만 의미가 있습니다.")
    parser.add_argument("--stop-on-error", action="store_true", help="여러 PDF 처리 중 하나라도 실패하면 즉시 중단")

    # 기존 명령어 호환용. 이제는 항상 AI 방식이므로 동작에는 영향이 없습니다.
    parser.add_argument("--use-ai", action="store_true", help=argparse.SUPPRESS)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.provider = normalize_provider(args.provider)
    if args.model is None:
        args.model = default_model_for(args.provider)
    args.model = normalize_model_for_provider(args.provider, args.model)

    if args.ai_fallback_models is None:
        args.ai_fallback_models = default_fallback_models_for(args.provider)
    args.ai_fallback_models = normalize_model_list_for_provider(args.provider, args.ai_fallback_models)

    if args.max_output_tokens is None:
        args.max_output_tokens = default_max_output_tokens_for(args.provider)

    args.max_depth = max(1, min(int(args.max_depth), 10))

    user_prompt = load_prompt(args)
    pdf_files = iter_pdfs(Path(args.path))

    if not pdf_files:
        print("처리할 PDF 파일이 없습니다.", flush=True)
        return 0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    success_count = 0
    for pdf_path in pdf_files:
        if process_pdf(pdf_path=pdf_path, args=args, user_prompt=user_prompt):
            success_count += 1

    failed_count = len(pdf_files) - success_count
    print(f"처리 결과: 성공 {success_count}개 / 실패 {failed_count}개", flush=True)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
