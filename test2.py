import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF

from PIL import Image
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


OUTPUT_DIR = Path("./data/output")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_AI_RETRIES = get_env_int("GEMINI_MAX_RETRIES", 5)
DEFAULT_AI_RETRY_BASE_DELAY = get_env_float("GEMINI_RETRY_BASE_DELAY", 2.0)
DEFAULT_AI_FALLBACK_MODELS = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash")
DEFAULT_GEMINI_TIMEOUT_SECONDS = get_env_float("GEMINI_TIMEOUT_SECONDS", 180.0)

BOLD_WORDS = ["bold", "black", "heavy", "semibold", "semi-bold", "demibold", "medium"]
ITALIC_WORDS = ["italic", "oblique"]

ROMAN_RE = r"[ivxlcdm]{1,12}"

FRONT_MATTER_WORDS = {"차례", "목차", "contents"}
BACK_MATTER_WORDS = {"찾아보기", "색인", "index"}

REFERENCE_WORDS = {
    "참고문헌",
    "references",
    "reference",
    "bibliography",
    "literaturecited",
}

COVER_NOISE_PHRASES = {
    "preventivemedicineandpublichealth",
}

QUESTION_LIKE_PHRASES = {
    "설명하시오",
    "제시하시오",
    "기술하시오",
    "정의하시오",
    "제안하시오",
    "나열할수있다",
    "설명할수있다",
    "비교하시오",
    "요약하시오",
    "알아보자",
}

TOC_SCHEMA = {
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


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", clean_text(text))


def strip_wrappers(text: str) -> str:
    return clean_text(text).strip("｜|[](){}<>〈〉「」『』·•∙ㆍ-–— ")


def norm_key(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[\s\-–—_:;,.·•∙ㆍ|/\\()\[\]{}]+", "", text)
    return text


def norm_header_footer_key(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[\s\-–—_:;,.·•∙ㆍ|/\\()\[\]{}]+", "", text)
    return text


def is_hangul_or_cjk(ch: str) -> bool:
    return (
        "\uac00" <= ch <= "\ud7a3"
        or "\u3130" <= ch <= "\u318f"
        or "\u1100" <= ch <= "\u11ff"
        or "\u4e00" <= ch <= "\u9fff"
    )


def useful_char_ratio(text: str) -> float:
    chars = [ch for ch in clean_text(text) if not ch.isspace()]
    if not chars:
        return 0.0

    useful = 0

    for ch in chars:
        if ch.isascii() and ch.isalnum():
            useful += 1
        elif is_hangul_or_cjk(ch):
            useful += 1
        elif ch in ".,:;!?()[]{}+-/%·ㆍ":
            useful += 1

    return useful / len(chars)


def is_caption(text: str) -> bool:
    text = clean_text(text)
    patterns = [
        r"^(그림|도표|표)\s*\d+[-.\s]",
        r"^(figure|fig\.?|table)\s*\d+[-.\s:]",
    ]
    return any(re.match(pattern, text, re.I) for pattern in patterns)


def is_front_back_exact_title(text: str) -> bool:
    key = compact_text(text).lower()
    return key in FRONT_MATTER_WORDS or key in BACK_MATTER_WORDS


def looks_like_page_marker(text: str) -> bool:
    """
    ∙903, ∙iii, x ∙, 차례∙ix, ii ∙찾아보기 같은
    쪽번호/머리말/꼬리말 형태를 제거합니다.
    """
    text = compact_text(text)
    lower = text.lower()

    if lower in FRONT_MATTER_WORDS or lower in BACK_MATTER_WORDS:
        return False

    if re.fullmatch(
        rf"[∙·•ㆍ\-–—]*({ROMAN_RE}|\d{{1,4}})[∙·•ㆍ\-–—]*",
        lower,
        re.I,
    ):
        return True

    if re.fullmatch(
        rf"({ROMAN_RE})?[∙·•ㆍ\-–—]*(차례|목차|찾아보기|색인)[∙·•ㆍ\-–—]*({ROMAN_RE})?",
        lower,
        re.I,
    ):
        return True

    return False


def is_reference_heading(text: str) -> bool:
    key = compact_text(strip_wrappers(text)).lower()
    return key in REFERENCE_WORDS


def looks_like_formula_or_corrupt(text: str) -> bool:
    """
    PDF 폰트 인코딩 깨짐, 수식 조각, private-use glyph 제거.
    """
    text = clean_text(text)
    chars = [ch for ch in text if not ch.isspace()]

    if not chars:
        return True

    if any(unicodedata.category(ch).startswith("C") for ch in chars):
        return True

    if any("\ue000" <= ch <= "\uf8ff" for ch in chars):
        return True

    if re.search(r"[=＋+÷×∑√≤≥≠≒∞]", text):
        return True

    if len(chars) >= 3 and useful_char_ratio(text) < 0.35:
        return True

    return False


def is_reference_like_item(text: str) -> bool:
    """
    1. Kasl, 1. FCTC, 1. AUDIT-K 같은 참고문헌/약어 항목 제거.
    한글 제목은 유지합니다.
    """
    text = clean_text(text)

    if re.fullmatch(r"\d+\.\s*[A-Z][A-Za-z0-9\-]{1,30}\.?", text):
        return True

    if re.fullmatch(r"\d+\.\s*[A-Z][A-Za-z\-]{1,30}\s+et\s+al\.?,?", text, re.I):
        return True

    return False


def heading_number_info(text: str):
    """
    제목 앞 번호 형식을 분석합니다.

    예:
    - 제4편       -> ko_part, level 1
    - 제1장       -> ko_chapter, level 2
    - 제2절       -> ko_section, level 3
    - 1. 제목     -> arabic_dot, level 3
    - 1.1 제목    -> arabic_decimal_1, level 4
    - 1.1.1 제목  -> arabic_decimal_2, level 5
    - 1) 제목     -> arabic_paren, level은 문맥에서 결정
    - (1) 제목     -> arabic_paren_wrap, 1) 다음에 오는 하위 항목
    """
    text = clean_text(text)

    match = re.match(r"^제\s*(\d+)\s*(편|부|장|절)\s*(.*)$", text)
    if match:
        number = int(match.group(1))
        kind = match.group(2)
        body = clean_text(match.group(3))

        if kind in {"편", "부"}:
            return {
                "style": "ko_part",
                "number": number,
                "number_text": f"제{number}{kind}",
                "body": body,
                "level": 1,
            }

        if kind == "장":
            return {
                "style": "ko_chapter",
                "number": number,
                "number_text": f"제{number}장",
                "body": body,
                "level": 2,
            }

        if kind == "절":
            return {
                "style": "ko_section",
                "number": number,
                "number_text": f"제{number}절",
                "body": body,
                "level": 3,
            }

    match = re.match(r"^(\d+(?:\.\d+){1,5})\.?\s*(\S.*)$", text)
    if match:
        number_text = match.group(1)
        numbers = [int(item) for item in number_text.split(".")]
        dot_count = number_text.count(".")
        level = min(3 + dot_count, 6)

        return {
            "style": f"arabic_decimal_{dot_count}",
            "number": numbers[-1],
            "numbers": numbers,
            "number_text": number_text,
            "body": clean_text(match.group(2)),
            "level": level,
        }

    match = re.match(r"^(\d{1,2})\.\s*(\S.*)$", text)
    if match:
        number = int(match.group(1))
        return {
            "style": "arabic_dot",
            "number": number,
            "number_text": f"{number}.",
            "body": clean_text(match.group(2)),
            "level": 3,
        }

    match = re.match(r"^\((\d{1,2})\)\s*(\S.*)$", text)
    if match:
        number = int(match.group(1))
        return {
            "style": "arabic_paren_wrap",
            "number": number,
            "number_text": f"({number})",
            "body": clean_text(match.group(2)),
            "level": None,
        }

    match = re.match(r"^(\d{1,2})\)\s*(\S.*)$", text)
    if match:
        number = int(match.group(1))
        return {
            "style": "arabic_paren",
            "number": number,
            "number_text": f"{number})",
            "body": clean_text(match.group(2)),
            "level": None,
        }

    match = re.match(r"^(\d{1,2})\s+(\S.*)$", text)
    if match:
        number = int(match.group(1))
        return {
            "style": "bare_number",
            "number": number,
            "number_text": str(number),
            "body": clean_text(match.group(2)),
            "level": None,
        }

    return None


def original_number_prefix_only_info(text: str):
    """
    원문 번호만 따로 추출된 경우를 감지합니다.

    예:
    - 1.
    - 1.1
    - 1.1.
    - 1)
    - (1)

    중요:
    이 번호는 새로 만드는 번호가 아니라 PDF에서 실제로 추출된 번호입니다.
    """
    text = clean_text(text)

    match = re.fullmatch(r"(\d+(?:\.\d+){1,5})\.?", text)
    if match:
        number_text = match.group(1)
        dot_count = number_text.count(".")
        return {
            "style": f"arabic_decimal_{dot_count}",
            "number_text": number_text,
            "level": min(3 + dot_count, 6),
        }

    match = re.fullmatch(r"(\d{1,2})\.", text)
    if match:
        number = int(match.group(1))
        return {
            "style": "arabic_dot",
            "number_text": f"{number}.",
            "level": 3,
        }

    match = re.fullmatch(r"\((\d{1,2})\)", text)
    if match:
        number = int(match.group(1))
        return {
            "style": "arabic_paren_wrap",
            "number_text": f"({number})",
            "level": None,
        }

    match = re.fullmatch(r"(\d{1,2})\)", text)
    if match:
        number = int(match.group(1))
        return {
            "style": "arabic_paren",
            "number_text": f"{number})",
            "level": None,
        }

    return None


def is_original_number_prefix_only(text: str) -> bool:
    return original_number_prefix_only_info(text) is not None


def major_label_level(text: str):
    info = heading_number_info(text)
    if not info:
        return None

    if info["style"] in {"ko_part", "ko_chapter", "ko_section"}:
        return info["level"]

    return None


def label_kind(text: str):
    text = compact_text(text)
    match = re.fullmatch(r"제\d+(편|부|장|절)", text)
    return match.group(1) if match else None


def is_label_only(text: str) -> bool:
    return label_kind(text) is not None


def is_bad_numbered_fragment(text: str) -> bool:
    """
    번호는 있지만 제목 본문이 너무 이상한 경우 제거.
    단, '제1장', '제4편'처럼 라벨만 있는 경우는 다음 줄과 병합해야 하므로 살립니다.
    """
    info = heading_number_info(text)
    if not info:
        return False

    body = clean_text(info.get("body", ""))
    body_compact = compact_text(body)

    if info["style"] in {"ko_part", "ko_chapter", "ko_section"} and not body_compact:
        return False

    if not body_compact:
        return True

    if len(body_compact) <= 1:
        return True

    if is_reference_heading(body):
        return True

    if looks_like_page_marker(body):
        return True

    if looks_like_formula_or_corrupt(body):
        return True

    if info["style"] != "arabic_paren":
        if re.fullmatch(r"[\d.,:/()%\[\]{}+\-\s]+", body):
            return True

    return False


def is_standalone_bad_number_fragment(text: str) -> bool:
    """
    병합되지 않고 혼자 남은 '1) 1960', '5) 21', '2) 「' 같은 조각 제거.
    """
    info = heading_number_info(text)
    if not info:
        return False

    body = compact_text(info.get("body", ""))

    if info["style"] == "arabic_paren":
        if re.fullmatch(r"\d{1,4}", body):
            return True

        if body in {"「", "」", "『", "』", "(", ")", "-", "–", "—"}:
            return True

    return False


def is_bad_heading_text(text: str) -> bool:
    """
    제목 후보로 쓰면 안 되는 텍스트를 한 번에 필터링합니다.
    """
    text = clean_text(text)
    compact = compact_text(text)
    key = norm_key(text)
    info = heading_number_info(text)

    if not text:
        return True

    if len(compact) <= 1:
        return True

    # 병합되지 않고 혼자 남은 원문 번호 조각은 최종 제목으로 쓰지 않습니다.
    if is_original_number_prefix_only(text):
        return True

    if key in COVER_NOISE_PHRASES:
        return True

    if any(phrase in key for phrase in QUESTION_LIKE_PHRASES):
        return True

    if is_front_back_exact_title(text):
        return True

    if looks_like_page_marker(text):
        return True

    if is_reference_heading(text):
        return True

    if is_reference_like_item(text):
        return True

    if is_bad_numbered_fragment(text):
        return True

    if looks_like_formula_or_corrupt(text):
        return True

    if not (info and info["style"] in {"arabic_paren", "arabic_dot", "bare_number"}):
        if re.fullmatch(r"[\d.,:/()%\[\]{}+\-\s]+", text):
            return True

    if re.fullmatch(r"\([A-Za-z\s\-]{1,60}\)", text):
        return True

    return False


def is_noise(text: str) -> bool:
    text = clean_text(text)

    if not text:
        return True

    # 원문 번호만 따로 추출된 줄은 버리지 않습니다.
    # 뒤에서 다음 제목 줄과 병합합니다.
    if is_original_number_prefix_only(text):
        return False

    if looks_like_page_marker(text):
        return True

    if looks_like_formula_or_corrupt(text):
        return True

    if re.fullmatch(r"[\d\s\-–—_.·•∙ㆍ|/\\]+", text):
        return True

    if re.match(r"^https?://|^www\.|^[\w.\-]+@[\w.\-]+$", text, re.I):
        return True

    return False


def is_bad_metadata_title(title: str, pdf_path: Path) -> bool:
    """
    <BED5BACE...hwp> 같은 HWP/PDF 변환 메타데이터 제목을 제거합니다.
    """
    title = clean_text(title)

    if not title:
        return True

    key = norm_key(title)

    bad = {
        norm_key(pdf_path.stem),
        "untitled",
        "document",
        "microsoftword",
        "hwp",
        "hwpx",
    }

    if key in bad:
        return True

    if re.fullmatch(r"<[A-Fa-f0-9]{10,}(?:\.[A-Za-z0-9]+)?>", title):
        return True

    if re.fullmatch(r"[A-Fa-f0-9]{12,}(?:\.[A-Za-z0-9]+)?", title):
        return True

    if re.fullmatch(r"[A-Za-z0-9_\-]{20,}\.(hwp|hwpx|doc|docx|pdf)", title, re.I):
        return True

    if looks_like_formula_or_corrupt(title):
        return True

    return False


def is_numbered_heading(text: str) -> bool:
    text = clean_text(text)
    info = heading_number_info(text)

    if not info:
        return False

    if info["style"] == "bare_number":
        return False

    if is_reference_heading(text):
        return False

    if is_reference_like_item(text):
        return False

    if looks_like_page_marker(text):
        return False

    if looks_like_formula_or_corrupt(text):
        return False

    if is_bad_numbered_fragment(text):
        return False

    return True


def numbered_level(text: str):
    """
    원문 번호가 명확한 항목의 level만 반환합니다.
    1) / 2) 형식은 문맥에 따라 level이 달라질 수 있으므로 여기서 고정하지 않습니다.
    """
    text = clean_text(text)
    info = heading_number_info(text)

    if not info:
        return None

    if info["style"] == "bare_number":
        return None

    if info["style"] == "arabic_paren":
        return None

    if is_bad_heading_text(text):
        return None

    level = info.get("level")
    if level is None:
        return None

    return max(1, min(int(level), 6))


def span_is_bold(span: dict) -> bool:
    font = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)
    return bool(flags & 16) or any(word in font for word in BOLD_WORDS)


def span_is_italic(span: dict) -> bool:
    font = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)
    return bool(flags & 2) or any(word in font for word in ITALIC_WORDS)


def get_metadata_title(pdf_path: Path):
    try:
        with fitz.open(pdf_path) as doc:
            title = clean_text((doc.metadata or {}).get("title", ""))
    except Exception:
        return None

    if is_bad_metadata_title(title, pdf_path):
        return None

    return title


def extract_builtin_toc(pdf_path: Path):
    """
    PDF 내부 북마크/outline 목차가 있으면 우선 사용합니다.
    """
    with fitz.open(pdf_path) as doc:
        toc = doc.get_toc(simple=True)

    if not toc:
        return None

    chapters = []

    for level, title, page in toc:
        title = clean_text(str(title))

        if title and page >= 1 and not is_bad_heading_text(title):
            chapters.append({
                "level": int(level),
                "chapter": title,
                "page": int(page),
            })

    return chapters or None


def extract_lines(pdf_path: Path, max_pages: int = 0):
    """
    PDF에서 line/span 단위 구조 정보를 추출합니다.
    """
    lines = []

    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        limit = page_count if max_pages <= 0 else min(max_pages, page_count)

        for page_index in range(limit):
            page = doc.load_page(page_index)
            page_dict = page.get_text("dict", sort=True)
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)

            for block_no, block in enumerate(page_dict.get("blocks", [])):
                if block.get("type") != 0:
                    continue

                for line_no, line in enumerate(block.get("lines", [])):
                    direction = line.get("dir", (1, 0))

                    if isinstance(direction, (tuple, list)) and len(direction) == 2:
                        if round(float(direction[0]), 2) != 1.0 or round(float(direction[1]), 2) != 0.0:
                            continue

                    spans = line.get("spans", [])

                    if not spans:
                        continue

                    text = clean_text("".join(str(span.get("text", "")) for span in spans))

                    if is_noise(text):
                        continue

                    bbox = tuple(float(value) for value in line.get("bbox", (0, 0, 0, 0)))

                    if len(bbox) != 4:
                        continue

                    max_size = 0.0
                    size_sum = 0.0
                    char_count = 0
                    fonts = []
                    flags = []
                    bold = False
                    italic = False

                    for span in spans:
                        span_text = str(span.get("text", ""))
                        span_len = max(1, len(span_text.strip()))
                        size = float(span.get("size", 0) or 0)

                        max_size = max(max_size, size)
                        size_sum += size * span_len
                        char_count += span_len
                        fonts.append(str(span.get("font", "")))
                        flags.append(int(span.get("flags", 0) or 0))
                        bold = bold or span_is_bold(span)
                        italic = italic or span_is_italic(span)

                    x0, y0, x1, y1 = bbox

                    lines.append({
                        "page": page_index + 1,
                        "text": text,
                        "size": round(max_size, 2),
                        "avg_size": round(size_sum / max(1, char_count), 2),
                        "bold": bold,
                        "italic": italic,
                        "fonts": sorted(set(fonts)),
                        "flags": sorted(set(flags)),
                        "bbox": [x0, y0, x1, y1],
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "page_width": page_width,
                        "page_height": page_height,
                        "rel_y0": y0 / page_height if page_height else 0,
                        "rel_y1": y1 / page_height if page_height else 0,
                        "block_no": block_no,
                        "line_no": line_no,
                        "prev_gap": 0.0,
                        "next_gap": 0.0,
                        "score": 0.0,
                        "reasons": [],
                    })

    compute_gaps(lines)
    return lines, page_count


def render_page_as_image(page, zoom: float = 2.0):
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def normalize_ocr_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return text

    # 한글 글자 사이에 잘못 들어간 공백 제거
    text = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", text)
    # 괄호나 구두점 앞뒤 공백 정리
    text = re.sub(r"\s+([,.:;?!\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_lines_with_ocr(
    pdf_path: Path,
    max_pages: int = 0,
    zoom: float = 2.0,
    psm: int = 3,
    lang: str = "kor+eng",
    workers: int = 1,
):
    try:
        import pytesseract
    except ImportError as error:
        raise RuntimeError("OCR을 사용하려면 pytesseract를 설치해야 합니다.") from error

    lines = []

    # Render pages first
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        limit = page_count if max_pages <= 0 else min(max_pages, page_count)

        images = []
        for page_index in range(limit):
            page = doc.load_page(page_index)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((page_index, img))

    def ocr_image(args_tuple):
        page_index, image = args_tuple
        page_width = float(image.width)
        page_height = float(image.height)
        try:
            config = f"--psm {int(psm)}"
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError:
            data = pytesseract.image_to_data(
                image,
                config=f"--psm {int(psm)}",
                output_type=pytesseract.Output.DICT,
            )

        rows = {}
        for i in range(len(data.get("level", []))):
            text = str(data.get("text", [])[i] or "").strip()
            if not text:
                continue

            line_num = int(data.get("line_num", [])[i] or 0)
            block_num = int(data.get("block_num", [])[i] or 0)
            left = float(data.get("left", [])[i] or 0)
            top = float(data.get("top", [])[i] or 0)
            width = float(data.get("width", [])[i] or 0)
            height = float(data.get("height", [])[i] or 0)

            key = (line_num, block_num)
            rows.setdefault(key, []).append({
                "text": text,
                "left": left,
                "top": top,
                "right": left + width,
                "bottom": top + height,
                "height": height,
            })

        page_lines = []
        for key in sorted(rows):
            items = rows[key]
            items.sort(key=lambda item: item["left"])
            text = " ".join(item["text"] for item in items)
            x0 = min(item["left"] for item in items)
            y0 = min(item["top"] for item in items)
            x1 = max(item["right"] for item in items)
            y1 = max(item["bottom"] for item in items)
            size = float(sum(item["height"] for item in items) / max(1, len(items)))

            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue

            if is_noise(cleaned_text):
                continue

            page_lines.append({
                "page": page_index + 1,
                "text": cleaned_text,
                "size": round(size, 2),
                "avg_size": round(size, 2),
                "bold": False,
                "italic": False,
                "fonts": [],
                "flags": [],
                "bbox": [x0, y0, x1, y1],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "page_width": page_width,
                "page_height": page_height,
                "rel_y0": y0 / page_height if page_height else 0,
                "rel_y1": y1 / page_height if page_height else 0,
                "block_no": key[1],
                "line_no": key[0],
                "prev_gap": 0.0,
                "next_gap": 0.0,
                "score": 0.0,
                "reasons": [],
            })

        return page_lines

    if workers is None or workers < 1:
        workers = 1

    if workers == 1:
        for tup in images:
            lines.extend(ocr_image(tup))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for page_lines in ex.map(ocr_image, images):
                lines.extend(page_lines)

    compute_gaps(lines)
    return lines, page_count


def compute_gaps(lines: list[dict]) -> None:
    by_page = defaultdict(list)

    for line in lines:
        by_page[line["page"]].append(line)

    for page_lines in by_page.values():
        page_lines.sort(key=lambda item: (item["y0"], item["x0"]))

        for index, line in enumerate(page_lines):
            if index == 0:
                line["prev_gap"] = max(0.0, line["y0"])
            else:
                previous = page_lines[index - 1]
                line["prev_gap"] = max(0.0, line["y0"] - previous["y1"])

            if index == len(page_lines) - 1:
                line["next_gap"] = max(0.0, line["page_height"] - line["y1"])
            else:
                next_line = page_lines[index + 1]
                line["next_gap"] = max(0.0, next_line["y0"] - line["y1"])


def lines_by_page(lines: list[dict]):
    by_page = defaultdict(list)

    for line in lines:
        by_page[line["page"]].append(line)

    return by_page


def detect_content_start_page(lines: list[dict]):
    """
    차례 앞부분을 버리기 위해 첫 실제 본문 시작 페이지를 찾습니다.
    보통 '제4편', '제1장' 같은 라벨이 처음 나오는 페이지입니다.
    """
    by_page = lines_by_page(lines)

    for page in sorted(by_page):
        page_lines = by_page[page]
        page_text = compact_text(" ".join(line["text"] for line in page_lines)).lower()

        if any(word in page_text for word in FRONT_MATTER_WORDS):
            continue

        has_major_label = any(
            major_label_level(line["text"]) in {1, 2}
            for line in page_lines
        )

        if has_major_label:
            return page

    return None


def detect_back_matter_start_page(lines: list[dict], min_page=None):
    """
    찾아보기/색인 시작 페이지를 찾습니다.
    찾으면 그 페이지부터 뒤는 제거합니다.
    """
    by_page = lines_by_page(lines)

    for page in sorted(by_page):
        if min_page is not None and page < min_page:
            continue

        for line in by_page[page]:
            key = compact_text(line["text"]).lower()

            if key in BACK_MATTER_WORDS:
                return page

    return None


def trim_front_back_matter(lines: list[dict]):
    """
    앞쪽 차례/목차 페이지와 뒤쪽 찾아보기/색인을 제거합니다.
    """
    if not lines:
        return lines

    content_start = detect_content_start_page(lines)
    back_start = detect_back_matter_start_page(lines, min_page=content_start)

    filtered = []

    for line in lines:
        page = line["page"]

        if content_start is not None and page < content_start:
            continue

        if back_start is not None and page >= back_start:
            continue

        filtered.append(line)

    compute_gaps(filtered)
    return filtered


def remove_headers_footers(lines: list[dict], min_repeat_pages: int = 3):
    """
    여러 페이지 상단/하단에 반복되는 머리말/꼬리말 제거.
    """
    pages_by_key = defaultdict(set)

    for line in lines:
        in_top_bottom = line["rel_y0"] <= 0.10 or line["rel_y1"] >= 0.90

        if not in_top_bottom:
            continue

        key = norm_header_footer_key(line["text"])

        if len(key) >= 3:
            pages_by_key[key].add(line["page"])

    repeated = {
        key for key, pages in pages_by_key.items()
        if len(pages) >= min_repeat_pages
    }

    filtered = []

    for line in lines:
        text = clean_text(line["text"])
        key = norm_header_footer_key(text)
        in_top_bottom = line["rel_y0"] <= 0.10 or line["rel_y1"] >= 0.90

        page_number_like = bool(
            re.match(r"^(page\s*)?\d+(\s*/\s*\d+)?$|^-\s*\d+\s*-$", text, re.I)
        ) or looks_like_page_marker(text)

        if in_top_bottom and (key in repeated or page_number_like):
            continue

        filtered.append(line)

    compute_gaps(filtered)
    return filtered


def merge_original_number_prefix_lines(lines: list[dict]):
    """
    PDF 원문 번호가 제목과 분리되어 추출된 경우 병합합니다.

    예:
    - "1.1" + "보건의료자원의종류"
      -> "1.1 보건의료자원의종류"

    - "1)" + "의료인력"
      -> "1) 의료인력"

    중요:
    이 함수는 번호를 새로 만들지 않습니다.
    PDF에서 실제로 추출된 번호 줄만 다음 제목 줄과 합칩니다.
    """
    if not lines:
        return lines

    by_page = defaultdict(list)

    for line in lines:
        by_page[line["page"]].append(line)

    merged_all = []

    for page in sorted(by_page):
        page_lines = sorted(by_page[page], key=lambda item: (item["y0"], item["x0"]))
        i = 0

        while i < len(page_lines):
            current = page_lines[i]
            current_text = clean_text(current["text"])
            prefix_info = original_number_prefix_only_info(current_text)

            if not prefix_info:
                merged_all.append(current)
                i += 1
                continue

            if i + 1 >= len(page_lines):
                i += 1
                continue

            nxt = page_lines[i + 1]
            nxt_text = clean_text(nxt["text"])

            if heading_number_info(nxt_text):
                i += 1
                continue

            if (
                is_front_back_exact_title(nxt_text)
                or looks_like_page_marker(nxt_text)
                or is_reference_heading(nxt_text)
                or looks_like_formula_or_corrupt(nxt_text)
            ):
                i += 1
                continue

            vertical_gap = max(0.0, nxt["y0"] - current["y1"])

            if vertical_gap > max(current["size"] * 3.0, 60):
                i += 1
                continue

            merged = dict(nxt)
            merged["text"] = f"{prefix_info['number_text']} {nxt_text}"
            merged["size"] = max(current.get("size", 0), nxt.get("size", 0))
            merged["avg_size"] = max(current.get("avg_size", 0), nxt.get("avg_size", 0))
            merged["bold"] = bool(current.get("bold")) or bool(nxt.get("bold"))
            merged["italic"] = bool(current.get("italic")) or bool(nxt.get("italic"))
            merged["x0"] = min(current["x0"], nxt["x0"])
            merged["y0"] = min(current["y0"], nxt["y0"])
            merged["x1"] = max(current["x1"], nxt["x1"])
            merged["y1"] = max(current["y1"], nxt["y1"])
            merged["bbox"] = [merged["x0"], merged["y0"], merged["x1"], merged["y1"]]
            merged["merged_from"] = [current_text, nxt_text]

            merged_all.append(merged)
            i += 2

    merged_all.sort(key=lambda item: (item["page"], item["y0"], item["x0"]))
    compute_gaps(merged_all)
    return merged_all


def merge_decimal_heading_with_previous_short_line(lines: list[dict]) -> list[dict]:
    """
    긴 1.2 형식의 본문 문장이 앞의 짧은 제목 라인과 함께 나올 때,
    번호는 유지하되 실제 제목으로 짧은 앞줄을 사용하도록 보정합니다.
    """
    if not lines:
        return lines

    by_page = defaultdict(list)

    for line in lines:
        by_page[line["page"]].append(line)

    skip_keys = set()

    for page, page_lines in by_page.items():
        page_lines.sort(key=lambda item: (item["y0"], item["x0"]))

        for index in range(1, len(page_lines)):
            current = page_lines[index]
            previous = page_lines[index - 1]

            if current["page"] != previous["page"]:
                continue

            current_info = heading_number_info(current["text"])
            if not current_info or not current_info["style"].startswith("arabic_decimal_"):
                continue

            current_text = clean_text(current["text"])
            if len(compact_text(current_text)) <= 40:
                continue

            previous_text = clean_text(previous["text"])
            if not previous_text:
                continue

            if heading_number_info(previous_text):
                continue

            if is_bad_heading_text(previous_text):
                continue

            if len(compact_text(previous_text)) > 20:
                continue

            gap = max(0.0, current["y0"] - previous["y1"])
            if gap > max(current["size"] * 2.0, 40):
                continue

            if abs(current["x0"] - previous["x0"]) > current["page_width"] * 0.20:
                continue

            merged_text = f"{current_info['number_text']} {previous_text}"
            current["text"] = merged_text
            current["size"] = max(current.get("size", 0), previous.get("size", 0))
            current["avg_size"] = max(current.get("avg_size", 0), previous.get("avg_size", 0))
            current["bold"] = bool(current.get("bold")) or bool(previous.get("bold"))
            current["x0"] = min(current["x0"], previous["x0"])
            current["y0"] = min(current["y0"], previous["y0"])
            current["x1"] = max(current["x1"], previous["x1"])
            current["y1"] = max(current["y1"], previous["y1"])
            current["bbox"] = [current["x0"], current["y0"], current["x1"], current["y1"]]

            skip_keys.add((
                previous["page"],
                round(previous["y0"], 2),
                round(previous["x0"], 2),
                compact_text(previous_text),
            ))

    merged = []
    for line in lines:
        key = (
            line["page"],
            round(line["y0"], 2),
            round(line["x0"], 2),
            compact_text(line["text"]),
        )

        if key in skip_keys:
            continue

        merged.append(line)

    return merged


def estimate_body_size(lines: list[dict]) -> float:
    counter = Counter()

    for line in lines:
        text = line["text"]

        if len(text) < 8 or is_caption(text) or is_bad_heading_text(text):
            continue

        counter[round(line["avg_size"], 1)] += min(len(text), 120)

    if counter:
        return float(counter.most_common(1)[0][0])

    fallback = Counter(
        round(line["avg_size"], 1)
        for line in lines
        if line["avg_size"] > 0
    )

    return float(fallback.most_common(1)[0][0]) if fallback else 10.0


def score_line(
    line: dict,
    body_size: float,
    max_heading_chars: int,
    include_captions: bool,
):
    text = line["text"]
    score = 0.0
    reasons = []

    if is_bad_heading_text(text):
        return -99, ["bad_heading_text"]

    if is_noise(text):
        return -99, ["noise"]

    if is_caption(text) and not include_captions:
        return -20, ["caption"]

    text_len = len(text)
    number_info = heading_number_info(text)
    numbered = is_numbered_heading(text)
    size_diff = line["size"] - body_size

    if text_len <= max_heading_chars:
        score += 1
        reasons.append("short")
    else:
        score -= 3
        reasons.append("too_long")

    if text_len <= 2:
        score -= 4
        reasons.append("too_short")

    if size_diff >= 6:
        score += 5
        reasons.append("much_larger")
    elif size_diff >= 3:
        score += 4
        reasons.append("larger")
    elif size_diff >= 1.2:
        score += 2.5
        reasons.append("slightly_larger")
    elif size_diff >= -0.2 and line["bold"] and numbered:
        score += 1
        reasons.append("body_size_bold_numbered")
    elif size_diff < -0.8:
        score -= 2
        reasons.append("smaller_than_body")

    if line["bold"]:
        score += 2
        reasons.append("bold")

    if numbered:
        score += 3
        reasons.append("numbered")

    if number_info and number_info["style"] == "arabic_dot":
        score += 1
        reasons.append("arabic_dot_style")

    if number_info and number_info["style"] == "arabic_paren":
        score += 2
        reasons.append("arabic_paren_style")

        heading_like = (
            line["bold"]
            or size_diff >= 0.8
            or line["prev_gap"] >= max(line["size"] * 0.65, 6)
        )

        if not heading_like:
            score -= 0.8
            reasons.append("paren_body_like")

    if number_info and number_info["style"].startswith("arabic_decimal_"):
        score += 1
        reasons.append("decimal_numbered")

        if line["prev_gap"] < max(line["size"] * 0.25, 4) and line["rel_y0"] > 0.15:
            score -= 1.5
            reasons.append("decimal_no_space_before")

        if text_len >= 40 and not line["bold"]:
            score -= 1.0
            reasons.append("long_decimal_nonbold")

    if major_label_level(text) is not None:
        score += 3
        reasons.append("major_label")

    if line["rel_y0"] < 0.35 and size_diff >= 1.2:
        score += 1
        reasons.append("upper_large")

    if line["prev_gap"] >= max(line["size"] * 0.65, 6):
        score += 1
        reasons.append("space_before")

    if line["next_gap"] >= max(line["size"] * 0.45, 4):
        score += 0.5
        reasons.append("space_after")

    page_center = line["page_width"] / 2
    line_center = (line["x0"] + line["x1"]) / 2
    centered = abs(line_center - page_center) <= line["page_width"] * 0.12

    if centered and size_diff >= 2 and text_len <= 80:
        score += 0.8
        reasons.append("centered")

    sentence_like = (
        text.endswith((".", "다.", "요.", "니다.", "다"))
        or any(phrase in text for phrase in QUESTION_LIKE_PHRASES)
    ) and text_len > 40

    if sentence_like:
        score -= 3
        reasons.append("sentence_like")

    if re.match(r"^[•\-–—*]\s+", text):
        score -= 1.5
        reasons.append("bullet_like")

    return round(score, 2), reasons


def pick_candidates(
    lines: list[dict],
    body_size: float,
    min_score: float,
    max_heading_chars: int,
    include_captions: bool,
):
    candidates = []

    for line in lines:
        score, reasons = score_line(
            line=line,
            body_size=body_size,
            max_heading_chars=max_heading_chars,
            include_captions=include_captions,
        )

        line["score"] = score
        line["reasons"] = reasons

        if score >= min_score and not is_bad_heading_text(line["text"]):
            candidates.append(line)

    grouped = {}

    for candidate in candidates:
        key = (candidate["page"], norm_key(candidate["text"]))
        old = grouped.get(key)

        if old is None or (
            candidate["score"],
            candidate["size"],
            -candidate["y0"],
        ) > (
            old["score"],
            old["size"],
            -old["y0"],
        ):
            grouped[key] = candidate

    result = list(grouped.values())
    result.sort(key=lambda item: (item["page"], item["y0"], item["x0"]))
    return result


def cluster_sizes(sizes, tolerance: float = 0.6, max_clusters: int = 5):
    sizes = sorted(
        {round(float(size), 1) for size in sizes if size > 0},
        reverse=True,
    )

    clusters = []

    for size in sizes:
        if not clusters:
            clusters.append([size])
            continue

        avg = sum(clusters[-1]) / len(clusters[-1])

        if abs(size - avg) <= tolerance:
            clusters[-1].append(size)
        else:
            clusters.append([size])

    return [
        round(sum(cluster) / len(cluster), 1)
        for cluster in clusters[:max_clusters]
    ]


def font_level(size: float, clusters: list[float], max_depth: int):
    if not clusters:
        return 1

    index = min(
        range(len(clusters)),
        key=lambda item: abs(size - clusters[item]),
    )

    return min(index + 1, max_depth)


def should_merge_after_numbered_fragment(text: str) -> bool:
    """
    1) 1960 + 년대이전의보건의료정책
    4) 2000 + 년대의주요의료정책
    같은 분리 제목을 합치기 위한 판단.
    """
    info = heading_number_info(text)

    if not info:
        return False

    if info["style"] not in {"arabic_paren", "arabic_dot", "bare_number"}:
        return False

    body = compact_text(info.get("body", ""))

    if not body:
        return False

    if len(body) <= 8:
        return True

    if re.fullmatch(r"\d{2,4}", body):
        return True

    if body.endswith(("과", "와", "및", "의", "에", "로", "으로")):
        return True

    if body.count("(") > body.count(")"):
        return True

    return False


def should_merge_after_plain_fragment(text: str) -> bool:
    """
    번호는 없지만 제목이 줄바꿈으로 잘린 경우를 합칩니다.
    예:
    보건의료서비스이용에영향을미치는요인과
    경제학적수요모형
    """
    text = clean_text(text)

    if heading_number_info(text):
        return False

    if is_bad_heading_text(text):
        return False

    compact = compact_text(text)

    if len(compact) < 4:
        return False

    if compact.endswith(("과", "와", "및", "의", "에", "로", "으로")):
        return True

    if compact.count("(") > compact.count(")"):
        return True

    return False


def join_heading_parts(parts: list[str]) -> str:
    """
    제목 조각을 자연스럽게 합칩니다.
    """
    if not parts:
        return ""

    result = clean_text(parts[0])

    for part in parts[1:]:
        part = clean_text(part)

        if not part:
            continue

        result_compact = compact_text(result)
        part_compact = compact_text(part)

        if result_compact and part_compact:
            if result_compact[-1].isdigit() and is_hangul_or_cjk(part_compact[0]):
                result = result + part_compact
                continue

        if result.endswith(("과", "와", "및", "의", "에", "로", "으로")):
            result = result + part
        else:
            result = result + " " + part

    return clean_text(result)


def merge_split_heading_labels(candidates: list[dict]):
    """
    1. '제1장' + 다음 줄 제목 병합
    2. '제4편' + 다음 줄 제목 병합
    3. '1) 1960' + '년대이전의...' 같은 분리 번호 제목 병합
    4. 번호 없는 제목 조각 병합

    중요:
    여기서도 번호를 새로 만들지 않습니다.
    이미 후보에 있는 조각만 합칩니다.
    """
    if not candidates:
        return candidates

    result = []
    i = 0

    while i < len(candidates):
        current = candidates[i]
        text = clean_text(current["text"])

        merge_mode = None
        max_extra = 0

        if is_label_only(text):
            kind = label_kind(text)
            merge_mode = "label"
            max_extra = 2 if kind in {"편", "부"} else 1

        elif should_merge_after_numbered_fragment(text):
            merge_mode = "number_fragment"
            max_extra = 1

        elif should_merge_after_plain_fragment(text):
            merge_mode = "plain_fragment"
            max_extra = 1

        if merge_mode:
            parts = [text]
            merged_items = [current]
            consumed = 0
            previous = current
            j = i + 1

            while j < len(candidates) and consumed < max_extra:
                nxt = candidates[j]
                nxt_text = clean_text(nxt["text"])

                if nxt["page"] != current["page"]:
                    break

                if nxt["y0"] < current["y0"]:
                    break

                if is_bad_heading_text(nxt_text):
                    break

                if is_label_only(nxt_text):
                    break

                if heading_number_info(nxt_text):
                    break

                vertical_gap = nxt["y0"] - previous["y1"]

                if vertical_gap > max(current["size"] * 3.0, 60):
                    break

                parts.append(nxt_text)
                merged_items.append(nxt)
                previous = nxt
                consumed += 1
                j += 1

            if len(parts) > 1:
                merged = dict(current)
                merged["text"] = join_heading_parts(parts)
                merged["size"] = max(item.get("size", 0) for item in merged_items)
                merged["score"] = max(item.get("score", 0) for item in merged_items)
                merged["merged_from"] = parts
                result.append(merged)
                i += consumed + 1
                continue

        result.append(current)
        i += 1

    return result


def fixed_level_from_number_info(info, max_depth: int):
    """
    원문 번호 형식만 보고 확정 가능한 level을 반환합니다.

    단, 1) / 2) 형식은 문맥에 따라 달라질 수 있으므로 여기서는 고정하지 않습니다.
    """
    if not info:
        return None

    style = info.get("style")

    if style == "bare_number":
        return None

    if style == "arabic_paren":
        return None

    level = info.get("level")

    if level is None:
        return None

    return max(1, min(int(level), max_depth))


def enforce_consistent_original_number_levels(chapters: list[dict], max_depth: int):
    """
    원문 번호 형식이 명확한 항목은 해당 형식의 level을 유지합니다.

    예:
    - 제4편 -> level 1
    - 제1장 -> level 2
    - 1. 제목 -> level 3
    - 1.1 제목 -> level 4
    - 1.1.1 제목 -> level 5

    1) 형식은 문맥에서 따로 처리합니다.
    """
    result = []

    for chapter in chapters:
        item = dict(chapter)
        text = clean_text(item.get("chapter", ""))
        info = heading_number_info(text)
        fixed_level = fixed_level_from_number_info(info, max_depth=max_depth)

        if fixed_level is not None:
            item["level"] = fixed_level
        else:
            item["level"] = max(1, min(int(item.get("level", 1)), max_depth))

        item["chapter"] = text
        result.append(item)

    return result


def normalize_chapter_levels_without_number_generation(chapters: list[dict], max_depth: int):
    """
    level 점프만 완화합니다.
    번호는 절대 생성하지 않습니다.

    paren 번호인 1) / 2)는 앞 단계에서 문맥으로 정한 level을 유지합니다.
    """
    if not chapters:
        return chapters

    normalized = []

    for chapter in chapters:
        item = dict(chapter)
        text = clean_text(item.get("chapter", ""))
        info = heading_number_info(text)
        style = info.get("style") if info else None

        level = max(1, min(int(item.get("level", 1)), max_depth))
        fixed_level = fixed_level_from_number_info(info, max_depth=max_depth)

        if style in {"arabic_paren", "arabic_paren_wrap"}:
            # 문맥으로 정한 level을 유지합니다.
            pass
        elif fixed_level is not None:
            level = fixed_level
        else:
            if normalized and level > normalized[-1]["level"] + 1:
                level = normalized[-1]["level"] + 1

        item["level"] = level
        item["chapter"] = text
        normalized.append(item)

    return normalized


def resolve_contextual_levels_from_original_numbers(chapters: list[dict], max_depth: int):
    """
    원문 번호를 보존한 채 level만 문맥으로 보정합니다.

    핵심:
    - 번호를 새로 만들지 않습니다.
    - 1) / 2)는 원문 문자열 그대로 두고 level만 조정합니다.
    - 1.1 같은 decimal 계층 아래에 있으면 1)는 level 5가 될 수 있습니다.
    - 원문에 번호가 없는 제목은 번호 없이 유지합니다.
    """
    if not chapters:
        return chapters

    result = []
    active_decimal_parent = False

    for chapter in chapters:
        item = dict(chapter)
        text = clean_text(item.get("chapter", ""))
        info = heading_number_info(text)
        style = info.get("style") if info else None
        level = max(1, min(int(item.get("level", 1)), max_depth))

        previous = result[-1] if result else None
        previous_level = previous["level"] if previous else None
        previous_info = heading_number_info(previous["chapter"]) if previous else None
        previous_style = previous_info.get("style") if previous_info else None

        if style == "ko_part":
            level = 1
            active_decimal_parent = False

        elif style == "ko_chapter":
            level = 2
            active_decimal_parent = False

        elif style == "ko_section":
            level = 3
            active_decimal_parent = False

        elif style == "arabic_dot":
            level = 3
            active_decimal_parent = False

        elif style and style.startswith("arabic_decimal_"):
            level = max(4, min(int(info["level"]), max_depth))
            active_decimal_parent = True

        elif style == "arabic_paren_wrap":
            if previous_style == "arabic_paren" and previous_level == 5 and max_depth >= 6:
                level = 6
            elif active_decimal_parent and max_depth >= 6:
                level = 6
            else:
                level = 5 if max_depth >= 5 else max_depth

            # (1), (2), (3) 형식은 1) / 2) 하위 항목으로 처리합니다.

        elif style == "arabic_paren":
            if active_decimal_parent and max_depth >= 5:
                level = 5
            else:
                level = 4 if max_depth >= 4 else max_depth

            # 1), 2), 3)가 이어지는 동안 active_decimal_parent는 유지합니다.

        else:
            # 번호 없는 제목: 번호는 붙이지 않고 level만 문맥으로 보정합니다.
            if previous_style == "arabic_dot" and previous_level == 3 and level <= 3:
                level = 4
                active_decimal_parent = True

            elif previous_level == 3 and level <= 3:
                level = 4
                active_decimal_parent = True

            elif previous_level == 4 and level <= 4:
                level = 4
                active_decimal_parent = True

            elif level <= 3:
                active_decimal_parent = False

            elif level >= 4:
                active_decimal_parent = True

        item["level"] = max(1, min(level, max_depth))
        item["chapter"] = text
        result.append(item)

    return result


def postprocess_chapters_for_output(
    chapters: list[dict],
    max_depth: int,
):
    """
    최종 목차 후처리.

    중요:
    여기서는 번호를 생성하지 않습니다.
    원문에서 추출된 번호만 보존합니다.
    """
    chapters = [
        chapter for chapter in chapters
        if not is_bad_heading_text(chapter.get("chapter", ""))
        and not is_standalone_bad_number_fragment(chapter.get("chapter", ""))
    ]

    chapters = enforce_consistent_original_number_levels(
        chapters=chapters,
        max_depth=max_depth,
    )

    chapters = normalize_chapter_levels_without_number_generation(
        chapters=chapters,
        max_depth=max_depth,
    )

    chapters = resolve_contextual_levels_from_original_numbers(
        chapters=chapters,
        max_depth=max_depth,
    )

    chapters = normalize_chapter_levels_without_number_generation(
        chapters=chapters,
        max_depth=max_depth,
    )

    return chapters


def infer_title(
    pdf_path: Path,
    metadata_title,
    candidates: list[dict],
    lines: list[dict],
    body_size: float,
):
    if metadata_title and not is_bad_metadata_title(metadata_title, pdf_path):
        return metadata_title

    first_page = min((line["page"] for line in lines), default=1)

    first_candidates = [
        candidate for candidate in candidates
        if candidate["page"] == first_page
        and candidate["rel_y0"] < 0.70
        and len(candidate["text"]) <= 120
        and not is_caption(candidate["text"])
        and not is_bad_heading_text(candidate["text"])
    ]

    if first_candidates:
        best = max(
            first_candidates,
            key=lambda item: (item["size"], item["score"], -item["y0"]),
        )

        if best["size"] >= body_size + 1.2 or best["bold"]:
            return best["text"]

    first_lines = [
        line for line in lines
        if line["page"] == first_page
        and line["rel_y0"] < 0.70
        and 3 <= len(line["text"]) <= 120
        and not is_bad_heading_text(line["text"])
    ]

    if first_lines:
        best = max(
            first_lines,
            key=lambda item: (item["size"], item["bold"], -item["y0"]),
        )

        if best["size"] >= body_size + 1.2 or best["bold"]:
            return best["text"]

    return pdf_path.stem


def build_chapters(
    candidates: list[dict],
    title: str,
    body_size: float,
    max_depth: int,
):
    candidates = merge_split_heading_labels(candidates)

    sizes = [
        candidate["size"]
        for candidate in candidates
        if candidate["size"] >= body_size + 0.8
    ]

    clusters = cluster_sizes(sizes, max_clusters=max_depth)

    chapters = []
    seen = set()

    for candidate in candidates:
        text = clean_text(candidate["text"])

        if is_bad_heading_text(text):
            continue

        if is_standalone_bad_number_fragment(text):
            continue

        if (
            candidate["page"] == 1
            and norm_key(text) == norm_key(title)
            and not is_numbered_heading(text)
        ):
            continue

        key = (norm_key(text), candidate["page"])

        if key in seen:
            continue

        seen.add(key)

        by_number = numbered_level(text)

        if by_number is not None:
            level = by_number
        else:
            level = font_level(candidate["size"], clusters, max_depth)

        level = max(1, min(int(level), max_depth))

        chapters.append({
            "level": level,
            "chapter": text,
            "page": int(candidate["page"]),
        })

    chapters = postprocess_chapters_for_output(
        chapters=chapters,
        max_depth=max_depth,
    )

    return chapters


def validate_toc(data, fallback_title: str):
    title = clean_text(str(data.get("title") or fallback_title)) or fallback_title
    chapters = []

    for item in data.get("chapters", []):
        if not isinstance(item, dict):
            continue

        chapter = clean_text(str(item.get("chapter", "")))

        if not chapter:
            continue

        if is_bad_heading_text(chapter):
            continue

        if is_standalone_bad_number_fragment(chapter):
            continue

        try:
            level = int(item.get("level", 1))
            page = int(item.get("page", 1))
        except Exception:
            continue

        chapters.append({
            "level": max(1, level),
            "chapter": chapter,
            "page": max(1, page),
        })

    chapters = postprocess_chapters_for_output(
        chapters=chapters,
        max_depth=6,
    )

    return {
        "title": title,
        "chapters": chapters,
    }


RETRYABLE_GEMINI_ERROR_MARKERS = (
    "503",
    "UNAVAILABLE",
    "HIGH DEMAND",
    "OVERLOADED",
    "TRY AGAIN LATER",
    "429",
    "RESOURCE_EXHAUSTED",
    "RATE LIMIT",
    "500",
    "INTERNAL",
    "504",
    "DEADLINE_EXCEEDED",
    "TIMEOUT",
    "TIMED OUT",
    "TEMPORAR",
)


CONFIGURATION_ERROR_MARKERS = (
    "GEMINI_API_KEY",
    "API KEY",
    "API_KEY",
    "UNAUTHENTICATED",
    "PERMISSION_DENIED",
    "401",
    "403",
    "GOOGLE-GENAI",
    "PIP INSTALL GOOGLE-GENAI",
)


def is_retryable_gemini_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".upper()
    return any(marker in message for marker in RETRYABLE_GEMINI_ERROR_MARKERS)


def is_ai_configuration_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".upper()
    return any(marker in message for marker in CONFIGURATION_ERROR_MARKERS)


def parse_model_list(primary_model: str, fallback_models) -> list[str]:
    """
    primary_model 뒤에 fallback 모델들을 중복 없이 붙입니다.
    fallback_models는 쉼표 문자열, 리스트, None 모두 허용합니다.
    """
    models = []

    def add_model(value):
        value = clean_text(value)
        if value and value not in models:
            models.append(value)

    add_model(primary_model)

    if fallback_models is None:
        fallback_models = DEFAULT_AI_FALLBACK_MODELS

    if isinstance(fallback_models, str):
        for item in fallback_models.split(","):
            add_model(item)
    else:
        for item in fallback_models:
            add_model(item)

    return models


def parse_gemini_json_response(response) -> dict:
    """
    google-genai 응답에서 JSON을 최대한 안전하게 꺼냅니다.
    SDK 버전에 따라 response.parsed 또는 response.text만 있을 수 있습니다.
    """
    parsed = getattr(response, "parsed", None)

    if isinstance(parsed, dict):
        return parsed

    text = str(getattr(response, "text", "") or "").strip()

    if not text:
        raise ValueError("Gemini 응답이 비어 있습니다.")

    # 혹시 모델이 ```json ... ``` 형태로 감싼 경우 제거합니다.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 앞뒤 설명이 섞였을 때 가장 바깥 JSON 객체만 다시 시도합니다.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def create_gemini_client(genai, api_key: str, timeout_seconds: float):
    timeout_ms = max(1, int(float(timeout_seconds) * 1000))

    try:
        return genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_ms},
        )
    except TypeError:
        # 구버전 google-genai에서 http_options를 받지 않는 경우를 대비합니다.
        return genai.Client(api_key=api_key)


def generate_content_with_retry(
    client,
    model: str,
    prompt: str,
    config: dict,
    max_retries: int,
    base_delay: float,
):
    """
    Gemini 503/429/500/504 등 일시 오류에 대해 exponential backoff로 재시도합니다.
    """
    max_retries = max(1, int(max_retries))
    base_delay = max(0.1, float(base_delay))
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as error:
            last_error = error

            if not is_retryable_gemini_error(error):
                raise

            if attempt >= max_retries:
                break

            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
            delay += random.uniform(0, min(1.0, base_delay))

            print(
                f"Gemini 일시 오류, 재시도 {attempt}/{max_retries - 1} "
                f"({model}, {delay:.1f}초 후): {error}",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise last_error


def refine_with_gemini(
    toc: dict,
    candidates: list[dict],
    model: str,
    ai_retries: int = DEFAULT_AI_RETRIES,
    ai_retry_base_delay: float = DEFAULT_AI_RETRY_BASE_DELAY,
    ai_fallback_models: str = DEFAULT_AI_FALLBACK_MODELS,
    gemini_timeout_seconds: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
):
    """
    선택 기능입니다.
    구조 기반 후보 안에서만 오탐 제거와 level 보정을 하도록 Gemini에게 요청합니다.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("--use-ai 옵션을 쓰려면 GEMINI_API_KEY 환경 변수가 필요합니다.")

    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError("--use-ai 옵션을 쓰려면 `pip install google-genai`가 필요합니다.") from error

    compact_candidates = []

    for candidate in candidates[:300]:
        text = candidate.get("text", "")

        if is_bad_heading_text(text):
            continue

        compact_candidates.append({
            "page": candidate["page"],
            "text": candidate["text"],
            "font_size": candidate["size"],
            "bold": candidate["bold"],
            "relative_y": round(candidate["rel_y0"], 3),
            "score": candidate["score"],
            "reasons": candidate["reasons"],
        })

    prompt = f"""
아래는 PDF의 시각적 구조를 분석해서 만든 목차 초안과 제목 후보 목록입니다.
본문 반복 단어나 의미 추론으로 새 주제를 만들지 마세요.
후보 목록에 없는 장 제목을 새로 만들지 마세요.
후보의 페이지 순서, 번호 패턴, 글자 크기, 굵기, 위치, 점수만 근거로 오탐을 제거하고 level을 보정하세요.
차례, 목차, 찾아보기, 색인, 참고문헌, 쪽번호, 수식 조각, 깨진 문자는 제거하세요.

번호는 절대 새로 만들지 마세요.
원문 후보에 있는 번호만 보존하세요.
원문에 번호가 없는 제목에는 번호를 붙이지 마세요.
'1. 제목', '1.1 제목', '1) 제목' 같은 번호 형식은 원문 문자열 그대로 유지하세요.
같은 번호 형식은 가능한 한 일관된 level로 유지하세요.
단, '1)' 형식은 문맥에 따라 '1. 제목' 아래일 수도 있고 '1.1 제목' 아래일 수도 있으므로 앞뒤 구조를 보고 level만 조정하세요.
chapter 문자열의 번호를 임의로 추가하거나 삭제하지 마세요.

반드시 JSON만 출력하세요.

[목차 초안]
{json.dumps(toc, ensure_ascii=False, indent=2)}

[구조 기반 제목 후보]
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}
""".strip()

    client = create_gemini_client(
        genai=genai,
        api_key=api_key,
        timeout_seconds=gemini_timeout_seconds,
    )

    config = {
        "response_mime_type": "application/json",
        "response_json_schema": TOC_SCHEMA,
    }

    models_to_try = parse_model_list(model, ai_fallback_models)
    last_error = None

    if not models_to_try:
        raise RuntimeError("Gemini 모델명이 비어 있습니다.")

    for index, model_name in enumerate(models_to_try):
        try:
            if index > 0:
                print(f"Gemini fallback 모델 시도: {model_name}", file=sys.stderr)

            response = generate_content_with_retry(
                client=client,
                model=model_name,
                prompt=prompt,
                config=config,
                max_retries=ai_retries,
                base_delay=ai_retry_base_delay,
            )

            parsed = parse_gemini_json_response(response)

            if index > 0:
                print(f"Gemini fallback 모델 성공: {model_name}", file=sys.stderr)

            return validate_toc(
                parsed,
                fallback_title=toc.get("title", "제목 미상"),
            )

        except Exception as error:
            last_error = error

            if is_ai_configuration_error(error):
                raise

            print(f"Gemini 모델 실패: {model_name} / {error}", file=sys.stderr)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Gemini 모델 호출에 실패했습니다.")


def create_toc(
    pdf_path: Path,
    max_pages: int,
    min_score: float,
    max_depth: int,
    max_heading_chars: int,
    ignore_builtin_toc: bool,
    include_captions: bool,
    use_ocr: bool,
    use_ai: bool,
    model: str,
    ai_retries: int = DEFAULT_AI_RETRIES,
    ai_retry_base_delay: float = DEFAULT_AI_RETRY_BASE_DELAY,
    ai_fallback_models: str = DEFAULT_AI_FALLBACK_MODELS,
    gemini_timeout_seconds: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
    ocr_workers: int = 1,
    ocr_zoom: float = 2.0,
    ocr_psm: int = 3,
    ocr_lang: str = "kor+eng",
):
    metadata_title = get_metadata_title(pdf_path)

    if not ignore_builtin_toc:
        builtin = extract_builtin_toc(pdf_path)

        if builtin:
            chapters = postprocess_chapters_for_output(
                chapters=builtin,
                max_depth=max_depth,
            )

            return {
                "title": metadata_title or pdf_path.stem,
                "chapters": chapters,
            }, []

    if use_ocr:
        lines, _ = extract_lines_with_ocr(
            pdf_path,
            max_pages=max_pages,
            zoom=ocr_zoom,
            psm=ocr_psm,
            lang=ocr_lang,
            workers=ocr_workers,
        )
    else:
        lines, _ = extract_lines(pdf_path, max_pages=max_pages)

    if not lines:
        return {
            "title": metadata_title or pdf_path.stem,
            "chapters": [],
        }, []

    # 차례/목차, 찾아보기/색인 같은 앞뒤 부속 페이지 제거
    lines = trim_front_back_matter(lines)

    # 반복 머리말/꼬리말 제거
    lines = remove_headers_footers(lines)

    # PDF 원문 번호가 제목과 분리되어 추출된 경우 병합
    # 번호를 새로 만드는 것이 아니라 실제 추출된 번호 줄만 붙입니다.
    lines = merge_original_number_prefix_lines(lines)
    lines = merge_decimal_heading_with_previous_short_line(lines)

    body_size = estimate_body_size(lines)

    candidates = pick_candidates(
        lines=lines,
        body_size=body_size,
        min_score=min_score,
        max_heading_chars=max_heading_chars,
        include_captions=include_captions,
    )

    title = infer_title(
        pdf_path=pdf_path,
        metadata_title=metadata_title,
        candidates=candidates,
        lines=lines,
        body_size=body_size,
    )

    if is_bad_metadata_title(title, pdf_path):
        title = pdf_path.stem

    chapters = build_chapters(
        candidates=candidates,
        title=title,
        body_size=body_size,
        max_depth=max_depth,
    )

    toc = {
        "title": title,
        "chapters": chapters,
    }

    if use_ai and candidates:
        try:
            toc = refine_with_gemini(
                toc=toc,
                candidates=candidates,
                model=model,
                ai_retries=ai_retries,
                ai_retry_base_delay=ai_retry_base_delay,
                ai_fallback_models=ai_fallback_models,
                gemini_timeout_seconds=gemini_timeout_seconds,
            )
        except Exception as error:
            if is_ai_configuration_error(error):
                raise

            print(
                f"AI 보정 실패, 구조 기반 목차로 계속 진행합니다: {error}",
                file=sys.stderr,
            )

    toc["chapters"] = postprocess_chapters_for_output(
        chapters=toc.get("chapters", []),
        max_depth=max_depth,
    )

    return toc, candidates


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def iter_pdfs(path: Path):
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {path}")

        return [path]

    if path.is_dir():
        return sorted(
            item for item in path.rglob("*")
            if item.suffix.lower() == ".pdf"
        )

    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")


def process_pdf(pdf_path: Path, args):
    print(f"처리 시작: {pdf_path.name}")

    try:
        toc, candidates = create_toc(
            pdf_path=pdf_path,
            max_pages=args.max_pages,
            min_score=args.min_score,
            max_depth=args.max_depth,
            max_heading_chars=args.max_heading_chars,
            ignore_builtin_toc=args.ignore_builtin_toc,
            include_captions=args.include_captions,
            use_ocr=args.use_ocr,
            use_ai=args.use_ai,
            model=args.model,
            ai_retries=getattr(args, 'ai_retries', DEFAULT_AI_RETRIES),
            ai_retry_base_delay=getattr(args, 'ai_retry_base_delay', DEFAULT_AI_RETRY_BASE_DELAY),
            ai_fallback_models=getattr(args, 'ai_fallback_models', DEFAULT_AI_FALLBACK_MODELS),
            gemini_timeout_seconds=getattr(args, 'gemini_timeout_seconds', DEFAULT_GEMINI_TIMEOUT_SECONDS),
            ocr_workers=getattr(args, 'ocr_workers', 1),
            ocr_zoom=getattr(args, 'ocr_zoom', 2.0),
            ocr_psm=getattr(args, 'ocr_psm', 3),
            ocr_lang=getattr(args, 'ocr_lang', 'kor+eng'),
        )

        output_dir = Path(args.output_dir)
        output_file = output_dir / f"{pdf_path.stem}_toc.json"

        save_json(output_file, toc)

        print(f"완료: {output_file}")
        print(f"  title: {toc.get('title')}")
        print(f"  chapters: {len(toc.get('chapters', []))}개")

        if args.write_candidates:
            candidate_file = output_dir / f"{pdf_path.stem}_candidates.json"
            save_json(candidate_file, candidates)
            print(f"후보 저장: {candidate_file}")

    except Exception as error:
        print(f"실패: {pdf_path.name} / {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="PDF의 시각적 구조를 분석해서 목차 JSON을 생성합니다."
    )

    parser.add_argument("path", nargs="?", default="./data/input")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-pages", type=int, default=0, help="0이면 전체 페이지")
    parser.add_argument("--min-score", type=float, default=7.5)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-heading-chars", type=int, default=80)
    parser.add_argument("--ignore-builtin-toc", action="store_true")
    parser.add_argument("--include-captions", action="store_true")
    parser.add_argument("--write-candidates", action="store_true")
    parser.add_argument("--use-ocr", action="store_true", help="OCR로 PDF를 읽습니다.")
    parser.add_argument("--use-ai", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES, help="Gemini 일시 오류 재시도 횟수")
    parser.add_argument("--ai-retry-base-delay", type=float, default=DEFAULT_AI_RETRY_BASE_DELAY, help="Gemini 재시도 기본 대기 시간(초)")
    parser.add_argument("--ai-fallback-models", default=DEFAULT_AI_FALLBACK_MODELS, help="기본 모델 실패 시 시도할 Gemini 모델 목록, 쉼표로 구분")
    parser.add_argument("--gemini-timeout-seconds", type=float, default=DEFAULT_GEMINI_TIMEOUT_SECONDS, help="Gemini 요청 타임아웃(초)")
    parser.add_argument("--ocr-workers", type=int, default=1, help="OCR 병렬 워커 수 (기본 1)")
    parser.add_argument("--ocr-zoom", type=float, default=2.0, help="페이지 렌더링 배율 (기본 2.0, 낮추면 속도 향상)")
    parser.add_argument("--ocr-psm", type=int, default=3, help="Tesseract --psm 값 (기본 3)")
    parser.add_argument("--ocr-lang", default="kor+eng", help="Tesseract 언어 옵션 (기본 'kor+eng')")

    args = parser.parse_args()

    pdf_files = iter_pdfs(Path(args.path))

    if not pdf_files:
        print("처리할 PDF 파일이 없습니다.")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        process_pdf(pdf_path, args)


if __name__ == "__main__":
    main()