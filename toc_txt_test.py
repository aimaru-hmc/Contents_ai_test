from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNK = ROOT / 'data/output/input_chunks/2_5-8_2021_gemma_20260708_153404/chunk_1_pages_1-39.json'
REFERENCE = ROOT / 'data/output/2편5-8장_2021_openai_gpt-5.5_20260710_144043_toc_39.json'
PARSED = ROOT / 'data/output/parsed_pdfs/2_5-8_2021_gemma_20260708_153404_parsed_pdf.txt'
TOC_OUTPUT_DIR = ROOT / 'data/output/toc_txt/toc'
LAYOUT_OUTPUT_DIR = ROOT / 'data/output/toc_txt/layout'
PAGE_RE = re.compile(r'^\[PAGE\s+(\d+)\]$')
LINE_RE = re.compile(
    r'^\[L s=(?P<s>[\d.]+) r=(?P<r>[\d.]+) x=(?P<x>-?[\d.]+) '
    r'y=(?P<y>-?[\d.]+) b=(?P<b>[01]) i=(?P<i>[01]) '
    r'f=(?P<f>[^]]+)\]\s*(?P<t>.*)$'
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


def clean(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip()


def norm(value):
    return re.sub(r'\s+', '', clean(value)).replace('–', '-').replace('—', '-')


def token_shape(token):
    """문자 내용을 고정하지 않고 유니코드 문자 종류로 시작 형태를 일반화한다."""
    result = []
    previous = None
    for char in clean(token):
        category = unicodedata.category(char)
        if category.startswith('N'):
            value = '#'
        elif category.startswith('L'):
            value = 'A'
        else:
            value = char
        if value != previous or value not in ('#', 'A'):
            result.append(value)
        previous = value
    return ''.join(result)


def tokens_with_offsets(text):
    return [(m.group(), m.start()) for m in re.finditer(r'\S+', text)]


def parse_layout(text):
    page = order = 0
    result = []
    for raw in text.splitlines():
        page_match = PAGE_RE.match(raw.strip())
        if page_match:
            page, order = int(page_match.group(1)), 0
            continue
        match = LINE_RE.match(raw.strip())
        if not match or not page:
            continue
        order += 1
        values = match.groupdict()
        title = clean(values['t'])
        if title:
            result.append(Line(page, order, float(values['s']), float(values['r']),
                float(values['x']), float(values['y']), int(values['b']),
                int(values['i']), values['f'], title))
    if not result:
        raise ValueError('레이아웃 줄을 찾지 못했습니다.')
    return result


def match_reference_title(title, page_lines):
    wanted = norm(title)
    ranked = []
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
        ranked.append((score, len(actual), line))
    return max(ranked, key=lambda item: (item[0], item[1], item[2].s))[2] if ranked else None


def learned_start_shape(reference_title, matched_line):
    wanted = norm(reference_title)
    for token, _ in tokens_with_offsets(matched_line.text):
        if norm(token) and norm(token) in wanted:
            return token_shape(token)
    tokens = tokens_with_offsets(reference_title)
    return token_shape(tokens[0][0]) if tokens else ''


def learn_layout_levels(reference, chunk_lines, reference_path, chunk_path):
    pages = defaultdict(list)
    for line in chunk_lines:
        pages[line.page].append(line)

    observations = []
    unmatched = []
    for item in reference.get('chapters', []):
        page = int(item['page'])
        line = match_reference_title(item['chapter'], pages[page])
        if line is None:
            unmatched.append({'level': int(item['level']), 'chapter': item['chapter'], 'page': page})
            continue
        observations.append({
            'level': int(item['level']),
            'layout': {'s': line.s, 'r': line.r, 'x': line.x, 'y': line.y,
                       'b': line.b, 'i': line.i, 'f': line.f},
            'start_shape': learned_start_shape(item['chapter'], line),
            'example': item['chapter'],
        })
    if not observations:
        raise ValueError('기준 TOC와 청크에서 일치하는 제목을 찾지 못했습니다.')

    grouped = {}
    for obs in observations:
        layout = obs['layout']
        key = (obs['level'], layout['s'], layout['r'], layout['b'], layout['i'], layout['f'])
        group = grouped.setdefault(key, {'level': obs['level'], 's': layout['s'], 'r': layout['r'],
            'b': layout['b'], 'i': layout['i'], 'f': layout['f'], 'xs': [], 'ys': [],
            'start_shapes': Counter(), 'examples': [], 'reference_count': 0})
        group['xs'].append(layout['x']); group['ys'].append(layout['y'])
        group['start_shapes'][obs['start_shape']] += 1
        group['examples'].append(obs['example']); group['reference_count'] += 1

    rules = []
    for group in grouped.values():
        rules.append({'level': group['level'], 's': group['s'], 'r': group['r'],
            'b': group['b'], 'i': group['i'], 'f': group['f'],
            'x_min': min(group['xs']), 'x_max': max(group['xs']),
            'y_min': min(group['ys']), 'y_max': max(group['ys']),
            'start_shapes': sorted(group['start_shapes']),
            'reference_count': group['reference_count'], 'examples': group['examples'][:8]})
    rules.sort(key=lambda rule: (rule['level'], -rule['s'], rule['f']))
    return {'source_reference': str(reference_path), 'source_chunk': str(chunk_path),
        'matching_priority': ['S/R/B/I/F', 'learned start shape', 'X', 'Y'],
        'rules': rules, 'unmatched_reference_titles': unmatched}


def rule_distance(line, rule):
    if (line.b, line.i, line.f) != (int(rule['b']), int(rule['i']), rule['f']):
        return math.inf
    if abs(line.s - float(rule['s'])) > .02 or abs(line.r - float(rule['r'])) > .02:
        return math.inf
    x_mid = (float(rule['x_min']) + float(rule['x_max'])) / 2
    return abs(line.s-float(rule['s'])) + .2*abs(line.r-float(rule['r'])) + .002*abs(line.x-x_mid)


def candidate_for_rule(line, rule):
    shapes = set(rule.get('start_shapes', []))
    for token, offset in tokens_with_offsets(line.text):
        if token_shape(token) in shapes:
            return clean(line.text[offset:])
    return None


def generate_toc(lines, title, learned):
    result = []
    for line in lines:
        matches = []
        for rule in learned['rules']:
            distance = rule_distance(line, rule)
            if math.isinf(distance):
                continue
            candidate = candidate_for_rule(line, rule)
            if candidate:
                matches.append((distance, rule, candidate))
        if not matches:
            continue
        distance, rule, candidate = min(matches, key=lambda item: item[0])
        result.append({'level': int(rule['level']), 'chapter': candidate, 'page': line.page,
            '_order': line.order,
            'level_reason': f"Matched learned layout (S={line.s:g}, R={line.r:g}, X={line.x:g}, Y={line.y:g}, B={line.b}, I={line.i}, F={line.f})."})

    result.sort(key=lambda item: (item['page'], item['_order']))
    unique, seen = [], set()
    for item in result:
        key = (item['page'], norm(item['chapter']))
        if key in seen:
            continue
        seen.add(key); item.pop('_order'); unique.append(item)
    return {'title': title, 'chapters': unique}


def timestamped_output(explicit, default_dir, default_stem, label, timestamp):
    """명시 경로에도 생성시각을 붙이고, 같은 초에 재실행해도 덮어쓰지 않는다."""
    if explicit:
        explicit = Path(explicit)
        if explicit.suffix:
            directory = explicit.parent
            stem = explicit.stem
            suffix = explicit.suffix
        else:
            directory = explicit
            stem = f'{default_stem}_{label}'
            suffix = '.json'
    else:
        directory = default_dir
        stem = f'{default_stem}_{label}'
        suffix = '.json'
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f'{stem}_{timestamp}{suffix}'
    sequence = 2
    while candidate.exists():
        candidate = directory / f'{stem}_{timestamp}_{sequence}{suffix}'
        sequence += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description='기준 파일에서 레이아웃-레벨을 학습해 LLM 없이 전체 TOC 생성')
    parser.add_argument('--chunk', type=Path, default=CHUNK,
                        help='학습 구간 레이아웃 청크 JSON 파일')
    parser.add_argument('--reference', type=Path, default=REFERENCE,
                        help='학습 기준 OpenAI TOC JSON 파일')
    parser.add_argument('--parsed', type=Path, default=PARSED,
                        help='전체 parsed PDF 텍스트 파일')
    parser.add_argument('--layout-level-output', type=Path,
                        help='레이아웃 결과 파일명 또는 저장 폴더(생성시간 자동 추가)')
    parser.add_argument('--output', type=Path,
                        help='TOC 결과 파일명 또는 저장 폴더(생성시간 자동 추가)')
    parser.add_argument('--title')
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    chunk = json.loads(args.chunk.read_text(encoding='utf-8'))
    learned = learn_layout_levels(reference, parse_layout(chunk['text']), args.reference, args.chunk)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    layout_output = timestamped_output(args.layout_level_output, LAYOUT_OUTPUT_DIR,
                                       args.parsed.stem, 'layout_levels', timestamp)
    layout_output.write_text(json.dumps(learned, ensure_ascii=False, indent=4)+'\n', encoding='utf-8')

    learned = json.loads(layout_output.read_text(encoding='utf-8'))
    toc = generate_toc(parse_layout(args.parsed.read_text(encoding='utf-8')),
                       args.title or reference.get('title') or args.parsed.stem, learned)
    output = timestamped_output(args.output, TOC_OUTPUT_DIR,
                                args.parsed.stem, 'toc', timestamp)
    output.write_text(json.dumps(toc, ensure_ascii=False, indent=4)+'\n', encoding='utf-8')
    print(f'Layout levels: {layout_output}')
    print(f'Unmatched reference titles: {len(learned["unmatched_reference_titles"])}')
    print(f'Created: {output}\nTOC entries: {len(toc["chapters"])}')


if __name__ == '__main__':
    main()
