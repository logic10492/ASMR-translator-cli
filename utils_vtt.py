# utils_vtt.py
# 简单的 vtt 读写工具。支持读取 whisper 的 segments 写入 VTT，
# 以及反向将 vtt 解析为 cues 列表：[{start, end, text}]
from typing import List, Dict, Union
import re
from pathlib import Path

PathLike = Union[str, Path]


def format_time_vtt(ts: float) -> str:
    """Format seconds (float) to WebVTT timestamp: HH:MM:SS.mmm"""
    try:
        ts_f = float(ts)
    except Exception:
        ts_f = 0.0
    if ts_f < 0:
        ts_f = 0.0
    h = int(ts_f // 3600)
    m = int((ts_f % 3600) // 60)
    s = ts_f % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"  # VTT uses dot as decimal separator


def srt_time(ts: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm (兼容用途)"""
    return format_time_vtt(ts).replace('.', ',')


def write_vtt_from_segments(path: PathLike, segments: List[Dict]) -> None:
    """
    Write a WebVTT file from whisper-style segments.
    Each segment expected to have keys: 'start', 'end', 'text'.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments):
            start = float(seg.get('start', 0.0))
            end = float(seg.get('end', start + 0.5))
            text = str(seg.get('text', '')).strip()
            # Include a numeric cue id for clarity (optional in VTT)
            f.write(f"{idx}\n")
            f.write(f"{format_time_vtt(start)} --> {format_time_vtt(end)}\n")
            f.write(text + "\n\n")


def read_vtt_cues(path: PathLike) -> List[Dict]:
    """
    Read a WebVTT file and return list of cues:
    [{'start': float, 'end': float, 'text': str}, ...]
    This parser is permissive and supports:
      - HH:MM:SS.mmm  (e.g. 00:01:23.456)
      - MM:SS.mmm     (e.g. 01:23.456)
      - decimal separator '.' or ','
    It ignores NOTE blocks and will skip parts without valid timecode.
    """
    p = Path(path)
    raw = p.read_text(encoding='utf-8')

    # remove BOM if present
    if raw.startswith('\ufeff'):
        raw = raw[1:]

    # normalize line endings
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')

    # strip initial WEBVTT header if present (and any header metadata until a blank line)
    raw = re.sub(r'^WEBVTT.*?\n\n', '', raw, count=1, flags=re.DOTALL)

    # split cues by two or more newlines
    parts = re.split(r'\n{2,}', raw.strip())

    cues: List[Dict] = []
    # regex to capture start and end (supports HH:MM:SS.mmm or MM:SS.mmm with . or ,)
    time_re = re.compile(
        r'(?P<start>\d{2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})\s*-->\s*'
        r'(?P<end>\d{2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})'
    )

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # ignore NOTE blocks
        if part.upper().startswith('NOTE'):
            continue

        lines = part.splitlines()
        # find the line containing '-->'
        time_line_idx = None
        for i, ln in enumerate(lines):
            if '-->' in ln:
                time_line_idx = i
                break
        if time_line_idx is None:
            # no timecode in this block
            continue

        time_line = lines[time_line_idx]
        m = time_re.search(time_line)
        if not m:
            # try to normalize tabs/spaces and retry
            tl = time_line.replace('\t', ' ').strip()
            m = time_re.search(tl)
            if not m:
                continue

        start = parse_time(m.group('start'))
        end = parse_time(m.group('end'))

        # text is all lines except numeric id lines and the time line
        text_lines = []
        for j, ln in enumerate(lines):
            if j == time_line_idx:
                continue
            # skip numeric id-only lines
            if re.fullmatch(r'\d+', ln.strip()):
                continue
            # skip some header-like lines (STYLE/REGION) — keep content lines
            if ln.strip().upper().startswith('STYLE') or ln.strip().upper().startswith('REGION'):
                continue
            text_lines.append(ln)
        text = '\n'.join(text_lines).strip()
        cues.append({'start': start, 'end': end, 'text': text})

    return cues


def parse_time(tstr: str) -> float:
    """
    Parse a VTT/SRT time string to seconds (float).
    Supports:
      - HH:MM:SS.mmm
      - MM:SS.mmm
    Accepts '.' or ',' as decimal separators.
    """
    if not tstr:
        return 0.0
    t = tstr.strip().replace(',', '.')
    parts = t.split(':')
    try:
        parts_f = [float(p) for p in parts]
    except Exception:
        return 0.0

    if len(parts_f) == 3:
        return parts_f[0] * 3600 + parts_f[1] * 60 + parts_f[2]
    elif len(parts_f) == 2:
        return parts_f[0] * 60 + parts_f[1]
    elif len(parts_f) == 1:
        return parts_f[0]
    else:
        return 0.0


def write_vtt_cues(path: PathLike, cues: List[Dict]) -> None:
    """
    Write a list of cues (dicts with 'start','end','text') to a .vtt file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for idx, c in enumerate(cues):
            start = float(c.get('start', 0.0))
            end = float(c.get('end', start + 0.5))
            text = str(c.get('text', '')).strip()
            f.write(f"{idx}\n")
            f.write(f"{format_time_vtt(start)} --> {format_time_vtt(end)}\n")
            f.write(text + "\n\n")


if __name__ == "__main__":
    # quick test
    sample = [
        {'start': 0.0, 'end': 1.23, 'text': 'こんにちは'},
        {'start': 1.5, 'end': 2.5, 'text': 'はぁ...'},
    ]
    test_path = Path('test_script.vtt')
    write_vtt_from_segments(test_path, sample)
    parsed = read_vtt_cues(test_path)
    print(parsed)
