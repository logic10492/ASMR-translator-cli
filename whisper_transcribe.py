#!/usr/bin/env python3
"""
批量将 inputs/*.wav 用 Whisper large-v3 转成日语 WebVTT。
- 可通过 --device 指定 'cpu' 或 'cuda'
- 每个 wav 会生成同名 .vtt
- 自动合并过短字幕行（适合 ASMR）
- 自动合并喘息为占位字幕 [BREATH]
- 音频切片功能：
    * 利用SEGMENT_LENGTH参数控制每片长度，默认30秒
    * 利用OVERLAP参数控制重叠长度，默认5秒
    * VTT 最终合并去重（还需要改进）
- 临时切片存放在 temp/，处理完一个文件后自动清理
- 后续将临时切片改为numpy直接输入到whisper,减少文件io
"""
import os
import argparse
from pathlib import Path
import whisper
from utils_vtt import write_vtt_from_segments
from tqdm import tqdm
import re
from pydub import AudioSegment
import shutil

INPUT_DIR = Path("inputs")
SCRIPT_DIR = Path("script_jp")
OUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

# 合并短段阈值（秒）
MERGE_SHORT_THRESHOLD = 1.0  # 1秒以内的短字幕

# 音频切片参数
SEGMENT_LENGTH = 30 * 1000  # 单位毫秒
OVERLAP = 5 * 1000          # 重叠 5 秒


def merge_short_segments(segments):
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        duration = seg['end'] - seg['start']
        if duration < MERGE_SHORT_THRESHOLD:
            prev['end'] = seg['end']
            prev['text'] = (prev['text'].strip() + ' ' + seg['text'].strip()).strip()
        else:
            merged.append(seg)
    return merged


def slice_audio(audio: AudioSegment):
    """
    将音频切成多片，每片 SEGMENT_LENGTH，重叠 OVERLAP 毫秒
    返回 [(start_ms, end_ms, AudioSegment), ...]
    """
    slices = []
    length = len(audio)
    start = 0
    while start < length:
        end = start + SEGMENT_LENGTH
        if start != 0:
            start -= OVERLAP  # 重叠前 5 秒
        end = min(end, length)
        slices.append((start, end, audio[start:end]))
        start += SEGMENT_LENGTH
    return slices


def transcribe_slice(model, audio_slice: AudioSegment, slice_idx, wav_stem, language='ja'):
    """
    whisper在长文件输入时，当某个段落空白过长，会认为已经结束，导致后续内容无法识别，所以引入切片机制
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TEMP_DIR / f"{wav_stem}_slice{slice_idx}.wav"
    audio_slice.export(tmp_path, format="wav")
    result = model.transcribe(str(tmp_path), language=language, task="transcribe", beam_size=5)
    segments = result.get('segments', [])
    return segments


def adjust_segments_for_overlap(segments, slice_start_ms, is_first_slice):
    """
    将片段时间戳调整为原始音频时间
    并去掉重叠部分（非首片段开头的 OVERLAP）
    """
    adjusted = []
    offset = slice_start_ms / 1000  # 毫秒转秒
    for seg in segments:
        start, end = seg['start'] + offset, seg['end'] + offset
        if not is_first_slice and start < offset + OVERLAP / 1000:
            start = offset + OVERLAP / 1000
            if start >= end:
                continue
        adjusted.append({'start': start, 'end': end, 'text': seg['text']})
    return adjusted


def transcribe_file(model, wav_path, language='ja'):
    audio = AudioSegment.from_file(wav_path)
    slices = slice_audio(audio)
    all_segments = []
    for i, (start_ms, end_ms, audio_slice_seg) in enumerate(slices):
        segments = transcribe_slice(model, audio_slice_seg, i, wav_path.stem, language)
        segments = adjust_segments_for_overlap(segments, start_ms, is_first_slice=(i == 0))
        all_segments.extend(segments)
    all_segments = merge_short_segments(all_segments)
    return all_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help="'cpu' 或 'cuda'")
    parser.add_argument('--model', default='large-v3', help='Whisper 模型')
    args = parser.parse_args()

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading Whisper model {args.model} on {args.device} ...")
    model = whisper.load_model(args.model, device=args.device)

    wavs = sorted(INPUT_DIR.glob('*.wav'))
    if not wavs:
        print('no wavs found in inputs/')
        return

    for wav in tqdm(wavs, desc='transcribing wavs'):
        segments = transcribe_file(model, wav, language='ja')
        out_vtt = SCRIPT_DIR / (wav.stem + '.vtt')
        write_vtt_from_segments(out_vtt, segments)
        print(f'written {out_vtt}')

        # 清理 temp 文件夹
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

    print('\nAll done. Whisper transcription complete.')


if __name__ == '__main__':
    main()
