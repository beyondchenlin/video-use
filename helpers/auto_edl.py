"""Build a cut EDL mechanically from a word-level transcript.

This is the deterministic brain of the fast unattended route. It drops
standalone filler words and long silences, keeps natural short pauses, and
never cuts inside a word. No LLM or network call is involved; semantic
editing still happens in the conversational route.

Usage:
    python helpers/auto_edl.py <transcript.json> --source-name S -o edl.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# standalone chinese filler characters and their repeats carry no meaning
FILLER_ZH = {"嗯", "呃", "哦", "啊", "呀", "诶", "唉"}

# standalone english filler words carry no meaning
FILLER_EN = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm"}

# keep natural pauses up to this gap and split ranges at longer silences
JOIN_GAP = 0.35

# pad kept words by this much so cuts never clip speech
EDGE_PAD = 0.06

# merge any range shorter than this back into a neighbor
MIN_SEG = 0.25

_STRIP = " ，。！？、；：,.!?;:…—\"'“”‘’（）()【】《》\n"


# strip surrounding punctuation for filler comparison
def core_text(text: str) -> str:
    return (text or "").strip(_STRIP)


# true when a word is a standalone filler to remove
def is_filler(word: dict) -> bool:
    core = core_text(word.get("text", ""))
    if not core:
        return False
    if core.lower() in FILLER_EN:
        return True
    # a run made only of chinese filler characters is also removable
    if all(ch in FILLER_ZH for ch in core):
        return True
    return False


# merge short ranges into a neighbor so the edit does not ship tiny fragments
def absorb_tiny(segments: list[list[float]]) -> list[list[float]]:
    if not segments:
        return segments
    result = [segments[0]]
    for seg in segments[1:]:
        if seg[1] - seg[0] < MIN_SEG:
            result[-1][1] = seg[1]
        else:
            result.append(seg)
    # fold a tiny first range into the second
    if len(result) > 1 and result[0][1] - result[0][0] < MIN_SEG:
        result[1][0] = result[0][0]
        result = result[1:]
    return result


# turn kept words into padded intervals split at filler words and long gaps
def build_intervals(words: list[dict]) -> list[tuple[float, float]]:
    all_words = [w for w in words if w.get("type") == "word"]
    kept = [(i, w) for i, w in enumerate(all_words) if not is_filler(w)]
    if not kept:
        return []

    # merge only words that were truly adjacent with a short natural gap
    merged: list[list[float]] = []
    prev_idx: int | None = None
    for idx, w in kept:
        start = max(0.0, float(w["start"]) - EDGE_PAD)
        end = float(w["end"]) + EDGE_PAD
        adjacent = prev_idx is not None and idx == prev_idx + 1
        if merged and adjacent and start - merged[-1][1] <= JOIN_GAP:
            merged[-1][1] = end
        else:
            merged.append([start, end])
        prev_idx = idx

    merged = absorb_tiny(merged)
    return [(a, b) for a, b in merged]


# build the edl payload for one source from a transcript payload
def build_edl(
    transcript: dict, source_name: str, source_video: str, grade: str | None = None
) -> dict:
    intervals = build_intervals(transcript.get("words") or [])
    ranges = [
        {
            "source": source_name,
            "start": round(a, 3),
            "end": round(b, 3),
            "beat": "",
            "quote": "",
            "reason": "auto fast cut",
        }
        for a, b in intervals
    ]
    edl = {
        "version": 1,
        "sources": {source_name: source_video},
        "ranges": ranges,
        "total_duration_s": round(sum(b - a for a, b in intervals), 3),
    }
    if grade:
        edl["grade"] = grade
    return edl


# cli entry point that reads a transcript and writes the edl
def main() -> None:
    ap = argparse.ArgumentParser(description="Build a fast cut EDL from a transcript")
    ap.add_argument("transcript", type=Path, help="Word-level transcript JSON")
    ap.add_argument("--source-name", default="S", help="Key used in EDL sources/ranges")
    ap.add_argument(
        "--source-video", default="",
        help="Absolute source video path for EDL sources (defaults to empty)",
    )
    ap.add_argument("--grade", default=None, help="Optional grade preset or filter")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    edl = build_edl(
        transcript, args.source_name, args.source_video, grade=args.grade
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"edl -> {args.output} ({len(edl['ranges'])} ranges, "
        f"{edl['total_duration_s']:.2f}s kept)"
    )


if __name__ == "__main__":
    main()
