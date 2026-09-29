"""Fast unattended video pipeline.

Runs the whole cut in one process without stopping for confirmation:

  1. precut mechanical silence and breath removal with gpu encoding
  2. transcribe the precut video (funasr on gpu for chinese, whisper for english)
  3. build a deterministic cut edl dropping fillers and long silences
  4. render the cut base
  5. optionally re transcribe the base and check both passes for consistency
  6. build sentence subtitles from the first pass timeline or the recheck pass
  7. burn subtitles last and normalize loudness to the social target

The conversational route in SKILL.md stays available for reviewed work; this is
the fast lane modeled on the FunClip stage pipeline.

Usage:
    python helpers/fast_pipeline.py <video>
    python helpers/fast_pipeline.py <video> --engine funasr --fps 30
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import auto_edl
import precut
import render
import transcribe


# pick an installed cjk font per operating system for burned subtitles
def cjk_force_style(margin: int = 90) -> str:
    font = {
        "Windows": "Microsoft YaHei",
        "Darwin": "PingFang SC",
    }.get(platform.system(), "Noto Sans CJK SC")
    return (
        f"FontName={font},FontSize=16,Bold=0,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV={margin}"
    )


# sentence punctuation that closes a cue during normal flow
CUE_BREAK_PUNCT = "。！？；.!?;"

# punctuation that can close a cue once the length cap is reached
OVERRUN_PUNCT = "。！？；，、,.!?;:"


# group transcript words into sentence cues on punctuation long gaps or length
def sentence_cues(
    words: list[dict], cue_gap: float, max_chars: int, max_overrun: int | None = None
) -> list[list[dict]]:
    """Group words into cues preferring punctuation over a hard length cut.

    A cue ends on sentence punctuation or a long gap. When the length cap is
    reached mid clause the cue extends to the next punctuation mark within the
    overrun budget. If none is that close it falls back to the last punctuation
    before the cap and then to the cap itself.
    """
    if max_overrun is None:
        max_overrun = max(4, max_chars // 5)
    hard_limit = max_chars + max_overrun

    cues: list[list[dict]] = []
    current: list[dict] = []
    cap_words = 0
    for i, w in enumerate(words):
        current.append(w)
        word_text = w.get("text") or ""
        text = "".join(x.get("text", "") for x in current)
        ends = bool(word_text) and word_text[-1] in CUE_BREAK_PUNCT
        if not ends and i + 1 < len(words):
            gap = float(words[i + 1]["start"]) - float(w["end"])
            ends = gap >= cue_gap
        if ends:
            cues.append(current)
            current = []
            cap_words = 0
            continue
        if len(text) < max_chars:
            continue
        if not cap_words:
            cap_words = len(current)
        # a punctuation mark inside the overrun budget can close the cue
        if word_text and word_text[-1] in OVERRUN_PUNCT and len(text) <= hard_limit:
            cues.append(current)
            current = []
            cap_words = 0
            continue
        if len(text) > hard_limit:
            # fall back to the last punctuation before the cap then to the cap
            cut = -1
            for k in range(min(cap_words, len(current)) - 1, -1, -1):
                prior = current[k].get("text") or ""
                if prior and prior[-1] in OVERRUN_PUNCT:
                    cut = k
                    break
            if cut < 0:
                cut = cap_words - 1
            cues.append(current[: cut + 1])
            current = current[cut + 1:]
            cap_words = 0
    if current:
        cues.append(current)
    return cues


# turn words on one timeline into cues carrying start end and text
def words_to_cues(
    words: list[dict], cue_gap: float = 0.6, max_chars: int = 20
) -> list[dict]:
    has_cjk = any(
        any("\u4e00" <= ch <= "\u9fff" for ch in (w.get("text") or "")) for w in words
    )
    cues: list[dict] = []
    for cue in sentence_cues(words, cue_gap, max_chars):
        start = float(cue[0]["start"])
        end = float(cue[-1]["end"])
        if end <= start:
            end = start + 0.4
        if has_cjk:
            text = "".join(w.get("text", "") for w in cue).strip()
        else:
            text = " ".join(w.get("text", "").strip() for w in cue).strip()
        cues.append({"start": start, "end": end, "text": text})
    return cues


# write timed cues to an srt file
def write_srt(cues: list[dict], srt_path: Path) -> None:
    lines: list[str] = []
    for n, c in enumerate(cues, start=1):
        lines.append(str(n))
        lines.append(f"{render._srt_timestamp(c['start'])} --> {render._srt_timestamp(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"subtitles -> {srt_path.name} ({len(cues)} cues)")


# build a sentence aligned srt from a post cut transcript
def build_sentence_srt(
    transcript: dict, srt_path: Path, cue_gap: float = 0.6, max_chars: int = 20
) -> None:
    words = [w for w in transcript.get("words") or [] if w.get("type") == "word"]
    write_srt(words_to_cues(words, cue_gap, max_chars), srt_path)


# map first pass words onto the output timeline using the edl offsets
def map_cues_to_output(
    transcript: dict, edl: dict, cue_gap: float = 0.6, max_chars: int = 20
) -> list[dict]:
    """Rule 5: output = word.start - segment_start + segment_offset.

    Fillers the EDL dropped are dropped here too so the text matches the cut.
    """
    words = [
        w for w in transcript.get("words") or []
        if w.get("type") == "word" and not auto_edl.is_filler(w)
    ]
    cues: list[dict] = []
    seg_offset = 0.0
    for r in edl.get("ranges") or []:
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_words = [
            w for w in words
            if float(w["end"]) > seg_start and float(w["start"]) < seg_end
        ]
        for c in words_to_cues(seg_words, cue_gap, max_chars):
            start = max(seg_start, c["start"]) - seg_start + seg_offset
            end = min(seg_end, c["end"]) - seg_start + seg_offset
            if end <= start:
                end = start + 0.4
            cues.append({"start": start, "end": end, "text": c["text"]})
        seg_offset += seg_end - seg_start
    return cues


# strip spaces and punctuation so two passes can be compared loosely
def normalize_cue_text(text: str) -> str:
    return re.sub(r"[\s，。！？；、,.!?;:…—\"'“”‘’（）()【】《》]", "", text or "")


# log sentences that differ between the first pass mapping and the recheck pass
def compare_cues(mapped: list[dict], recheck: list[dict], limit: int = 5) -> list[dict]:
    diffs: list[dict] = []
    for rc in recheck:
        best = None
        best_overlap = 0.0
        for mc in mapped:
            overlap = min(mc["end"], rc["end"]) - max(mc["start"], rc["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best = mc
        if best is None:
            continue
        if normalize_cue_text(best["text"]) != normalize_cue_text(rc["text"]):
            diffs.append(
                {"start": rc["start"], "first": best["text"], "recheck": rc["text"]}
            )
    if diffs:
        print(f"  asr consistency: {len(diffs)} sentence(s) differ between the two passes")
        for d in diffs[:limit]:
            print(f"    {d['start']:6.2f}s  first pass: {d['first']}")
            print(f"            recheck:    {d['recheck']}")
    else:
        print("  asr consistency: both passes agree")
    return diffs


# run the full fast pipeline and return the final video path
def run_fast(
    video: Path,
    edit_dir: Path,
    engine: str,
    source_name: str = "S",
    fps: str = "30/1",
    recheck: bool = True,
    caption_margin: int = 90,
) -> Path:
    edit_dir.mkdir(parents=True, exist_ok=True)
    use_gpu = precut.nvidia_available()
    codec = "h264_nvenc" if use_gpu else "libx264"

    # 1. mechanical pre cut
    precut_video = edit_dir / (video.stem + ".precut.mp4")
    precut.precut(
        video,
        precut_video,
        margin="0.2sec",
        bitrate="8000k" if use_gpu else None,
        use_gpu=use_gpu,
        keep_intermediate=False,
    )

    # 2. transcribe the precut video
    tr1_path = transcribe.transcribe_one(
        video=precut_video,
        edit_dir=edit_dir,
        engine=engine,
        local_options={},
        funasr_options={},
    )
    tr1 = json.loads(tr1_path.read_text(encoding="utf-8"))

    # 3. deterministic cut edl against the precut video
    edl = auto_edl.build_edl(tr1, source_name, str(precut_video.resolve()))
    edl_path = edit_dir / "edl.json"
    edl_path.write_text(
        json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. render the cut base without subtitles
    segment_paths = render.extract_all_segments(
        edl, edit_dir, preview=False, fps=fps, codec=codec
    )
    base = edit_dir / "base.mp4"
    render.concat_segments(segment_paths, base, edit_dir)

    # 5. map the first pass onto the output timeline and optionally recheck
    first_pass_cues = map_cues_to_output(tr1, edl)
    srt = edit_dir / "master.srt"
    if recheck:
        tr2_path = transcribe.transcribe_one(
            video=base,
            edit_dir=edit_dir,
            engine=engine,
            local_options={},
            funasr_options={},
            force=True,
        )
        tr2 = json.loads(tr2_path.read_text(encoding="utf-8"))
        tr2_words = [w for w in tr2.get("words") or [] if w.get("type") == "word"]
        recheck_cues = words_to_cues(tr2_words)
        compare_cues(first_pass_cues, recheck_cues)
        write_srt(recheck_cues, srt)
    else:
        write_srt(first_pass_cues, srt)

    # 6. burn subtitles last into the final video and normalize loudness
    final = edit_dir / "final.mp4"
    render.build_final_composite(
        base, [], srt, final, edit_dir,
        force_style=cjk_force_style(caption_margin), codec=codec, loudnorm=True,
    )
    return final


# cli entry point that resolves the engine from flags or dotenv and runs the pipeline
def main() -> None:
    ap = argparse.ArgumentParser(description="Fast unattended video pipeline")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--engine", choices=transcribe.ENGINES, default=None)
    ap.add_argument("--fps", default="30/1", help="Output frame rate, default 30")
    ap.add_argument(
        "--no-recheck",
        action="store_true",
        help="Skip the post-cut re-transcription and build subtitles from the "
             "first pass mapped onto the output timeline (less accurate text).",
    )
    ap.add_argument(
        "--caption-margin",
        type=int,
        default=90,
        help="Subtitle MarginV in libass units, default 90 (platform safe zone).",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    env = transcribe.load_env()
    engine = args.engine
    if engine is None:
        engine, source = transcribe.resolve_engine(None, env)
    print(f"engine: {engine}", flush=True)

    try:
        final = run_fast(
            video, edit_dir, engine,
            fps=render.parse_fps(args.fps),
            recheck=not args.no_recheck,
            caption_margin=args.caption_margin,
        )
    except ValueError as exc:
        sys.exit(str(exc))
    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"\ndone: {final} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
