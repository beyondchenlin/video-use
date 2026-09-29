"""Fast unattended video pipeline.

Runs the whole cut in one process without stopping for confirmation:

  1. precut mechanical silence and breath removal with gpu encoding
  2. transcribe the precut video (funasr on gpu for chinese, whisper for english)
  3. build a deterministic cut edl dropping fillers and long silences
  4. render the cut base
  5. re transcribe the base reusing the already loaded model so timestamps match
  6. build sentence subtitles and burn them last

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import auto_edl
import precut
import render
import transcribe


# pick an installed cjk font per operating system for burned subtitles
def cjk_force_style() -> str:
    font = {
        "Windows": "Microsoft YaHei",
        "Darwin": "PingFang SC",
    }.get(platform.system(), "Noto Sans CJK SC")
    return (
        f"FontName={font},FontSize=16,Bold=0,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=90"
    )


# group transcript words into sentence cues on punctuation long gaps or length
def sentence_cues(words: list[dict], cue_gap: float, max_chars: int) -> list[list[dict]]:
    cues: list[list[dict]] = []
    current: list[dict] = []
    for i, w in enumerate(words):
        current.append(w)
        text = "".join(x.get("text", "") for x in current)
        ends = bool(w.get("text")) and w["text"][-1] in "。！？；.!?;"
        long = len(text) >= max_chars
        if not ends and i + 1 < len(words):
            gap = float(words[i + 1]["start"]) - float(w["end"])
            ends = gap >= cue_gap
        if ends or long:
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


# build a sentence aligned srt from a post cut transcript
def build_sentence_srt(
    transcript: dict, srt_path: Path, cue_gap: float = 0.6, max_chars: int = 20
) -> None:
    words = [w for w in transcript.get("words") or [] if w.get("type") == "word"]
    has_cjk = any(
        any("\u4e00" <= ch <= "\u9fff" for ch in (w.get("text") or "")) for w in words
    )
    lines: list[str] = []
    for n, cue in enumerate(sentence_cues(words, cue_gap, max_chars), start=1):
        start = float(cue[0]["start"])
        end = float(cue[-1]["end"])
        if end <= start:
            end = start + 0.4
        if has_cjk:
            text = "".join(w.get("text", "") for w in cue).strip()
        else:
            text = " ".join(w.get("text", "").strip() for w in cue).strip()
        lines.append(str(n))
        lines.append(f"{render._srt_timestamp(start)} --> {render._srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"subtitles -> {srt_path.name} ({len(lines) // 4} cues)")


# run the full fast pipeline and return the final video path
def run_fast(
    video: Path,
    edit_dir: Path,
    engine: str,
    source_name: str = "S",
    fps: str = "30/1",
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

    # 5. re transcribe the base reusing the cached model then build sentence srt
    tr2_path = transcribe.transcribe_one(
        video=base,
        edit_dir=edit_dir,
        engine=engine,
        local_options={},
        funasr_options={},
        force=True,
    )
    tr2 = json.loads(tr2_path.read_text(encoding="utf-8"))
    srt = edit_dir / "master.srt"
    build_sentence_srt(tr2, srt)

    # 6. burn subtitles last into the final video
    final = edit_dir / "final.mp4"
    render.build_final_composite(
        base, [], srt, final, edit_dir,
        force_style=cjk_force_style(), codec=codec,
    )
    return final


# cli entry point that resolves the engine from flags or dotenv and runs the pipeline
def main() -> None:
    ap = argparse.ArgumentParser(description="Fast unattended video pipeline")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--engine", choices=transcribe.ENGINES, default=None)
    ap.add_argument("--fps", default="30/1", help="Output frame rate, default 30")
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

    final = run_fast(video, edit_dir, engine, fps=render.parse_fps(args.fps))
    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"\ndone: {final} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
