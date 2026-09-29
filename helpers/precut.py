"""Mechanically pre-cut a source before transcription.

Bakes phone rotation into the frames and removes silence plus breath gaps with
auto-editor so the expensive ASR and LLM stages only process the parts that
stay in the edit. This is the fast deterministic stage; semantic cutting still
happens later from the transcript.

Usage:
    python helpers/precut.py <video> -o <dir or file>
    python helpers/precut.py <video> --margin 0.2sec
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# probe coded dimensions rotation pixel format audio presence and duration
def probe_video(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries",
         "format=duration:stream=codec_type,width,height,pix_fmt:stream_side_data=rotation",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    info = {
        "width": 0, "height": 0, "rotation": 0,
        "pix_fmt": "", "has_audio": False, "duration": 0.0,
    }
    for st in streams:
        ctype = st.get("codec_type")
        if ctype == "video" and not info["width"]:
            info["width"] = int(st.get("width") or 0)
            info["height"] = int(st.get("height") or 0)
            info["pix_fmt"] = st.get("pix_fmt") or ""
            for sd in st.get("side_data_list") or []:
                if sd.get("rotation") is not None:
                    info["rotation"] = int(round(float(sd["rotation"])))
                    break
        elif ctype == "audio":
            info["has_audio"] = True
    try:
        info["duration"] = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        info["duration"] = 0.0
    return info


# report whether an nvidia gpu plus driver is reachable so nvenc can be used
def nvidia_available() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


# return true when auto editor would drop orientation or the pixel format is unsafe
def needs_normalize(info: dict) -> bool:
    if info["rotation"] % 360 in (90, 270):
        return True
    if info["pix_fmt"] and info["pix_fmt"] != "yuv420p":
        return True
    return False


# bake rotation into the frames and normalize to a widely compatible mp4
def normalize_video(video: Path, out: Path, use_gpu: bool, has_audio: bool) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video)]
    if use_gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4",
                "-profile:v", "high", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-pix_fmt", "yuv420p"]
    cmd += ["-metadata:s:v:0", "rotate="]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    cmd += ["-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


# cut silence and breath gaps keeping the sounded parts at full speed
def cut_silence(
    video: Path, out: Path, use_gpu: bool, margin: str, bitrate: str | None
) -> None:
    cmd = [
        sys.executable, "-m", "auto_editor", str(video),
        "-o", str(out),
        "--margin", margin,
        "--silent-speed", "99999",
        "--sounded-speed", "1.0",
        "--no-open", "--quiet",
    ]
    if use_gpu:
        cmd += ["--video-codec", "h264_nvenc"]
    if bitrate:
        cmd += ["--video-bitrate", bitrate]
    subprocess.run(cmd, check=True)


# normalize orientation if needed then run auto editor and return the cut file
def precut(
    video: Path, output: Path, margin: str, bitrate: str | None,
    use_gpu: bool, keep_intermediate: bool
) -> Path:
    info = probe_video(video)
    work = video
    intermediate: Path | None = None
    if needs_normalize(info):
        intermediate = output.parent / (video.stem + ".oriented.mp4")
        normalize_video(video, intermediate, use_gpu, info["has_audio"])
        work = intermediate
    cut_silence(work, output, use_gpu, margin, bitrate)
    if intermediate is not None and not keep_intermediate:
        intermediate.unlink(missing_ok=True)
    return output


# cli entry point that pre cuts one video and reports the output path
def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-cut a video before transcription")
    ap.add_argument("video", help="Input video file")
    ap.add_argument(
        "-o", "--output", default=None,
        help="Output file or directory (default: alongside the input as <name>.precut.mp4)",
    )
    ap.add_argument("--margin", default="0.2sec", help="Keep this much around loud parts")
    ap.add_argument(
        "--bitrate", default="8000k",
        help="Video bitrate for the cut output, empty to disable",
    )
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU encoding")
    ap.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep the orientation normalized intermediate file",
    )
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        sys.exit(f"no input video at {video}")

    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir() or args.output.endswith(("\\", "/")):
            out_path = out_path / (video.stem + ".precut.mp4")
    else:
        out_path = video.with_name(video.stem + ".precut.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_gpu = not args.no_gpu and nvidia_available()
    bitrate = args.bitrate or None
    result = precut(video, out_path, args.margin, bitrate, use_gpu, args.keep_intermediate)
    print(f"precut -> {result}")


if __name__ == "__main__":
    main()
