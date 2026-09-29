# Fast route

An opt-in, unattended lane that cuts a talking-head video end to end without
stopping for review. It complements the conversational process; it does not
replace it. Use the fast route when the user wants speed and accepts
mechanical editing; use the conversational route whenever judgment, retake
selection, or careful wording matters.

## Why it is fast

Four ideas, taken from the FunClip stage pipeline:

1. **Mechanical work first.** Silence and breath are removed deterministically
   with `auto-editor` before any model runs. This needs no AI and is repeatable.
2. **Everything on the GPU.** Transcoding and the final encode use `h264_nvenc`;
   FunASR runs on CUDA when it is available.
3. **Re-transcribe after the cut.** Subtitles are built from the edited video,
   reusing the already loaded model, so they are always in sync.
4. **One process, model loaded once.** The ASR model is the expensive part;
   loading it once and reusing it for both passes avoids a second cold start.

## Steps

1. `precut.py` — probe rotation and pixel format. A phone clip with a
   quarter-turn rotation (or non-`yuv420p` pixels) is first re-encoded with
   ffmpeg autorotate so the orientation is baked in. `auto-editor` then removes
   silence/breath. Normal sources skip the extra pass.
2. `transcribe.py` — word-level transcript of the precut video (FunASR for
   Chinese, Whisper for English).
3. `auto_edl.py` — deterministic EDL: drop standalone filler words and long
   silences, keep natural short pauses, pad word edges, merge tiny fragments.
4. `render.py` — per-segment extract and lossless concat into `base.mp4`.
5. Re-transcribe `base.mp4` (model is cached in the same process) and build a
   sentence-aligned SRT.
6. Burn subtitles last into `final.mp4`.

## Key parameters

These reuse values that were tuned on the reference FunClip project; do not
re-tune them blindly.

- precut `auto-editor`: `--margin 0.2sec --silent-speed 99999 --sounded-speed
  1.0 --video-codec h264_nvenc --video-bitrate 8000k --no-open --quiet`
- conditional autorotate: `h264_nvenc -preset p4 -profile high -pix_fmt yuv420p
  -metadata:s:v:0 rotate=""`, audio `aac -b:a 128k -ar 48000 -ac 2`
- `auto_edl.py`: filler sets defined in the file; `JOIN_GAP 0.35`, `EDGE_PAD
  0.06`, `MIN_SEG 1.0`
- final nvenc: `-preset p6 -tune hq -rc vbr -cq 21 -b:v 0`, `-pix_fmt yuv420p`

The renderer keeps `libx264` as the portable default; nvenc is selected only on
machines where the encoder is advertised.

## Usage

```
python helpers/fast_pipeline.py <video>
python helpers/fast_pipeline.py <video> --engine funasr --fps 30
python helpers/fast_pipeline.py <video> --edit-dir /path/to/edit
```

Outputs land in `<video_parent>/edit/` (or `--edit-dir`): `edl.json`,
`<name>.precut.mp4`, `transcripts/`, `clips_graded/`, `base.mp4`, `master.srt`,
and `final.mp4`.

## Requirements

- ffmpeg/ffprobe on PATH with `h264_nvenc` for the GPU path
- CUDA torch for GPU FunASR (CPU torch still works, just slower)
- `auto-editor` (the reference project pins `>=24,<30`)
- FunASR extras for Chinese: `pip install -e '.[stt-funasr]'`

## Limits

- **Mechanical, not semantic.** It removes standalone fillers and long
  silences; it does not understand meaning and cannot pick the best take.
- **First run pays model load.** Reading the roughly 2 GB FunASR stack from
  disk is a fixed cold-start cost per process; only the inference itself is
  fast. A long-running resident process would avoid it.
- **Rotation must be baked before `auto-editor`.** Re-encoding a phone clip
  with a display-matrix rotation can drop that matrix; the conditional
  autorotate step prevents a sideways result.
- **Subtitle breaks are heuristic.** Sentence cues split on punctuation, long
  gaps, and a length cap; line breaks may land earlier than a hand-made SRT.
