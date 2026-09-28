# FunASR local Chinese transcription engine

`--engine funasr` runs Alibaba Tongyi's FunASR Paraformer stack fully on-device.
It is the engine to reach for Chinese and Chinese-heavy material; the local
Whisper engine stays the better choice for English.

## Which engine to use

| Engine | Best for | Runs where | Speaker labels | Hotwords |
| --- | --- | --- | --- | --- |
| `elevenlabs` | any language, fastest setup | cloud (paid) | yes | no |
| `local` (Whisper large-v3-turbo) | English | on-device | no | no |
| `funasr` (Paraformer + CAM++) | Chinese, mixed zh/en | on-device | yes | yes |

There is no silent fallback. The engine is chosen by `--engine`, then
`VIDEO_USE_TRANSCRIBER` in `.env` or the environment, then ElevenLabs when an
`ELEVENLABS_API_KEY` is present. Every run prints which engine it used.

## What FunASR gives you

- **Strong Chinese ASR.** Paraformer-Large with the FSMN VAD and CT punctuation
  model is one of the best open Chinese recognizers, and it handles mixed
  Chinese/English speech.
- **Per-character timestamps.** Paraformer returns a timestamp for every
  character. The engine groups characters into words with jieba so
  `takes_packed.md` reads as words instead of one character per entry, while
  keeping word-boundary precision.
- **Speaker diarization.** The CAM++ speaker model tags every word with a
  `speaker_id` (`speaker_0`, `speaker_1`, ...), so speaker changes split phrases
  in the packed transcript.
- **Hotwords.** Pass names, brands, or jargon with `--hotwords "人名 术语"` to
  bias recognition (the SeACo-Paraformer hotword feature).
- **Silence as spacing.** Gaps between words are emitted as explicit `spacing`
  entries, so long pauses split phrases and are easy to find as cut points.
- **Offline after the first run.** Once the weights are cached, transcription
  needs no network.

## Install

From the repo root:

```bash
uv sync --extra stt-funasr
# or, with an existing virtualenv:
pip install -e ".[stt-funasr]"
```

This pulls `funasr`, `modelscope`, `torch` / `torchaudio` (CPU build), and
`jieba`. For English material also add the Whisper extra for your platform
(`stt-cpu` on Windows/Linux, `stt-mlx` on Apple Silicon).

Select the engine persistently by adding one line to `.env` at the repo root:

```
VIDEO_USE_TRANSCRIBER=funasr
```

## Use

```bash
# one file
python helpers/transcribe.py path/to/video.mp4 --engine funasr

# a whole directory (the local engine runs one file at a time)
python helpers/transcribe_batch.py path/to/videos --engine funasr

# bias toward specific names or terms
python helpers/transcribe.py video.mp4 --engine funasr --hotwords "张三 通义实验室"

# hint the number of speakers to help diarization
python helpers/transcribe.py video.mp4 --engine funasr --num-speakers 2
```

The output transcript keeps the standard shape, so `pack_transcripts.py`,
`render.py`, and every other helper work unchanged. The top-level `engine` key is
`funasr`.

## Models

| Role | Model id |
| --- | --- |
| ASR | `damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| VAD | `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| Punctuation | `damo/punc_ct-transformer_cn-en-common-vocab471067-large` |
| Speaker | `damo/speech_campplus_sv_zh-cn_16k-common` |

The first run downloads roughly 1 GB of weights through ModelScope; later runs
load them from the local cache (typically `~/.cache/modelscope`). Point
`--model` at a local directory or a different model id to override the ASR model.

## Limits

- **English is weaker than Whisper.** For English footage prefer `--engine local`.
- **CPU transcription is slower than the cloud**, roughly a fraction of realtime
  on a modern CPU; long takes take minutes, not seconds.
- **No `(laughter)` / `(applause)` audio events.** Find reaction beats from the
  waveform with `timeline_view.py` instead. The engine does emit `spacing` for
  silence.
- **Diarization is best effort.** CAM++ is strong but not perfect on overlapping
  speech; confirm who speaks when the edit depends on it.
