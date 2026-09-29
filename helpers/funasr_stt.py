"""Local Chinese speech to text through FunASR (Alibaba Tongyi speech lab).

Runs Paraformer-Large with the FSMN VAD, CT punctuation, and CAM++ speaker
models fully on-device, and returns the same transcript shape the rest of
the pipeline reads: a top-level ``words`` list of
``{"type": "word", "text", "start", "end", "speaker_id"}`` entries plus
``spacing`` entries for the gaps between words.

Why this engine exists: the local Whisper engine (local_stt.py) is strong on
English but weak on Chinese; FunASR Paraformer is one of the best open
Chinese ASR models, ships per-character timestamps, hotword support, and
local speaker diarization through CAM++. Use --engine funasr for Chinese /
mixed-language material and --engine local for English.

Chinese characters are grouped into words with jieba so the packed markdown
reads as real words instead of one-character-per-entry. Punctuation is glued
onto the preceding word. Nothing here installs packages: a missing funasr
install exits with the exact extra to add.

Usage:
    python helpers/funasr_stt.py transcribe <audio.wav> -o <out.json>
    python helpers/funasr_stt.py transcribe <audio.wav> --hotwords "人名 术语"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


EXTRA = "stt-funasr"

# gaps at least this long between words are emitted as explicit spacing entries
SPACING_MIN = 0.15

# model ids (same stack the FunClip project ships, damo/ aliases stay valid)
MODEL_IDS = {
    "asr": "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "vad": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": "damo/punc_ct-transformer_cn-en-common-vocab471067-large",
    "spk": "damo/speech_campplus_sv_zh-cn_16k-common",
}


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[\s，。！？、；：,.!?;:…—\-\"'“”‘’（）()【】《》]")
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:\"]")


# true when the text contains a chinese character
def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


# true when the single character is punctuation or whitespace
def is_punct(char: str) -> bool:
    return bool(_PUNCT_RE.fullmatch(char))


# split one recognized sentence into word and punctuation tokens
def tokenize_sentence(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if has_cjk(text):
        import jieba

        return [tok for tok in jieba.lcut(text) if tok.strip()]
    return _EN_TOKEN_RE.findall(text)


# turn one funasr sentence dict into word entries by aligning tokens to per-char timestamps
def words_from_sentence(sentence: dict) -> list[dict]:
    text = sentence.get("text", "")
    char_ts = sentence.get("timestamp") or []
    spk = sentence.get("spk")
    speaker_id = f"speaker_{spk}" if spk is not None else None

    words: list[dict] = []
    ts_index = 0
    for token in tokenize_sentence(text):
        core = "".join(c for c in token if not is_punct(c))
        punct = "".join(c for c in token if is_punct(c))
        if not core:
            # pure punctuation: glue onto the previous word
            if punct and words:
                words[-1]["text"] += punct
            continue
        n = len(core)
        chunk = char_ts[ts_index : ts_index + n]
        if not chunk:
            # timestamps ran out: keep the word glued to the last known boundary
            if words:
                words[-1]["text"] += core + punct
            continue
        start = float(chunk[0][0]) / 1000.0
        end = float(chunk[-1][1]) / 1000.0
        if end <= start:
            end = start + 0.05
        words.append(
            {
                "type": "word",
                "text": core + punct,
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker_id": speaker_id,
            }
        )
        ts_index += n
    return words


# insert explicit spacing entries wherever the gap between words is long enough
def add_spacing(words: list[dict], spacing_min: float = SPACING_MIN) -> list[dict]:
    out: list[dict] = []
    prev = None
    for word in words:
        if prev is not None:
            gap = word["start"] - prev["end"]
            if gap >= spacing_min:
                out.append(
                    {
                        "type": "spacing",
                        "text": "",
                        "start": round(prev["end"], 3),
                        "end": round(word["start"], 3),
                        "speaker_id": prev.get("speaker_id"),
                    }
                )
        out.append(word)
        prev = word
    return out


# join word text into a readable transcript with no spaces between chinese words
def join_text(words: list[dict]) -> str:
    parts: list[str] = []
    for word in words:
        if word.get("type") != "word":
            continue
        text = word["text"]
        if parts and has_cjk(parts[-1]) and has_cjk(text):
            parts.append(text)
        elif parts and has_cjk(parts[-1][-1]) and has_cjk(text[0]):
            parts.append(text)
        else:
            parts.append((" " if parts else "") + text)
    return "".join(parts)


# the funasr model is kept for the life of the process so batch runs load it once
_MODEL = None


# pick cuda when a gpu is available and fall back to cpu everywhere else
def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# turn on offline hub mode once every configured model is already cached locally
def enable_offline_if_cached() -> bool:
    import os

    roots = [Path.home() / ".cache" / "modelscope" / "models"]
    env_root = os.environ.get("MODELSCOPE_CACHE")
    if env_root:
        roots.insert(0, Path(env_root) / "models")
    needed = [mid.replace("/", "--") for mid in MODEL_IDS.values()]
    if all(any((root / name).exists() for root in roots) for name in needed):
        os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return True
    return False


# load the paraformer vad punctuation and speaker models once and cache them
def get_model(model_path: str | None = None, device: str | None = None):
    global _MODEL
    if _MODEL is None:
        enable_offline_if_cached()
        from funasr import AutoModel

        kwargs = dict(
            model=model_path or MODEL_IDS["asr"],
            vad_model=MODEL_IDS["vad"],
            punc_model=MODEL_IDS["punc"],
            spk_model=MODEL_IDS["spk"],
            device=device or detect_device(),
            disable_update=True,
        )
        _MODEL = AutoModel(**kwargs)
    return _MODEL


# fail first with the exact install command when funasr is not importable
def preflight(library: str | None = None) -> str:
    try:
        import funasr  # noqa: F401
    except ImportError:
        raise SystemExit(
            "funasr is not installed. From the repo root run:\n"
            f"  uv sync --extra {EXTRA}    (or: pip install -e '.[{EXTRA}]')"
        ) from None
    return EXTRA


# transcribe a mono wav and return the canonical transcript payload
def transcribe_wav(
    wav: Path,
    library: str | None = None,
    language: str | None = None,
    model: str | None = None,
    hotwords: str = "",
    num_speakers: int | None = None,
    **_unused,
) -> dict:
    preflight(library)
    asr = get_model(model)

    generate_kwargs = dict(
        sentence_timestamp=True,
        return_spk_res=True,
        return_raw_text=True,
        is_final=True,
        hotword=hotwords or "",
        cache={},
    )
    result = asr.generate(str(wav), **generate_kwargs)
    sentence_info = result[0].get("sentence_info") or []

    words: list[dict] = []
    for sentence in sentence_info:
        words.extend(words_from_sentence(sentence))
    words = add_spacing(words)

    return {
        "engine": "funasr",
        "library": "funasr",
        "model": model or MODEL_IDS["asr"],
        "language_code": language or "zh",
        "text": join_text(words),
        "words": words,
    }


# cli entry point that parses arguments and writes the transcript json
def main() -> None:
    ap = argparse.ArgumentParser(description="FunASR local Chinese transcription")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("transcribe", help="Transcribe a mono 16 kHz wav to transcript json")
    run.add_argument("audio", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--language", default=None, help="ISO language code; defaults to zh")
    run.add_argument("--model", default=None, help="Override the Paraformer model id or local dir")
    run.add_argument("--hotwords", default="", help="Hotwords separated by spaces")
    run.add_argument("--num-speakers", type=int, default=None)
    args = ap.parse_args()

    if not args.audio.exists():
        sys.exit(f"audio not found: {args.audio}")
    payload = transcribe_wav(
        args.audio,
        language=args.language,
        model=args.model,
        hotwords=args.hotwords,
        num_speakers=args.num_speakers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    word_count = sum(1 for w in payload["words"] if w["type"] == "word")
    print(f"saved: {args.output} ({word_count} words)")


if __name__ == "__main__":
    main()
