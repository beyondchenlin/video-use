# Quick mode

Quick mode is the one-shot fast lane. The user uploads a video and says one
sentence; the agent returns `final.mp4` with no strategy round, no questions,
and no preview loop.

Underneath it is exactly the fast lane from `references/fast-route.md`, so every
hard rule in `SKILL.md` still applies. Only the interaction changes.

## The rule

1. **What the user asks for wins.** Every explicit requirement is honored, even
   when it differs from the defaults below.
2. **What the user does not say takes the default.** Never ask about it.

So a bare "帮我剪辑这个口播视频" runs the default command. "帮我剪辑这个口播视频,
字幕低一点" runs the same command with `--caption-margin 60`. Nothing else changes,
and nothing is asked.

## When to enter quick mode

Enter as soon as the user asks you to cut a video, with or without a brief. A
single talking-head / 口播 clip is the canonical case.

- "帮我剪辑这个口播视频" / "帮我剪一下这个视频" / "把这个视频剪了" / "剪一下"
- "快速剪一下" / "一键出片" / "直接生成" / "出个初版" / "随便剪一下"
- an attached video or a path, with or without extra wording
- "不用问我" / "别问了直接做"

Do **not** ask a strategy question first. Run the command and report.

## When it is not quick mode

Leave quick mode only when the brief needs editorial judgment the fast lane
cannot express:

- pick the best take across several clips
- hit a target runtime or follow a script
- match a brand, add animations, "剪得好看一点"

Then use the conversational route in `SKILL.md`, and plan **exactly** that
request — do not drag the rest of the conversational workflow into it.

## Attached video

If the user attached the video, use the path the harness reports for the
attachment. If you only have a filename, look beside the current working
directory. Ask for the path once only when the file truly cannot be found.

## The command

One command, defaults only:

    python -X utf8 helpers/fast_pipeline.py <video>

- transcription engine: from `.env` / `VIDEO_USE_TRANSCRIBER`; never ask
- output: `<video dir>/edit/final.mp4`
- frame rate: 30

## Natural language to flags

| the user says | add |
|---|---|
| 不要重转写 / 直接出 / 快一点 | `--no-recheck` |
| 字幕低一点 / 高一点 | `--caption-margin 60` / `--caption-margin 120` |
| 输出到别的地方 | `--edit-dir <dir>` |
| 指定帧率 | `--fps 25` |
| 换转写引擎 | `--engine funasr` (or `local`, `elevenlabs`) |

Add only the flags the sentence asks for. Do not invent flags that do not
exist, and do not ask for the rest.

## What to answer

Reply with:

1. the final path
2. duration and size (`ffprobe`)
3. one sentence on what quick mode did — removed standalone fillers and long
   silences, burned sentence subtitles, normalized to -14 LUFS
4. the quick-mode limits in one sentence: mechanical cuts, no take selection,
   ASR errors are burned in (`references/fast-route.md` Limits)

Ask a follow-up only when the command fails.

## Cost

About 45 s of wall time for a 33 s clip on an RTX 4090; most of it is the ASR
cold start. Report the actual wall time.
