# OpenAI Transcription (fork)

This fork (`darinkishore/hex`, branch `vendor`) adds OpenAI transcription the
way Dusk's Voxtype and Hyprscrawl use it: `gpt-live-transcribe` streamed for
dictation, and `gpt-transcribe` for recordings, with every word of a recording
timed on this Mac by WhisperX's forced alignment. It is macOS only. Nothing is
sent to OpenAI unless an OpenAI model or transcriber is chosen.

## Setup

```ts
~/.config/openai/api_key                  // openai.key
├── The key, trimmed; owner-only (0600) and owned by you, else refused
└── Another path: apiKeyFile in openai.json (absolute or ~/…)

~/.config/hex/openai.json                 // openai.config; every key optional
├── prompt          sent when not blank (dictation and recordings)
├── keywords        keywords[] / Realtime keywords; no <, >, or line breaks
├── delay           gpt-live-transcribe latency: minimal|low (default)|medium|high|xhigh
├── noiseReduction  null (default: off) | "near_field" | "far_field"
├── apiKeyFile      default ~/.config/openai/api_key
└── An unknown key or a bad value -> an error naming the file, never ignored
```

Read on every use, so edits apply at the next dictation or recording without
a restart. The noise filter is off by default because on Dusk the near-field
filter dropped whole passages of real dictation (`docs/voxtype.md` there).

## Dictation

```ts
Choose a model (picker, menu bar, `hex model use <id>`)   // openai.models
├── openai_gpt_live_transcribe  "GPT Live Transcribe (OpenAI)", streams
├── openai_gpt_transcribe       "GPT Transcribe (OpenAI)", on release
├── Listed once the key file is valid ("installed"); never recommended
├── "Use" (nothing downloads); a missing key fails the switch and says why
└── Language: the selection's (auto -> none sent, OpenAI detects)

GPT Live Transcribe, shortcut held past 300 ms   // openai.dictation-live
├── One Realtime session: wss://api.openai.com/v1/realtime?intent=transcription
│   session.update: gpt-live-transcribe, delay, languages, prompt, keywords,
│   noise filter only when configured, turn_detection null (manual turn)
├── 16 kHz capture -> Voxtype's linear 2:3 resampler -> 24 kHz PCM16, 80 ms frames
├── Backlog (450 ms pre-roll + 300 ms hold + connecting) drains at <= 2x real time
├── Release -> remaining frames -> input_audio_buffer.commit
│   -> the committed item's final transcript (a final may precede its ack)
└── Failure, refusal, or no final within 10 s -> the whole clip with gpt-transcribe
    (source=fallback), logged as a warning; short taps, chords, and Escape
    close the session, logged as stopped by Hex rather than as a failure

GPT Transcribe, or any whole-clip run          // openai.dictation-file
└── Release -> the clip as recordings are sent (below), without word timing
```

Voice Action uses the same model. Voice-delimited captures keep their control
phrase cut as Apple Speech's did: on the text, not word times.

## Recordings with Timed Words

```ts
gpt-transcribe file transcription              // openai.file
├── 16 kHz PCM16, the source's own samples (never re-encoded)
├── Chunks of at most 30 s, each ending at the centre of the quietest 100 ms
│   window starting in its last five seconds (earliest on a tie); contiguous,
│   no sample dropped or repeated: Voxtype's cut, verbatim
├── POST /v1/audio/transcriptions per chunk, up to four at once:
│   model, response_format=json, prompt, keywords[], languages[]=en, audio.wav
├── Through /usr/bin/curl: URL and headers (the key) on descriptor 3, body on
│   stdin; nothing on argv or disk
├── Network, 429 rate limit, 500/502/503/504 -> retry after 1 s, then 2 s
└── insufficient_quota (no credit, spend limit), 4xx -> fail at once with the reason

Acoustic word timing                           // openai.alignment
├── wav2vec2 base 960h: torchaudio's WAV2VEC2_ASR_BASE_960H weights, as the
│   pinned ONNX export Xenova/wav2vec2-base-960h@a19f851b (378 MB, SHA-256
│   e4661427…), in Application Support/hex/models; downloaded once, verified,
│   then trusted by size and modification time
├── WhisperX 3.8.6's align, ported: text cleaning and wildcard column, trellis,
│   backtrack, repeat merge, frame ratio, round(x, 3), pandas' pairwise mean
├── Each chunk is a segment: [start, end] in seconds and its transcript
├── Every word must come back with a finite start < end, in order, inside the
│   audio, scored in [0, 1], and the words must be the transcript's exactly
└── Otherwise the whole transcription fails: no interpolated, spread, or guessed time
```

Two differences from Dusk, both deliberate. Voxtype re-encodes decoded audio as
`(s × 32767) as i16` before upload and alignment; that truncation moved Dusk's
own scores by up to 0.085 on the fixture below (times unchanged), so Hex keeps
the source samples. And WhisperX groups words by NLTK Punkt sentences, which
never changes a word's timing but duplicates a word Punkt splits without
whitespace (`"Yes.(No)"`), failing Dusk's validation; Hex splits at spaces only.

```ts
hex transcribe <wav> [--scrawl] | --prepare    // openai.scrawl
├── No running Hex, lock, or app data written; logs to stderr and logs/transcribe.log
│   (never what was said)
├── <wav>: 16 kHz; Scrawl's mono PCM16 is used as is, else downmixed
├── default -> Voxtype's --json --timestamps reply (seconds, scores)
├── --scrawl -> Scrawl's command-transcriber JSON v1: words and sentence
│   segments in ms (round(seconds × 1000)), estimated: false, model gpt-transcribe
├── --prepare -> check the key, install the alignment model
└── Any failure -> exit 1; Scrawl then uses Apple Speech for that capture
    (source "speech-analyzer" in scrawl.json)

Scrawl.app setup (~/.config/scrawl/config.json)
├── "transcriber": "command"
└── "transcribeCommand": ["/Applications/Hex.app/Contents/MacOS/hex",
                          "transcribe", "--scrawl", "{audio}"]
```

## Captures in Hex

```ts
Settings › Experiments › Scrawl › Transcriber -> OpenAI    // openai.capture
├── capture.transcriber = openai; the same path as hex transcribe --scrawl
├── source "openai:gpt-transcribe+whisperx:wav2vec2-base-960h", estimated false
├── Parakeet (the dictation model) while dictation is OpenAI -> this path too
└── Imported from Scrawl when its transcribeCommand is hex transcribe --scrawl
```

## Privacy Boundaries

```ts
Nothing leaves the Mac unless chosen           // openai.privacy
├── Meetings record, finalize, or retranscribe with an OpenAI dictation model
│   -> refused: "Meetings transcribe on this Mac …"; choose a local model
├── Local API (/models, /transcriptions; Scrawl's `hex` transcriber, the SDK)
│   -> cloud models unlisted and refused (unsupported-model)
├── Segment timing for an OpenAI model outside captures -> an error, no upload
└── Setup's welcome says the voice is sent to OpenAI, never "on this Mac",
    while an OpenAI model is selected; its model row offers "Use", not
    "Download", and a bad key or openai.json is named as the failure
```

Sources: [openai_transcription.rs](../../src/openai_transcription.rs),
[openai_realtime.rs](../../src/openai_realtime.rs),
[forced_alignment.rs](../../src/forced_alignment.rs),
[onnx_runtime.rs](../../src/onnx_runtime.rs), the remote stream and its
deadline in [transcription_engine.rs](../../src/transcription_engine.rs), the
catalog in [transcription_models.rs](../../src/transcription_models.rs),
`OpenAiTranscriber` in [capture_host.rs](../../src/capture_host.rs), the
refusals in [meeting.rs](../../src/meeting.rs) and
[local_api.rs](../../src/local_api.rs), and setup's copy in
[rune_ui/onboarding.rs](../../src/rune_ui/onboarding.rs).

Checks:

- Alignment: `recorded_emissions_reproduce_dusks_whisperx_alignment_exactly`
  replays torch emissions recorded on Dusk (fixtures in
  `tests/fixtures/forced-alignment`) and requires Dusk's every text, time, and
  score bit for bit, including digits, punctuation, and a three-segment file;
  plus padding, slicing, ffmpeg-equivalent reading, rounding, pairwise mean,
  Python whitespace, refused backtracks, and segment bounds in
  [forced_alignment_tests.rs](../../src/forced_alignment_tests.rs).
  `the_onnx_model_reproduces_dusks_alignment` (ignored; needs the model and
  `HEX_ALIGNMENT_WAVS`) checks the real model on the fixtures' audio.
- File client, against a local server through the real curl
  ([openai_transcription_tests.rs](../../src/openai_transcription_tests.rs)):
  the exact multipart fields and uploaded PCM, auto language, retry and
  permanent failures (including spend limits), an unreachable server,
  parallel chunks in order, one failed chunk, cancellation, the chunk cut
  (including Dusk's 25.05 s fixture cut), alignment of chunks, Scrawl's JSON,
  WAV reading, config parsing, and the key file's permissions.
- Realtime, against a local WebSocket server
  ([openai_realtime_tests.rs](../../src/openai_realtime_tests.rs)): the session,
  Voxtype's resampler counts, every PCM byte in order in ≤ 80 ms frames, the
  ≤ 2x pacing, the commit after the last frame, a final before its ack, a
  refused session, no server, and cancellation.
  `the_live_api_transcribes_a_streamed_capture` (ignored) calls OpenAI.
- Engine: `a_remote_stream_gets_the_longer_finish_deadline`. Catalog:
  `openai_models_take_any_language_download_nothing_and_are_never_recommended`.
  Refusals: `cloud_models_are_refused_so_no_client_audio_leaves_the_mac`,
  `meetings_refuse_a_dictation_model_that_uploads_audio`,
  `the_welcome_says_an_openai_model_sends_the_voice_to_openai`. Import:
  `scrawls_transcriber_imports_as_the_matching_capture_transcriber`.

**Observed 2026-10-26 (echo, Mac Studio):** on Dusk's own torch emissions the
port reproduced `voxtype-align` exactly on four fixtures (63 words, three
segments). The ONNX model on the same audio matched every time; scores within
0.002. `hex transcribe` on a 23.6 s fixture took 2.2 s (debug build) and
returned the same text as Dusk's Voxtype; all 56 word times and scores equalled
Dusk's aligner on the same audio exactly. Streaming that fixture to
`gpt-live-transcribe` as a hotkey capture does (750 ms backlog, then 160 ms
blocks at real time) returned the complete transcript 522 ms after release.
The installed app's `hex transcribe --scrawl` on Dusk's 37 s two-chunk
fixture (espeak-ng) gave the same 39 word times, to the millisecond, as Dusk's
Voxtype run on it; Scrawl.app 0.3.13's `scrawl reprocess` of a synthetic
bundle with that audio called it and wrote `source: command:hex`,
`model: gpt-transcribe`, and those 39 measured words, with keyframes cut at the
spoken sentences. These are synthetic fixtures (macOS `say`, espeak-ng), not
microphone accuracy.

**Known gaps:** no reconnect inside a Realtime session (the whole clip is the
recovery); a capture longer than OpenAI's 60-minute session limit falls back to
the whole clip. Alignment supports English only, as on Dusk. Upstream Hex
builds that do not know these model ids fail to load a `settings.json` that
selects one: switch to a local model (`hex model use parakeet_unified_en`)
before going back to an upstream build.
