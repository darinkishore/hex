//! OpenAI transcription, set up as Dusk's Voxtype is: one owner-only API key
//! file and one config, `~/.config/hex/openai.json`, whose prompt, keywords,
//! and Realtime settings every request shares.
//!
//! File transcription (`gpt-transcribe`) follows Voxtype's `--file-model`
//! path: 16 kHz PCM16 in contiguous chunks of at most 30 s, each cut at the
//! quietest 100 ms window of its last five seconds (never dropping or
//! repeating a sample), each sent to `/v1/audio/transcriptions` as JSON
//! with the prompt, `keywords[]`, and `languages[]`. Hex sends the chunks
//! in parallel and retries transient failures; Voxtype does neither. With
//! word timing ([`transcribe_with_words`]) the chunks are English and the
//! [forced aligner](crate::forced_alignment) times every word, or the
//! transcription fails: there is no fallback to guessed timing.
//!
//! Hex uploads and aligns the source's own PCM16. Voxtype re-encodes its
//! decoded audio as `(sample * 32767) as i16`, shrinking every sample by up
//! to one step; that is the one place its audio and Hex's differ.
//!
//! Requests go through the system curl, as Hex's other HTTP does. The key
//! reaches curl in a config on a private pipe (descriptor 3) and the body on
//! stdin, so neither touches argv or the disk.

use std::fs;
use std::io::{Cursor, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use serde::{Deserialize, Serialize};

use crate::forced_alignment::{self, Aligner, Word};

pub const FILE_MODEL: &str = "gpt-transcribe";
pub const LIVE_MODEL: &str = "gpt-live-transcribe";
pub const SAMPLE_RATE: usize = 16_000;
const API_BASE: &str = "https://api.openai.com/v1";

const MAX_CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE;
const CUT_SEARCH_SAMPLES: usize = 5 * SAMPLE_RATE;
const CUT_WINDOW_SAMPLES: usize = SAMPLE_RATE / 10;

/// Voxtype's per-request client timeout.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT_SECONDS: &str = "15";
/// Transient failures (network, rate limit, server errors) retry after one
/// and then two seconds; anything else fails at once.
const ATTEMPTS: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_secs(1);
const PARALLEL_REQUESTS: usize = 4;
const MAX_RESPONSE_BYTES: u64 = 8 * 1024 * 1024;
const POLL: Duration = Duration::from_millis(20);

/// `gpt-live-transcribe`'s latency tier.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Delay {
    Minimal,
    #[default]
    Low,
    Medium,
    High,
    Xhigh,
}

impl Delay {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }
}

/// Realtime's server-side input filter. Off by default: on Dusk the
/// near-field filter dropped whole passages of real dictation.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NoiseReduction {
    NearField,
    FarField,
}

impl NoiseReduction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NearField => "near_field",
            Self::FarField => "far_field",
        }
    }
}

/// `~/.config/hex/openai.json`. Every key is optional; an unknown one is an
/// error rather than a silently ignored typo.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ConfigFile {
    api_key_file: Option<String>,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    delay: Delay,
    #[serde(default)]
    noise_reduction: Option<NoiseReduction>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiConfig {
    pub api_key_file: PathBuf,
    /// Sent when not blank.
    pub prompt: String,
    pub keywords: Vec<String>,
    pub delay: Delay,
    pub noise_reduction: Option<NoiseReduction>,
    /// `https://api.openai.com/v1`; tests point it at a local server.
    pub(crate) api_base: String,
}

impl OpenAiConfig {
    /// The prompt, when it says anything.
    pub fn prompt(&self) -> Option<&str> {
        Some(self.prompt.as_str()).filter(|prompt| !prompt.trim().is_empty())
    }

    pub fn api_key(&self) -> Result<ApiKey> {
        read_api_key(&self.api_key_file)
    }

    fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.api_base)
    }

    /// The Realtime transcription endpoint (`?intent=transcription`; the
    /// model goes in the session, never the URL).
    pub(crate) fn realtime_url(&self) -> String {
        let base = self
            .api_base
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1);
        format!("{base}/realtime?intent=transcription")
    }
}

/// Reads the config, or the defaults when there is no file.
pub fn load_config() -> Result<OpenAiConfig> {
    let path = crate::app_paths::openai_config_file()?;
    let home = dirs::home_dir().ok_or_else(|| eyre!("home directory is unavailable"))?;
    let bytes = match fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).wrap_err_with(|| format!("could not read {}", path.display()));
        }
    };
    parse_config(bytes.as_deref(), &home).wrap_err_with(|| format!("invalid {}", path.display()))
}

fn parse_config(bytes: Option<&[u8]>, home: &Path) -> Result<OpenAiConfig> {
    let file: ConfigFile = match bytes {
        Some(bytes) => serde_json::from_slice(bytes)?,
        None => ConfigFile::default(),
    };
    for keyword in &file.keywords {
        // The Realtime API rejects a session whose keywords hold these.
        ensure!(
            !keyword.trim().is_empty() && !keyword.contains(['<', '>', '\r', '\n']),
            "keyword {keyword:?} is empty or contains <, >, or a line break"
        );
    }
    let api_key_file = match file.api_key_file.as_deref() {
        None => home.join(".config/openai/api_key"),
        Some(path) => match path.strip_prefix("~/") {
            Some(rest) => home.join(rest),
            None if Path::new(path).is_absolute() => PathBuf::from(path),
            None => bail!("apiKeyFile must be absolute or start with ~/"),
        },
    };
    Ok(OpenAiConfig {
        api_key_file,
        prompt: file.prompt,
        keywords: file.keywords,
        delay: file.delay,
        noise_reduction: file.noise_reduction,
        api_base: API_BASE.into(),
    })
}

/// An API key; it is never printed.
#[derive(Clone)]
pub struct ApiKey(String);

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl ApiKey {
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn for_test(key: &str) -> Self {
        Self(key.into())
    }
}

/// The trimmed key from a file only its owner, the current user, can read.
fn read_api_key(path: &Path) -> Result<ApiKey> {
    let metadata = fs::metadata(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => eyre!("no OpenAI API key at {}", path.display()),
        _ => eyre!("could not read {}: {error}", path.display()),
    })?;
    ensure!(metadata.is_file(), "{} is not a file", path.display());
    ensure!(
        metadata.uid() == unsafe { libc::getuid() },
        "{} must belong to you",
        path.display()
    );
    ensure!(
        metadata.mode() & 0o077 == 0,
        "{} must be readable only by you (chmod 600)",
        path.display()
    );
    let key =
        fs::read_to_string(path).wrap_err_with(|| format!("could not read {}", path.display()))?;
    let key = key.trim();
    ensure!(!key.is_empty(), "{} is empty", path.display());
    ensure!(
        key.bytes()
            .all(|byte| byte.is_ascii_graphic() && byte != b'"' && byte != b'\\'),
        "{} does not hold an API key",
        path.display()
    );
    Ok(ApiKey(key.into()))
}

/// A 16 kHz float sample as PCM16, as Scrawl writes `audio.wav` and as
/// WhisperX reads it back (`/ 32768`).
pub fn pcm16(sample: f32) -> i16 {
    let sample = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    (sample * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16
}

/// Voxtype's chunks: contiguous, at most 30 s, each ending at the centre of
/// the quietest 100 ms window (the first, on a tie) among those starting in
/// its last five seconds. The final chunk ends with the audio.
pub(crate) fn chunk_ranges(pcm: &[i16]) -> Vec<(usize, usize)> {
    let samples: Vec<f32> = pcm.iter().map(|&sample| sample as f32 / 32768.0).collect();
    let energy = |start: usize| {
        samples[start..start + CUT_WINDOW_SAMPLES]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
    };
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let limit = (start + MAX_CHUNK_SAMPLES).min(samples.len());
        let end = if limit < samples.len() {
            ((limit - CUT_SEARCH_SAMPLES)..(limit - CUT_WINDOW_SAMPLES))
                .step_by(CUT_WINDOW_SAMPLES)
                .min_by(|&a, &b| energy(a).total_cmp(&energy(b)))
                .map(|window| window + CUT_WINDOW_SAMPLES / 2)
                .unwrap_or(limit)
        } else {
            limit
        };
        ranges.push((start, end));
        start = end;
    }
    ranges
}

fn wav(pcm: &[i16]) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::with_capacity(44 + pcm.len() * 2));
    let mut writer = hound::WavWriter::new(
        &mut bytes,
        hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for &sample in pcm {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(bytes.into_inner())
}

/// What one file transcription request asks for.
pub(crate) struct FileRequest<'a> {
    pub model: &'a str,
    pub languages: &'a [String],
}

/// One chunk's span of the audio, in samples, and its transcript.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Chunk {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Transcribes `pcm` (16 kHz) chunk by chunk, up to four requests at once.
/// `cancelled` is polled while waiting.
pub(crate) fn transcribe_chunks(
    config: &OpenAiConfig,
    key: &ApiKey,
    pcm: &[i16],
    request: &FileRequest<'_>,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Vec<Chunk>> {
    let ranges = chunk_ranges(pcm);
    let texts: Vec<Mutex<Option<String>>> = ranges.iter().map(|_| Mutex::new(None)).collect();
    // The failure earliest in the audio; requests it stopped are not failures.
    let failure: Mutex<Option<(usize, color_eyre::Report)>> = Mutex::new(None);
    let failed = AtomicBool::new(false);
    let next = AtomicUsize::new(0);
    let stop = || cancelled() || failed.load(Ordering::Acquire);
    thread::scope(|scope| {
        for _ in 0..PARALLEL_REQUESTS.min(ranges.len()) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::AcqRel);
                    let Some(&(start, end)) = ranges.get(index) else {
                        return;
                    };
                    let result = match wav(&pcm[start..end]) {
                        Ok(wav) => post_file(config, key, request, wav, &stop),
                        Err(error) => Err(Stop::Failed(error)),
                    };
                    match result {
                        Ok(text) => {
                            *texts[index]
                                .lock()
                                .unwrap_or_else(|error| error.into_inner()) = Some(text);
                        }
                        Err(Stop::Stopped) => {}
                        Err(Stop::Failed(error)) => {
                            failed.store(true, Ordering::Release);
                            let mut first =
                                failure.lock().unwrap_or_else(|error| error.into_inner());
                            if first.as_ref().is_none_or(|(earlier, _)| index < *earlier) {
                                *first = Some((index, error));
                            }
                        }
                    }
                }
            });
        }
    });
    if let Some((_, error)) = failure
        .into_inner()
        .unwrap_or_else(|error| error.into_inner())
    {
        return Err(error);
    }
    ensure!(!cancelled(), "transcription was cancelled");
    ranges
        .iter()
        .zip(texts)
        .map(|(&(start, end), text)| {
            let text = text
                .into_inner()
                .unwrap_or_else(|error| error.into_inner())
                .ok_or_else(|| eyre!("a chunk was not transcribed"))?;
            Ok(Chunk { start, end, text })
        })
        .collect()
}

/// The chunks' non-empty transcripts joined by single spaces, as Voxtype
/// joins them.
pub(crate) fn joined(chunks: &[Chunk]) -> String {
    chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Transcribes 16 kHz PCM16 with `gpt-transcribe` in `languages` (none:
/// detected).
pub fn transcribe_text(
    config: &OpenAiConfig,
    key: &ApiKey,
    pcm: &[i16],
    languages: &[String],
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<String> {
    let request = FileRequest {
        model: FILE_MODEL,
        languages,
    };
    Ok(joined(&transcribe_chunks(
        config, key, pcm, &request, cancelled,
    )?))
}

/// A transcript with WhisperX timing for every word.
#[derive(Clone, Debug, PartialEq)]
pub struct TimedTranscript {
    pub text: String,
    pub words: Vec<Word>,
}

/// `voxtype transcribe --file-model gpt-transcribe --timestamps`: English
/// `gpt-transcribe` chunks, then acoustic timing for every word of them.
/// The aligner loads while the requests are out.
pub fn transcribe_with_words(
    config: &OpenAiConfig,
    key: &ApiKey,
    pcm: &[i16],
    load_aligner: impl FnOnce() -> Result<Aligner> + Send,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<TimedTranscript> {
    ensure!(!pcm.is_empty(), "the audio is empty");
    let english = ["en".to_string()];
    let request = FileRequest {
        model: FILE_MODEL,
        languages: &english,
    };
    let (chunks, aligner) = thread::scope(|scope| {
        let loading = scope.spawn(load_aligner);
        let chunks = transcribe_chunks(config, key, pcm, &request, cancelled);
        (chunks, loading.join())
    });
    let chunks = chunks?;
    let mut aligner = aligner.map_err(|_| eyre!("loading the aligner panicked"))??;
    align_chunks(pcm, &chunks, |pcm, segments| aligner.align(pcm, segments))
}

/// Times the chunks' words; the alignment itself is `align`.
pub(crate) fn align_chunks(
    pcm: &[i16],
    chunks: &[Chunk],
    align: impl FnOnce(&[i16], &[forced_alignment::Segment]) -> Result<Vec<Word>>,
) -> Result<TimedTranscript> {
    let text = joined(chunks);
    if text.is_empty() {
        return Ok(TimedTranscript {
            text,
            words: Vec::new(),
        });
    }
    let segments: Vec<_> = chunks
        .iter()
        .map(|chunk| forced_alignment::Segment {
            start: chunk.start as f64 / SAMPLE_RATE as f64,
            end: chunk.end as f64 / SAMPLE_RATE as f64,
            text: chunk.text.clone(),
        })
        .collect();
    let words =
        align(pcm, &segments).wrap_err("acoustic alignment failed; no word times were guessed")?;
    // Voxtype's own check, after `voxtype-align`'s.
    ensure!(
        text.split_whitespace()
            .eq(words.iter().map(|word| word.text.as_str())),
        "alignment did not preserve every transcript word"
    );
    Ok(TimedTranscript { text, words })
}

/// Scrawl's command-transcriber protocol, version 1: measured word times
/// and sentence segments in milliseconds, as Dusk's Scrawl builds them.
pub fn scrawl_output(transcript: &TimedTranscript) -> serde_json::Value {
    let words: Vec<Span> = transcript
        .words
        .iter()
        .map(|word| (word.start_ms(), word.end_ms(), word.text.clone()))
        .collect();
    let spans = |spans: &[Span]| {
        spans
            .iter()
            .map(|(t0, t1, text)| serde_json::json!({"t0": t0, "t1": t1, "text": text}))
            .collect::<Vec<_>>()
    };
    serde_json::json!({
        "version": 1,
        "model": FILE_MODEL,
        "timing": forced_alignment::TIMING,
        "text": transcript.text,
        "estimated": false,
        "words": spans(&words),
        "segments": spans(&sentences(&words)),
    })
}

/// `t0`, `t1` in milliseconds, and text.
type Span = (u64, u64, String);

/// Sentences ending at `. ! ? ; :` before any closing quotes or brackets.
fn sentences(words: &[Span]) -> Vec<Span> {
    let ends_sentence = |word: &str| {
        word.trim_end_matches(['"', '\'', '”', '’', ')', ']'])
            .ends_with(['.', '!', '?', ';', ':'])
    };
    let mut sentences = Vec::new();
    let mut start = 0;
    for (index, word) in words.iter().enumerate() {
        if ends_sentence(&word.2) || index + 1 == words.len() {
            let phrase = &words[start..=index];
            sentences.push((
                phrase[0].0,
                phrase
                    .iter()
                    .map(|word| word.1)
                    .max()
                    .unwrap_or(phrase[0].1),
                phrase
                    .iter()
                    .map(|word| word.2.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            ));
            start = index + 1;
        }
    }
    sentences
}

/// Why a request failed, and whether trying again could help.
#[derive(Debug)]
enum Failure {
    Transient(String),
    Permanent(String),
    Cancelled,
}

/// How a request ended without a transcript.
enum Stop {
    /// Cancelled, or stopped because another request failed.
    Stopped,
    Failed(color_eyre::Report),
}

fn post_file(
    config: &OpenAiConfig,
    key: &ApiKey,
    request: &FileRequest<'_>,
    wav: Vec<u8>,
    stop: &dyn Fn() -> bool,
) -> Result<String, Stop> {
    let boundary = boundary().map_err(Stop::Failed)?;
    let body = multipart(&boundary, config, request, &wav);
    let mut delay = RETRY_DELAY;
    for attempt in 1..=ATTEMPTS {
        match post_once(config, key, &boundary, &body, stop) {
            Ok(text) => return Ok(text),
            Err(Failure::Cancelled) => return Err(Stop::Stopped),
            Err(Failure::Permanent(reason)) => return Err(Stop::Failed(eyre!(reason))),
            Err(Failure::Transient(reason)) if attempt == ATTEMPTS => {
                return Err(Stop::Failed(eyre!(reason)));
            }
            Err(Failure::Transient(reason)) => {
                tracing::warn!(attempt, %reason, "OpenAI transcription failed; retrying");
                let until = Instant::now() + delay;
                while Instant::now() < until {
                    if stop() {
                        return Err(Stop::Stopped);
                    }
                    thread::sleep(POLL);
                }
                delay *= 2;
            }
        }
    }
    unreachable!("the last attempt returns")
}

fn boundary() -> Result<String> {
    let mut random = [0_u8; 16];
    getrandom::fill(&mut random).map_err(|error| eyre!("no secure random source: {error}"))?;
    Ok(format!(
        "hex-{}",
        random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn multipart(
    boundary: &str,
    config: &OpenAiConfig,
    request: &FileRequest<'_>,
    wav: &[u8],
) -> Vec<u8> {
    let mut body = Vec::with_capacity(wav.len() + 1024);
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    };
    field("model", request.model);
    field("response_format", "json");
    if let Some(prompt) = config.prompt() {
        field("prompt", prompt);
    }
    for keyword in &config.keywords {
        field("keywords[]", keyword);
    }
    for language in request.languages {
        field("languages[]", language);
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

fn post_once(
    config: &OpenAiConfig,
    key: &ApiKey,
    boundary: &str,
    body: &[u8],
    stop: &dyn Fn() -> bool,
) -> Result<String, Failure> {
    let url = config.transcriptions_url();
    let curl_config = format!(
        "url = \"{url}\"\nheader = \"Authorization: Bearer {}\"\nheader = \"Content-Type: multipart/form-data; boundary={boundary}\"\n",
        key.expose()
    );
    let output = curl(&url, &curl_config, body.to_vec(), REQUEST_TIMEOUT, stop)?;
    if !output.status.success() {
        return Err(Failure::Transient(format!(
            "could not reach OpenAI ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let (status, response) = split_status(&output.stdout)
        .ok_or_else(|| Failure::Transient("OpenAI returned no HTTP status".into()))?;
    let json: serde_json::Value = serde_json::from_slice(response).unwrap_or_default();
    if status == 200 {
        return json["text"]
            .as_str()
            .map(|text| text.trim().to_string())
            .ok_or_else(|| Failure::Permanent("OpenAI's transcription has no text".into()));
    }
    let error = &json["error"];
    let code = error["code"].as_str().unwrap_or("unknown_error");
    let message = error["message"].as_str().unwrap_or("");
    let reason = format!("OpenAI transcription failed ({status} {code}): {message}");
    match status {
        // An empty balance or a spend limit is not a rate limit; waiting
        // seconds will not lift it.
        429 if error["type"] == "insufficient_quota" => Err(Failure::Permanent(reason)),
        429 | 500 | 502 | 503 | 504 => Err(Failure::Transient(reason)),
        _ => Err(Failure::Permanent(reason)),
    }
}

/// curl appends `\n<status>` after the body.
fn split_status(stdout: &[u8]) -> Option<(u16, &[u8])> {
    let newline = stdout.iter().rposition(|byte| *byte == b'\n')?;
    let status = std::str::from_utf8(&stdout[newline + 1..])
        .ok()?
        .parse()
        .ok()?;
    Some((status, &stdout[..newline]))
}

struct CurlOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Everything curl gets on its command line. The URL and the headers,
/// the key among them, are in the config on descriptor 3.
fn curl_arguments(url: &str, timeout: Duration) -> Vec<String> {
    // Plain HTTP only for the tests' loopback server.
    let protocol = if url.starts_with("http://127.0.0.1:") {
        "=http"
    } else {
        "=https"
    };
    [
        "--disable",
        "--silent",
        "--show-error",
        "--write-out",
        "\n%{http_code}",
        "--proto",
        protocol,
        "--connect-timeout",
        CONNECT_TIMEOUT_SECONDS,
        "--max-time",
        &timeout.as_secs().to_string(),
        "--config",
        "/dev/fd/3",
        "--data-binary",
        "@-",
    ]
    .map(String::from)
    .to_vec()
}

/// Runs `/usr/bin/curl` with `config` on descriptor 3 and `body` on stdin,
/// killing it at `timeout` or once `stop` says so.
fn curl(
    url: &str,
    config: &str,
    body: Vec<u8>,
    timeout: Duration,
    stop: &dyn Fn() -> bool,
) -> Result<CurlOutput, Failure> {
    if stop() {
        return Err(Failure::Cancelled);
    }
    let (config_reader, mut config_writer) = std::io::pipe()
        .map_err(|error| Failure::Permanent(format!("no pipe for curl: {error}")))?;
    // A few hundred bytes always fit the pipe's buffer before curl runs.
    config_writer
        .write_all(config.as_bytes())
        .map_err(|error| Failure::Permanent(format!("could not prepare curl: {error}")))?;
    drop(config_writer);
    let config_fd = config_reader.as_raw_fd();
    let mut command = Command::new("/usr/bin/curl");
    command
        .args(curl_arguments(url, timeout))
        .process_group(0)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    unsafe {
        command.pre_exec(move || {
            // dup2 clears close-on-exec on the copy; a descriptor already
            // at 3 keeps it unless cleared.
            let result = if config_fd == 3 {
                libc::fcntl(3, libc::F_SETFD, 0)
            } else {
                libc::dup2(config_fd, 3)
            };
            if result == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|error| Failure::Permanent(format!("could not start curl: {error}")))?;
    drop(config_reader);
    let mut stdin = child.stdin.take().expect("curl stdin is piped");
    let stdout = child.stdout.take().expect("curl stdout is piped");
    let stderr = child.stderr.take().expect("curl stderr is piped");
    let writer = thread::spawn(move || {
        let _ = stdin.write_all(&body);
    });
    let stdout = thread::spawn(move || read_bounded(stdout));
    let stderr = thread::spawn(move || read_bounded(stderr));
    let started = Instant::now();
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(error) => {
                break Err(Failure::Transient(format!(
                    "could not wait for curl: {error}"
                )));
            }
        }
        if stop() {
            break Err(Failure::Cancelled);
        }
        // curl's own --max-time ends it first; this covers a stuck process.
        if started.elapsed() >= timeout + Duration::from_secs(5) {
            break Err(Failure::Transient(format!(
                "OpenAI did not answer within {} seconds",
                timeout.as_secs()
            )));
        }
        thread::sleep(POLL);
    };
    if outcome.is_err() {
        terminate(&mut child);
    }
    let _ = writer.join();
    let panicked = |_| Err("curl's output reader panicked".to_string());
    let stdout = stdout.join().unwrap_or_else(panicked);
    let stderr = stderr.join().unwrap_or_else(panicked);
    let status = outcome?;
    let stdout = stdout.map_err(Failure::Permanent)?;
    Ok(CurlOutput {
        status,
        stdout,
        stderr: stderr.unwrap_or_default(),
    })
}

fn read_bounded(mut output: impl Read) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    output
        .by_ref()
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("could not read curl's output: {error}"))?;
    if bytes.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(format!(
            "OpenAI's answer exceeded {MAX_RESPONSE_BYTES} bytes"
        ));
    }
    Ok(bytes)
}

fn terminate(child: &mut Child) {
    // The child leads its own process group.
    unsafe {
        libc::killpg(child.id() as libc::pid_t, libc::SIGKILL);
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// 16 kHz PCM16 from a WAV file: Scrawl's `audio.wav` exactly, or any
/// other 16 kHz WAV downmixed and converted with [`pcm16`].
pub fn read_wav(path: &Path) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path)
        .wrap_err_with(|| format!("could not read {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate == SAMPLE_RATE as u32,
        "{} is {} Hz; Scrawl's audio is {SAMPLE_RATE} Hz",
        path.display(),
        spec.sample_rate
    );
    let channels = usize::from(spec.channels.max(1));
    if spec.channels == 1
        && spec.sample_format == hound::SampleFormat::Int
        && spec.bits_per_sample == 16
    {
        return reader
            .samples::<i16>()
            .collect::<Result<_, _>>()
            .wrap_err("invalid WAV samples");
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(samples
        .chunks(channels)
        .map(|frame| pcm16(frame.iter().sum::<f32>() / frame.len() as f32))
        .collect())
}

/// `hex transcribe`: transcribe a 16 kHz WAV with timed words and print
/// Voxtype's `--json --timestamps` reply, or with `scrawl` Scrawl's
/// command-transcriber JSON. `prepare` only checks the key and installs
/// the alignment model. Runs without the app; nothing here logs what was
/// said.
pub fn run_cli(audio: Option<&Path>, scrawl: bool, prepare: bool) -> Result<()> {
    init_cli_logging();
    let config = load_config()?;
    let key = config.api_key()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let never = AtomicBool::new(false);
    if prepare {
        if !forced_alignment::is_installed() {
            eprintln!("Downloading the alignment model (378 MB)…");
        }
        Aligner::load(root, &never)?;
        eprintln!(
            "Ready: OpenAI key at {}, alignment model installed.",
            config.api_key_file.display()
        );
        return Ok(());
    }
    let audio = audio.ok_or_else(|| eyre!("name a WAV to transcribe"))?;
    let pcm = read_wav(audio)?;
    let started = Instant::now();
    let transcript =
        transcribe_with_words(&config, &key, &pcm, || Aligner::load(root, &never), &|| {
            false
        })
        .inspect_err(|error| tracing::error!("transcription failed: {error:#}"))?;
    tracing::info!(
        audio_ms = pcm.len() / 16,
        words = transcript.words.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "transcribed with OpenAI and aligned every word"
    );
    let output = if scrawl {
        scrawl_output(&transcript)
    } else {
        serde_json::json!({
            "text": transcript.text,
            "engine": "openai-file",
            "model": FILE_MODEL,
            "timing": forced_alignment::TIMING,
            "words": transcript.words.iter().map(|word| serde_json::json!({
                "text": word.text, "start": word.start, "end": word.end, "score": word.score,
            })).collect::<Vec<_>>(),
        })
    };
    println!("{output}");
    Ok(())
}

/// Warnings and errors to stderr and `transcribe.log` in Hex's logs.
fn init_cli_logging() {
    use tracing_subscriber::fmt::writer::MakeWriterExt;
    let filter = tracing_subscriber::EnvFilter::try_from_env("HEX_TRANSCRIBE_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("voice_control=info"));
    let log = crate::app_paths::logs_dir().ok().and_then(|directory| {
        fs::create_dir_all(&directory).ok()?;
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join("transcribe.log"))
            .ok()
    });
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false);
    let _ = match log {
        Some(log) => subscriber
            .with_writer(std::io::stderr.and(Mutex::new(log)))
            .try_init(),
        None => subscriber.with_writer(std::io::stderr).try_init(),
    };
}

/// The languages a dictation selection asks for: none when detecting.
pub fn languages_for(language: &str) -> Vec<String> {
    if language == crate::transcription_models::AUTO_LANGUAGE {
        Vec::new()
    } else {
        vec![language.to_string()]
    }
}

#[cfg(test)]
#[path = "openai_transcription_tests.rs"]
mod tests;
