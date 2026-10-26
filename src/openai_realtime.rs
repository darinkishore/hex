//! Streaming dictation over OpenAI's Realtime transcription API
//! (`gpt-live-transcribe`), as Dusk's Voxtype streams it: one manual turn
//! per capture (`turn_detection: null`), the capture's 16 kHz audio
//! resampled to 24 kHz PCM16 and appended in 80 ms frames, a commit once
//! the capture ends, and the committed item's final transcript as the text.
//! The prompt, keywords, latency tier, and noise filter come from
//! `~/.config/hex/openai.json`.
//!
//! Two deliberate differences from Voxtype:
//! - Backlog drains at up to twice real time; Voxtype paces at real time.
//!   A hotkey stream starts 750 ms behind (pre-roll and the hold threshold)
//!   plus the connection time, and at real time that lag lasts to the end
//!   of the capture. On Dusk, a 48 s fixture uploaded at twice real time
//!   kept every passage, as real time did; only an instant upload lost
//!   text.
//! - No reconnect or replay: any failure ends the stream, and the engine
//!   transcribes the complete clip with `gpt-transcribe` instead.

use std::sync::mpsc::{self as std_mpsc, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use transcribe_cpp::CancelToken;

use crate::openai_transcription::{ApiKey, LIVE_MODEL, OpenAiConfig};
use crate::transcription_engine::StreamFeed;

const SOURCE_RATE: u32 = 16_000;
/// The only rate Realtime's `audio/pcm` accepts.
const TARGET_RATE: u32 = 24_000;
/// 80 ms of 24 kHz PCM16, OpenAI's example frame.
const CHUNK_BYTES: usize = 3_840;
/// How much faster than real time a backlog may drain.
const CATCH_UP_RATE: f64 = 2.0;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const SESSION_UPDATE_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);
/// Voxtype's bound on a manual turn's final transcript. The engine's own
/// finish deadline is shorter and falls back to the whole clip first.
const FINAL_TIMEOUT: Duration = Duration::from_secs(30);
const POLL: Duration = Duration::from_millis(20);

/// What a stream needs: the shared config, its key, and the selection's
/// languages (none when detecting).
#[derive(Clone)]
pub struct RealtimeSettings {
    pub config: OpenAiConfig,
    pub key: ApiKey,
    pub languages: Vec<String>,
}

impl RealtimeSettings {
    /// Voxtype's `session.update`: a transcription session with manual turns.
    pub(crate) fn session_update(&self) -> serde_json::Value {
        let config = &self.config;
        let mut transcription = serde_json::json!({
            "model": LIVE_MODEL,
            "delay": config.delay.as_str(),
        });
        if !self.languages.is_empty() {
            transcription["languages"] = serde_json::json!(self.languages);
        }
        if let Some(prompt) = config.prompt() {
            transcription["prompt"] = serde_json::json!(prompt);
        }
        if !config.keywords.is_empty() {
            transcription["keywords"] = serde_json::json!(config.keywords);
        }
        let mut input = serde_json::json!({
            "format": { "type": "audio/pcm", "rate": TARGET_RATE },
            "transcription": transcription,
            // Server VAD is refused by gpt-live-transcribe; the commit ends the turn.
            "turn_detection": null,
        });
        if let Some(filter) = config.noise_reduction {
            input["noise_reduction"] = serde_json::json!({ "type": filter.as_str() });
        }
        serde_json::json!({
            "type": "session.update",
            "session": { "type": "transcription", "audio": { "input": input } },
        })
    }
}

/// One capture's Realtime session, fed from the engine thread. Its network
/// work runs on its own thread; feeding never blocks.
pub struct RealtimeStream {
    audio: Option<mpsc::UnboundedSender<Vec<f32>>>,
    result: std_mpsc::Receiver<Result<String, String>>,
    /// The session's outcome once it arrived early (a failure).
    ended: Option<Result<String, String>>,
    cancel: CancelToken,
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl RealtimeStream {
    /// Connects in the background; audio fed meanwhile waits for the
    /// session. `cancel` aborts a finish in progress.
    pub fn start(settings: RealtimeSettings, cancel: CancelToken) -> Self {
        let (audio, audio_rx) = mpsc::unbounded_channel();
        let (result_tx, result) = std_mpsc::channel();
        let (stop, stop_rx) = oneshot::channel();
        let thread = thread::Builder::new()
            .name("openai-realtime".into())
            .spawn(move || {
                let outcome = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| format!("could not start the Realtime runtime: {error}"))
                    .and_then(|runtime| runtime.block_on(run(settings, audio_rx, stop_rx)));
                if let Err(reason) = &outcome {
                    tracing::warn!(%reason, "OpenAI Realtime stream ended without a transcript");
                }
                let _ = result_tx.send(outcome);
            });
        let (thread, ended) = match thread {
            Ok(thread) => (Some(thread), None),
            Err(error) => (
                None,
                Some(Err(format!("could not start the Realtime thread: {error}"))),
            ),
        };
        Self {
            audio: Some(audio),
            result,
            ended,
            cancel,
            stop: Some(stop),
            thread,
        }
    }

    fn early_failure(&mut self) -> Option<String> {
        if self.ended.is_none()
            && let Ok(outcome) = self.result.try_recv()
        {
            self.ended = Some(outcome);
        }
        match &self.ended {
            Some(Err(reason)) => Some(reason.clone()),
            Some(Ok(_)) => Some("the Realtime session ended early".into()),
            None => None,
        }
    }
}

impl StreamFeed for RealtimeStream {
    fn feed(&mut self, samples: &[f32]) -> Result<(), String> {
        if let Some(reason) = self.early_failure() {
            return Err(reason);
        }
        let audio = self
            .audio
            .as_ref()
            .ok_or_else(|| "the Realtime stream was already finished".to_string())?;
        audio
            .send(samples.to_vec())
            .map_err(|_| "the Realtime session stopped".to_string())
    }

    /// Ends the capture's audio, then waits for the committed turn's final
    /// transcript, or until the engine cancels the stream.
    fn finalize(&mut self) -> Result<String, String> {
        self.audio = None;
        if let Some(outcome) = self.ended.take() {
            return outcome.map(|text| text.trim().to_string());
        }
        loop {
            match self.result.recv_timeout(POLL) {
                Ok(outcome) => return outcome.map(|text| text.trim().to_string()),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("the Realtime session stopped".into());
                }
                Err(RecvTimeoutError::Timeout) if self.cancel.is_cancelled() => {
                    if let Some(stop) = self.stop.take() {
                        let _ = stop.send(());
                    }
                    return Err("the Realtime stream was cancelled".into());
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

impl Drop for RealtimeStream {
    fn drop(&mut self) {
        self.audio = None;
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        // The session ends at once on stop; never wait on the network here.
        drop(self.thread.take());
    }
}

async fn run(
    settings: RealtimeSettings,
    audio: mpsc::UnboundedReceiver<Vec<f32>>,
    stop: oneshot::Receiver<()>,
) -> Result<String, String> {
    tokio::select! {
        biased;
        _ = stop => Err("the Realtime stream was stopped".into()),
        outcome = session(&settings, audio) => outcome,
    }
}

async fn session(
    settings: &RealtimeSettings,
    mut audio: mpsc::UnboundedReceiver<Vec<f32>>,
) -> Result<String, String> {
    let started = Instant::now();
    let mut request = settings
        .config
        .realtime_url()
        .into_client_request()
        .map_err(|error| format!("invalid Realtime URL: {error}"))?;
    let authorization = HeaderValue::from_str(&format!("Bearer {}", settings.key.expose()))
        .map_err(|_| "the API key is not a valid header value".to_string())?;
    request.headers_mut().insert("Authorization", authorization);
    let (socket, _) =
        tokio::time::timeout(CONNECT_TIMEOUT, tokio_tungstenite::connect_async(request))
            .await
            .map_err(|_| "could not connect to OpenAI Realtime in time".to_string())?
            .map_err(|error| format!("could not connect to OpenAI Realtime: {error}"))?;
    let (mut write, mut read) = socket.split();
    send(
        &mut write,
        Message::text(settings.session_update().to_string()),
    )
    .await?;
    wait_for_session_updated(&mut read).await?;
    let connected_ms = started.elapsed().as_millis() as u64;

    let mut resampler = Resampler::new(SOURCE_RATE, TARGET_RATE);
    let mut pending: Vec<u8> = Vec::with_capacity(CHUNK_BYTES * 2);
    let mut source_samples = 0_usize;
    let mut audio_closed = false;
    let mut committed = false;
    let mut next_append_at = Instant::now();
    let mut final_deadline: Option<Instant> = None;
    // Item ids in commit order, and finals by item; a final may arrive
    // before its commit is acknowledged.
    let mut committed_item: Option<String> = None;
    let mut finals = std::collections::HashMap::<String, String>::new();
    loop {
        if let Some(item) = &committed_item
            && let Some(text) = finals.remove(item)
        {
            let _ = send(&mut write, Message::Close(None)).await;
            tracing::info!(
                audio_ms = source_samples / 16,
                connected_ms,
                session_ms = started.elapsed().as_millis() as u64,
                chars = text.chars().count(),
                "OpenAI Realtime final received"
            );
            return Ok(text);
        }
        let final_timer = async {
            match final_deadline {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = final_timer => {
                return Err("OpenAI Realtime sent no final transcript in time".into());
            }
            chunk = audio.recv(), if !audio_closed && pending.len() < CHUNK_BYTES => {
                match chunk {
                    Some(chunk) => {
                        source_samples += chunk.len();
                        pending.extend_from_slice(&resampler.encode(&chunk));
                    }
                    None => audio_closed = true,
                }
            }
            // Full frames, and the final short one, are paced; the commit
            // follows the last frame's duration.
            _ = tokio::time::sleep_until(next_append_at),
                if pending.len() >= CHUNK_BYTES || (audio_closed && !pending.is_empty()) => {
                let count = pending.len().min(CHUNK_BYTES);
                let frame: Vec<u8> = pending.drain(..count).collect();
                let append = serde_json::json!({
                    "type": "input_audio_buffer.append",
                    "audio": base64::engine::general_purpose::STANDARD.encode(&frame),
                });
                send(&mut write, Message::text(append.to_string())).await?;
                let duration = Duration::from_secs_f64(count as f64 / f64::from(TARGET_RATE * 2));
                next_append_at = (next_append_at + duration.div_f64(CATCH_UP_RATE)).max(Instant::now());
            }
            _ = tokio::time::sleep_until(next_append_at),
                if audio_closed && pending.is_empty() && !committed => {
                send(&mut write, Message::text(r#"{"type":"input_audio_buffer.commit"}"#)).await?;
                committed = true;
                final_deadline = Some(Instant::now() + FINAL_TIMEOUT);
            }
            message = read.next() => {
                let message = match message {
                    Some(Ok(message)) => message,
                    Some(Err(error)) => return Err(format!("OpenAI Realtime connection failed: {error}")),
                    None => return Err("OpenAI Realtime closed before the final transcript".into()),
                };
                let text = match message {
                    Message::Text(text) => text,
                    Message::Ping(payload) => {
                        send(&mut write, Message::Pong(payload)).await?;
                        continue;
                    }
                    Message::Close(frame) => {
                        return Err(format!("OpenAI Realtime closed before the final transcript: {frame:?}"));
                    }
                    _ => continue,
                };
                let Ok(event) = serde_json::from_str::<serde_json::Value>(&text) else {
                    continue;
                };
                let item = event["item_id"].as_str().unwrap_or_default().to_string();
                match event["type"].as_str().unwrap_or_default() {
                    "input_audio_buffer.committed" => committed_item = Some(item),
                    "conversation.item.input_audio_transcription.completed" => {
                        finals.insert(item, event["transcript"].as_str().unwrap_or_default().to_string());
                    }
                    "conversation.item.input_audio_transcription.failed" => {
                        return Err(format!("OpenAI Realtime transcription failed: {}", event["error"]));
                    }
                    "error" => return Err(format!("OpenAI Realtime error: {}", event["error"])),
                    _ => {}
                }
            }
        }
    }
}

type Writer = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;
type Reader = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

async fn send(write: &mut Writer, message: Message) -> Result<(), String> {
    tokio::time::timeout(WRITE_TIMEOUT, write.send(message))
        .await
        .map_err(|_| "writing to OpenAI Realtime timed out".to_string())?
        .map_err(|error| format!("could not write to OpenAI Realtime: {error}"))
}

/// Waits for `session.updated`; an `error` first is the configuration's.
async fn wait_for_session_updated(read: &mut Reader) -> Result<(), String> {
    let deadline = Instant::now() + SESSION_UPDATE_TIMEOUT;
    loop {
        let message = tokio::time::timeout_at(deadline, read.next())
            .await
            .map_err(|_| "OpenAI Realtime did not accept the session in time".to_string())?;
        let text = match message {
            Some(Ok(Message::Text(text))) => text,
            Some(Ok(Message::Close(frame))) => {
                return Err(format!("OpenAI Realtime closed during setup: {frame:?}"));
            }
            Some(Ok(_)) => continue,
            Some(Err(error)) => {
                return Err(format!("OpenAI Realtime failed during setup: {error}"));
            }
            None => return Err("OpenAI Realtime closed during setup".into()),
        };
        let Ok(event) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        match event["type"].as_str().unwrap_or_default() {
            "session.updated" => return Ok(()),
            "error" => {
                return Err(format!(
                    "OpenAI Realtime refused the session: {}",
                    event["error"]
                ));
            }
            _ => {}
        }
    }
}

/// Voxtype's realtime sample encoding: `round(s * 32767)`.
fn pcm16_le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Voxtype's stateful linear-interpolation resampler: it carries the last
/// sample and the read phase across chunks, so output does not depend on
/// how the audio was split.
struct Resampler {
    step: f64,
    phase: f64,
    carry: Option<f32>,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        Self {
            step: f64::from(from) / f64::from(to),
            phase: 0.0,
            carry: None,
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        let carried = usize::from(self.carry.is_some());
        let length = carried + input.len();
        let at = |index: usize| {
            if index < carried {
                self.carry.expect("a carried sample exists")
            } else {
                input[index - carried]
            }
        };
        let mut output =
            Vec::with_capacity(((length as f64 - self.phase) / self.step) as usize + 1);
        let mut position = self.phase;
        // Grid points landing on the last sample (the 2:3 grid hits
        // integers) stay on one side despite accumulated float drift.
        const BOUNDARY_EPSILON: f64 = 1e-9;
        while position <= (length - 1) as f64 + BOUNDARY_EPSILON {
            let index = position.floor() as usize;
            let fraction = (position - index as f64) as f32;
            let a = at(index.min(length - 1));
            let b = at((index + 1).min(length - 1));
            output.push(a * (1.0 - fraction) + b * fraction);
            position += self.step;
        }
        self.phase = position - (length - 1) as f64;
        self.carry = Some(input[input.len() - 1]);
        output
    }

    fn encode(&mut self, samples: &[f32]) -> Vec<u8> {
        pcm16_le(&self.process(samples))
    }
}

#[cfg(test)]
#[path = "openai_realtime_tests.rs"]
mod tests;
