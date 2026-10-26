use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant as StdInstant;

use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

use super::*;
use crate::openai_transcription::{Delay, NoiseReduction};

fn config(base: &str) -> OpenAiConfig {
    OpenAiConfig {
        api_key_file: PathBuf::from("/unused"),
        prompt: "Technical dictation from a single speaker.".into(),
        keywords: vec!["NixOS".into(), "jj".into()],
        delay: Delay::Low,
        noise_reduction: None,
        api_base: base.into(),
    }
}

fn settings(base: &str, languages: &[&str]) -> RealtimeSettings {
    RealtimeSettings {
        config: config(base),
        key: ApiKey::for_test("sk-test"),
        languages: languages
            .iter()
            .map(|language| language.to_string())
            .collect(),
    }
}

#[test]
fn the_session_is_dusks_manual_turn_transcription_session() {
    assert_eq!(
        settings("https://api.openai.com/v1", &["en"]).session_update(),
        serde_json::json!({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": { "input": {
                    "format": { "type": "audio/pcm", "rate": 24000 },
                    "transcription": {
                        "model": "gpt-live-transcribe",
                        "delay": "low",
                        "languages": ["en"],
                        "prompt": "Technical dictation from a single speaker.",
                        "keywords": ["NixOS", "jj"],
                    },
                    "turn_detection": null,
                }},
            },
        })
    );
    // Detected language, a noise filter, and no prompt or keywords.
    let mut detected = settings("https://api.openai.com/v1", &[]);
    detected.config.noise_reduction = Some(NoiseReduction::NearField);
    detected.config.prompt = " ".into();
    detected.config.keywords.clear();
    detected.config.delay = Delay::Minimal;
    let input = &detected.session_update()["session"]["audio"]["input"];
    assert_eq!(
        input["transcription"],
        serde_json::json!({ "model": "gpt-live-transcribe", "delay": "minimal" })
    );
    assert_eq!(
        input["noise_reduction"],
        serde_json::json!({ "type": "near_field" })
    );
}

#[test]
fn resampling_matches_voxtype_and_ignores_chunking() {
    let input: Vec<f32> = (0..16_000)
        .map(|index| ((index as f32) * 0.01).sin() * 0.5)
        .collect();
    let mut whole = Resampler::new(SOURCE_RATE, TARGET_RATE);
    let all = whole.process(&input);
    let mut split = Resampler::new(SOURCE_RATE, TARGET_RATE);
    let mut pieces = Vec::new();
    for (index, chunk) in input.chunks(1_600).enumerate() {
        let out = split.process(chunk);
        // Voxtype's counts: 2,399 first, then 2,400 per 1,600 samples.
        assert_eq!(out.len(), if index == 0 { 2_399 } else { 2_400 });
        pieces.extend(out);
    }
    // Chunking moves the read phase by float rounding only (Voxtype's
    // bound: 1e-6); never a sample more or less.
    let same = |a: &[f32], b: &[f32]| {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-6)
    };
    assert!(same(&all, &pieces));
    let mut uneven = Resampler::new(SOURCE_RATE, TARGET_RATE);
    let mut odd = Vec::new();
    for chunk in input.chunks(337) {
        odd.extend(uneven.process(chunk));
    }
    assert!(same(&all, &odd));
    // Voxtype's wire encoding rounds `s * 32767`.
    assert_eq!(
        pcm16_le(&[1.0, -1.0, 0.5, 2.0]),
        [0xff, 0x7f, 0x01, 0x80, 0x00, 0x40, 0xff, 0x7f]
    );
}

/// What the fake Realtime server received.
#[derive(Default)]
struct Received {
    authorization: Option<String>,
    path: String,
    session: Option<serde_json::Value>,
    frames: Vec<Vec<u8>>,
    frame_times: Vec<StdInstant>,
    committed_after_frames: Option<usize>,
}

/// How the fake server answers.
#[derive(Clone, Copy)]
enum Script {
    /// committed, then the final.
    Normal,
    /// The final before the commit acknowledgement.
    FinalFirst,
    /// An error event instead of session.updated.
    RefuseSession,
    /// Accept everything, never answer the commit.
    Silent,
}

// tungstenite's header callback fixes its large error type.
#[allow(clippy::result_large_err)]
fn serve(script: Script) -> (String, Arc<Mutex<Received>>) {
    let received = Arc::new(Mutex::new(Received::default()));
    let record = received.clone();
    let (port_tx, port_rx) = std_mpsc::channel();
    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            port_tx.send(listener.local_addr().unwrap().port()).unwrap();
            let (stream, _) = listener.accept().await.unwrap();
            let headers = record.clone();
            let socket = tokio_tungstenite::accept_hdr_async(
                stream,
                move |request: &Request, response: Response| {
                    let mut seen = headers.lock().unwrap();
                    seen.path = request.uri().to_string();
                    seen.authorization = request
                        .headers()
                        .get("authorization")
                        .map(|value| value.to_str().unwrap().to_string());
                    Ok(response)
                },
            )
            .await
            .unwrap();
            let (mut write, mut read) = socket.split();
            let reply = |value: serde_json::Value| Message::text(value.to_string());
            while let Some(Ok(message)) = read.next().await {
                let Message::Text(text) = message else {
                    continue;
                };
                let event: serde_json::Value = serde_json::from_str(&text).unwrap();
                match event["type"].as_str().unwrap() {
                    "session.update" => {
                        record.lock().unwrap().session = Some(event.clone());
                        let answer = match script {
                            Script::RefuseSession => serde_json::json!({
                                "type": "error", "error": { "message": "bad keyword" }
                            }),
                            _ => serde_json::json!({ "type": "session.updated" }),
                        };
                        write.send(reply(answer)).await.unwrap();
                    }
                    "input_audio_buffer.append" => {
                        let audio = base64::engine::general_purpose::STANDARD
                            .decode(event["audio"].as_str().unwrap())
                            .unwrap();
                        let mut seen = record.lock().unwrap();
                        seen.frames.push(audio);
                        seen.frame_times.push(StdInstant::now());
                    }
                    "input_audio_buffer.commit" => {
                        {
                            let mut seen = record.lock().unwrap();
                            seen.committed_after_frames = Some(seen.frames.len());
                        }
                        let committed = serde_json::json!({
                            "type": "input_audio_buffer.committed", "item_id": "item_1",
                            "previous_item_id": null,
                        });
                        let done = serde_json::json!({
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "item_1", "content_index": 0,
                            "transcript": "  hello there  ",
                        });
                        match script {
                            Script::Normal => {
                                write.send(reply(committed)).await.unwrap();
                                write.send(reply(done)).await.unwrap();
                            }
                            Script::FinalFirst => {
                                write.send(reply(done)).await.unwrap();
                                write.send(reply(committed)).await.unwrap();
                            }
                            Script::RefuseSession | Script::Silent => {}
                        }
                    }
                    _ => {}
                }
            }
        });
    });
    let port = port_rx.recv().unwrap();
    (format!("http://127.0.0.1:{port}/v1"), received)
}

fn tone(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|index| ((index as f32) * 0.05).sin() * 0.25)
        .collect()
}

#[test]
fn a_capture_streams_paced_frames_then_commits_and_takes_the_final() {
    let (base, received) = serve(Script::Normal);
    let audio = tone(16_000);
    let mut stream = RealtimeStream::start(settings(&base, &["en"]), CancelToken::new());
    // The whole second arrives at once, as a backlog does.
    for block in audio.chunks(2_560) {
        stream.feed(block).unwrap();
    }
    let started = StdInstant::now();
    assert_eq!(stream.finalize().unwrap(), "hello there");
    let received = received.lock().unwrap();
    assert_eq!(received.authorization.as_deref(), Some("Bearer sk-test"));
    assert_eq!(received.path, "/v1/realtime?intent=transcription");
    assert_eq!(
        received.session.as_ref().unwrap()["session"]["audio"]["input"]["transcription"]["model"],
        "gpt-live-transcribe"
    );
    // Every sample arrived once, in order, in frames of at most 80 ms.
    let mut expected = Resampler::new(SOURCE_RATE, TARGET_RATE);
    let expected: Vec<u8> = audio
        .chunks(2_560)
        .flat_map(|block| expected.encode(block))
        .collect();
    assert_eq!(received.frames.concat(), expected);
    assert!(
        received
            .frames
            .iter()
            .all(|frame| frame.len() <= CHUNK_BYTES)
    );
    assert_eq!(received.committed_after_frames, Some(received.frames.len()));
    // One second of audio drains no faster than twice real time.
    let first = received.frame_times[0];
    let last = *received.frame_times.last().unwrap();
    let spread = last.duration_since(first);
    let audio_seconds = expected.len() as f64 / f64::from(TARGET_RATE * 2);
    let last_frame = received.frames.last().unwrap().len() as f64 / f64::from(TARGET_RATE * 2);
    let minimum = (audio_seconds - last_frame) / CATCH_UP_RATE;
    assert!(
        spread.as_secs_f64() >= minimum - 0.02,
        "{spread:?} < {minimum}"
    );
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn a_final_before_its_commit_acknowledgement_still_counts() {
    let (base, _) = serve(Script::FinalFirst);
    let mut stream = RealtimeStream::start(settings(&base, &["en"]), CancelToken::new());
    stream.feed(&tone(4_000)).unwrap();
    assert_eq!(stream.finalize().unwrap(), "hello there");
}

#[test]
fn a_refused_session_fails_the_stream() {
    let (base, _) = serve(Script::RefuseSession);
    let mut stream = RealtimeStream::start(settings(&base, &["en"]), CancelToken::new());
    let _ = stream.feed(&tone(4_000));
    let error = stream.finalize().unwrap_err();
    assert!(
        error.contains("refused the session") && error.contains("bad keyword"),
        "{error}"
    );
}

#[test]
fn no_server_fails_fast_and_feeding_reports_it() {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut stream = RealtimeStream::start(
        settings(&format!("http://127.0.0.1:{port}/v1"), &["en"]),
        CancelToken::new(),
    );
    let started = StdInstant::now();
    // Feeding keeps working until the failure is known, then reports it.
    let mut failed = None;
    while started.elapsed() < Duration::from_secs(5) {
        if let Err(reason) = stream.feed(&tone(160)) {
            failed = Some(reason);
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    assert!(failed.unwrap().contains("could not connect"));
    assert!(stream.finalize().is_err());
}

#[test]
fn cancelling_ends_a_finish_that_gets_no_final() {
    let (base, received) = serve(Script::Silent);
    let cancel = CancelToken::new();
    let mut stream = RealtimeStream::start(settings(&base, &["en"]), cancel.clone());
    stream.feed(&tone(1_600)).unwrap();
    let canceller = {
        let cancel = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(400));
            cancel.cancel();
        })
    };
    let started = StdInstant::now();
    let error = stream.finalize().unwrap_err();
    canceller.join().unwrap();
    assert!(error.contains("cancelled"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(2));
    // The audio was sent and committed before the wait.
    assert!(received.lock().unwrap().committed_after_frames.is_some());
}

/// The real API: `HEX_REALTIME_WAV` (16 kHz mono PCM16) streamed as a
/// hotkey capture streams it, a 750 ms backlog then 160 ms blocks at real
/// time, with the user's ~/.config/hex/openai.json and key. Prints the
/// transcript and timings for comparison with `hex transcribe`.
#[test]
#[ignore = "calls OpenAI and needs HEX_REALTIME_WAV"]
fn the_live_api_transcribes_a_streamed_capture() {
    let path = PathBuf::from(std::env::var("HEX_REALTIME_WAV").unwrap());
    let pcm = crate::openai_transcription::read_wav(&path).unwrap();
    let audio: Vec<f32> = pcm
        .iter()
        .map(|&sample| f32::from(sample) / 32768.0)
        .collect();
    let config = crate::openai_transcription::load_config().unwrap();
    let key = config.api_key().unwrap();
    let mut stream = RealtimeStream::start(
        RealtimeSettings {
            config,
            key,
            languages: vec!["en".into()],
        },
        CancelToken::new(),
    );
    let backlog = 12_000;
    stream.feed(&audio[..backlog.min(audio.len())]).unwrap();
    for block in audio[backlog.min(audio.len())..].chunks(2_560) {
        thread::sleep(Duration::from_millis(160));
        stream.feed(block).unwrap();
    }
    let released = StdInstant::now();
    let text = stream.finalize().unwrap();
    eprintln!(
        "release to final: {} ms\n{text}",
        released.elapsed().as_millis()
    );
    assert!(!text.is_empty());
}
