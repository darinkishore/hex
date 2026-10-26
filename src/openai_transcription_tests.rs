use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use super::*;

fn temp_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let directory = std::env::temp_dir().join(format!(
        "hex-openai-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

#[test]
fn the_config_defaults_to_dusks_key_file_and_rejects_typos() {
    let home = Path::new("/Users/someone");
    let defaults = parse_config(None, home).unwrap();
    assert_eq!(defaults.api_key_file, home.join(".config/openai/api_key"));
    assert_eq!(defaults.delay, Delay::Low);
    assert_eq!(defaults.noise_reduction, None);
    assert_eq!(defaults.prompt(), None);
    assert!(defaults.keywords.is_empty());

    let config = parse_config(
        Some(
            br#"{"apiKeyFile":"~/keys/openai","prompt":"Nix talk.","keywords":["NixOS","jj"],
                "delay":"xhigh","noiseReduction":"far_field"}"#,
        ),
        home,
    )
    .unwrap();
    assert_eq!(config.api_key_file, home.join("keys/openai"));
    assert_eq!(config.prompt(), Some("Nix talk."));
    assert_eq!(config.keywords, ["NixOS", "jj"]);
    assert_eq!(config.delay, Delay::Xhigh);
    assert_eq!(config.noise_reduction, Some(NoiseReduction::FarField));

    assert!(parse_config(Some(br#"{"promt":"typo"}"#), home).is_err());
    assert!(parse_config(Some(br#"{"apiKeyFile":"relative/key"}"#), home).is_err());
    assert!(parse_config(Some(br#"{"keywords":["a<b"]}"#), home).is_err());
    assert!(parse_config(Some(br#"{"keywords":["two\nlines"]}"#), home).is_err());
    assert!(parse_config(Some(br#"{"delay":"fast"}"#), home).is_err());
    assert_eq!(
        parse_config(Some(br#"{"prompt":"  "}"#), home)
            .unwrap()
            .prompt(),
        None
    );
}

#[test]
fn realtime_and_file_endpoints_share_the_base() {
    let mut config = parse_config(None, Path::new("/home")).unwrap();
    assert_eq!(
        config.realtime_url(),
        "wss://api.openai.com/v1/realtime?intent=transcription"
    );
    assert_eq!(
        config.transcriptions_url(),
        "https://api.openai.com/v1/audio/transcriptions"
    );
    config.api_base = "http://127.0.0.1:9/v1".into();
    assert_eq!(
        config.realtime_url(),
        "ws://127.0.0.1:9/v1/realtime?intent=transcription"
    );
}

#[test]
fn the_key_file_must_be_private_and_hold_one_key() {
    let directory = temp_dir("key");
    let path = directory.join("api_key");
    let write = |contents: &str, mode: u32| {
        fs::write(&path, contents).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
    };
    write("sk-test-123\n", 0o600);
    let key = read_api_key(&path).unwrap();
    assert_eq!(key.expose(), "sk-test-123");
    assert_eq!(format!("{key:?}"), "ApiKey(<redacted>)");
    write("sk-test-123\n", 0o644);
    assert!(format!("{:#}", read_api_key(&path).unwrap_err()).contains("chmod 600"));
    write("  \n", 0o600);
    assert!(read_api_key(&path).is_err());
    write("sk-\"quoted\"", 0o600);
    assert!(read_api_key(&path).is_err());
    assert!(
        format!(
            "{:#}",
            read_api_key(&directory.join("missing")).unwrap_err()
        )
        .contains("no OpenAI API key")
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn pcm16_matches_scrawls_writer() {
    assert_eq!(pcm16(0.0), 0);
    assert_eq!(pcm16(1.0), 32_767);
    assert_eq!(pcm16(-1.0), -32_768);
    assert_eq!(pcm16(2.0), 32_767);
    assert_eq!(pcm16(f32::NAN), 0);
    assert_eq!(pcm16(0.5), 16_384);
    assert_eq!(pcm16(-1.0 / 32_768.0), -1);
}

#[test]
fn chunks_cover_every_sample_once_and_stay_within_30_seconds() {
    // Voxtype's own check.
    let audio = vec![0_i16; SAMPLE_RATE * 97 + 3];
    let ranges = chunk_ranges(&audio);
    assert_eq!(ranges.first().unwrap().0, 0);
    assert_eq!(ranges.last().unwrap().1, audio.len());
    for pair in ranges.windows(2) {
        assert_eq!(pair[0].1, pair[1].0);
    }
    assert!(
        ranges
            .iter()
            .all(|(a, b)| b > a && b - a <= MAX_CHUNK_SAMPLES)
    );
    assert_eq!(chunk_ranges(&[]), []);
    assert_eq!(chunk_ranges(&[0; 100]), [(0, 100)]);
}

#[test]
fn a_chunk_ends_at_the_centre_of_its_quietest_late_window() {
    // Steady sound, with one quiet 100 ms window three seconds before the
    // 30 s limit: the cut lands 50 ms into it.
    let mut audio = vec![8_000_i16; SAMPLE_RATE * 40];
    let quiet = 27 * SAMPLE_RATE;
    audio[quiet..quiet + CUT_WINDOW_SAMPLES].fill(0);
    assert_eq!(
        chunk_ranges(&audio),
        [(0, quiet + 800), (quiet + 800, audio.len())]
    );
    // Silence everywhere ties; the earliest window wins, at 25.05 s, as on
    // Dusk's 37 s fixture.
    let silent = vec![0_i16; 594_021];
    assert_eq!(chunk_ranges(&silent), [(0, 400_800), (400_800, 594_021)]);
}

/// One HTTP request as the mock server saw it.
struct Seen {
    head: String,
    body: Vec<u8>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    /// The multipart fields, in order, with the file's bytes.
    fn fields(&self) -> Vec<(String, Vec<u8>)> {
        let boundary = self
            .header("content-type")
            .and_then(|value| value.split_once("boundary="))
            .map(|(_, boundary)| format!("--{boundary}"))
            .expect("a multipart boundary");
        let mut fields = Vec::new();
        let body = &self.body;
        let mut rest = &body[..];
        while let Some(start) = find(rest, boundary.as_bytes()) {
            rest = &rest[start + boundary.len()..];
            if rest.starts_with(b"--") {
                break;
            }
            let header_end = find(rest, b"\r\n\r\n").unwrap();
            let headers = String::from_utf8_lossy(&rest[..header_end]).to_string();
            let name = headers
                .split("name=\"")
                .nth(1)
                .and_then(|after| after.split('"').next())
                .unwrap()
                .to_string();
            let content = &rest[header_end + 4..];
            let end = find(content, format!("\r\n{boundary}").as_bytes()).unwrap();
            fields.push((name, content[..end].to_vec()));
            rest = &content[end + 2..];
        }
        fields
    }

    fn text_fields(&self) -> Vec<(String, String)> {
        self.fields()
            .into_iter()
            .filter(|(name, _)| name != "file")
            .map(|(name, value)| (name, String::from_utf8(value).unwrap()))
            .collect()
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn read_request(stream: &mut TcpStream) -> Seen {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut head = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line.is_empty() {
            break;
        }
        head.push_str(&line);
    }
    let seen = Seen {
        head,
        body: Vec::new(),
    };
    let length: usize = seen.header("content-length").unwrap().parse().unwrap();
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    Seen { body, ..seen }
}

fn respond(stream: &mut TcpStream, status: u16, body: &str) {
    write!(
        stream,
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}

/// A server answering each request with `reply(request number, request)`.
fn serve(
    reply: impl Fn(u32, &Seen) -> (u16, String) + Send + Sync + 'static,
) -> (String, Arc<Mutex<Vec<Seen>>>, Arc<AtomicU32>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    let seen = Arc::new(Mutex::new(Vec::new()));
    let count = Arc::new(AtomicU32::new(0));
    let (record, counter) = (seen.clone(), count.clone());
    let reply = Arc::new(reply);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let (record, counter, reply) = (record.clone(), counter.clone(), reply.clone());
            thread::spawn(move || {
                let request = read_request(&mut stream);
                let number = counter.fetch_add(1, Ordering::AcqRel) + 1;
                let (status, body) = reply(number, &request);
                respond(&mut stream, status, &body);
                record.lock().unwrap().push(request);
            });
        }
    });
    (base, seen, count)
}

fn config_at(base: &str) -> OpenAiConfig {
    OpenAiConfig {
        api_key_file: PathBuf::from("/unused"),
        prompt: "Technical dictation.".into(),
        keywords: vec!["NixOS".into(), "jj".into()],
        delay: Delay::Low,
        noise_reduction: None,
        api_base: base.into(),
    }
}

fn never() -> bool {
    false
}

#[test]
fn a_file_request_carries_voxtypes_fields_and_the_chunks_own_pcm() {
    let (base, seen, _) = serve(|_, _| (200, r#"{"text":"  hello there  "}"#.into()));
    let pcm: Vec<i16> = (0..16_000)
        .map(|index| (index % 2000) as i16 - 1000)
        .collect();
    let languages = ["en".to_string()];
    let text = transcribe_text(
        &config_at(&base),
        &ApiKey::for_test("sk-secret"),
        &pcm,
        &languages,
        &never,
    )
    .unwrap();
    assert_eq!(text, "hello there");
    let seen = seen.lock().unwrap();
    let [request] = seen.as_slice() else {
        panic!("one request")
    };
    assert!(request.head.starts_with("POST /v1/audio/transcriptions "));
    assert_eq!(request.header("authorization"), Some("Bearer sk-secret"));
    assert_eq!(
        request.text_fields(),
        [
            ("model".into(), "gpt-transcribe".into()),
            ("response_format".into(), "json".into()),
            ("prompt".into(), "Technical dictation.".into()),
            ("keywords[]".into(), "NixOS".into()),
            ("keywords[]".into(), "jj".into()),
            ("languages[]".into(), "en".into()),
        ]
    );
    let (_, file) = request
        .fields()
        .into_iter()
        .find(|(name, _)| name == "file")
        .unwrap();
    let mut reader = hound::WavReader::new(Cursor::new(file)).unwrap();
    assert_eq!(reader.spec().sample_rate, 16_000);
    assert_eq!(reader.spec().channels, 1);
    let uploaded: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    assert_eq!(uploaded, pcm);
}

#[test]
fn curl_gets_no_secret_on_its_command_line() {
    // `--disable` must come first to skip ~/.curlrc; the URL and headers,
    // with the key, travel in the config on descriptor 3.
    let arguments = curl_arguments(
        "https://api.openai.com/v1/audio/transcriptions",
        REQUEST_TIMEOUT,
    );
    assert_eq!(arguments[0], "--disable");
    assert!(
        arguments
            .windows(2)
            .any(|pair| pair == ["--config", "/dev/fd/3"])
    );
    assert!(
        arguments
            .windows(2)
            .any(|pair| pair == ["--proto", "=https"])
    );
    assert!(
        !arguments
            .iter()
            .any(|argument| argument.contains("openai.com"))
    );
    assert!(
        curl_arguments("http://127.0.0.1:9/v1/x", REQUEST_TIMEOUT)
            .windows(2)
            .any(|pair| pair == ["--proto", "=http"])
    );
    assert!(
        curl_arguments("http://example.com/v1/x", REQUEST_TIMEOUT)
            .windows(2)
            .any(|pair| pair == ["--proto", "=https"])
    );
}

#[test]
fn auto_language_sends_no_languages_and_a_blank_prompt_none() {
    let (base, seen, _) = serve(|_, _| (200, r#"{"text":"bonjour"}"#.into()));
    let mut config = config_at(&base);
    config.prompt = "   ".into();
    config.keywords.clear();
    let languages = languages_for("auto");
    assert!(languages.is_empty());
    assert_eq!(languages_for("fr"), ["fr"]);
    transcribe_text(
        &config,
        &ApiKey::for_test("k"),
        &[0; 1600],
        &languages,
        &never,
    )
    .unwrap();
    assert_eq!(
        seen.lock().unwrap()[0].text_fields(),
        [
            ("model".into(), "gpt-transcribe".into()),
            ("response_format".into(), "json".into()),
        ]
    );
}

#[test]
fn transient_failures_retry_and_permanent_ones_do_not() {
    // A server error, then success.
    let (base, _, count) = serve(|number, _| match number {
        1 => (
            503,
            r#"{"error":{"code":"overloaded","message":"busy"}}"#.into(),
        ),
        _ => (200, r#"{"text":"second try"}"#.into()),
    });
    let text = transcribe_text(
        &config_at(&base),
        &ApiKey::for_test("k"),
        &[0; 1600],
        &[],
        &never,
    )
    .unwrap();
    assert_eq!(
        (text.as_str(), count.load(Ordering::Acquire)),
        ("second try", 2)
    );

    // An empty balance is not a rate limit: no retry, and the reason shows.
    let (base, _, count) = serve(|_, _| {
        (
            429,
            r#"{"error":{"type":"insufficient_quota","code":"credit_balance_exhausted","message":"You have no credits remaining."}}"#
                .into(),
        )
    });
    let error = transcribe_text(
        &config_at(&base),
        &ApiKey::for_test("k"),
        &[0; 1600],
        &[],
        &never,
    )
    .unwrap_err();
    assert_eq!(count.load(Ordering::Acquire), 1);
    let message = format!("{error:#}");
    assert!(message.contains("429 credit_balance_exhausted") && message.contains("no credits"));

    let (base, _, count) = serve(|_, _| {
        (
            429,
            r#"{"error":{"type":"insufficient_quota","code":"organization_spend_limit_exceeded"}}"#
                .into(),
        )
    });
    assert!(
        transcribe_text(
            &config_at(&base),
            &ApiKey::for_test("k"),
            &[0; 1600],
            &[],
            &never
        )
        .is_err()
    );
    assert_eq!(count.load(Ordering::Acquire), 1);

    let (base, _, count) = serve(|_, _| (401, r#"{"error":{"code":"invalid_api_key"}}"#.into()));
    assert!(
        transcribe_text(
            &config_at(&base),
            &ApiKey::for_test("k"),
            &[0; 1600],
            &[],
            &never
        )
        .is_err()
    );
    assert_eq!(count.load(Ordering::Acquire), 1);
}

#[test]
fn an_unreachable_server_fails_after_three_attempts() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    drop(listener);
    let started = Instant::now();
    let error = transcribe_text(
        &config_at(&base),
        &ApiKey::for_test("k"),
        &[0; 1600],
        &[],
        &never,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("could not reach OpenAI"));
    // One and then two seconds between the attempts.
    assert!(started.elapsed() >= Duration::from_secs(3));
}

#[test]
fn long_audio_is_transcribed_in_parallel_chunks_joined_in_order() {
    // Each answer names its chunk by length; the first is slowest.
    let (base, seen, _) = serve(|_, request| {
        let file = request
            .fields()
            .into_iter()
            .find(|(name, _)| name == "file")
            .unwrap()
            .1;
        let samples = (file.len() - 44) / 2;
        if samples == 400_800 {
            thread::sleep(Duration::from_millis(300));
        }
        let text = if samples == 80_000 {
            String::new()
        } else {
            format!("chunk{samples}")
        };
        (200, serde_json::json!({ "text": text }).to_string())
    });
    // Silence cuts at the earliest window: 25.05 s, twice, then the rest.
    let pcm = vec![0_i16; 400_800 * 2 + 80_000];
    let chunks = transcribe_chunks(
        &config_at(&base),
        &ApiKey::for_test("k"),
        &pcm,
        &FileRequest {
            model: FILE_MODEL,
            languages: &[],
        },
        &never,
    )
    .unwrap();
    assert_eq!(seen.lock().unwrap().len(), 3);
    assert_eq!(
        chunks,
        [
            Chunk {
                start: 0,
                end: 400_800,
                text: "chunk400800".into()
            },
            Chunk {
                start: 400_800,
                end: 801_600,
                text: "chunk400800".into()
            },
            Chunk {
                start: 801_600,
                end: 881_600,
                text: String::new()
            },
        ]
    );
    // Empty chunks drop out of the text.
    assert_eq!(joined(&chunks), "chunk400800 chunk400800");
}

#[test]
fn one_failed_chunk_fails_the_transcription_with_its_reason() {
    let (base, _, _) = serve(|_, request| {
        let file_bytes = request
            .fields()
            .into_iter()
            .find(|(n, _)| n == "file")
            .unwrap()
            .1
            .len();
        if file_bytes == 44 + 2 * 80_000 {
            (
                400,
                r#"{"error":{"code":"bad_audio","message":"no"}}"#.into(),
            )
        } else {
            (200, r#"{"text":"fine"}"#.into())
        }
    });
    let pcm = vec![0_i16; 400_800 * 2 + 80_000];
    let error =
        transcribe_text(&config_at(&base), &ApiKey::for_test("k"), &pcm, &[], &never).unwrap_err();
    assert!(format!("{error:#}").contains("400 bad_audio"));
}

#[test]
fn cancelling_kills_the_request() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    // Accept and never answer.
    let _hold = thread::spawn(move || {
        let held: Vec<_> = listener.incoming().take(1).collect();
        thread::sleep(Duration::from_secs(10));
        drop(held);
    });
    let cancelled = AtomicBool::new(false);
    let started = Instant::now();
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(300));
            cancelled.store(true, Ordering::Release);
        });
        let error = transcribe_text(
            &config_at(&base),
            &ApiKey::for_test("k"),
            &[0; 1600],
            &[],
            &|| cancelled.load(Ordering::Acquire),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("cancelled"));
    });
    assert!(started.elapsed() < Duration::from_secs(3));
}

fn word(text: &str, start: f64, end: f64) -> Word {
    Word {
        text: text.into(),
        start,
        end,
        score: 0.9,
    }
}

#[test]
fn chunks_align_as_dusks_segments_and_must_keep_every_word() {
    let chunks = [
        Chunk {
            start: 0,
            end: 400_800,
            text: "First open the window.".into(),
        },
        Chunk {
            start: 400_800,
            end: 594_021,
            text: String::new(),
        },
    ];
    let pcm = vec![0_i16; 594_021];
    let timed = align_chunks(&pcm, &chunks, |audio, segments| {
        assert_eq!(audio.len(), 594_021);
        assert_eq!(
            segments,
            [
                forced_alignment::Segment {
                    start: 0.0,
                    end: 25.05,
                    text: "First open the window.".into()
                },
                forced_alignment::Segment {
                    start: 25.05,
                    end: 37.1263125,
                    text: String::new()
                },
            ]
        );
        Ok(vec![
            word("First", 1.0, 1.2),
            word("open", 1.3, 1.5),
            word("the", 1.6, 1.7),
            word("window.", 1.8, 2.2),
        ])
    })
    .unwrap();
    assert_eq!(timed.text, "First open the window.");
    assert_eq!(timed.words.len(), 4);

    // A word the aligner lost is an error, never a guess.
    assert!(align_chunks(&pcm, &chunks, |_, _| Ok(vec![word("First", 1.0, 1.2)])).is_err());
    // Nothing said: no alignment at all.
    let silent = [Chunk {
        start: 0,
        end: 10,
        text: String::new(),
    }];
    let timed = align_chunks(&pcm[..10], &silent, |_, _| panic!("nothing to align")).unwrap();
    assert!(timed.text.is_empty() && timed.words.is_empty());
}

#[test]
fn scrawl_gets_measured_words_and_sentences_in_milliseconds() {
    let transcript = TimedTranscript {
        text: "Look here. Then (there.) click".into(),
        words: vec![
            word("Look", 1.2, 1.5),
            word("here.", 1.6, 2.0),
            word("Then", 2.5004, 2.6),
            word("(there.)", 2.7, 3.0005),
            word("click", 3.2, 3.4),
        ],
    };
    assert_eq!(
        scrawl_output(&transcript),
        serde_json::json!({
            "version": 1,
            "model": "gpt-transcribe",
            "timing": "whisperx:wav2vec2-base-960h",
            "text": "Look here. Then (there.) click",
            "estimated": false,
            "words": [
                {"t0": 1200, "t1": 1500, "text": "Look"},
                {"t0": 1600, "t1": 2000, "text": "here."},
                {"t0": 2500, "t1": 2600, "text": "Then"},
                {"t0": 2700, "t1": 3001, "text": "(there.)"},
                {"t0": 3200, "t1": 3400, "text": "click"},
            ],
            "segments": [
                {"t0": 1200, "t1": 2000, "text": "Look here."},
                {"t0": 2500, "t1": 3001, "text": "Then (there.)"},
                {"t0": 3200, "t1": 3400, "text": "click"},
            ],
        })
    );
    let empty = TimedTranscript {
        text: String::new(),
        words: Vec::new(),
    };
    assert_eq!(scrawl_output(&empty)["segments"], serde_json::json!([]));
}

#[test]
fn wav_files_read_as_16_khz_pcm16() {
    let directory = temp_dir("wav");
    let write = |name: &str, spec: hound::WavSpec, samples: &[f32]| {
        let path = directory.join(name);
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for sample in samples {
            match spec.sample_format {
                hound::SampleFormat::Float => writer.write_sample(*sample).unwrap(),
                hound::SampleFormat::Int => writer.write_sample(pcm16(*sample)).unwrap(),
            }
        }
        writer.finalize().unwrap();
        path
    };
    let mono = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let path = write("mono.wav", mono, &[0.5, -0.25, 1.0]);
    assert_eq!(read_wav(&path).unwrap(), [16_384, -8_192, 32_767]);
    let stereo = hound::WavSpec {
        channels: 2,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
        ..mono
    };
    let path = write("stereo.wav", stereo, &[0.5, 0.0, -1.0, -1.0]);
    assert_eq!(read_wav(&path).unwrap(), [8_192, -32_768]);
    let path = write(
        "fast.wav",
        hound::WavSpec {
            sample_rate: 48_000,
            ..mono
        },
        &[0.0],
    );
    assert!(format!("{:#}", read_wav(&path).unwrap_err()).contains("48000 Hz"));
    fs::remove_dir_all(directory).unwrap();
}
