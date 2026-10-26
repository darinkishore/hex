use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::*;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forced-alignment")
}

#[derive(Deserialize)]
struct Fixture {
    samples: usize,
    segments: Vec<FixtureSegment>,
    words: Vec<FixtureWord>,
}

#[derive(Deserialize)]
struct FixtureSegment {
    start: f64,
    end: f64,
    text: String,
    emissions: String,
    frames: usize,
}

#[derive(Deserialize)]
struct FixtureWord {
    text: String,
    start: f64,
    end: f64,
    score: f64,
}

/// Replays recorded emissions in order, checking each segment's slice.
struct Recorded(VecDeque<(usize, Emission)>);

impl Emissions for Recorded {
    fn emissions(&mut self, samples: &[f32]) -> Result<Emission> {
        let (expected, emission) = self.0.pop_front().expect("one emission per segment");
        assert_eq!(samples.len(), expected, "segment slice length");
        Ok(emission)
    }
}

fn load(name: &str) -> (Fixture, Recorded) {
    let directory = fixtures();
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(directory.join(format!("{name}.json"))).unwrap()).unwrap();
    let recorded = fixture
        .segments
        .iter()
        .map(|segment| {
            let bytes = fs::read(directory.join(&segment.emissions)).unwrap();
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();
            let slice = (segment.end * 16_000.0) as usize - (segment.start * 16_000.0) as usize;
            (
                slice.min(fixture.samples),
                Emission::new(segment.frames, values),
            )
        })
        .collect();
    (fixture, Recorded(recorded))
}

fn segments(fixture: &Fixture) -> Vec<Segment> {
    fixture
        .segments
        .iter()
        .map(|segment| Segment {
            start: segment.start,
            end: segment.end,
            text: segment.text.clone(),
        })
        .collect()
}

/// Dusk's `voxtype-align` (WhisperX 3.8.6, torchaudio's model) produced
/// each fixture's words from these exact emissions. Hex must reproduce
/// every text, time, and score bit for bit: digits and punctuation take
/// the wildcard, and fixture `d` aligns three segments at offsets.
#[test]
fn recorded_emissions_reproduce_dusks_whisperx_alignment_exactly() {
    for name in ["a", "b", "c", "d"] {
        let (fixture, mut recorded) = load(name);
        let pcm = vec![0_i16; fixture.samples];
        let words = align(&mut recorded, &pcm, &segments(&fixture)).unwrap();
        assert!(recorded.0.is_empty(), "{name}: every segment was aligned");
        assert_eq!(words.len(), fixture.words.len(), "{name}");
        for (word, expected) in words.iter().zip(&fixture.words) {
            assert_eq!(word.text, expected.text, "{name}");
            assert_eq!(
                word.start.to_bits(),
                expected.start.to_bits(),
                "{name} {word:?}"
            );
            assert_eq!(
                word.end.to_bits(),
                expected.end.to_bits(),
                "{name} {word:?}"
            );
            assert_eq!(
                word.score.to_bits(),
                expected.score.to_bits(),
                "{name} {word:?}"
            );
        }
    }
}

#[test]
fn words_convert_to_milliseconds_as_dusks_scrawl_adapter_does() {
    let word = Word {
        text: "x".into(),
        start: 1.0005,
        end: 2.4995,
        score: 0.5,
    };
    assert_eq!((word.start_ms(), word.end_ms()), (1001, 2500));
}

fn constant(frames: usize) -> Emission {
    // Uniform log probabilities: every path scores the same.
    Emission::new(frames, vec![-(CLASSES as f32).ln(); frames * CLASSES])
}

struct Constant(usize);

impl Emissions for Constant {
    fn emissions(&mut self, _samples: &[f32]) -> Result<Emission> {
        Ok(constant(self.0))
    }
}

#[test]
fn more_characters_than_frames_fail_instead_of_guessing() {
    let pcm = vec![0_i16; 16_000];
    let segment = Segment {
        start: 0.0,
        end: 1.0,
        text: "far too many characters".into(),
    };
    let error = align(&mut Constant(3), &pcm, &[segment]).unwrap_err();
    assert!(
        format!("{error:#}").contains("backtrack failed"),
        "{error:#}"
    );
}

#[test]
fn empty_transcripts_are_skipped_and_nothing_aligns_to_nothing() {
    let pcm = vec![0_i16; 32_000];
    let segments = [
        Segment {
            start: 0.0,
            end: 1.0,
            text: " \u{1c} ".into(),
        },
        Segment {
            start: 1.0,
            end: 2.0,
            text: String::new(),
        },
    ];
    assert!(
        align(&mut Constant(50), &pcm, &segments)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn segments_must_be_ordered_bounded_and_inside_the_audio() {
    let pcm = vec![0_i16; 16_000 * 40];
    let segment = |start: f64, end: f64| Segment {
        start,
        end,
        text: "a".into(),
    };
    for segments in [
        vec![segment(1.0, 0.5)],
        vec![segment(0.0, 31.0)],
        vec![segment(0.0, 2.0), segment(1.0, 3.0)],
        vec![segment(39.0, 40.02)],
        vec![segment(f64::NAN, 1.0)],
    ] {
        let error = align(&mut Constant(50), &pcm, &segments).unwrap_err();
        assert!(
            format!("{error:#}").contains("invalid or unbounded"),
            "{error:#}"
        );
    }
    // Up to 10 ms past the end is accepted, as `voxtype-align` does.
    assert_eq!(
        align(&mut Constant(500), &pcm, &[segment(39.0, 40.01)])
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn words_keep_their_punctuation_and_split_only_at_spaces() {
    let pcm = vec![0_i16; 16_000 * 2];
    let segment = Segment {
        start: 0.0,
        end: 2.0,
        text: "  Yes.(No)\tok,  3 ".into(),
    };
    let words = align(&mut Constant(99), &pcm, &[segment]).unwrap();
    let texts: Vec<_> = words.iter().map(|word| word.text.as_str()).collect();
    assert_eq!(texts, ["Yes.(No)", "ok,", "3"]);
    let mut end = 0.0;
    for word in &words {
        assert!(end <= word.start && word.start < word.end && word.end <= 2.0);
        end = word.end;
    }
}

#[test]
fn a_segment_shorter_than_one_frame_is_padded_like_whisperx() {
    struct Expect(usize);
    impl Emissions for Expect {
        fn emissions(&mut self, samples: &[f32]) -> Result<Emission> {
            assert_eq!(samples.len(), MIN_SEGMENT_SAMPLES);
            Ok(constant(self.0))
        }
    }
    let pcm = vec![0_i16; 16_000];
    let segment = Segment {
        start: 0.99,
        end: 1.0,
        text: "a".into(),
    };
    assert_eq!(align(&mut Expect(1), &pcm, &[segment]).unwrap().len(), 1);
}

#[test]
fn audio_is_read_as_whisperxs_ffmpeg_loader_reads_it() {
    struct Capture(Vec<f32>);
    impl Emissions for Capture {
        fn emissions(&mut self, samples: &[f32]) -> Result<Emission> {
            self.0 = samples.to_vec();
            Ok(constant(40))
        }
    }
    let mut pcm = vec![0_i16; 16_000];
    pcm[0] = i16::MIN;
    pcm[1] = i16::MAX;
    pcm[2] = -1;
    let mut capture = Capture(Vec::new());
    let segment = Segment {
        start: 0.0,
        end: 1.0,
        text: "a".into(),
    };
    align(&mut capture, &pcm, &[segment]).unwrap();
    assert_eq!(capture.0[..3], [-1.0, 32767.0 / 32768.0, -1.0 / 32768.0]);
}

#[test]
fn aux_classes_are_removed_before_the_softmax() {
    let mut logits = vec![0.0_f32; MODEL_CLASSES];
    logits[1] = 50.0; // `<s>`, which torchaudio drops
    logits[4] = 2.0; // `|`, torchaudio's class 1
    let values = log_softmax_without_aux(&logits, 1);
    assert_eq!(values.len(), CLASSES);
    let total: f64 = values.iter().map(|value| f64::from(*value).exp()).sum();
    assert!((total - 1.0).abs() < 1e-6);
    assert!(values[1] > values[0] && values[0] == values[2]);
}

#[test]
fn rounding_matches_python() {
    // Ties go to even; values just under a tie stay under.
    assert_eq!(python_round3(0.0625), 0.062);
    assert_eq!(python_round3(2.675), 2.675);
    assert_eq!(python_round3(1.0005), 1.0);
    assert_eq!(python_round3(0.0005), 0.001);
    assert_eq!(python_round3(0.1235), 0.123);
}

#[test]
fn word_scores_sum_pairwise_as_numpy_does() {
    // NumPy's pairwise sum and a running sum disagree on these values;
    // pandas' mean, and so WhisperX, uses the pairwise one.
    let values = [
        0.47, 0.836, 0.476, 0.639, 0.151, 0.635, 0.868, 0.523, 0.741, 0.671, 0.064,
    ];
    assert_eq!(values.iter().sum::<f64>(), 6.074);
    assert_eq!(pairwise_sum(&values), 6.074000000000001);
    let short = [0.1, 0.2, 0.3];
    assert_eq!(pairwise_sum(&short), short.iter().sum::<f64>());
    let long: Vec<f64> = (0..300).map(|index| f64::from(index) * 0.001).collect();
    let mut half = long.len() / 2;
    half -= half % 8;
    assert_eq!(
        pairwise_sum(&long),
        pairwise_sum(&long[..half]) + pairwise_sum(&long[half..])
    );
}

#[test]
fn python_whitespace_includes_the_information_separators() {
    assert_eq!(python_split("a\u{1f}b\u{a0}c  d"), ["a", "b", "c", "d"]);
    assert_eq!(python_split("a\u{200b}b"), ["a\u{200b}b"]);
}

/// The real model on audio: `HEX_ALIGNMENT_WAVS` names a folder with
/// `a.wav`..`d.wav`, the fixtures' audio (16 kHz mono PCM16). Times must
/// match Dusk's alignment exactly; scores may differ by ONNX Runtime's
/// arithmetic, at most 0.002.
#[test]
#[ignore = "downloads the 378 MB model and needs the fixtures' audio"]
fn the_onnx_model_reproduces_dusks_alignment() {
    let directory = PathBuf::from(std::env::var("HEX_ALIGNMENT_WAVS").unwrap());
    let mut aligner = Aligner::load(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        &AtomicBool::new(false),
    )
    .unwrap();
    for name in ["a", "b", "c", "d"] {
        let (fixture, _) = load(name);
        let mut reader = hound::WavReader::open(directory.join(format!("{name}.wav"))).unwrap();
        let pcm: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        assert_eq!(pcm.len(), fixture.samples);
        let words = aligner.align(&pcm, &segments(&fixture)).unwrap();
        assert_eq!(words.len(), fixture.words.len());
        for (word, expected) in words.iter().zip(&fixture.words) {
            assert_eq!(
                (&word.text, word.start, word.end),
                (&expected.text, expected.start, expected.end),
                "{name}"
            );
            assert!(
                (word.score - expected.score).abs() <= 0.002,
                "{name} {word:?}"
            );
        }
    }
}
