//! Acoustic word timing for transcripts that arrive without any, such as
//! OpenAI's file transcription: WhisperX 3.8.6's CTC forced alignment,
//! ported exactly, on the model Dusk's `voxtype-align` loads (torchaudio's
//! `WAV2VEC2_ASR_BASE_960H`). Hex runs the same weights from their Hugging
//! Face conversion as ONNX, through the process's one ONNX Runtime.
//!
//! The contract is `voxtype-align`'s: contiguous segments of at most 30 s,
//! each with its transcript, and every word of every transcript comes back
//! with an acoustic start, end, and score, or the whole alignment fails.
//! Nothing is interpolated, spread by length, or guessed.
//!
//! Exact parts: text cleaning (lowercase, `|` for spaces, a wildcard column
//! for characters outside the model's alphabet), the trellis and backtrack,
//! repeat merging, the frame-to-seconds ratio, Python's `round(x, 3)`, and
//! pandas' pairwise mean for word scores. The emissions are the one
//! approximate step: ONNX Runtime's arithmetic differs from torch's by
//! about 0.002 in log probability, which on the fixtures moves no word
//! boundary and one score by 0.001.
//!
//! One deliberate difference: WhisperX groups words by NLTK Punkt
//! sentences, which never changes a word's timing but, when Punkt splits a
//! sentence without whitespace (`"Yes.(No)"`), duplicates the straddling
//! word and fails validation. Hex groups by spaces alone and aligns it.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use fs2::FileExt;
use serde::{Deserialize, Serialize};

pub const SAMPLE_RATE: usize = 16_000;

/// torchaudio's labels for `WAV2VEC2_ASR_BASE_960H`, lowercased as WhisperX
/// keys its dictionary. The first is the CTC blank.
const LABELS: [char; CLASSES] = [
    '-', '|', 'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'u', 'm', 'w', 'c', 'f', 'g',
    'y', 'p', 'b', 'v', 'k', '\'', 'x', 'j', 'q', 'z',
];
const CLASSES: usize = 29;
const BLANK: usize = 0;
/// The Hugging Face model's 32 logits are torchaudio's 29 with fairseq's
/// `<s>`, `</s>` and `<unk>` at 1, 2 and 3, which torchaudio removes.
const MODEL_CLASSES: usize = 32;
const REMOVED_CLASSES: std::ops::RangeInclusive<usize> = 1..=3;
/// wav2vec2's first convolution needs this many samples for one frame.
const MIN_SEGMENT_SAMPLES: usize = 400;
/// `voxtype-align`'s bounds: segments at most 30 s, ends up to 10 ms past
/// the audio.
const MAX_SEGMENT_SECONDS: f64 = 30.01;
const END_SLACK_SECONDS: f64 = 0.01;

const MODEL_FILENAME: &str = "wav2vec2-base-960h.onnx";
const MODEL_URL: &str = "https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/a19f851b3d42865797e410752b4c570c871e4825/onnx/model.onnx";
const MODEL_BYTES: u64 = 377_887_594;
const MODEL_SHA256: &str = "e46614273f03ff4b87923a965e417fa72004825522cb007c9c25633b8475490d";
/// Identifies the alignment in transcripts, as Dusk's `timing` field does.
pub const TIMING: &str = "whisperx:wav2vec2-base-960h";

/// One contiguous span of the audio, in seconds, and what was said in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// A word with WhisperX's acoustic timing: seconds rounded to the
/// millisecond and the mean probability of its characters' frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub score: f64,
}

impl Word {
    /// Milliseconds as Dusk's Scrawl adapter converts them.
    pub fn start_ms(&self) -> u64 {
        (self.start * 1000.0).round() as u64
    }

    pub fn end_ms(&self) -> u64 {
        (self.end * 1000.0).round() as u64
    }
}

/// Frame log probabilities over [`LABELS`] for one segment's samples.
pub(crate) trait Emissions {
    fn emissions(&mut self, samples: &[f32]) -> Result<Emission>;
}

pub(crate) struct Emission {
    frames: usize,
    /// `frames` rows of [`CLASSES`] log probabilities.
    values: Vec<f32>,
}

impl Emission {
    #[cfg(test)]
    pub(crate) fn new(frames: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), frames * CLASSES);
        Self { frames, values }
    }
}

/// The wav2vec2 model on ONNX Runtime.
pub struct Aligner {
    session: ort::session::Session,
}

impl Aligner {
    /// Loads the model, downloading and verifying it first when it is
    /// missing. The download stops once `cancelled` is set.
    pub fn load(project_root: &Path, cancelled: &AtomicBool) -> Result<Self> {
        crate::onnx_runtime::init(project_root)?;
        let model = install_model(cancelled)?;
        let threads = thread::available_parallelism().map_or(4, |count| count.get().min(8));
        let builder_error =
            |error: ort::Error<ort::session::builder::SessionBuilder>| eyre!("{error}");
        let session = ort::session::Session::builder()
            .map_err(|error| eyre!("{error}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
            .map_err(builder_error)?
            .with_intra_threads(threads)
            .map_err(builder_error)?
            .with_inter_threads(1)
            .map_err(builder_error)?
            .commit_from_file(&model)
            .map_err(|error| eyre!("could not load {}: {error}", model.display()))?;
        Ok(Self { session })
    }

    /// Aligns `segments` of 16 kHz PCM16 audio; see [`align`].
    pub fn align(&mut self, pcm: &[i16], segments: &[Segment]) -> Result<Vec<Word>> {
        align(self, pcm, segments)
    }
}

impl Emissions for Aligner {
    fn emissions(&mut self, samples: &[f32]) -> Result<Emission> {
        let input = ort::value::Tensor::from_array((
            [1_usize, samples.len()],
            samples.to_vec().into_boxed_slice(),
        ))
        .map_err(|error| eyre!("{error}"))?;
        let outputs = self
            .session
            .run(ort::inputs!["input_values" => input])
            .map_err(|error| eyre!("{error}"))?;
        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| eyre!("{error}"))?;
        ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[2] == MODEL_CLASSES as i64,
            "the alignment model returned logits of shape {shape:?}"
        );
        let frames = usize::try_from(shape[1])?;
        Ok(Emission {
            frames,
            values: log_softmax_without_aux(logits, frames),
        })
    }
}

/// torchaudio's 29 classes from the model's 32, then `log_softmax` over
/// them, computed in f64 and stored as torch's f32.
fn log_softmax_without_aux(logits: &[f32], frames: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(frames * CLASSES);
    for row in logits.as_chunks::<MODEL_CLASSES>().0.iter().take(frames) {
        let kept = row
            .iter()
            .enumerate()
            .filter(|(class, _)| !REMOVED_CLASSES.contains(class))
            .map(|(_, logit)| f64::from(*logit));
        let max = kept.clone().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = kept
            .clone()
            .map(|logit| (logit - max).exp())
            .sum::<f64>()
            .ln();
        values.extend(kept.map(|logit| (logit - max - log_sum) as f32));
    }
    values
}

/// Aligns each segment's transcript to its span of `pcm` (16 kHz mono,
/// read as WhisperX's ffmpeg loader does: `sample / 32768`) and validates
/// the result as `voxtype-align` does. Segments must be in order, at most
/// 30 s, and within the audio; empty transcripts are skipped.
pub(crate) fn align(
    model: &mut impl Emissions,
    pcm: &[i16],
    segments: &[Segment],
) -> Result<Vec<Word>> {
    let audio: Vec<f32> = pcm.iter().map(|&sample| sample as f32 / 32768.0).collect();
    let duration = pcm.len() as f64 / SAMPLE_RATE as f64;
    let mut selected = Vec::new();
    let mut previous = 0.0;
    for segment in segments {
        let (start, end) = (segment.start, segment.end);
        ensure!(
            start.is_finite()
                && end.is_finite()
                && previous <= start
                && start < end
                && end <= duration + END_SLACK_SECONDS
                && end - start <= MAX_SEGMENT_SECONDS,
            "invalid or unbounded alignment segment"
        );
        previous = end;
        let text = python_split(&segment.text).join(" ");
        if !text.is_empty() {
            selected.push(Segment { start, end, text });
        }
    }
    let mut words = Vec::new();
    for (index, segment) in selected.iter().enumerate() {
        let aligned = align_segment(model, &audio, segment)
            .wrap_err_with(|| format!("could not align segment {}", index + 1))?;
        words.extend(aligned);
    }
    let expected = selected
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    validate(&expected, &words, duration)?;
    Ok(words)
}

/// `voxtype-align`'s check: the words are exactly the transcript's, each
/// with finite acoustic timing inside the audio, in order, scored in [0, 1].
fn validate(expected: &str, words: &[Word], duration: f64) -> Result<()> {
    let tokens = python_split(expected);
    ensure!(
        tokens.len() == words.len() && tokens.iter().zip(words).all(|(t, w)| *t == w.text),
        "alignment omitted or changed transcript words"
    );
    let mut end = 0.0;
    for word in words {
        ensure!(
            word.start.is_finite() && word.end.is_finite() && word.score.is_finite(),
            "word {} lacks acoustic alignment",
            word.text
        );
        ensure!(
            end <= word.start
                && word.start < word.end
                && word.end <= duration + END_SLACK_SECONDS
                && (0.0..=1.0).contains(&word.score),
            "invalid acoustic word boundaries for {}",
            word.text
        );
        end = word.end;
    }
    Ok(())
}

/// One segment as `whisperx.align` handles it.
fn align_segment(
    model: &mut impl Emissions,
    audio: &[f32],
    segment: &Segment,
) -> Result<Vec<Word>> {
    let text: Vec<char> = segment.text.chars().collect();
    // 1. Keep characters in the dictionary, `|` for spaces, and every
    // other non-space as a wildcard. A character whose lowercase is
    // several code points contributes several tokens, as in WhisperX.
    let mut clean = String::new();
    let mut clean_indices = Vec::new();
    for (index, character) in text.iter().enumerate() {
        let lowered: String = if *character == ' ' {
            "|".into()
        } else {
            character.to_lowercase().collect()
        };
        if lowered != " " {
            clean.push_str(&lowered);
            clean_indices.push(index);
        }
    }
    ensure!(
        !clean_indices.is_empty(),
        "no characters to align in this segment"
    );
    let duration = audio.len() as f64 / SAMPLE_RATE as f64;
    ensure!(
        segment.start < duration,
        "the segment starts after the audio ends"
    );
    // 2. The segment's samples, exactly as WhisperX slices them.
    let first = ((segment.start * SAMPLE_RATE as f64) as usize).min(audio.len());
    let last = ((segment.end * SAMPLE_RATE as f64) as usize).clamp(first, audio.len());
    let mut samples = audio[first..last].to_vec();
    if samples.len() < MIN_SEGMENT_SAMPLES {
        // WhisperX pads with zeros too; its Hugging Face path, like this
        // model, has no length mask.
        samples.resize(MIN_SEGMENT_SAMPLES, 0.0);
    }
    let emission = model.emissions(&samples)?;
    ensure!(
        emission.frames > 0,
        "the alignment model returned no frames"
    );
    // 3. Tokens, with a wildcard column (each frame's best non-blank class)
    // for characters outside the alphabet.
    let index_of = |character: char| LABELS.iter().position(|label| *label == character);
    let wildcard = clean.chars().any(|character| index_of(character).is_none());
    let (columns, emission) = if wildcard {
        let mut extended = Vec::with_capacity(emission.frames * (CLASSES + 1));
        for row in emission.values.as_chunks::<CLASSES>().0 {
            extended.extend_from_slice(row);
            extended.push(row[1..].iter().copied().fold(f32::NEG_INFINITY, f32::max));
        }
        (CLASSES + 1, extended)
    } else {
        (CLASSES, emission.values)
    };
    let tokens: Vec<usize> = clean
        .chars()
        .map(|character| index_of(character).unwrap_or(CLASSES))
        .collect();
    let frames = emission.len() / columns;
    let at = |frame: usize, class: usize| emission[frame * columns + class];
    // 4. Viterbi trellis, backtrack, and per-token segments.
    let trellis = Trellis::new(frames, &tokens, &at);
    let path = trellis
        .backtrack(&tokens, &at)
        .ok_or_else(|| eyre!("the alignment backtrack failed"))?;
    let tokens_aligned = merge_repeats(&path);
    // 5. Character times, then words split at spaces.
    let ratio = (segment.end - segment.start) / frames as f64;
    let mut words: Vec<Word> = Vec::new();
    let mut current = WordChars::default();
    for (index, character) in text.iter().enumerate() {
        if *character != ' ' {
            let position = clean_indices
                .iter()
                .position(|clean| *clean == index)
                .expect("every non-space character is aligned");
            let token = tokens_aligned
                .get(position)
                .ok_or_else(|| eyre!("the alignment path skipped a character"))?;
            current.add(
                *character,
                python_round3(token.start as f64 * ratio + segment.start),
                python_round3(token.end as f64 * ratio + segment.start),
                python_round3(token.score),
            );
        } else {
            current.text.push(*character);
        }
        if (index + 1 == text.len() || text[index + 1] == ' ')
            && let Some(word) = std::mem::take(&mut current).finish()
        {
            words.push(word);
        }
    }
    Ok(words)
}

/// The characters of one WhisperX word: its text (stripped at the end)
/// and the timing of its non-space characters.
#[derive(Default)]
struct WordChars {
    text: String,
    start: f64,
    end: f64,
    scores: Vec<f64>,
}

impl WordChars {
    fn add(&mut self, character: char, start: f64, end: f64, score: f64) {
        if self.scores.is_empty() {
            (self.start, self.end) = (start, end);
        } else {
            self.start = self.start.min(start);
            self.end = self.end.max(end);
        }
        self.text.push(character);
        self.scores.push(score);
    }

    fn finish(self) -> Option<Word> {
        let text = python_strip(&self.text);
        if text.is_empty() {
            return None;
        }
        // Every character of a non-empty word is aligned, so it has times.
        Some(Word {
            text: text.to_string(),
            start: self.start,
            end: self.end,
            score: python_round3(pairwise_sum(&self.scores) / self.scores.len() as f64),
        })
    }
}

/// The trellis of `whisperx.alignment.get_trellis`, in torch's f32.
struct Trellis {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

impl Trellis {
    fn new(frames: usize, tokens: &[usize], at: &impl Fn(usize, usize) -> f32) -> Self {
        let (rows, columns) = (frames + 1, tokens.len() + 1);
        let mut values = vec![0.0_f32; rows * columns];
        let index = |row: usize, column: usize| row * columns + column;
        values[index(0, 0)] = 0.0;
        // torch.cumsum accumulates f32 in f64 on the CPU.
        let mut sum = 0.0_f64;
        for frame in 0..frames {
            sum += f64::from(at(frame, BLANK));
            values[index(frame + 1, 0)] = sum as f32;
        }
        let token_count = tokens.len();
        for column in columns.saturating_sub(token_count)..columns {
            values[index(0, column)] = f32::NEG_INFINITY;
        }
        // Python's `trellis[-num_tokens:, 0]` covers every row when the
        // tokens outnumber them.
        for row in rows.saturating_sub(token_count)..rows {
            values[index(row, 0)] = f32::INFINITY;
        }
        for frame in 0..frames {
            let blank = at(frame, BLANK);
            for column in 1..columns {
                let stayed = values[index(frame, column)] + blank;
                let changed = values[index(frame, column - 1)] + at(frame, tokens[column - 1]);
                values[index(frame + 1, column)] = torch_maximum(stayed, changed);
            }
        }
        Self {
            rows,
            columns,
            values,
        }
    }

    fn get(&self, row: usize, column: usize) -> f32 {
        self.values[row * self.columns + column]
    }

    /// `whisperx.alignment.backtrack`: from the best final frame back to
    /// the first token, one point per frame. `None` when it runs out of
    /// frames first.
    fn backtrack(&self, tokens: &[usize], at: &impl Fn(usize, usize) -> f32) -> Option<Vec<Point>> {
        let mut column = self.columns - 1;
        // torch.argmax returns the first maximum; NaN counts as the maximum.
        let mut start = 0;
        for row in 1..self.rows {
            let (value, best) = (self.get(row, column), self.get(start, column));
            if value > best || (value.is_nan() && !best.is_nan()) {
                start = row;
            }
        }
        let mut path = Vec::new();
        let mut reached = false;
        for row in (1..=start).rev() {
            let stayed = self.get(row - 1, column) + at(row - 1, BLANK);
            let changed = self.get(row - 1, column - 1) + at(row - 1, tokens[column - 1]);
            let class = if changed > stayed {
                tokens[column - 1]
            } else {
                BLANK
            };
            path.push(Point {
                token: column - 1,
                frame: row - 1,
                score: f64::from(at(row - 1, class).exp()),
            });
            if changed > stayed {
                column -= 1;
                if column == 0 {
                    reached = true;
                    break;
                }
            }
        }
        if !reached {
            return None;
        }
        path.reverse();
        Some(path)
    }
}

/// `torch.maximum`, which propagates NaN.
fn torch_maximum(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

struct Point {
    token: usize,
    frame: usize,
    score: f64,
}

/// One token's frames: `[start, end)` and their mean probability.
struct TokenSpan {
    start: usize,
    end: usize,
    score: f64,
}

/// `whisperx.alignment.merge_repeats`.
fn merge_repeats(path: &[Point]) -> Vec<TokenSpan> {
    let mut spans = Vec::new();
    let mut first = 0;
    while first < path.len() {
        let mut last = first;
        while last < path.len() && path[first].token == path[last].token {
            last += 1;
        }
        let mut sum = 0.0;
        for point in &path[first..last] {
            sum += point.score;
        }
        spans.push(TokenSpan {
            start: path[first].frame,
            end: path[last - 1].frame + 1,
            score: sum / (last - first) as f64,
        });
        first = last;
    }
    spans
}

/// NumPy's pairwise summation of f64 (`pairwise_sum_DOUBLE`), which pandas'
/// `mean` uses before dividing by the count.
fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        let mut sum = 0.0;
        for value in values {
            sum += value;
        }
        sum
    } else if n <= BLOCK {
        let mut partial = [0.0; 8];
        partial.copy_from_slice(&values[..8]);
        let mut index = 8;
        while index < n - n % 8 {
            for (lane, sum) in partial.iter_mut().enumerate() {
                *sum += values[index + lane];
            }
            index += 8;
        }
        let mut sum = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for value in &values[index..] {
            sum += value;
        }
        sum
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

/// Python's `round(x, 3)`: the decimal nearest the exact binary value, ties
/// to even, read back as the nearest f64. Rust's formatter rounds the same
/// way.
fn python_round3(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.3}")
        .parse()
        .expect("a formatted finite float parses")
}

/// Python's `str.isspace`: Unicode whitespace plus the ASCII information
/// separators U+001C to U+001F.
fn python_isspace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// Python's `str.split()` with no separator.
pub(crate) fn python_split(text: &str) -> Vec<&str> {
    text.split(python_isspace)
        .filter(|part| !part.is_empty())
        .collect()
}

fn python_strip(text: &str) -> &str {
    text.trim_matches(python_isspace)
}

/// The pinned model in Application Support, downloaded and verified on
/// first use. A receipt of the verified file's size and modification time
/// spares hashing 378 MB on every load.
pub fn install_model(cancelled: &AtomicBool) -> Result<PathBuf> {
    let directory = crate::transcription_models::models_dir()?;
    fs::create_dir_all(&directory)?;
    let lock = File::create(directory.join(".wav2vec2.lock"))?;
    lock.lock_exclusive()?;
    let destination = directory.join(MODEL_FILENAME);
    if receipt_matches(&destination) {
        return Ok(destination);
    }
    if fs::metadata(&destination).is_ok() {
        if verify_model(&destination, cancelled).is_ok() {
            write_receipt(&destination)?;
            return Ok(destination);
        }
        fs::remove_file(&destination)?;
    }
    let partial = directory.join(format!("{MODEL_FILENAME}.partial"));
    if fs::metadata(&partial).is_ok_and(|metadata| metadata.len() >= MODEL_BYTES) {
        fs::remove_file(&partial)?;
    }
    tracing::info!("downloading the wav2vec2 alignment model");
    let mut child = Command::new("/usr/bin/curl")
        .args([
            "--fail",
            "--location",
            "--retry",
            "3",
            "--connect-timeout",
            "15",
            "--max-time",
            "1800",
            "--continue-at",
            "-",
            "--silent",
            "--show-error",
            "--output",
        ])
        .arg(&partial)
        .arg(MODEL_URL)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .wrap_err("could not start the alignment model download")?;
    let status = loop {
        if cancelled.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("the alignment model download was cancelled");
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        thread::sleep(Duration::from_millis(100));
    };
    if !status.success() {
        let mut stderr = String::new();
        if let Some(mut pipe) = child.stderr.take() {
            let _ = pipe.read_to_string(&mut stderr);
        }
        bail!(
            "could not download the alignment model ({status}): {}",
            stderr.trim()
        );
    }
    if let Err(error) = verify_model(&partial, cancelled) {
        let _ = fs::remove_file(&partial);
        return Err(error);
    }
    File::open(&partial)?.sync_all()?;
    fs::rename(&partial, &destination)?;
    crate::transcription_models::sync_directory(&directory)?;
    write_receipt(&destination)?;
    Ok(destination)
}

/// Whether the model is installed and verified, without hashing it.
pub fn is_installed() -> bool {
    crate::transcription_models::models_dir()
        .is_ok_and(|directory| receipt_matches(&directory.join(MODEL_FILENAME)))
}

fn verify_model(path: &Path, cancelled: &AtomicBool) -> Result<()> {
    let bytes = fs::metadata(path)?.len();
    ensure!(
        bytes == MODEL_BYTES,
        "the alignment model has {bytes} bytes, expected {MODEL_BYTES}"
    );
    ensure!(
        crate::transcription_models::sha256_hex(path, Some(cancelled))? == MODEL_SHA256,
        "the alignment model's checksum does not match"
    );
    Ok(())
}

#[derive(Deserialize, Serialize, PartialEq)]
struct Receipt {
    bytes: u64,
    sha256: String,
    modified_ns: u128,
}

fn receipt_path(model: &Path) -> PathBuf {
    model.with_extension("onnx.verified.json")
}

fn current_receipt(model: &Path) -> Option<Receipt> {
    let metadata = fs::metadata(model).ok()?;
    let modified_ns = metadata
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some(Receipt {
        bytes: metadata.len(),
        sha256: MODEL_SHA256.into(),
        modified_ns,
    })
}

fn receipt_matches(model: &Path) -> bool {
    let Some(current) = current_receipt(model) else {
        return false;
    };
    current.bytes == MODEL_BYTES
        && fs::read(receipt_path(model))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Receipt>(&bytes).ok())
            .is_some_and(|stored| stored == current)
}

fn write_receipt(model: &Path) -> Result<()> {
    let receipt = current_receipt(model).ok_or_else(|| eyre!("the alignment model is missing"))?;
    let path = receipt_path(model);
    let partial = path.with_extension("json.partial");
    fs::write(&partial, serde_json::to_vec(&receipt)?)?;
    fs::rename(&partial, &path)?;
    Ok(())
}

#[cfg(test)]
#[path = "forced_alignment_tests.rs"]
mod tests;
