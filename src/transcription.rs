use color_eyre::eyre::Result;
use serde::{Deserialize, Serialize};

use crate::apple_speech::AppleSpeech;
use crate::dictation::{DictationProtocol, VoiceTranscript};
use crate::openai_transcription::OpenAiTranscriber;
use crate::parakeet::Parakeet;
use crate::transcription_models::{
    ModelRuntime, TranscriptionModelId, TranscriptionSelection, validate,
};

const UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES: usize = 3_200;

pub enum Transcriber {
    Gguf(Box<Parakeet>),
    AppleSpeech(AppleSpeech),
    OpenAi(OpenAiTranscriber),
}

#[derive(Default)]
pub struct WarmTranscriber {
    active: Option<Transcriber>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    /// Word timings, when the model produces them.
    #[serde(default)]
    pub words: Vec<TranscriptSegment>,
}

impl Transcriber {
    pub fn load() -> Result<Self> {
        let (_, selection) = crate::app_settings::transcription_selection();
        Self::load_selection(&selection)
    }

    pub fn load_selection(selection: &TranscriptionSelection) -> Result<Self> {
        match validate(selection)?.runtime {
            ModelRuntime::Gguf(_) => Parakeet::load_selection(selection)
                .map(Box::new)
                .map(Self::Gguf),
            ModelRuntime::AppleSpeech => AppleSpeech::load(selection).map(Self::AppleSpeech),
            ModelRuntime::OpenAi(model) => {
                OpenAiTranscriber::load(selection, model).map(Self::OpenAi)
            }
        }
    }

    pub fn matches_selection(&self, selection: &TranscriptionSelection) -> bool {
        match self {
            Self::Gguf(model) => model.matches_selection(selection),
            Self::AppleSpeech(model) => model.matches_selection(selection),
            Self::OpenAi(model) => model.matches_selection(selection),
        }
    }

    /// Brings an idle GGUF model's weights back before audio arrives (see
    /// [`Parakeet::prewarm`]). Apple Speech and OpenAI have no weights of
    /// Hex's to warm.
    pub fn prewarm(&mut self) -> Result<()> {
        match self {
            Self::Gguf(model) => model.prewarm(),
            Self::AppleSpeech(_) | Self::OpenAi(_) => Ok(()),
        }
    }

    /// Transcribe normalized 16 kHz audio. Backend padding is applied once here,
    /// before GGUF chunking or any voice-control retranscription.
    pub fn transcribe(&mut self, mut samples: Vec<f32>) -> Result<String> {
        match self {
            Self::Gguf(model) => {
                prepare_gguf_samples(&mut samples, model.model_id());
                model.transcribe(&samples)
            }
            Self::AppleSpeech(model) => model.transcribe(&samples).map(|result| result.text),
            Self::OpenAi(model) => model.transcribe(&samples),
        }
    }

    pub fn transcribe_voice(
        &mut self,
        mut samples: Vec<f32>,
        protocol: &DictationProtocol,
    ) -> Result<VoiceTranscript> {
        match self {
            Self::Gguf(model) => {
                prepare_gguf_samples(&mut samples, model.model_id());
                model.transcribe_voice(&samples, protocol)
            }
            Self::AppleSpeech(model) => model
                .transcribe(&samples)
                .map(|result| VoiceTranscript::heard(result.text, protocol)),
            Self::OpenAi(model) => model
                .transcribe(&samples)
                .map(|text| VoiceTranscript::heard(text, protocol)),
        }
    }

    /// Transcribe one normalized segment with the same input preparation as
    /// plain dictation. The caller retains ownership of segment offsets.
    pub fn transcribe_segments(&mut self, mut samples: Vec<f32>) -> Result<Transcript> {
        match self {
            Self::Gguf(model) => {
                prepare_gguf_samples(&mut samples, model.model_id());
                model
                    .transcribe_segments(&samples)
                    .map(|result| Transcript {
                        text: result.text,
                        segments: result
                            .segments
                            .into_iter()
                            .map(|segment| TranscriptSegment {
                                start_ms: segment.t0_ms,
                                end_ms: segment.t1_ms,
                                text: segment.text,
                            })
                            .collect(),
                        words: result
                            .words
                            .into_iter()
                            .map(|word| TranscriptSegment {
                                start_ms: word.t0_ms,
                                end_ms: word.t1_ms,
                                text: word.text,
                            })
                            .collect(),
                    })
            }
            Self::AppleSpeech(model) => model.transcribe(&samples),
            // Segment and word times need the forced aligner, which the
            // capture transcriber runs itself; nothing else may send this
            // audio to OpenAI.
            Self::OpenAi(_) => Err(color_eyre::eyre::eyre!(
                "OpenAI dictation has no segment timing; choose a local model"
            )),
        }
    }
}

fn prepare_gguf_samples(samples: &mut Vec<f32>, model: Option<TranscriptionModelId>) {
    crate::dictation::pad_for_parakeet(samples);
    if model == Some(TranscriptionModelId::ParakeetUnifiedEnglish) {
        samples.resize(
            samples.len() + UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES,
            0.0,
        );
    }
}

impl WarmTranscriber {
    /// A warm transcriber already serving `model`, for opt-in harnesses.
    #[cfg(test)]
    pub(crate) fn serving(model: Transcriber) -> Self {
        Self {
            active: Some(model),
        }
    }

    pub fn load() -> Result<Self> {
        Ok(Self {
            active: Some(Transcriber::load()?),
        })
    }

    /// The selection of the warm model when it is a GGUF model.
    pub(crate) fn gguf_selection(&self) -> Option<&TranscriptionSelection> {
        match &self.active {
            Some(Transcriber::Gguf(model)) => model.selection(),
            _ => None,
        }
    }

    /// Whether the warm model streams captures: a GGUF model on a device that
    /// streams, or OpenAI's live model.
    pub(crate) fn can_stream(&self) -> bool {
        self.stream_options().is_some() || self.streams_remotely()
    }

    /// Whether its streams go to OpenAI Realtime.
    pub(crate) fn streams_remotely(&self) -> bool {
        matches!(&self.active, Some(Transcriber::OpenAi(model)) if model.streams())
    }

    /// What a Realtime stream needs, when the warm model streams there.
    pub(crate) fn realtime(&self) -> Option<Result<crate::openai_realtime::RealtimeSettings>> {
        match &self.active {
            Some(Transcriber::OpenAi(model)) if model.streams() => Some(model.realtime_settings()),
            _ => None,
        }
    }

    /// Stream options of the warm model, when it can stream on its device.
    pub(crate) fn stream_options(&self) -> Option<&transcribe_cpp::StreamOptions> {
        match &self.active {
            Some(Transcriber::Gguf(model)) => model.stream_options(),
            _ => None,
        }
    }

    pub(crate) fn with_stream<R>(
        &mut self,
        stream_options: &transcribe_cpp::StreamOptions,
        cancel: &transcribe_cpp::CancelToken,
        drive: impl FnOnce(&mut transcribe_cpp::Stream<'_>) -> R,
    ) -> transcribe_cpp::Result<R> {
        match &mut self.active {
            Some(Transcriber::Gguf(model)) => model.with_stream(stream_options, cancel, drive),
            _ => Err(transcribe_cpp::Error::NotImplemented(
                "the active transcriber cannot stream".into(),
            )),
        }
    }

    /// The loaded model, without switching or loading one.
    pub(crate) fn active(&mut self) -> Option<&mut Transcriber> {
        self.active.as_mut()
    }

    pub fn activate(&mut self, selection: &TranscriptionSelection) -> Result<&mut Transcriber> {
        if !self
            .active
            .as_ref()
            .is_some_and(|model| model.matches_selection(selection))
        {
            let candidate = Transcriber::load_selection(selection)?;
            self.active = Some(candidate);
        }
        Ok(self
            .active
            .as_mut()
            .expect("activated transcriber must be available"))
    }
}

#[cfg(test)]
mod tests {
    use super::{Transcript, UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES, prepare_gguf_samples};
    use crate::transcription_models::TranscriptionModelId;

    #[test]
    fn apple_bridge_transcript_uses_the_canonical_wire_shape() {
        let transcript: Transcript = serde_json::from_str(
            r#"{"text":"hello","segments":[{"startMs":10,"endMs":40,"text":"hello"}]}"#,
        )
        .unwrap();

        assert_eq!(transcript.text, "hello");
        assert_eq!(transcript.segments.len(), 1);
        assert_eq!(transcript.segments[0].start_ms, 10);
        assert_eq!(transcript.segments[0].end_ms, 40);
    }

    #[test]
    fn unified_english_gets_trailing_silence_for_final_token_context() {
        let mut samples = vec![0.5; 32_000];

        prepare_gguf_samples(
            &mut samples,
            Some(TranscriptionModelId::ParakeetUnifiedEnglish),
        );

        assert_eq!(
            samples.len(),
            32_000 + UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES
        );
        assert!(samples[..32_000].iter().all(|sample| *sample == 0.5));
        assert!(samples[32_000..].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn other_gguf_models_do_not_get_trailing_silence() {
        let mut samples = vec![0.5; 32_000];

        prepare_gguf_samples(&mut samples, Some(TranscriptionModelId::ParakeetV2));

        assert_eq!(samples.len(), 32_000);
    }

    #[test]
    fn short_unified_input_is_minimum_padded_before_trailing_context() {
        let mut samples = vec![0.5; 1_600];
        prepare_gguf_samples(
            &mut samples,
            Some(TranscriptionModelId::ParakeetUnifiedEnglish),
        );

        // The 200 ms context is additional to the 1.5 s minimum, not part of it.
        assert_eq!(
            samples.len(),
            24_000 + UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES
        );
        assert!(samples[..1_600].iter().all(|sample| *sample == 0.5));
        assert!(samples[1_600..].iter().all(|sample| *sample == 0.0));
    }
}
