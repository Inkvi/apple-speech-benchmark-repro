# Reproducing the Apple SpeechAnalyzer vs Whisper benchmark

An independent reproduction of Inscribe's benchmark
["Apple's New Speech API vs Whisper"](https://get-inscribe.com/blog/apple-speech-api-benchmark.html)
(2026-07-13), which measured Apple's new on-device `SpeechAnalyzer` API against
the legacy `SFSpeechRecognizer` and several Whisper models on LibriSpeech.

The original is a solid, unusually honest benchmark (it publishes its raw
transcripts). This repo checks it two ways and extends it to the larger Whisper
models the original did not run.

## What you can reproduce

**1. Re-score their published Apple transcripts (no audio, ~1 minute).**
Inscribe released every per-utterance transcript for both Apple engines. Each
record has the LibriSpeech `reference` and the engine `hypothesis`, so you can
recompute WER from scratch and check it against their `summary.json`.

```bash
./.venv/bin/python rescore_published.py
```

This verifies their **scoring** is honest. It does not prove the transcripts are
genuine engine output, because you are trusting their `hypothesis` text. For that
you have to run the audio yourself, which is step 2.

**2. Independently run Whisper on the real audio (WhisperKit, the engine they used).**
This trusts no one's transcripts. It runs Whisper on the actual LibriSpeech audio
through WhisperKit CoreML, then scores against LibriSpeech's own references.

```bash
./setup.sh                       # venv, LibriSpeech test-clean, WhisperKit CLI
./.venv/bin/python run_whisperkit.py tiny base small large-v3-v20240930 medium large-v3
```

Apple's own `SpeechAnalyzer` engine only runs on macOS 26+, so re-running that
column requires a Swift harness on the new Speech framework; this repo currently
reproduces the **Whisper** column (the one that can be checked against OpenAI's
published numbers) and extends it to bigger models. PRs adding a SpeechAnalyzer
harness are welcome.

## Methodology (kept identical to the original where possible)

- **Corpus:** LibriSpeech `test-clean` (2620 utterances). Public, from OpenSLR.
- **Metric:** corpus WER (total word edits / total reference words), not the mean
  of per-utterance WERs. Empty output scores as 100% WER for that utterance.
- **Engine:** WhisperKit CoreML, same as the original. `mlx-whisper` is available
  as a cross-implementation check (`run_mlx.py`); its numbers differ, which is the
  point.
- **Normalizer:** OpenAI's own `EnglishTextNormalizer`. We use it because OpenAI
  published Whisper's LibriSpeech WER with it, which makes our numbers directly
  comparable to OpenAI's. **Inscribe's normalizer is slightly stricter than this**
  (they say so), which is why their absolute numbers run ~0.3pp higher than both
  ours and OpenAI's. A normalizer applies equally to every engine, so it shifts
  all absolute numbers together and leaves the ranking unchanged.
- **Decoding:** `--language en`, greedy, no VAD chunking (LibriSpeech utterances
  are short, so chunking would only trim them).

## Results

### Whisper column, test-clean

Our numbers (WhisperKit CoreML + OpenAI normalizer) next to Inscribe's reported
numbers and OpenAI's published numbers.

| Model | This repo | Inscribe | OpenAI published |
|-------|----------:|---------:|-----------------:|
| Whisper Tiny  | 7.45% | 7.88% | 7.6% |
| Whisper Base  | 5.01% | 5.42% | 5.0% |
| Whisper Small | _pending_ | 3.74% | 3.4% |

Extended to models Inscribe did not benchmark:

| Model | This repo | Inscribe | OpenAI published |
|-------|----------:|---------:|-----------------:|
| Whisper Medium         | _pending_ | not tested | - |
| Whisper Large-v3-turbo | _pending_ | not tested | - |
| Whisper Large-v3       | _pending_ | not tested | - |

Reading: our WhisperKit numbers land on OpenAI's published figures, and Inscribe
sits a few tenths above both, consistent with their stricter normalizer. The
harness reproduces.

### Re-score of Inscribe's published Apple transcripts (test-clean / test-other)

Recomputing WER from their released transcripts with OpenAI's normalizer:

| Engine | Split | This repo | Inscribe | Delta |
|--------|-------|----------:|---------:|------:|
| Apple SpeechAnalyzer | test-clean | 1.83% | 2.12% | -0.29 |
| Apple SpeechAnalyzer | test-other | 4.24% | 4.56% | -0.32 |
| SFSpeechRecognizer (legacy) | test-clean | 8.65% | 9.02% | -0.37 |
| SFSpeechRecognizer (legacy) | test-other | 15.80% | 16.25% | -0.45 |

The consistent negative delta is the normalizer difference, not a scoring error:
with a hand-matched stricter normalizer these land on Inscribe's numbers to within
0.06pp, and we independently reproduce their footnote (exactly 1 empty hypothesis
in legacy test-other).

## Requirements

- macOS on Apple Silicon (for WhisperKit CoreML). `rescore_published.py` runs
  anywhere with Python.
- Xcode / Swift toolchain (for the WhisperKit CLI build).
- Python 3.10+.
- ~1 GB disk for LibriSpeech test-clean plus the CoreML models.

## Credits

- Original benchmark and published transcripts: [Inscribe](https://get-inscribe.com/blog/apple-speech-api-benchmark.html).
- [LibriSpeech](https://www.openslr.org/12) (Panayotov et al., 2015).
- [WhisperKit](https://github.com/argmaxinc/WhisperKit) by Argmax.
- [Whisper](https://github.com/openai/whisper) and its English normalizer by OpenAI.

MIT licensed. Not affiliated with Inscribe, Apple, Argmax, or OpenAI.
