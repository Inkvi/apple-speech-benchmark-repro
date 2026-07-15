# Reproducing the Apple SpeechAnalyzer vs Whisper benchmark

An independent reproduction of Inscribe's benchmark
["Apple's New Speech API vs Whisper"](https://get-inscribe.com/blog/apple-speech-api-benchmark.html)
(2026-07-13), which measured Apple's new on-device `SpeechAnalyzer` API against
the legacy `SFSpeechRecognizer` and several Whisper models on LibriSpeech.

The original is a solid, unusually honest benchmark (it publishes its raw
transcripts). This repo re-scores it, independently re-runs every engine on the
real audio (including Apple SpeechAnalyzer via a small macOS 26 harness), adds the
engine it left out (NVIDIA Parakeet), and documents a contamination caveat that
changes how the numbers should be read: LibriSpeech is in Parakeet's training set
but held out by Whisper, so it is not an apples-to-apples comparison.

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

**3. Extend to Parakeet, the engine the original left out.**
NVIDIA Parakeet TDT was the most-requested missing model in the discussion of the
original benchmark. Same corpus, normalizer, and metric, on-device via MLX:

```bash
./.venv/bin/python run_parakeet.py v2 v3
```

**4. Independently run Apple SpeechAnalyzer on the real audio (macOS 26+).**
Inscribe only published LibriSpeech transcripts, so to check Apple's number
end-to-end you have to run the engine yourself. `SpeechAnalyzerCLI/` is a small
Swift harness (macOS 26 Speech framework, fully on-device) that does exactly that:

```bash
(cd SpeechAnalyzerCLI && swift build -c release)
./.venv/bin/python run_apple.py
```

Our run scores **1.82%** on test-clean, matching our re-score of Inscribe's
published transcripts (1.83%) to 0.01pp, which confirms their Apple transcripts
were genuine, not just their scoring.

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

WER is word error rate (lower better). RTF is real-time factor = compute time /
audio duration (lower faster; 0.02 means ~50x real time). RTF is hardware- and
concurrency-dependent, so treat it as directional, not a cross-machine ranking.
Inscribe themselves flagged their RTF as provisional.

### All engines, test-clean (our independent runs)

RTF is single-stream (concurrency 1). The last column is the crucial caveat, see
"Test-set contamination" below.

| Engine | WER% | RTF | LibriSpeech in training? |
|--------|-----:|----:|--------------------------|
| Parakeet TDT 0.6b v2 (English) | 1.71 | 0.025 | **yes (in-domain)** |
| Apple SpeechAnalyzer            | 1.82 | 0.019 | undisclosed |
| Whisper large-v3-turbo          | 1.92 | 0.064 | held out (zero-shot) |
| Parakeet TDT 0.6b v3 (multilingual) | 2.03 | 0.019 | **yes (in-domain)** |
| Whisper Small                   | 3.33 | 0.040 | held out (zero-shot) |
| Whisper Base                    | 5.01 | 0.017 | held out (zero-shot) |
| Whisper Tiny                    | 7.45 | 0.014 | held out (zero-shot) |

Parakeet v2 posts the lowest WER, but read that against the last column before
concluding it is "the best on-device engine": see below.

### Test-set contamination / domain overlap (read this before ranking)

LibriSpeech is **not** cleanly held out across these models, so the comparison is
not apples-to-apples:

- **Parakeet (v2, v3): trained on LibriSpeech.** NVIDIA's model card lists
  "LibriSpeech (960 hours)" in the training mix. The test-clean *utterances* are a
  held-out, speaker-disjoint split, but the **domain is in-distribution** (same
  LibriVox audiobook read-speech). NVIDIA's own card reports LibriSpeech-clean at
  1.69% (we got 1.71%) while reporting 9-11% on AMI/Earnings/GigaSpeech: their own
  numbers show LibriSpeech is their easiest, most in-domain set.
- **Whisper: OpenAI reports LibriSpeech as zero-shot** and did transcript-level
  dedup. So Whisper's numbers are out-of-distribution here. (680k h of web audio
  means perfect exclusion is unverifiable, but this is OpenAI's stated diligence.)
- **Apple SpeechAnalyzer: undisclosed.** Apple publishes nothing about training
  data, so we cannot say whether LibriSpeech/LibriVox was seen.

Net: LibriSpeech test-clean favors LibriSpeech-trained models (Parakeet), so part
of Parakeet's lead measures training overlap, not pure capability. A fair ranking
needs an out-of-domain set none of them trained on (e.g. Earnings-22, TED-LIUM
held-out). The same caveat applies to the original benchmark's Apple-beats-Whisper
claim if Apple trained on LibriSpeech.

### Whisper column verification (why we trust the harness)

Our WhisperKit + OpenAI-normalizer numbers land on OpenAI's published zero-shot
figures; Inscribe sits ~0.4 above both, consistent with their stricter normalizer:

| Model | This repo | Inscribe | OpenAI published |
|-------|----------:|---------:|-----------------:|
| Whisper Tiny  | 7.45% | 7.88% | 7.6% |
| Whisper Base  | 5.01% | 5.42% | 5.0% |
| Whisper Small | 3.33% | 3.74% | 3.4% |

Whisper large-v3-turbo (full) added at 1.92%. The quantized turbo variant
(632MB palettized) did not run cleanly through the WhisperKit CLI via
`--model-path` (it stalled on load), so it is omitted rather than reported.

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

- macOS on Apple Silicon (for WhisperKit CoreML and Parakeet MLX).
  `rescore_published.py` runs anywhere with Python.
- **macOS 26+** for `run_apple.py` (the SpeechAnalyzer API is macOS 26 only).
- Xcode / Swift toolchain (for the WhisperKit CLI and the SpeechAnalyzer harness).
- Python 3.10+.
- ~1 GB disk for LibriSpeech test-clean plus the CoreML models.

## Credits

- Original benchmark and published transcripts: [Inscribe](https://get-inscribe.com/blog/apple-speech-api-benchmark.html).
- [LibriSpeech](https://www.openslr.org/12) (Panayotov et al., 2015).
- [WhisperKit](https://github.com/argmaxinc/WhisperKit) by Argmax.
- [Whisper](https://github.com/openai/whisper) and its English normalizer by OpenAI.
- [Parakeet](https://huggingface.co/nvidia) by NVIDIA, run via [parakeet-mlx](https://github.com/senstella/parakeet-mlx).

MIT licensed. Not affiliated with Inscribe, Apple, Argmax, or OpenAI.
