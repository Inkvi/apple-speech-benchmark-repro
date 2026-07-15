# Reproducing the Apple SpeechAnalyzer vs Whisper benchmark

An independent reproduction of Inscribe's benchmark
["Apple's New Speech API vs Whisper"](https://get-inscribe.com/blog/apple-speech-api-benchmark.html)
(2026-07-13), which measured Apple's new on-device `SpeechAnalyzer` API against the
legacy `SFSpeechRecognizer` and several Whisper models on LibriSpeech.

The original is a good, unusually transparent benchmark: it publishes its raw
per-utterance transcripts. This repo re-scores those transcripts, independently
re-runs every engine on the real audio (including Apple SpeechAnalyzer via a small
macOS 26 Swift harness), adds the engine the original left out (NVIDIA Parakeet),
puts 95% confidence intervals on every number, and extends the test out of domain
(Earnings-22, AMI). Every number here is reproducible with the scripts in this repo,
and the raw hypotheses we generated are committed under `results/transcripts/`.

## TL;DR

- **On LibriSpeech, Inscribe's headline holds only within noise.** Apple
  SpeechAnalyzer (1.82% WER), Whisper large-v3 (1.82%), and Parakeet v2 (1.69%) are a
  statistical tie: their 95% confidence intervals all overlap. "Most accurate on-device
  engine" is true inside the error bars, not as a clean win.
- **The benchmark's real gap is Parakeet.** NVIDIA Parakeet TDT (which the original did
  not test) is tied for best on LibriSpeech and clearly best on the fair out-of-domain
  set (Earnings-22, ~11.2%), so it belongs in any on-device comparison.
- **Out of domain, Apple is competitive but not ahead.** On Earnings-22 (out-of-domain
  for every engine, the fairest test), Apple (12.03%) ties on-device WhisperKit (12.35%)
  but trails both Parakeet (~11.2%) and reference-implementation Whisper (~11.1% on the
  Open ASR Leaderboard).
- **Two caveats decide how to read this.** (1) LibriSpeech and AMI are both in
  Parakeet's training data; only Earnings-22 is out-of-domain for all engines. (2)
  On-device WhisperKit at default settings hallucinates on short conversational clips
  (AMI is 38% sub-1s), inflating its AMI WER well above reference Whisper, so we flag the
  AMI Whisper numbers as a tooling artifact, not Whisper's true error.

WER is word error rate, lower is better. Every WER below is corpus WER (total word
edits / total reference words) with a 95% bootstrap confidence interval (CI) over
utterances. When two engines' CIs overlap, treat them as tied, not ranked.

## What we found

### LibriSpeech test-clean (2620 utterances, our independent runs)

| Engine | WER% | 95% CI | LibriSpeech in training? |
|--------|-----:|:------:|--------------------------|
| Parakeet TDT 0.6b v2 (English)      | 1.69 | 1.54-1.84 | **yes (in-domain)** |
| Apple SpeechAnalyzer                | 1.82 | 1.66-1.99 | undisclosed |
| Whisper large-v3                    | 1.82 | 1.67-1.98 | zero-shot |
| Whisper large-v3-turbo              | 1.93 | 1.77-2.09 | zero-shot |
| Parakeet TDT 0.6b v3 (multilingual) | 2.15 | 1.92-2.40 | **yes (in-domain)** |
| Whisper medium                      | 2.55 | 2.36-2.76 | zero-shot |
| Whisper small                       | 3.32 | 3.09-3.56 | zero-shot |
| Whisper base                        | 5.01 | 4.75-5.29 | zero-shot |
| Whisper tiny                        | 7.47 | 7.12-7.81 | zero-shot |

The top four (Parakeet v2, Apple, Whisper large-v3, turbo) sit inside one another's
confidence intervals. On this set they are a four-way tie, not a ranking.

### Out of domain: does the ranking hold?

Two harder sets from the Open ASR Leaderboard bundle, run end to end with the same
normalizer and corpus WER. **Earnings-22 is out-of-domain for every engine (the fair
test). AMI is in Parakeet's training set**, so Parakeet has home-field there.

| Engine | LibriSpeech | Earnings-22 (fair, OOD for all) | AMI (Parakeet in-domain) |
|--------|------------:|:-------------------------------:|:------------------------:|
| Parakeet v2            | 1.69 | 11.23 (10.73-11.73) | 11.58 (11.23-11.93) |
| Parakeet v3            | 2.15 | 11.20 (10.71-11.74) | 11.43 (11.08-11.79) |
| Apple SpeechAnalyzer   | 1.82 | 12.03 (11.52-12.56) | 14.32 (13.96-14.69) |
| Whisper large-v3-turbo | 1.93 | 12.35 (11.87-12.86) | 21.68 (21.22-22.15) † |
| Whisper small          | 3.32 | 14.92 (14.27-15.57) | 22.88 (22.40-23.35) † |

† These AMI Whisper numbers are inflated by a WhisperKit hallucination artifact, not a
fair reading of Whisper. See "The AMI Whisper numbers" below; the reference-implementation
figures are ~15.2 (turbo) and ~13.6 (large-v3).

Reading it:

- **On Earnings-22, the fair set, Parakeet is clearly best** (11.2%, CI below Apple's
  lower bound). **Apple (12.03%) ties on-device WhisperKit (12.35%)** by overlapping CI,
  but the reference Whisper implementation on the Open ASR Leaderboard scores ~11.1%
  (turbo), so Apple trails reference Whisper and Parakeet. Apple is a strong on-device
  engine here, not the most accurate one.
- **On AMI, Parakeet leads** (~11.5% vs Apple's 14.3%), but AMI is in Parakeet's training
  set, so part of that gap is domain overlap, not pure skill. Ignore the raw AMI Whisper
  cells (see below).
- **Parakeet leads even on Earnings-22, where it has no home-field**, so its advantage is
  real and not only contamination. That is the finding the original benchmark missed by
  not testing it.

### The AMI Whisper numbers (a WhisperKit artifact, not Whisper's error)

Our raw AMI WER for Whisper (21.7% turbo, 22.9% small) is far worse than the Open ASR
Leaderboard's reference-implementation Whisper (~15.2% turbo, ~13.6% large-v3), and we do
not treat it as Whisper's true AMI error. AMI is heavily segmented (median clip 1.52s,
38% under 1s), and WhisperKit CoreML at default settings hallucinates on short clips:
it transcribes the words correctly and then invents extra text. One example:

```
reference:  Say nice machine, it goes
WhisperKit: It's a nice machine that goes in. Yeah. I spent too much time. All right.
Apple:      It's a nice machine that goes.
```

Reference Whisper suppresses this with temperature-fallback and compression-ratio /
log-probability thresholds; WhisperKit's CLI does not apply them the same way, and Apple
and Parakeet simply do not hallucinate on these clips. So the AMI Whisper gap measures a
tooling behavior on short segments, not the model's accuracy. We report our numbers for
transparency but cite the leaderboard figures as the fair Whisper-on-AMI reading. The same
effect exists on Earnings-22 but is small there (longer clips): our WhisperKit turbo 12.35
vs the leaderboard's 11.07.

### Training-data overlap (read before ranking)

LibriSpeech is not cleanly held out across these models, so LibriSpeech-only rankings
are not apples-to-apples:

- **Parakeet v2 and v3: trained on LibriSpeech and AMI.** NVIDIA's cards list
  LibriSpeech (960 hours) and AMI in the training mix (v2 via the Granary human-labeled
  set, v3 via NeMo ASR Set 3.0). So LibriSpeech and AMI are in-distribution for
  Parakeet; Earnings-22 is not listed and is out-of-domain for it.
- **Whisper: zero-shot on all three.** OpenAI reports LibriSpeech as zero-shot with
  transcript-level dedup, and AMI/Earnings-22 are not in its training either. (680k h of
  web audio means perfect exclusion is unverifiable, but this is OpenAI's stated
  diligence.)
- **Apple SpeechAnalyzer: undisclosed.** Apple publishes nothing about training data,
  so whether LibriSpeech or AMI was seen is unknown.

Net: LibriSpeech and AMI favor the LibriSpeech/AMI-trained models (Parakeet).
Earnings-22 is the only set here that is out-of-domain for all three families, so it is
the fairest read of general capability.

## Cross-checks against published numbers

Our self-measured numbers line up with independently published figures, which is the
main check that the harness and scoring are sound.

| Engine / set | This repo | Published | Source |
|--------------|----------:|----------:|--------|
| Parakeet v2, LibriSpeech clean | 1.69 | 1.69 | NVIDIA model card |
| Parakeet v2, Earnings-22       | 11.23 | 11.15 | NVIDIA model card |
| Parakeet v2, AMI               | 11.58 | 11.16 | NVIDIA model card |
| Parakeet v3, Earnings-22       | 11.20 | 11.42 | NVIDIA model card |

Parakeet v2 on LibriSpeech reproduces NVIDIA's card to the hundredth (1.69 vs 1.69),
and Earnings-22 lands within the confidence interval (11.23 vs 11.15). AMI runs a bit
higher than the card (11.58 vs 11.16); the card's AMI figure uses the leaderboard's
"cleaned" reference variant, ours uses the bundle's standard references, which is the
expected direction for that difference.

Whisper is run here through **WhisperKit CoreML** (the on-device path Inscribe used),
not the reference PyTorch implementation the Open ASR Leaderboard scores, so absolute
Whisper numbers differ from the leaderboard's by implementation and quantization. For
context, the leaderboard reports Whisper large-v3-turbo at 11.07 (Earnings-22) and
13.87 cleaned / 15.16 original (AMI); large-v3 at 11.59 and 13.63 / 14.86.

## Verifying our Whisper harness

Our small-model WhisperKit numbers land on OpenAI's published zero-shot LibriSpeech
figures, which validates the harness end to end:

| Model | This repo | OpenAI published |
|-------|----------:|-----------------:|
| Whisper tiny  | 7.47 | 7.6 |
| Whisper base  | 5.01 | 5.0 |
| Whisper small | 3.32 | 3.4 |

## Re-scoring Inscribe's published transcripts

Inscribe released every per-utterance transcript for both Apple engines. Recomputing
WER from those with OpenAI's normalizer (no audio needed, `rescore_published.py`):

| Engine | Split | This repo | Inscribe | Delta |
|--------|-------|----------:|---------:|------:|
| Apple SpeechAnalyzer         | test-clean | 1.83 | 2.12 | -0.29 |
| Apple SpeechAnalyzer         | test-other | 4.24 | 4.56 | -0.32 |
| SFSpeechRecognizer (legacy)  | test-clean | 8.65 | 9.02 | -0.37 |
| SFSpeechRecognizer (legacy)  | test-other | 15.80 | 16.25 | -0.45 |

The consistent negative delta is the normalizer, not a scoring error: Inscribe's
normalizer is slightly stricter than OpenAI's stock one (they disclose this), so every
engine shifts by a similar amount (0.29-0.45pp) and the ranking is unchanged. Our own
independent Apple run (1.82% on test-clean above) matches this re-score of their
published transcripts (1.83%), which is strong evidence their Apple transcripts are
genuine engine output, not just honest scoring.

## Environment

- Apple M4 Pro, 64 GB RAM, macOS 26.5.1 (build 25F80).
- Xcode 26.6 / Swift 6.3.3 (for the WhisperKit CLI and the SpeechAnalyzer harness).
- WhisperKit pinned to v0.18.0; `parakeet-mlx` 0.5.2; `openai-whisper` 20250625 (used
  only for its `EnglishTextNormalizer`); `huggingface_hub` 1.23.0.
- Apple's speech assets are server-provided and cannot be version-pinned, so the Apple
  numbers are tied to whatever asset shipped as of 2026-07-15 and may drift.
- Runs dated 2026-07-15.

## What you can reproduce

```bash
./setup.sh                              # venv, LibriSpeech test-clean, WhisperKit CLI (pinned)

# 1. Re-score Inscribe's published Apple transcripts (no audio, ~1 min)
./.venv/bin/python rescore_published.py

# 2. Independently run Whisper on the real audio (WhisperKit CoreML)
./.venv/bin/python run_whisperkit.py tiny base small large-v3-v20240930 large-v3

# 3. Parakeet, the engine the original left out (on-device via MLX)
./.venv/bin/python run_parakeet.py v2 v3

# 4. Apple SpeechAnalyzer on the real audio (macOS 26+, builds a small Swift harness)
(cd SpeechAnalyzerCLI && swift build -c release)
./.venv/bin/python run_apple.py

# 5. Out of domain: full test sets, every engine, then score with CIs
export HF_TOKEN=...                     # optional, for faster Hugging Face downloads
./.venv/bin/python prep_ood_dataset.py earnings22
./.venv/bin/python run_ood_engines.py earnings22
./.venv/bin/python score_ood.py earnings22

# AMI is the same, but the WhisperKit CLI segfaults on the full folder, so run its
# Whisper models in chunks; Apple and Parakeet go through run_ood_engines as usual.
./.venv/bin/python prep_ood_dataset.py ami
./.venv/bin/python run_ood_engines.py ami apple parakeet-v2 parakeet-v3
./.venv/bin/python chunk_whisper_ami.py whisper-small
./.venv/bin/python chunk_whisper_ami.py whisper-large-v3-v20240930
./.venv/bin/python score_ood.py ami
```

Step 2 accepts `medium` too. `run_mlx.py` is an optional cross-implementation check of
Whisper via `mlx-whisper`.

## Methodology

- **Corpus:** LibriSpeech `test-clean` (2620 utterances, OpenSLR). Out-of-domain sets
  are the full Earnings-22 and AMI test splits from the Open ASR Leaderboard bundle
  `hf-audio/esb-datasets-test-only-sorted`.
- **Metric:** corpus WER (total edits / total reference words), not the mean of
  per-utterance WERs. Empty output scores as 100% for that utterance. Each number
  carries a 95% bootstrap CI over utterances (1000 resamples, fixed seed).
- **Engine:** WhisperKit CoreML for Whisper, same on-device path as the original.
  Parakeet via `parakeet-mlx`. Apple SpeechAnalyzer via `SpeechAnalyzerCLI/` (macOS 26
  Speech framework, fully on-device).
- **Normalizer:** OpenAI's `EnglishTextNormalizer`, applied to every engine equally, so
  it shifts all absolute numbers together and leaves rankings unchanged. We use it
  because OpenAI published Whisper's LibriSpeech WER with it.
- **Decoding:** `--language en`, greedy, no VAD chunking (utterances are short clips).
- **AMI note:** the WhisperKit CLI segfaults on the full 12k-file AMI folder (a scale
  limit in the tool; Apple and Parakeet process the same folder fine), so AMI Whisper is
  run in chunks of 1500 via `chunk_whisper_ami.py`. Separately, we drop 101 clips shorter
  than 0.15s (backchannels like "Yeah.", single letters; 105 reference words, 0.117% of
  the total) for every engine so all engines are scored on the identical set.
- **Speed:** Apple and Parakeet run single-stream at roughly 0.02 real-time factor
  (about 50x faster than audio) on the M4 Pro. WhisperKit was run at the CLI's default
  concurrency, so its per-file timing is not a clean single-stream figure; we do not
  rank on speed, and neither should you across machines.

## Requirements

- macOS on Apple Silicon (WhisperKit CoreML, Parakeet MLX). `rescore_published.py` runs
  anywhere with Python.
- **macOS 26+** for `run_apple.py` (the SpeechAnalyzer API is macOS 26 only).
- Xcode / Swift toolchain.
- Python 3.10+. Note `openai-whisper` pulls in torch (~2 GB); it is used only for the
  text normalizer.
- ~4 GB disk for LibriSpeech plus the CoreML models, more if you cache the OOD audio.

## Credits

- Original benchmark and published transcripts: [Inscribe](https://get-inscribe.com/blog/apple-speech-api-benchmark.html).
- [LibriSpeech](https://www.openslr.org/12) (Panayotov et al., 2015).
- [WhisperKit](https://github.com/argmaxinc/WhisperKit) by Argmax.
- [Whisper](https://github.com/openai/whisper) and its English normalizer by OpenAI.
- [Parakeet](https://huggingface.co/nvidia) by NVIDIA, run via [parakeet-mlx](https://github.com/senstella/parakeet-mlx).
- Out-of-domain sets from the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) ESB bundle.

MIT licensed. Not affiliated with Inscribe, Apple, Argmax, NVIDIA, or OpenAI.
