"""Tier A: independently reproduce the Whisper column with WhisperKit (CoreML).

This does NOT trust anyone's transcripts. It runs Whisper on the real LibriSpeech
audio through WhisperKit (the same engine Inscribe used), then scores the output
against LibriSpeech's own reference transcripts with OpenAI's normalizer.

Usage:
    python run_whisperkit.py tiny base small large-v3-v20240930 medium large-v3

Model names are WhisperKit variants; the "openai_whisper-" prefix is added by the
CLI. Requires ./setup.sh to have downloaded LibriSpeech test-clean and built the
WhisperKit CLI. Results append to results/whisperkit.json.
"""
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRI = os.path.join(ROOT, "data", "LibriSpeech", "test-clean")
AUDIO_FLAT = os.path.join(ROOT, "data", "audio_flat")
WHISPERKIT = os.path.join(ROOT, "WhisperKit")
RESULTS = os.path.join(ROOT, "results", "whisperkit.json")


def load_references() -> dict[str, str]:
    """LibriSpeech ground truth from its own *.trans.txt files (the canonical source)."""
    refs = {}
    for tf in glob.glob(os.path.join(LIBRI, "**", "*.trans.txt"), recursive=True):
        with open(tf) as fh:
            for line in fh:
                uid, text = line.strip().split(" ", 1)
                refs[uid] = text
    return refs


def run(model: str, refs: dict[str, str]):
    from normalize_wer import corpus_wer

    report_dir = os.path.join(ROOT, "results", "reports", model)
    os.makedirs(report_dir, exist_ok=True)
    print(f"=== {model}: transcribing {len(refs)} utterances ===", flush=True)
    t0 = time.time()
    cmd = [
        "swift", "run", "-c", "release", "whisperkit-cli", "transcribe",
        "--audio-folder", AUDIO_FLAT,
        "--model", model,
        "--language", "en",
        "--chunking-strategy", "none",
        "--report", "--report-path", report_dir,
    ]
    proc = subprocess.run(cmd, cwd=WHISPERKIT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"!!! {model} FAILED: {proc.stderr[-400:]}", flush=True)
        return

    missing = 0

    def pairs():
        nonlocal missing
        for uid, ref in refs.items():
            rf = os.path.join(report_dir, f"{uid}.json")
            if os.path.exists(rf):
                hyp = json.load(open(rf)).get("text", "")
            else:
                hyp = ""
                missing += 1
            yield ref, hyp

    wer, errors, ref_words = corpus_wer(pairs())
    result = {
        "engine": model, "split": "test-clean", "werPercent": round(wer, 2),
        "utterances": len(refs), "missingReports": missing,
        "computeSeconds": round(time.time() - t0),
    }
    allr = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
    allr = [x for x in allr if x["engine"] != model] + [result]
    json.dump(allr, open(RESULTS, "w"), indent=2)
    print(f"DONE {model}: WER {result['werPercent']}%  missing={missing}  "
          f"{result['computeSeconds']}s", flush=True)


def main():
    models = sys.argv[1:] or ["tiny", "base", "small"]
    models = [f"whisper-{m}" if not m.startswith("whisper-") else m for m in models]
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    refs = load_references()
    assert refs, f"no references found under {LIBRI}; run ./setup.sh first"
    for m in models:
        run(m, refs)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
