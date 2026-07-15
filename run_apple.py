"""Run Apple's on-device SpeechAnalyzer over LibriSpeech test-clean and score it.

This actually runs the engine on the audio (unlike rescore_published.py, which only
re-scores Inscribe's published transcripts), so it independently verifies Apple's
number end-to-end. Requires macOS 26+ and the built `sacli` harness:

    (cd SpeechAnalyzerCLI && swift build -c release)
    python run_apple.py

sacli is single-stream, so its wall time / audio seconds is already a clean RTF.
"""
import glob
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRI = os.path.join(ROOT, "data", "LibriSpeech", "test-clean")
FLAT = os.path.join(ROOT, "data", "audio_flat")
SACLI = os.path.join(ROOT, "SpeechAnalyzerCLI", ".build", "release", "sacli")
OUT = os.path.join(ROOT, "results", "reports", "apple-speechanalyzer")
RESULTS = os.path.join(ROOT, "results", "apple.json")


def load_references() -> dict[str, str]:
    refs = {}
    for tf in glob.glob(os.path.join(LIBRI, "**", "*.trans.txt"), recursive=True):
        for line in open(tf):
            uid, text = line.strip().split(" ", 1)
            refs[uid] = text
    return refs


def main():
    from normalize_wer import tokens, edit_distance

    assert os.path.exists(SACLI), f"build the harness first: (cd SpeechAnalyzerCLI && swift build -c release)"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    refs = load_references()

    t0 = time.time()
    subprocess.run([SACLI, "--audio-folder", FLAT, "--out", OUT], check=True)
    wall = time.time() - t0

    errors = ref_words = missing = 0
    audio_s = 0.0
    for uid, ref in refs.items():
        rf = os.path.join(OUT, f"{uid}.json")
        hyp = json.load(open(rf)).get("text", "") if os.path.exists(rf) else ""
        if not os.path.exists(rf):
            missing += 1
        r, h = tokens(ref), tokens(hyp)
        errors += edit_distance(r, h)
        ref_words += len(r)
    # LibriSpeech test-clean is 19,452 s of audio; RTF = wall / audio (sacli is single-stream)
    res = {
        "engine": "apple-speechanalyzer", "split": "test-clean",
        "werPercent": round(100 * errors / ref_words, 2),
        "missing": missing, "utterances": len(refs),
        "wallSeconds": round(wall), "realTimeFactor": round(wall / 19452, 4),
    }
    json.dump(res, open(RESULTS, "w"), indent=2)
    print(f"Apple SpeechAnalyzer test-clean: WER {res['werPercent']}%  "
          f"RTF {res['realTimeFactor']}  missing={missing}")


if __name__ == "__main__":
    main()
