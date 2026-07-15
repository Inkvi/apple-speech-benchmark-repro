"""Run every engine on a prepared out-of-domain sample and score with corpus WER.

Expects data_<config>/audio/*.wav + data_<config>/refs.json (see prep_ood_dataset.py),
the WhisperKit CLI built under WhisperKit/, and the SpeechAnalyzer harness built under
SpeechAnalyzerCLI/. Parakeet runs via parakeet-mlx.

    python run_ood_engines.py earnings22

Writes results_<config>.json.
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

WHISPER_MODELS = ["whisper-small", "whisper-large-v3-v20240930"]  # small + turbo, by name
PARAKEET = [("parakeet-v2", "mlx-community/parakeet-tdt-0.6b-v2"),
            ("parakeet-v3", "mlx-community/parakeet-tdt-0.6b-v3")]


def main():
    from normalize_wer import tokens, edit_distance

    cfg = sys.argv[1]
    audio = f"{ROOT}/data_{cfg}/audio"
    refs = json.load(open(f"{ROOT}/data_{cfg}/refs.json"))
    results_path = f"{ROOT}/results_{cfg}.json"
    sacli = f"{ROOT}/SpeechAnalyzerCLI/.build/release/sacli"

    def score(get):
        errs = words = miss = 0
        for uid, ref in refs.items():
            h = get(uid)
            if h is None:
                miss += 1
                h = ""
            r, hy = tokens(ref), tokens(h)
            errs += edit_distance(r, hy)
            words += len(r)
        return round(100 * errs / words, 2), miss

    def save(rec):
        allr = json.load(open(results_path)) if os.path.exists(results_path) else []
        allr = [x for x in allr if x["engine"] != rec["engine"]] + [rec]
        json.dump(allr, open(results_path, "w"), indent=2)

    def report_getter(rdir):
        return lambda uid: (json.load(open(f"{rdir}/{uid}.json")).get("text")
                            if os.path.exists(f"{rdir}/{uid}.json") else None)

    # Apple SpeechAnalyzer
    if os.path.exists(sacli):
        ad = f"{ROOT}/results/reports/{cfg}/apple"
        os.makedirs(ad, exist_ok=True)
        subprocess.run([sacli, "--audio-folder", audio, "--out", ad],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wer, miss = score(report_getter(ad))
        save({"engine": "apple-speechanalyzer", "werPercent": wer, "missing": miss})
        print(f"apple-speechanalyzer: {wer}%", flush=True)

    # WhisperKit
    for model in WHISPER_MODELS:
        rdir = f"{ROOT}/results/reports/{cfg}/{model}"
        os.makedirs(rdir, exist_ok=True)
        p = subprocess.run(
            ["swift", "run", "-c", "release", "whisperkit-cli", "transcribe",
             "--audio-folder", audio, "--model", model, "--language", "en",
             "--chunking-strategy", "none", "--report", "--report-path", rdir],
            cwd=f"{ROOT}/WhisperKit", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            print(f"!!! {model} FAILED: {p.stderr[-150:]}", flush=True)
            continue
        wer, miss = score(report_getter(rdir))
        save({"engine": model, "werPercent": wer, "missing": miss})
        print(f"{model}: {wer}%", flush=True)

    # Parakeet
    from parakeet_mlx import from_pretrained
    for name, repo in PARAKEET:
        m = from_pretrained(repo)
        hyp = {uid: m.transcribe(f"{audio}/{uid}.wav").text for uid in refs}
        wer, miss = score(lambda uid: hyp.get(uid))
        save({"engine": name, "werPercent": wer, "missing": miss})
        print(f"{name}: {wer}%", flush=True)


if __name__ == "__main__":
    main()
