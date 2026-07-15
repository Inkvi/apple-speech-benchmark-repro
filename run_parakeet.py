"""Extend the benchmark to NVIDIA Parakeet TDT (the engine Inscribe left out).

Parakeet was the most-requested missing model in the discussion of the original
benchmark, so this runs it on the same LibriSpeech test-clean with the same
OpenAI normalizer and corpus-WER, on-device via parakeet-mlx (Apple MLX).

    python run_parakeet.py v2 v3

Notes:
- v2 (parakeet-tdt-0.6b-v2) is ENGLISH-ONLY and is the fair comparison on
  LibriSpeech. v3 is multilingual and usually a touch worse on English.
- This is MLX, not CoreML. The production on-device path on Mac is often
  FluidAudio (CoreML); a CoreML number would be closer to WhisperKit's setup.
- RTF here is summed per-file transcription wall-time / total audio seconds.
  Run it with nothing else using the GPU or the timing will be polluted.
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRI = os.path.join(ROOT, "data", "LibriSpeech", "test-clean")
RESULTS = os.path.join(ROOT, "results", "parakeet.json")
REPO = {"v2": "mlx-community/parakeet-tdt-0.6b-v2",
        "v3": "mlx-community/parakeet-tdt-0.6b-v3"}


def load_references() -> dict[str, str]:
    refs = {}
    for tf in glob.glob(os.path.join(LIBRI, "**", "*.trans.txt"), recursive=True):
        for line in open(tf):
            uid, text = line.strip().split(" ", 1)
            refs[uid] = text
    return refs


def audio_seconds(paths) -> float:
    import soundfile as sf
    total = 0.0
    for p in paths.values():
        info = sf.info(p)
        total += info.frames / info.samplerate
    return total


def main():
    from parakeet_mlx import from_pretrained
    from normalize_wer import tokens, edit_distance

    variants = sys.argv[1:] or ["v2", "v3"]
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    refs = load_references()
    paths = {p.split("/")[-1][:-5]: p
             for p in glob.glob(os.path.join(LIBRI, "**", "*.flac"), recursive=True)}
    total_audio = audio_seconds(paths)

    for v in variants:
        repo = REPO[v]
        model = from_pretrained(repo)
        errors = ref_words = 0
        compute = 0.0
        for uid, ref in refs.items():
            t = time.time()
            hyp = model.transcribe(paths[uid]).text
            compute += time.time() - t
            r, h = tokens(ref), tokens(hyp)
            errors += edit_distance(r, h)
            ref_words += len(r)
        res = {
            "engine": f"parakeet-tdt-0.6b-{v}", "split": "test-clean",
            "werPercent": round(100 * errors / ref_words, 2),
            "audioSeconds": round(total_audio), "computeSeconds": round(compute),
            "realTimeFactor": round(compute / total_audio, 4),
        }
        allr = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
        allr = [x for x in allr if x["engine"] != res["engine"]] + [res]
        json.dump(allr, open(RESULTS, "w"), indent=2)
        print(f"DONE {res['engine']}: WER {res['werPercent']}%  "
              f"RTF {res['realTimeFactor']}", flush=True)


if __name__ == "__main__":
    main()
