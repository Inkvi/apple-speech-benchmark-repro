"""Build a capped, evenly-spaced sample from an Open ASR Leaderboard test set.

Downloads one parquet shard of hf-audio/esb-datasets-test-only-sorted, samples N
segments evenly across it, decodes the audio bytes with soundfile (no torchcodec
needed), writes 16-bit wavs, and saves references. Used for the out-of-domain
comparison (Earnings-22, AMI, ...).

    python prep_ood_dataset.py earnings22 test-00002-of-00005.parquet 300
    python prep_ood_dataset.py ami       test-00007-of-00015.parquet 300

Set HF_TOKEN in the environment for faster/authenticated downloads (optional).
Writes to data_<config>/audio/*.wav and data_<config>/refs.json.
"""
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "hf-audio/esb-datasets-test-only-sorted"


def main():
    cfg = sys.argv[1]
    shard = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    text_col = sys.argv[4] if len(sys.argv) > 4 else "text"

    out = f"data_{cfg}"
    audio_dir = f"{out}/audio"
    os.makedirs(audio_dir, exist_ok=True)

    print(f"downloading {cfg}/{shard} ...", flush=True)
    path = hf_hub_download(REPO, f"{cfg}/{shard}", repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    targets = set(int(i) for i in np.linspace(0, total - 1, min(n, total)).astype(int))

    refs = {}
    gi = -1
    audio_s = 0.0
    for batch in pf.iter_batches(batch_size=64, columns=["audio", text_col]):
        d = batch.to_pydict()
        for au, tx in zip(d["audio"], d[text_col]):
            gi += 1
            if gi not in targets:
                continue
            raw = au["bytes"] if isinstance(au, dict) else au
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            uid = f"{cfg}_{gi:06d}"
            sf.write(f"{audio_dir}/{uid}.wav", arr, sr, subtype="PCM_16")
            refs[uid] = tx
            audio_s += len(arr) / sr
    json.dump(refs, open(f"{out}/refs.json", "w"))
    print(f"wrote {len(refs)} wavs, ~{audio_s/60:.1f} min audio -> {out}/refs.json", flush=True)


if __name__ == "__main__":
    main()
