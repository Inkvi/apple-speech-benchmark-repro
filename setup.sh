#!/usr/bin/env bash
# One-time setup: Python deps, LibriSpeech test-clean, and the WhisperKit CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Python venv + deps"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip >/dev/null
./.venv/bin/pip install -r requirements.txt

echo "==> LibriSpeech test-clean (~346 MB)"
mkdir -p data && cd data
if [ ! -d LibriSpeech/test-clean ]; then
  curl -L -o test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
  tar xzf test-clean.tar.gz && rm test-clean.tar.gz
fi

echo "==> flatten audio into one folder (absolute symlinks)"
mkdir -p audio_flat
find "$ROOT/data/LibriSpeech/test-clean" -name '*.flac' | while read -r f; do
  ln -sf "$f" "audio_flat/$(basename "$f")"
done
echo "    $(ls audio_flat | wc -l | tr -d ' ') files linked"
cd "$ROOT"

echo "==> WhisperKit CLI (only needed for the audio reproduction, requires Xcode/Swift)"
if command -v swift >/dev/null 2>&1; then
  if [ ! -d WhisperKit ]; then
    git clone --depth 1 https://github.com/argmaxinc/WhisperKit.git
  fi
  ( cd WhisperKit && swift build -c release --product whisperkit-cli )
else
  echo "    swift not found; skip. rescore_published.py still works without it."
fi

echo "==> done. Try:  ./.venv/bin/python rescore_published.py"
