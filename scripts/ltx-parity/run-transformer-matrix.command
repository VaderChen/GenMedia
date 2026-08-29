#!/bin/zsh

set -euo pipefail

ROOT_DIR="${0:A:h:h:h}"
MODEL_DIR="${1:?用法：$0 <model-dir> [fixture-dir]}"
FIXTURE_DIR="${2:-$ROOT_DIR/scripts/ltx-parity/fixtures}"
PYTHON_BIN="${LTX_PARITY_PYTHON:-$HOME/Library/Application Support/GenImage/Runtime/ltx-2-mlx/.venv/bin/python}"
POC_BIN="${LTX_PARITY_SWIFT:-$ROOT_DIR/RuntimeSupport/LTXVideoWorker/.build/out/Products/Release/GenImageLTXVideoPoC}"
WEIGHTS_NAME="transformer-distilled-1.1.safetensors"

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "找不到 LTX Python 參考環境：$PYTHON_BIN"
  exit 1
fi
if [[ ! -x "$POC_BIN" ]]; then
  print -u2 "找不到 Swift PoC：$POC_BIN"
  print -u2 "請先執行 swift build --package-path RuntimeSupport/LTXVideoWorker -c release"
  exit 1
fi
if [[ ! -f "$MODEL_DIR/$WEIGHTS_NAME" ]]; then
  print -u2 "找不到權重：$MODEL_DIR/$WEIGHTS_NAME"
  exit 1
fi

mkdir -p "$FIXTURE_DIR"

typeset -a CASES=(
  "small:4:8:8:256"
  "medium:8:16:16:2048"
  "large:16:16:32:8192"
)

for dtype in bfloat16 float32; do
  for specification in $CASES; do
    IFS=: read -r name frames height width token_count <<< "$specification"
    reference="$FIXTURE_DIR/ltx-transformer-${name}-${dtype}.safetensors"
    actual="${TMPDIR:-/tmp}/ltx-transformer-${name}-${dtype}-swift.safetensors"

    print "== transformer-forward $name $dtype ($token_count video tokens) =="
    "$PYTHON_BIN" "$ROOT_DIR/scripts/ltx-parity/dump.py" transformer-forward \
      --model-dir "$MODEL_DIR" \
      --weights "$WEIGHTS_NAME" \
      --output "$reference" \
      --input-dtype "$dtype" \
      --video-frames "$frames" \
      --video-height "$height" \
      --video-width "$width" \
      --audio-tokens 256 \
      --text-tokens 64 \
      --seed 7000
    "$POC_BIN" transformer \
      --model-dir "$MODEL_DIR" \
      --weights "$WEIGHTS_NAME" \
      --input "$reference" \
      --output "$actual"
    "$POC_BIN" compare --reference "$reference" --actual "$actual" --key video_velocity
    "$POC_BIN" compare --reference "$reference" --actual "$actual" --key audio_velocity

    print "== transformer-tiled $name $dtype ($token_count video tokens) =="
    tiled_reference="$FIXTURE_DIR/ltx-transformer-tiled-${name}-${dtype}.safetensors"
    tiled_actual="${TMPDIR:-/tmp}/ltx-transformer-tiled-${name}-${dtype}-swift.safetensors"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/ltx-parity/dump.py" transformer-tiled \
      --model-dir "$MODEL_DIR" \
      --weights "$WEIGHTS_NAME" \
      --output "$tiled_reference" \
      --input-dtype "$dtype" \
      --video-frames "$frames" \
      --video-height "$height" \
      --video-width "$width" \
      --audio-tokens 256 \
      --text-tokens 64 \
      --tile-frames 2 \
      --tile-frame-overlap 1 \
      --tile-height 1 \
      --tile-height-overlap 0 \
      --tile-width 1 \
      --tile-width-overlap 0 \
      --seed 7000
    "$POC_BIN" transformer-tiled \
      --model-dir "$MODEL_DIR" \
      --weights "$WEIGHTS_NAME" \
      --input "$tiled_reference" \
      --output "$tiled_actual"
    "$POC_BIN" compare --reference "$tiled_reference" --actual "$tiled_actual" --key video_velocity
    "$POC_BIN" compare --reference "$tiled_reference" --actual "$tiled_actual" --key audio_velocity
  done
done
