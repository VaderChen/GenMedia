#!/bin/zsh
# Downloads the ComfyUI MiniMax H3 reference sources used by the parity scripts.
# They are third-party sources, so they are fetched on demand rather than vendored.
set -euo pipefail
DEST="${1:-${TMPDIR:-/tmp}/minimax-h3-ref}"
BASE="https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master"
mkdir -p "$DEST"
typeset -A FILES
FILES=(
  "comfy_extras/nodes_minimax_h3.py"  "nodes_minimax_h3.py"
  "comfy/ldm/minimax/model.py"        "model.py"
  "comfy/ldm/minimax/vae.py"          "vae.py"
  "comfy/ldm/minimax/audio_vae.py"    "audio_vae.py"
  "comfy/text_encoders/minimax.py"    "minimax.py"
  "comfy/text_encoders/qwen3vl.py"    "qwen3vl.py"
  "comfy/samplers.py"                 "samplers.py"
  "comfy/model_sampling.py"           "model_sampling.py"
)
for src dst in ${(kv)FILES}; do
  curl -sSLf -o "$DEST/$dst" "$BASE/$src"
  print "  $(wc -l < "$DEST/$dst" | tr -d ' ') lines  $dst"
done
print "reference in $DEST"
