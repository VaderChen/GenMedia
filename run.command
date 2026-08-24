#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

export COPYFILE_DISABLE=1

if ! command -v swift >/dev/null 2>&1; then
  print -u2 "錯誤：找不到 Swift。請先安裝 Xcode，並完成 Command Line Tools 設定。"
  exit 1
fi

print "正在確認 GenImage 為最新版本…"
"$SCRIPT_DIR/build.command" --no-app

BIN_DIR="$(swift build -c release --show-bin-path)"
APP_EXECUTABLE="$BIN_DIR/GenImage"
QWEN_WORKER_PACKAGE="$SCRIPT_DIR/RuntimeSupport/Qwen2511Worker"
QWEN_WORKER_BIN_DIR="$(swift build --package-path "$QWEN_WORKER_PACKAGE" -c release --show-bin-path)"
export GENIMAGE_QWEN_WORKER="$QWEN_WORKER_BIN_DIR/GenImageQwen2511Worker"

print "正在啟動 GenImage…"
exec "$APP_EXECUTABLE" "$@"
