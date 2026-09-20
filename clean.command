#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ ! -f "$SCRIPT_DIR/Package.swift" || ! -d "$SCRIPT_DIR/Sources" ]]; then
  print -u2 "錯誤：$SCRIPT_DIR 不是 GenImage 專案目錄。"
  exit 1
fi

print "正在清除 GenImage 中間產物…"

source "$SCRIPT_DIR/scripts/project-packages.sh"
while IFS= read -r package_root; do
  if [[ -d "$package_root/.build" ]]; then
    rm -rf -- "$package_root/.build"
    print "已清除：${package_root#$SCRIPT_DIR/}/.build"
  fi
done < <(genimage_package_roots "$SCRIPT_DIR")

find "$SCRIPT_DIR" \
  -type d \( -name '.git' -o -name 'Backups' -o -name '*.bak' \
    -o -name 'Models' -o -name 'Generated' -o -name 'Outputs' \
    -o -name 'dist' -o -name 'third_party' -o -name 'cert' \) -prune -o \
  -type f -name '.DS_Store' -exec rm -f -- {} +

print "清理完成。已清除主程式及 Worker 編譯快取與 .DS_Store；保留原始碼、模型、輸出與所有 .bak 備份。"
