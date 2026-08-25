#!/bin/zsh

# 套用 Patches/manifest.txt 列出的相依套件原始碼修正。
#
# 這些修正貼在 .build/checkouts/ 內，那是 SwiftPM 隨時可以重建的目錄，所以每次建置都要重跑。
# 本腳本的重點是「絕不安靜地失敗」：checkout 不存在、patch 檔遺失、Package.resolved 的版本與
# patch 撰寫時不符、patch 套用失敗、或套用後找不到預期的標記，一律中止並說明原因。過去這裡
# 只用 grep 判斷是否已套用，任何一個環節出錯都會直接以「未修正的原始碼」繼續建置。
#
#   apply-runtime-patches.command            套用並驗證
#   apply-runtime-patches.command --verify   只驗證，不修改任何檔案

set -euo pipefail
# 下方以 [[:space:]]# 去除欄位前後空白，需要 zsh 的延伸萬用字元。
setopt extended_glob

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
MANIFEST="$REPO_ROOT/Patches/manifest.txt"

VERIFY_ONLY=false
case "${1:-}" in
  --verify) VERIFY_ONLY=true ;;
  "") ;;
  *)
    print -u2 "用法：${0:t} [--verify]"
    exit 2
    ;;
esac

fail() {
  print -u2 "錯誤：$1"
  shift
  for line in "$@"; do print -u2 "  $line"; done
  exit 1
}

# Package.resolved 內某個 identity 的版本；沒有 version 欄位（以 revision 釘選）時回傳 revision。
resolved_pin() {
  local resolved="$1" identity="$2"
  [[ -f "$resolved" ]] || return 1
  /usr/bin/awk -v want="$identity" '
    /"identity"[[:space:]]*:/ {
      value = $0
      sub(/.*"identity"[[:space:]]*:[[:space:]]*"/, "", value)
      sub(/".*/, "", value)
      inside = (value == want)
      revision = ""
      next
    }
    inside && /"revision"[[:space:]]*:/ {
      value = $0
      sub(/.*"revision"[[:space:]]*:[[:space:]]*"/, "", value)
      sub(/".*/, "", value)
      revision = value
      next
    }
    inside && /"version"[[:space:]]*:/ {
      value = $0
      sub(/.*"version"[[:space:]]*:[[:space:]]*"/, "", value)
      sub(/".*/, "", value)
      print value
      found = 1
      exit
    }
    inside && /^[[:space:]]*}[[:space:]]*$/ && revision != "" {
      print revision
      found = 1
      exit
    }
    END { if (!found) exit 1 }
  ' "$resolved"
}

typeset -a package_paths pin_identities expected_pins checkout_dirs
typeset -a patch_files marker_files markers descriptions
typeset -A checked_pins

while IFS= read -r raw_line; do
  line="${raw_line##[[:space:]]#}"
  line="${line%%[[:space:]]#}"
  # 只支援整行註解：欄位內容可以包含 #，不做行內註解裁切。
  [[ -z "$line" || "$line" == \#* ]] && continue

  typeset -a fields
  fields=("${(@s:|:)line}")
  [[ ${#fields} -eq 8 ]] || fail "Patches/manifest.txt 每行需要 8 個欄位，這行有 ${#fields} 個：" "$line"

  typeset -a trimmed
  trimmed=()
  for field in "${fields[@]}"; do
    field="${field##[[:space:]]#}"
    trimmed+=("${field%%[[:space:]]#}")
  done

  package_paths+=("${trimmed[1]}")
  pin_identities+=("${trimmed[2]}")
  expected_pins+=("${trimmed[3]}")
  checkout_dirs+=("${trimmed[4]}")
  patch_files+=("${trimmed[5]}")
  marker_files+=("${trimmed[6]}")
  markers+=("${trimmed[7]}")
  descriptions+=("${trimmed[8]}")
done < "$MANIFEST"

[[ ${#patch_files} -gt 0 ]] || fail "Patches/manifest.txt 沒有任何項目。"

applied_count=0
for index in {1..${#patch_files}}; do
  package_path="${package_paths[$index]}"
  pin_identity="${pin_identities[$index]}"
  expected_pin="${expected_pins[$index]}"
  checkout_dir="${checkout_dirs[$index]}"
  patch_file="$REPO_ROOT/Patches/${patch_files[$index]}"
  description="${descriptions[$index]}"

  package_root="$REPO_ROOT/$package_path"
  [[ "$package_path" == "." ]] && package_root="$REPO_ROOT"
  checkout="$package_root/.build/checkouts/$checkout_dir"
  marker_file="$checkout/${marker_files[$index]}"
  marker="${markers[$index]}"

  # 相依套件版本一旦更動，patch 就可能套到不同的程式碼上；先擋下來要求人確認。
  pin_key="$package_path/$pin_identity"
  if [[ -z "${checked_pins[$pin_key]:-}" ]]; then
    actual_pin="$(resolved_pin "$package_root/Package.resolved" "$pin_identity" || true)"
    [[ -n "$actual_pin" ]] || fail \
      "在 $package_path/Package.resolved 找不到相依套件 $pin_identity。" \
      "請先執行：swift package --package-path $package_path resolve"
    [[ "$actual_pin" == "$expected_pin" ]] || fail \
      "$pin_identity 的版本與修正檔不符。" \
      "Package.resolved：$actual_pin" \
      "Patches/manifest.txt：$expected_pin" \
      "請確認 Patches/ 內的修正仍適用於新版本，再更新 manifest 的 expected_pin。"
    checked_pins[$pin_key]="$actual_pin"
  fi

  [[ -f "$patch_file" ]] || fail "找不到修正檔：Patches/${patch_files[$index]}（$description）"
  [[ -d "$checkout" ]] || fail \
    "找不到相依套件原始碼：$checkout" \
    "請先執行：swift package --package-path $package_path resolve"
  [[ -f "$marker_file" ]] || fail \
    "$description：找不到要修正的檔案 ${marker_files[$index]}。" \
    "$pin_identity 的目錄結構可能已改變。"

  if /usr/bin/grep -qF -- "$marker" "$marker_file"; then
    continue
  fi

  if [[ "$VERIFY_ONLY" == true ]]; then
    fail "$description 尚未套用（$checkout_dir）。" "請執行：scripts/${0:t}"
  fi

  print "正在套用 $description…"
  # -N 已套用的 hunk 直接略過、-t 不互動詢問、-F 0 禁止模糊比對（避免貼到錯誤的位置）。
  # 舊版流程用 stdin 餵 patch，遇到 "Assume -R?" 會把 patch 內容當成回答，可能反向套用。
  if ! /usr/bin/patch -p1 -d "$checkout" -i "$patch_file" -N -t -F 0 -V none; then
    fail "$description 套用失敗（$checkout_dir）。" \
      "若 $pin_identity 已升級，請重新產生 Patches/${patch_files[$index]}。"
  fi
  applied_count=$((applied_count + 1))
done

# 後置驗證：無論這次有沒有動作，每個標記都必須存在，否則不允許繼續建置。
for index in {1..${#patch_files}}; do
  package_path="${package_paths[$index]}"
  package_root="$REPO_ROOT/$package_path"
  [[ "$package_path" == "." ]] && package_root="$REPO_ROOT"
  marker_file="$package_root/.build/checkouts/${checkout_dirs[$index]}/${marker_files[$index]}"
  /usr/bin/grep -qF -- "${markers[$index]}" "$marker_file" || fail \
    "${descriptions[$index]} 套用後仍找不到預期的標記。" \
    "檔案：$marker_file" \
    "標記：${markers[$index]}"
done

if [[ "$VERIFY_ONLY" == true ]]; then
  print "相依套件修正已全部套用（${#patch_files} 項）。"
elif [[ $applied_count -gt 0 ]]; then
  # 套用失敗時 patch 會留下 .rej 供人檢查；成功後清掉殘留的備份檔，避免下次誤判。
  for index in {1..${#patch_files}}; do
    package_path="${package_paths[$index]}"
    package_root="$REPO_ROOT/$package_path"
    [[ "$package_path" == "." ]] && package_root="$REPO_ROOT"
    checkout="$package_root/.build/checkouts/${checkout_dirs[$index]}"
    [[ -d "$checkout" ]] || continue
    /usr/bin/find "$checkout" -name "*.orig" -type f -delete 2>/dev/null || true
  done
  print "相依套件修正完成（本次套用 $applied_count 項，共 ${#patch_files} 項）。"
fi
