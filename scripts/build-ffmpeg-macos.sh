#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
FFMPEG_VERSION="${FFMPEG_VERSION:-8.1.2}"
LAME_VERSION="${LAME_VERSION:-4.0}"
PREFIX="${FFMPEG_PREFIX:-$PROJECT_DIR/third_party/ffmpeg}"
PREFIX_BACKUP="$PREFIX.bak"
SOURCE_ROOT="${FFMPEG_SOURCE_ROOT:-${TMPDIR:-/tmp}/genmedia-ffmpeg-source}"
INSTALL_PREFIX="${FFMPEG_INSTALL_PREFIX:-/opt/genmedia-ffmpeg}"
STAGING_ROOT="$SOURCE_ROOT/staging-$FFMPEG_VERSION"
STAGED_PREFIX="$STAGING_ROOT$INSTALL_PREFIX"
ARCHIVE="$SOURCE_ROOT/ffmpeg-$FFMPEG_VERSION.tar.xz"
SOURCE_DIR="$SOURCE_ROOT/ffmpeg-$FFMPEG_VERSION"
DOWNLOAD_URL="https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
LAME_ARCHIVE="$SOURCE_ROOT/lame-$LAME_VERSION.tar.gz"
LAME_SOURCE_DIR="$SOURCE_ROOT/lame-$LAME_VERSION"
LAME_PREFIX="$SOURCE_ROOT/lame-$LAME_VERSION-macos14"
LAME_DOWNLOAD_URL="https://downloads.sourceforge.net/project/lame/lame/$LAME_VERSION/lame-$LAME_VERSION.tar.gz"
LAME_LIB_DIR="$LAME_PREFIX/lib"
BUILD_SUCCEEDED=false

for command_name in curl tar make clang pkg-config otool vtool install_name_tool; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    print -u2 "缺少 FFmpeg 建置必要指令：$command_name"
    exit 1
  fi
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "此 FFmpeg 建置腳本只支援 Apple Silicon macOS。"
  exit 1
fi
restore_previous_prefix() {
  if [[ "$BUILD_SUCCEEDED" == false ]]; then
    rm -rf -- "$PREFIX"
    if [[ -d "$PREFIX_BACKUP" ]]; then
      mv -- "$PREFIX_BACKUP" "$PREFIX"
    fi
  fi
}
trap restore_previous_prefix EXIT INT TERM

mkdir -p "$SOURCE_ROOT"
if [[ ! -d "$SOURCE_DIR" ]]; then
  if [[ ! -s "$ARCHIVE" ]]; then
    print "正在下載 FFmpeg $FFMPEG_VERSION…"
    curl --fail --location --retry 3 --output "$ARCHIVE" "$DOWNLOAD_URL"
  fi
  tar -xJf "$ARCHIVE" -C "$SOURCE_ROOT"
fi
if [[ ! -d "$LAME_SOURCE_DIR" ]]; then
  if [[ ! -s "$LAME_ARCHIVE" ]]; then
    print "正在下載 LAME $LAME_VERSION…"
    curl --fail --location --retry 3 --output "$LAME_ARCHIVE" "$LAME_DOWNLOAD_URL"
  fi
  tar -xzf "$LAME_ARCHIVE" -C "$SOURCE_ROOT"
fi

if [[ -e "$PREFIX_BACKUP" ]]; then
  print -u2 "備份目錄已存在，請先確認後移除：$PREFIX_BACKUP"
  exit 1
fi
if [[ -e "$PREFIX" ]]; then
  mv -- "$PREFIX" "$PREFIX_BACKUP"
fi
mkdir -p "$PREFIX"

rm -rf -- "$LAME_PREFIX"
cd "$LAME_SOURCE_DIR"
if [[ -f Makefile ]]; then
  make distclean
fi
CFLAGS="-O3 -mmacosx-version-min=14.0" \
LDFLAGS="-mmacosx-version-min=14.0" \
./configure \
  --prefix="$LAME_PREFIX" \
  --enable-shared \
  --disable-static \
  --disable-decoder \
  --disable-frontend
make -j"$(sysctl -n hw.ncpu)"
make install

if [[ ! -f "$LAME_PREFIX/include/lame/lame.h" || \
      ! -e "$LAME_LIB_DIR/libmp3lame.0.dylib" ]]; then
  print -u2 "LAME 建置完成但缺少標頭或動態函式庫。"
  exit 1
fi

cd "$SOURCE_DIR"
if [[ -f ffbuild/config.mak ]]; then
  make distclean
fi

rm -rf -- "$STAGING_ROOT"

PKG_CONFIG_PATH="$LAME_LIB_DIR/pkgconfig" ./configure \
  --prefix="$INSTALL_PREFIX" \
  --arch=arm64 \
  --target-os=darwin \
  --cc=clang \
  --enable-shared \
  --disable-static \
  --disable-debug \
  --disable-doc \
  --disable-ffplay \
  --disable-network \
  --disable-autodetect \
  --enable-ffmpeg \
  --enable-ffprobe \
  --enable-videotoolbox \
  --enable-audiotoolbox \
  --enable-libmp3lame \
  --extra-cflags="-I$LAME_PREFIX/include -mmacosx-version-min=14.0" \
  --extra-ldflags="-L$LAME_LIB_DIR -mmacosx-version-min=14.0 -Wl,-rpath,@executable_path/../lib"

make -j"$(sysctl -n hw.ncpu)"
make install DESTDIR="$STAGING_ROOT"
if [[ ! -d "$STAGED_PREFIX" ]]; then
  print -u2 "FFmpeg 暫存安裝目錄不存在：$STAGED_PREFIX"
  exit 1
fi
/bin/cp -R "$STAGED_PREFIX/." "$PREFIX/"

typeset -a EXTERNAL_LIBRARIES
EXTERNAL_LIBRARIES=(
  "$LAME_LIB_DIR"/libmp3lame*.dylib(N)
)
if (( ${#EXTERNAL_LIBRARIES[@]} == 0 )); then
  print -u2 "找不到自行建置的 LAME 動態函式庫。"
  exit 1
fi
for library in "${EXTERNAL_LIBRARIES[@]}"; do
  /bin/cp -P "$library" "$PREFIX/lib/${library:t}"
done

while IFS= read -r library; do
  /usr/bin/install_name_tool -id "@rpath/${library:t}" "$library"
  while IFS= read -r dependency; do
    [[ -e "$PREFIX/lib/${dependency:t}" ]] || continue
    /usr/bin/install_name_tool \
      -change "$dependency" "@rpath/${dependency:t}" "$library"
  done < <(
    /usr/bin/otool -L "$library" \
      | sed -n 's/^[[:space:]]*\([^[:space:]]*\.dylib\).*$/\1/p'
  )
done < <(find "$PREFIX/lib" -maxdepth 1 -type f -name '*.dylib' -print)

for tool in ffmpeg ffprobe; do
  /usr/bin/install_name_tool \
    -add_rpath '@executable_path/../lib' "$PREFIX/bin/$tool" 2>/dev/null || true
  while IFS= read -r dependency; do
    [[ -e "$PREFIX/lib/${dependency:t}" ]] || continue
    /usr/bin/install_name_tool \
      -change "$dependency" "@rpath/${dependency:t}" "$PREFIX/bin/$tool"
  done < <(
    /usr/bin/otool -L "$PREFIX/bin/$tool" \
      | sed -n 's/^[[:space:]]*\([^[:space:]]*\.dylib\).*$/\1/p'
  )
done

mkdir -p \
  "$PREFIX/share/licenses/ffmpeg" \
  "$PREFIX/share/licenses/lame"
/bin/cp "$SOURCE_DIR/COPYING.LGPLv2.1" \
  "$PREFIX/share/licenses/ffmpeg/COPYING.LGPLv2.1"
/bin/cp "$LAME_SOURCE_DIR/COPYING" "$PREFIX/share/licenses/lame/COPYING"
/bin/cp "$LAME_SOURCE_DIR/LICENSE" "$PREFIX/share/licenses/lame/LICENSE"

cat > "$PREFIX/share/licenses/ffmpeg/BUILD-INFO.txt" <<EOF
FFmpeg version: $FFMPEG_VERSION
Source: $DOWNLOAD_URL
LAME version: $LAME_VERSION
LAME source: $LAME_DOWNLOAD_URL
Build script: scripts/build-ffmpeg-macos.sh
Configuration: shared LGPL build; GPL and nonfree components are disabled.
EOF

if [[ ! -x "$PREFIX/bin/ffmpeg" || ! -x "$PREFIX/bin/ffprobe" ]]; then
  print -u2 "FFmpeg 建置完成但找不到 ffmpeg 或 ffprobe。"
  exit 1
fi

CONFIGURATION="$($PREFIX/bin/ffmpeg -version | sed -n 's/^configuration: //p')"
if [[ "$CONFIGURATION" == *"--enable-gpl"* || "$CONFIGURATION" == *"--enable-nonfree"* ]]; then
  print -u2 "拒絕保留啟用 GPL 或 nonfree 的 FFmpeg 建置。"
  exit 1
fi

ENCODERS="$($PREFIX/bin/ffmpeg -hide_banner -encoders 2>/dev/null)"
for encoder in libmp3lame aac flac h264_videotoolbox pcm_s16le; do
  if [[ "$ENCODERS" != *"$encoder"* ]]; then
    print -u2 "FFmpeg 缺少必要 Encoder：$encoder"
    exit 1
  fi
done

typeset -a DEPLOYMENT_TARGET_BINARIES
DEPLOYMENT_TARGET_BINARIES=(
  "$PREFIX/bin/ffmpeg"
  "$PREFIX/bin/ffprobe"
  "$PREFIX/lib"/*.dylib(N.)
)
for binary in "${DEPLOYMENT_TARGET_BINARIES[@]}"; do
  BUILD_INFO="$(/usr/bin/vtool -show-build "$binary")"
  if [[ "$BUILD_INFO" != *"minos 14.0"* ]]; then
    print -u2 "建置產物不是 macOS 14 相容版本：$binary"
    print -u2 "$BUILD_INFO"
    exit 1
  fi
done

BUILD_SUCCEEDED=true
if [[ -d "$PREFIX_BACKUP" ]]; then
  rm -rf -- "$PREFIX_BACKUP"
fi
trap - EXIT INT TERM

print "完成 GenMedia LGPL FFmpeg：$PREFIX"
print "版本：$FFMPEG_VERSION"
print "已包含 ffmpeg、ffprobe、MP3、AAC、FLAC、PCM 與 VideoToolbox H.264。"
