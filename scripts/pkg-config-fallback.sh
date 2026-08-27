#!/bin/sh

# LAME 4.0's configure script requires a pkg-config executable even when its
# optional sndfile/mpg123 integrations and frontends are disabled.  The
# GenMedia build only needs libmp3lame itself, so this fallback advertises the
# pkg-config protocol version while reporting every optional package as absent.
case "${1:-}" in
  --version)
    printf '%s\n' '0.29.2-genmedia-fallback'
    exit 0
    ;;
  --atleast-pkgconfig-version)
    exit 0
    ;;
  *)
    exit 1
    ;;
esac
