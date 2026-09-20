#!/bin/zsh

# Shared by source backup and cleanup; discover new Worker packages automatically.
genimage_package_roots() {
  local project_root="$1"
  local manifest
  print -r -- "$project_root"
  for manifest in "$project_root"/RuntimeSupport/*/Package.swift(N); do
    print -r -- "${manifest:h}"
  done
}
