#!/usr/bin/env bash

# Read-only repository hygiene report for EvoVILA.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${EVO_PYTHON:-python}"
cd "$repo_root"

printf '%s\n' "== EvoVILA repository check =="
printf 'root: %s\n' "$repo_root"
printf 'branch: '
git branch --show-current 2>/dev/null || printf '%s\n' "unavailable"

printf '%s\n' "-- remotes --"
git remote -v 2>/dev/null || true

printf '%s\n' "-- submodules --"
git submodule status 2>/dev/null || true

printf '%s\n' "-- working tree --"
if git status --short 2>/dev/null | sed -n '1,120p'; then
    :
else
    printf '%s\n' "git status unavailable"
fi

printf '%s\n' "-- tracked files at least 100 MiB --"
large_files=0
while IFS= read -r -d '' path; do
    if [[ -f "$path" ]]; then
        size_bytes="$(stat -c '%s' "$path" 2>/dev/null || printf '0')"
        if [[ "$size_bytes" -ge 104857600 ]]; then
            printf '%s bytes %s\n' "$size_bytes" "$path"
            large_files=1
        fi
    fi
done < <(git ls-files -z 2>/dev/null)
if [[ "$large_files" -eq 0 ]]; then
    printf '%s\n' "none"
fi

printf '%s\n' "-- tracked or non-ignored suspicious weight files --"
weight_files=0
while IFS= read -r path; do
    if [[ "$path" =~ (^|/)(checkpoints?|weights?|model_weights?)(/|$) || "$path" =~ \.(bin|ckpt|onnx|pt|pth|safetensors|safetensor)$ ]]; then
        printf '%s\n' "$path"
        weight_files=1
    fi
done < <(git ls-files -co --exclude-standard 2>/dev/null)
if [[ "$weight_files" -eq 0 ]]; then
    printf '%s\n' "none"
fi

printf '%s\n' "-- Python environment --"
if command -v "$python_bin" >/dev/null 2>&1 || [[ -x "$python_bin" ]]; then
    "$python_bin" --version 2>&1
    "$python_bin" -c 'import sys; print("executable:", sys.executable); print("version:", sys.version.split()[0])' 2>/dev/null || true
else
    printf 'python not found: %s\n' "$python_bin"
fi

printf '%s\n' "== check complete (read-only) =="
