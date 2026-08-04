# Development Log

## Current Status

- M0 repository audit, baseline protection, and code map: complete.
- M1 capability request, read-only media context, registry, and no-op path:
  complete.
- M2 image segmentation warm-up: complete as an opt-in static-grid decoder.

## M2 Decisions

The branch consumes projected VILA image features, uses BCE plus Dice loss,
and keeps capability parameters in `capabilities.bin`. It does not implement
text conditioning, `[SEG]` tokens, video segmentation, SAM2, or dynamic
routing.

## Running Checks

From the repository root:

```bash
/public/duyinglong/miniconda3/envs/chatts/bin/python -m pytest -q tests/test_capabilities.py
/public/duyinglong/miniconda3/envs/chatts/bin/python -m py_compile llava/capabilities/*.py llava/model/llava_arch.py llava/model/configuration_llava.py llava/remote_code/modeling_vila.py llava/remote_code/configuration_vila.py llava/data/collate.py llava/model/language_model/llava_llama.py llava/model/language_model/llava_topdown_llama.py
EVO_PYTHON=/9950backfile/chenjiahui/.conda/envs/evovila/bin/python bash scripts/evo/check_repo.sh
git diff --check
```

Full model loading still depends on the project environment and local VILA
model assets; no weights or datasets are downloaded by these checks.

## Environment Update — 2026-08-04

The authoritative EvoVILA environment is installed under `/9950backfile`:

```bash
conda activate /9950backfile/chenjiahui/.conda/envs/evovila
```

Use the environment's interpreter explicitly in scripts and checks:

```bash
EVO_PYTHON=/9950backfile/chenjiahui/.conda/envs/evovila/bin/python \
  bash scripts/evo/check_repo.sh
/9950backfile/chenjiahui/.conda/envs/evovila/bin/python -m pytest -q tests -ra
```

The environment uses Python 3.10.14, PyTorch 2.3.0 with CUDA 12.1,
Transformers 4.46.0, Hydra 1.3.4, Loguru 0.7.3, `s2wrapper` 0.1, and the
VILA-compatible FlashAttention 2.5.8 wheel. `ps3-torch` remains intentionally
uninstalled for M0-M2 because its current `timm==1.0.15` requirement conflicts
with VILA's pinned `timm==0.9.12`; PS3 is not used by the M0-M2 paths.
