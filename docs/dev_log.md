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

## Baseline Asset Update — 2026-08-04

The environment was completed far enough to import the full VILA model path:
`ps3-torch==0.1.3`, `ftfy`, `tiktoken`, `triton==3.1.0`,
`protobuf==3.20.3`, `s2wrapper==0.1`, and `flash-attn==2.5.8` are installed.
The earlier note about PS3 being uninstalled is superseded by this update.
`pip check` still reports two upstream constraints: Torch 2.3.0 declares
`triton==2.3.0`, while the VILA FP8 import path requires Triton 3.1.0;
PS3 declares `timm==1.0.15`, while the VILA baseline pins `timm==0.9.12`.
Neither conflict affects the ordinary M0-M2 smoke paths.

External baseline assets are stored outside the repository:

```text
/9950backfile/chenjiahui/evo_artifacts/models/VILA1.5-3b/
/9950backfile/chenjiahui/evo_artifacts/data/VILA-inference-demos/
```

The checkpoint contains 17 files with no incomplete downloads. The demo
directory contains `imagenet_cat.jpg` and `OAI-sora-tokyo-walk.mp4`.
An image caption smoke passed with the local checkpoint. An 8-frame video
caption smoke also passed after setting the protocol value `config.fps = 0`.
The stock `vila-infer` CLI currently leaves `config.fps` as `None`, so its
video path raises a `TypeError` before decoding; this is recorded as an
upstream baseline CLI issue and is not changed in M0-M2.
