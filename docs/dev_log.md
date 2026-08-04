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

## Environment Compatibility Update - 2026-08-04

The authoritative environment now uses a dependency set that satisfies both
VILA and the PS3/S2 integrations:

```text
torch==2.4.1+cu121
torchvision==0.19.1
triton==3.0.0
timm==1.0.15
flash-attn==2.7.4.post1 (PyTorch 2.4, CUDA 12 wheel)
transformers==4.46.0
ps3-torch==0.1.3
s2wrapper==0.1
```

Torch 2.4.1 declares Triton 3.0.0, and Triton 3.0.0 provides the
`libdevice` API used by VILA's FP8 kernels. PS3's exact `timm==1.0.15`
requirement is now the project pin, and the VILA imports used by the standard
path remain compatible. `pip check` reports no broken requirements.

The video media helper now treats a null `fps` configuration as `0.0`, so the
stock CLI uses uniform frame sampling when no FPS override is configured.

## External Asset Inventory - 2026-08-04

Assets remain outside Git under `/9950backfile`:

| Asset | Local path | Status |
| --- | --- | --- |
| VILA1.5-3b | `/9950backfile/chenjiahui/evo_artifacts/models/VILA1.5-3b/` | 17 files, about 5.9G; HF revision `42d1dda6807cc521ef27674ca2ae157539d17026` |
| VILA inference demos | `/9950backfile/chenjiahui/evo_artifacts/data/VILA-inference-demos/` | cat image and Sora Tokyo video |
| SAM2.1 tiny | `/9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_tiny.pt` | SHA256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |
| SAM2.1 base-plus | `/9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_base_plus.pt` | SHA256 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` |
| DAVIS 2017 | `/9950backfile/zhangyafei/DAVIS-2017/` | about 5.5G; 60 train and 30 validation sequences with JPEG frames and annotations |
| YouTube-VOS 2019 | `/9950backfile/zhangyafei/YouTubeVOS2019/` | train, valid, test, and test ground-truth archives/metadata are present |

The SAM2 files are the official SAM2.1 checkpoints. DAVIS is the first M3
validation target; YouTube-VOS is reserved for the later training and
retention protocol. No dataset or checkpoint is copied into the repository.

## M3 Video Segmentation Update - 2026-08-04

The M3 implementation adds an opt-in `video_segmentation` capability backed
by the official SAM2 video predictor. It accepts a JPEG frame directory or
MP4, supports anchor-frame points/boxes for one or multiple objects, and
returns CPU boolean masks shaped `[T,H,W]` for one object or `[T,N,H,W]` for
multiple objects. `return_result=True` additionally exposes object IDs, video
size, anchor frame, and instrumentation.

SAM2 is not imported by ordinary capability/package initialization. The
predictor is built only inside an explicit `segment_videos()` request. The
default local configuration uses:

```text
SAM2 root: /9950backfile/zhangyafei/sam2
checkpoint: /9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_tiny.pt
config: configs/sam2.1/sam2.1_hiera_t.yaml
```

Focused capability tests pass (`13 passed`). The external-asset command is:

```bash
PYTHONPATH=/9950backfile/zhangyafei/sam2:/9950backfile/chenjiahui/EvoVILA \
  /9950backfile/chenjiahui/.conda/envs/evovila/bin/python \
  scripts/evo/smoke_video_segmentation.py --sequence bear --device cuda
```

The DAVIS 2017 `bear` validation sequence completed on an NVIDIA A800: 82
frames, mask shape `[82,480,854]`, mean IoU `0.968115`, predictor latency
`3.739s`, anchor latency `0.824s`, propagation latency `4.289s`, and total
latency `16.546s`. SAM2 reported that its optional compiled `_C` post-processing
extension was unavailable and skipped hole filling; core predictor propagation
completed successfully, and this warning is retained as a reproducibility
note.

The ordinary VILA regression also passed with the local VILA1.5-3b checkpoint:
image caption and 8-frame video caption both completed on the same environment.
The CLI had a separate default-mode issue: `--conv-mode auto` overwrote the
model-detected legacy template with AUTO even when the checkpoint had no
Transformers `chat_template`. `llava/cli/infer.py` now preserves the detected
template for default/explicit `auto`; an explicit non-auto mode still overrides
it.

### M3 Running Checks

```bash
/9950backfile/chenjiahui/.conda/envs/evovila/bin/python -m pytest -q tests -ra
/9950backfile/chenjiahui/.conda/envs/evovila/bin/python -m py_compile \
  llava/capabilities/*.py llava/model/llava_arch.py \
  llava/remote_code/modeling_vila.py scripts/evo/smoke_video_segmentation.py
EVO_PYTHON=/9950backfile/chenjiahui/.conda/envs/evovila/bin/python \
  bash scripts/evo/check_repo.sh
git diff --check
```
