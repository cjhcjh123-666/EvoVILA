# EvoVILA Roadmap

The roadmap is staged so the original VILA capability set can be measured
before and after every extension. M0, M1, M2, and the M3 video segmentation
warm-up are active in the current branch. Conditional compute remains future
work.

## M0: VILA Audit and Baseline Preservation

**Input:** The imported VILA repository, its Git history, and the pinned
`long_rl` submodule reference.

**Output:** Repository remotes and development branch are initialized;
`AGENTS.md`, `docs/PROJECT_SCOPE.md`, `docs/VILA_CODE_MAP.md`, this roadmap,
`scripts/evo/check_repo.sh`, `configs/evo/README.md`, and
`docs/BASELINE_PROTOCOL.md` document the baseline and extension rules.

**Acceptance:** `origin` points to EvoVILA, `upstream` points to the original
VILA repository, the working tree has no weights/data/cache additions, the
submodule is pinned to the commit recorded by VILA, and the code map identifies
real files/functions for model construction, media, fusion, training, and
evaluation. Text-only and multimodal smoke-test contracts are specified before
new capability code is added.

**Main risks:** Upstream/network access may prevent fetching the submodule;
the large VILA surface can hide alternate inference and training paths; model
weights are intentionally not downloaded for this audit.

## M1: Modular Capability-Extension Interface

**Status:** Implemented as an observation-only interface; no segmentation
implementation is enabled.

**Input:** The M0 call map and baseline smoke tests.

**Output:** `llava.capabilities` provides a registry, explicit request object,
read-only media context, and no-op pipeline. The normal and remote-code model
paths notify this pipeline at `_embed` without handing control of media-token
fusion to an extension. See `docs/CAPABILITY_INTERFACE.md`.

**Acceptance:** Ordinary VILA loading and generation work without optional
dense dependencies; an extension can receive media context without changing
default tensor/token alignment; interface tests cover image and video-shaped
inputs.

**Main risks:** Coupling a new interface to `_embed` may regress media-token
ordering, checkpoint compatibility, or distributed dummy calls.

## M2: Image Segmentation Warm-Up

**Status:** Implemented as an opt-in static-grid mask branch. It consumes
projected image features and does not condition on language hidden states.

**Input:** M1 interface and image grounding/segmentation training schema.

**Output:** `image_segmentation` provides a lightweight mask decoder, BCE plus
Dice loss, an optional `segmentation_masks` batch field, `segment_images()`,
and a separate `capabilities.bin` checkpoint. The default VILA path remains
unchanged when the capability is not requested.

**Acceptance:** Static square visual-token smoke tests cover mask outputs,
losses, gradients, and explicit activation; ordinary image QA/captioning
retains the baseline path when the capability is disabled; no SAM2 dependency
is required for image-only execution.

**Main risks:** Image/text alignment, mask resolution, added memory, and
unintended gradient flow into frozen VILA modules. Full referring/reasoning
segmentation is deferred until a text-conditioned decoder and data contract
are designed.

## M3: Video Segmentation with SAM2

**Status:** Implemented as an explicit `video_segmentation` capability. The
branch accepts a frame directory or MP4, applies anchor-frame point/box
prompts, propagates with the official SAM2.1 video predictor, returns binary
masks shaped `[T,H,W]` or `[T,N,H,W]`, and records predictor, anchor,
propagation, and total latency. SAM2 import and predictor construction remain
lazy, and the default VILA path does not load them.

**Input:** M2 image branch and a versioned video segmentation data contract.

**Output:** An optional video branch with anchor-frame prediction, SAM2
propagation, and explicit per-component instrumentation.

**Acceptance:** Focused tests cover lazy import, prompt validation, output
normalization, missing-dependency errors, and both model entry points. A
SAM2.1 tiny plus DAVIS 2017 `bear` smoke passed on an NVIDIA A800 with 82
frames, output shape `[82,480,854]`, mean IoU `0.9681`, and separate anchor,
propagation, and total latency. Referring/reasoning segmentation and full
vision/LLM/mask-decoder training accounting remain deferred to later stages.

**Main risks:** Temporal identity drift, frame sampling mismatch, propagation
memory growth, and a hidden dependency in generic model loading.

## M4: Capability Retention and Mixed Training

**Input:** M2/M3 branches and the original VILA training/data registry.

**Output:** Mixed-capability training recipes, retention datasets, and
checkpoint policies that keep optional modules separable.

**Acceptance:** Image, multi-image, video, QA, captioning, and reasoning
regression suites meet predefined retention thresholds while segmentation
metrics improve against the warm-up branches.

**Main risks:** Catastrophic forgetting, data mixture imbalance, LoRA/module
freezing mistakes, and incompatible checkpoint layouts.

## M5: Compute Response-Surface Analysis

**Input:** Retained-capability checkpoints and controlled task/compute
budgets.

**Output:** Reproducible measurements of quality, latency, memory, and FLOPs
across visual, language, mask, and propagation budgets.

**Acceptance:** Results distinguish component and end-to-end costs on fixed
hardware and report real execution measurements rather than visual-token
counts alone.

**Main risks:** Non-comparable kernels, caching effects, batch-size confounds,
and incomplete accounting of preprocessing or propagation.

## M6: Capability-Conditioned Computation

**Input:** M5 response surfaces and explicit capability requests.

**Output:** A router/policy that selects reasoning frames, visual tokens,
anchors, and dense compute according to task, sample, and budget.

**Acceptance:** The policy is opt-in, has deterministic budget controls, falls
back to the baseline path, and improves the measured quality-cost tradeoff
without violating retention thresholds.

**Main risks:** Routing instability, reward/metric mismatch, hard-to-debug
conditional execution, and hidden baseline regressions.

## M7: Full Evaluation and Ablation

**Input:** The complete conditional-compute system and frozen evaluation
protocols.

**Output:** Full benchmark results, ablations, efficiency tables, retention
analysis, and release documentation.

**Acceptance:** Every claimed capability and efficiency result has a reproducible
command/configuration, baseline comparison, component-level accounting, and
ablation. The default installation remains usable for ordinary VILA tasks.

**Main risks:** Dataset/license constraints, benchmark leakage, unreproducible
hardware effects, and overclaiming from one operating point.
