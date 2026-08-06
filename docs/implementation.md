# Implementation Guide - EvoVILA-Seg Spatiotemporal Segmentation

> Generated: 2026-08-05 | Strategy: extend the VILA baseline through opt-in adapters | Status: S0-S3_COMPLETE; S4A_MASK_DATA_CONTRACT_DESIGN_FROZEN; S4B_T1_IMAGE_OVERFIT_IMPLEMENTED_AND_RUNNING; S4B_T2_VIDEO_DEFERRED
> Basis: user-confirmed architecture and repository audit. A formal `docs/idea_report.md` Part 3 does not yet exist, so benchmark-scale training remains outside this implementation milestone.

## 1 Original Project And Scope

EvoVILA-Seg extends the existing VILA repository rather than creating a
segmentation-only fork. The implementation is divided into five gates:

| Gate | Scope | VILA core changes | External assets |
|---|---|---:|---:|
| S0 | Contracts, query-conditioned decoder, losses, fake-feature tests | None | None |
| S1 | Request-scoped VILA feature extraction and fused provenance | Narrow local hook | Local VILA checkpoint only for smoke |
| S2 | Lazy SAM2 image feature/refinement adapter | None on ordinary path | Existing local SAM2 source/checkpoint |
| S3 | Predicted-anchor video propagation | Public opt-in entry point | Existing local video assets |
| S4 | Image/video training and retention evaluation | Training adapters only initially | External datasets, separately approved |

S0-S1 and the S2 image plumbing are complete, including a local VILA1.5-3B plus
SAM2 real-checkpoint smoke. The decoder remains randomly initialized, so no
mask-quality claim is made. S3 predicted-anchor forward and bidirectional video
propagation have passed real local-checkpoint smokes. Benchmark-scale training
is not authorized by this document.

## 2 Proposed Repository Structure

```text
docs/
├── user_requirements.md
├── implementation.md
└── dev_log.md
llava/
├── model/
│   └── fusion_observer.py
└── evo_seg/
    ├── __init__.py
    ├── contracts.py
    ├── decoder.py
    ├── losses.py
    ├── capability.py
    ├── provenance.py
    ├── vila_adapter.py
    ├── sam2_adapter.py
    ├── sam2_video_adapter.py
    ├── image_pipeline.py
    ├── video_pipeline.py
    ├── training_contracts.py
    └── training.py
tests/
├── test_evo_seg_contracts.py
├── test_evo_seg_decoder.py
├── test_evo_seg_losses.py
├── test_evo_seg_capability.py
├── test_evo_seg_provenance.py
├── test_evo_seg_vila_adapter.py
├── test_evo_seg_sam2_adapter.py
├── test_evo_seg_image_pipeline.py
├── test_evo_seg_video_pipeline.py
├── test_evo_seg_training_contracts.py
└── test_evo_seg_baseline.py
scripts/
└── evo_seg/
    ├── smoke_decoder.py
    ├── smoke_sam2_image.py
    ├── smoke_image_segmentation.py
    └── smoke_video_segmentation.py
configs/
└── evo_seg/
    ├── s0_decoder.yaml
    ├── s2_image.yaml
    └── s3_video.yaml
```

> `sam2_adapter.py`, the real-asset smoke scripts, and their configs are added
> only at their corresponding gates. S0 creates no SAM2 dependency.

## 3 File Responsibilities

| File | Responsibility | Input | Output | First gate |
|---|---|---|---|---|
| `llava/evo_seg/__init__.py` | Export the stable, SAM2-free public S0 API | package import | contract/decoder/capability symbols | S0 |
| `llava/evo_seg/contracts.py` | Validate the unified T=1/T>1 data contract | query, dense features, frame masks, targets | immutable typed batches/results | S0 |
| `llava/evo_seg/decoder.py` | Cross-attend language/object queries to dense spatial features | `[B,L,Dq]`, `[B,T,C,H,W]` | mask logits and object logits | S0 |
| `llava/evo_seg/losses.py` | Mask, objectness, query-swap, and consistency losses | result plus targets/controls | named scalar losses | S0 |
| `llava/evo_seg/capability.py` | Request-scoped lifecycle and lazy component ownership | explicit request | segmentation result | S0/S1 |
| `llava/model/fusion_observer.py` | Hold one request-local, lightweight fusion observer without importing the extension | context manager | current observer or `None` | S1 |
| `llava/evo_seg/provenance.py` | Validate fused token provenance and gather explicit multi-token query states | fused rows, hidden states, query mask | query-state batch | S1 |
| `llava/evo_seg/vila_adapter.py` | Run frozen VILA teacher forcing and connect an injected dense provider to S0 capability | tokenized VILA inputs and query spans | segmentation result | S1 |
| `llava/model/llava_arch.py` | Add one inactive observer notification inside both repository-native `_embed` paths | current fusion scope | unchanged return tuples | S1 |
| `llava/evo_seg/sam2_adapter.py` | Lazy image dense encoding and mask refinement | RGB frames and coarse masks | SAM2 image features/refined masks | S2 |
| `llava/evo_seg/sam2_video_adapter.py` | Lazy shared-weight video builder and predicted-anchor propagation | RGB videos and anchor logits | padded propagated mask tubes | S3 |
| `llava/evo_seg/image_pipeline.py` | Compose the opt-in VILA adapter, spatial decoder result, and SAM2 image refinement | tokenized VILA media/query plus raw RGB | coarse and refined image masks | S2 |
| `llava/evo_seg/video_pipeline.py` | Select a fixed predicted anchor and compose frozen VILA, spatial decoder, and SAM2 bidirectional propagation | tokenized VILA video/query plus raw RGB | coarse anchor and propagated video masks | S3 |
| `llava/evo_seg/training_contracts.py` | Freeze external mask records, same-media query swaps, negative controls, split isolation, and ordinary-VILA retention probes without reading data | immutable metadata | validated manifest/probe contracts | S4a |
| `llava/evo_seg/training.py` | Freeze policy and optimizer parameter selection | VILA, decoder, optional SAM2 | checked parameter groups | S4 |
| `tests/test_evo_seg_contracts.py` | Reject malformed requests/batches/results and cover T=1/T>1 | synthetic tensors | pass/fail | S0 |
| `tests/test_evo_seg_decoder.py` | Check shapes, invalid frames, gradients, and query sensitivity | synthetic tensors | pass/fail | S0 |
| `tests/test_evo_seg_losses.py` | Check loss numerics and negative controls | synthetic logits/targets | pass/fail | S0 |
| `tests/test_evo_seg_capability.py` | Check explicit opt-in, public entry, timings, and import isolation | fake decoder/batch | pass/fail | S0 |
| `tests/test_evo_seg_provenance.py` | Check text/media expansion, left/right padding, truncation, and query gathering | fake fused rows/hidden states | pass/fail | S1 |
| `tests/test_evo_seg_vila_adapter.py` | Check frozen teacher forcing, multi-token extraction, dense provider bridge, and default isolation | no-weight VILA harness | pass/fail | S1 |
| `tests/test_evo_seg_sam2_adapter.py` | Check raw RGB validation, lazy local build, frozen encoder, padding reconstruction, and mask refinement | fake official predictor API | pass/fail | S2 |
| `tests/test_evo_seg_image_pipeline.py` | Check composed image output, explicit opt-in, T=1 enforcement, state cleanup, and import isolation | fake VILA/SAM2 harness | pass/fail | S2 |
| `tests/test_evo_seg_video_pipeline.py` | Check predicted-mask prompting, fixed anchors, bidirectional propagation, padding reconstruction, cleanup, and import isolation | fake VILA/video-predictor harness | pass/fail | S3 |
| `tests/test_evo_seg_training_contracts.py` | Check image/video alignment, local external paths, same-media different-target swaps, negative coverage, split leakage, retention probes, and import isolation | metadata only | pass/fail | S4a |
| `tests/test_evo_seg_baseline.py` | Check original text/image/multi-image/video contracts with extension disabled | draft media wrappers | pass/fail | S0 |
| `scripts/evo_seg/smoke_decoder.py` | Reproducible no-weight S0 smoke entry point | CLI dimensions/seed/output path | JSON outside Git | S0 |
| `scripts/evo_seg/smoke_sam2_image.py` | Exercise the real frozen SAM2 encoder/refinement provider without VILA weights | config plus local SAM2 source/checkpoint paths | JSON outside Git or stdout | S2 |
| `scripts/evo_seg/smoke_image_segmentation.py` | Exercise the real opt-in image path | config and local asset paths | JSON outside Git | S2 |
| `scripts/evo_seg/smoke_video_segmentation.py` | Exercise predicted-anchor propagation | config and local asset paths | JSON outside Git | S3 |
| `configs/evo_seg/s0_decoder.yaml` | Record default synthetic smoke dimensions and seed | configuration | decoder smoke settings | S0 |
| `configs/evo_seg/s2_image.yaml` | Record image adapter options without embedding asset paths | configuration | image smoke settings | S2 |
| `configs/evo_seg/s3_video.yaml` | Record video propagation options without embedding asset paths | configuration | video smoke settings | S3 |

## 4 Unified Data Contract

### 4.1 `SegmentationRequest`

Defined in `llava/evo_seg/contracts.py` as a frozen dataclass.

- `enabled: bool = False`: must be true for dense execution.
- `task: Literal["image", "video"]`: explicit measurement category.
- `options: Mapping[str, Any]`: immutable capability options.
- `request_id: Optional[str]`: diagnostics only; never used for routing logic.

`SegmentationRequest.disabled()` returns the default no-op request.
`SegmentationRequest.from_value(value)` accepts `None`, a request object, or a
mapping and rejects unknown task names and an enabled request without a task.

### 4.2 `GroundingBatch`

Defined as a validated dataclass with:

- `query_states: Tensor[B,L,Dq]`
- `query_mask: BoolTensor[B,L]`
- `dense_features: Tensor[B,T,C,H,W]`
- `frame_mask: BoolTensor[B,T]`
- optional `target_masks: Tensor[B,N,T,Ht,Wt]`
- optional `target_presence: BoolTensor[B,N,T]`
- optional `sample_ids: Tuple[str, ...]`

The constructor checks finite floating tensors, non-empty queries, at least one
valid frame per sample, positive spatial dimensions, and exact batch/time
alignment. Image inputs use `T=1`; no image-only tensor rank is introduced.

### 4.3 `SegmentationResult`

- `mask_logits: Tensor[B,N,T,H,W]`
- `object_logits: Tensor[B,N]`
- `object_embeddings: Tensor[B,N,D]`
- `frame_embeddings: Tensor[B,N,T,D]`
- `frame_mask: BoolTensor[B,T]`
- `diagnostics: Mapping[str, Any]`

`frame_embeddings` are mask-weighted dense features for each object and frame;
`object_embeddings` are their valid-frame pooled representation. Invalid frames
must have zeroed logits and embeddings and must not contribute to losses.

### 4.4 S4a Mask Training And Retention Metadata

`MaskTrainingRecord` is one referring-expression target or negative control. It
contains stable sample/media IDs, `image` or `video` type, an absolute external
media path, strictly increasing sampled source-frame indices, aligned optional
absolute mask paths, per-frame presence, one fixed anchor position, split,
query, optional target ID, and `positive`, `no_object`, or `empty_query` kind.
Image records have exactly one frame at source index zero. Positive records have
at least one present mask and a present anchor. Negative controls contain no
mask paths or presence; `no_object` keeps a non-empty query while `empty_query`
requires an empty query and is evaluation-only because `GroundingBatch`
correctly rejects an empty query token mask.

`QuerySwapPair` names two distinct positive samples. A validated pair must use
the same media, split, sampled frames, and anchor, but different query text and
target IDs.
It represents a symmetric same-media query swap; a loader must never construct
this control by pairing unrelated media.

`MaskTrainingManifest` freezes records and pairs, rejects duplicate sample IDs
or source-media leakage across splits (checked by both stable media ID and exact
media path), and requires every represented training or validation split to
contain a valid query-swap pair, a same-media `no_object` control, and a
same-media `empty_query` control. It performs metadata validation only and never
opens media or mask files.

`RetentionProbe` and `RetentionProbeSet` freeze ordinary VILA text,
single-image, multi-image, and video inputs together with the exact frozen
baseline model ID/revision and comparison mode. All four tasks are mandatory;
segmentation capability fields are intentionally absent. Baseline outputs are
captured outside Git before any training and compared using the same probes
afterward.

## 5 Function-Level Implementation

### 5.0 `llava/evo_seg/__init__.py`

Exports `SegmentationRequest`, `GroundingBatch`, `SegmentationResult`,
`QueryConditionedSpatialDecoder`, and `SegmentationCapability`. It must not
import future SAM2 adapters, VILA builders, or optional external packages.

### 5.1 `llava/evo_seg/contracts.py`

**`freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]`**

Recursively freezes mapping/list/set structure while retaining tensor objects.

**`SegmentationRequest.from_value(value: Any) -> SegmentationRequest`**

Normalizes request values and fails closed on malformed opt-in configuration.

**`GroundingBatch.validate() -> None`**

Checks ranks, dtypes, finite values, masks, and B/T/N alignment. Validation is
called from `__post_init__` so no decoder receives an ambiguous batch.

**`SegmentationResult.validate() -> None`**

Checks output ranks, batch/time consistency, finite logits, and invalid-frame
zeroing for both logits and frame embeddings.

### 5.2 `llava/evo_seg/decoder.py`

**`QueryConditionedSpatialDecoder(nn.Module)`**

Constructor parameters:

- `query_dim: int`, `feature_dim: int`, `model_dim: int`
- `num_heads: int`, `num_layers: int`
- `num_object_queries: int = 1`
- `dropout: float = 0.0`

Modules:

1. Layer-normalize and project language tokens into `model_dim`.
2. Project dense features with a `1x1` convolution without discarding H/W.
3. Add learned object queries to a pooled language context.
4. For each decoder layer, apply object-to-language attention followed by
   object-to-spatial attention and a feed-forward residual block.
5. Produce one mask embedding per object query and compute dense mask logits by
   normalized dot product with projected per-pixel features.
6. Use mask probabilities to pool a frame embedding for every object/frame,
   then pool valid frames into the request-level object embedding.
7. Produce request-level objectness logits from the final object embeddings.

**`forward(batch: GroundingBatch) -> SegmentationResult`**

The forward path flattens only attention axes, restores `[B,N,T,H,W]`, masks
invalid frames, derives `[B,N,T,D]` frame embeddings, and returns diagnostics
containing attention source lengths and output shapes. It must never import
SAM2 or call VILA.

> The decoder uses the full query token sequence. `[SEG]`, when later present,
> is a capability marker and may initialize an object query, but cannot replace
> the language sequence.

### 5.3 `llava/evo_seg/losses.py`

**`binary_mask_loss(logits, targets, valid) -> Tensor`**

Computes masked BCEWithLogits over valid object/frame entries.

**`dice_loss(logits, targets, valid, eps=1e-6) -> Tensor`**

Computes sample/object/frame Dice loss and averages only valid entries.

**`objectness_loss(logits, presence, frame_mask) -> Tensor`**

Reduces `[B,N,T]` target presence over valid frames with `any`, then supervises
whether a referred object exists anywhere in the request.

**`query_swap_margin_loss(correct_logits, swapped_logits, targets, valid, margin) -> Tensor`**

Requires correct-query masks to score their target above same-image swapped
queries. The score is the target-region mean logit minus the non-target-region
mean logit, averaged only over valid object/frames. This control directly
targets the shortcut observed on the old branch.

**`temporal_consistency_loss(frame_embeddings, presence, frame_mask) -> Tensor`**

Reserved for S3/S4. Returns zero for `T=1`; for video it penalizes cosine drift
between adjacent `[B,N,T,D]` embeddings only when both frames are valid and the
target is present in both.

**`compute_segmentation_loss(result, batch, weights, controls=None) -> Mapping[str, Tensor]`**

Returns `bce`, `dice`, `objectness`, `query_swap`, `temporal`, and `total`.
`controls` must provide swapped-query mask logits when query-swap weight is
nonzero. Missing required targets or controls raise an error rather than
silently removing a loss.

### 5.4 `llava/evo_seg/capability.py`

**`SegmentationCapability(nn.Module)`**

Owns the decoder and optional providers but does not own or replace VILA's
ordinary forward path.

**`is_enabled(request: Any) -> bool`**

Parses a request without constructing dense modules. `None` is always false.

**`forward(batch: GroundingBatch, request: SegmentationRequest) -> SegmentationResult`**

Rejects disabled requests, invokes the decoder, and records component timing.

**`clear_request_state() -> None`**

Clears only diagnostics. Training tensors remain local to `forward`; no global
mutable media context is used, keeping concurrent requests independent.

### 5.5 `llava/evo_seg/sam2_adapter.py` - S2

SAM2 imports occur inside builder functions only.

**`RGBFrameBatch`** accepts only CPU `uint8` RGB pixels with shape
`[B,T,3,H,W]` and a boolean `[B,T]` frame mask. VILA-normalized tensors are
rejected so SAM2 always performs its own resize/normalization.

**`build_sam2_image_predictor(options) -> SAM2ImagePredictor`** lazily
validates a local official source tree, package-relative config, explicit local
checkpoint, and device. It never accepts a remote model ID or downloads an
asset. The returned model is put in eval mode and all parameters are frozen.

**`SAM2ImageFeatureProvider.initialize() -> None`** explicitly builds the
frozen predictor without setting request images. Ordinary requests never call
this method; the real-provider smoke uses it to separate model initialization
from image encoder timing.

**`SAM2ImageFeatureProvider.encode_frames(batch) -> DenseFeatureBatch`** calls
the official `set_image_batch`, reads `image_embed [N,C,H,W]`, reconstructs
`[B,T,C,H,W]`, zeroes invalid frames, clones the result, and immediately clears
predictor image state under a lock.

**`SAM2ImageFeatureProvider.refine_masks(batch, coarse_masks) -> Tensor`**
resizes decoder logits to SAM2's mask-prompt size and calls `predict_batch`
without point/box/human prompts. The returned shape is `[B,N,T,Hrgb,Wrgb]`.
The current image predictor API re-encodes the RGB image during refinement, so
the smoke reports this time as `sam2_mask_refinement_with_reencode` rather than
claiming decoder-only latency.

### 5.6 `llava/evo_seg/sam2_video_adapter.py` - S3

**`build_sam2_video_image_predictor(options)`** builds the official frozen
video predictor and wraps that same model with `SAM2ImagePredictor`. S3 uses
this factory so anchor dense encoding and video propagation share one SAM2
weight instance. Neither the builder nor the external `sam2` package is reached
until the video capability is explicitly initialized or executed.

**`SAM2VideoMaskPropagator(provider)`** requires the shared video-capable image
provider. **`propagate(batch, anchor_indices, anchor_mask_logits)`** validates a
CPU RGB batch, one valid fixed anchor per sample, and floating logits
`[B,N,Hmask,Wmask]`. It thresholds predicted logits at zero, never accepts a GT
mask or point/box prompt, then calls the official `init_state`, `add_new_mask`,
and forward/reverse `propagate_in_video` APIs. Official SAM2 accepts MP4 bytes or
JPEG directories rather than in-memory frame tensors, so each valid sample is
materialized in an automatically removed request-local JPEG directory. Batch
and padded videos are processed as independent predictor states and rebuilt as
`[B,N,T,Hrgb,Wrgb]` with invalid frames equal to zero. Predictor states are
reset and cleared in `finally` blocks. Diagnostics report temporary frame I/O,
state initialization, anchor prompting, forward propagation, reverse
propagation, and total propagation time separately.

### 5.7 `llava/evo_seg/image_pipeline.py` - S2

This module is imported only by an explicit image-segmentation caller. It does
not import the external `sam2` package at module load time.

**`ImageSegmentationOutput`** contains the validated coarse
`SegmentationResult`, refined mask logits `[B,N,1,Hrgb,Wrgb]`, and immutable
pipeline diagnostics. It rejects non-finite tensors, shape mismatches, or
refined values on invalid frames.

**`VILAImageSegmentationPipeline(adapter, refiner)`** requires the adapter's
dense provider and the SAM2 refiner to be the same request-local provider. Its
**`segment(...) -> ImageSegmentationOutput`** rejects disabled, video, or
multi-frame requests before executing VILA or SAM2; calls the existing VILA
adapter with raw RGB supplied only through `dense_input`; refines the predicted
coarse mask without point/box/human prompts; and reports total plus refinement
time. **`clear_request_state()`** clears capability diagnostics and any
predictor-side image state without unloading weights.

### 5.8 `llava/evo_seg/video_pipeline.py` - S3

**`VideoSegmentationOutput`** contains a T=1 coarse anchor result, the validated
SAM2 propagation result, and immutable diagnostics. The object count and batch
axes must agree, while the propagation frame mask must match the original raw
video contract.

**`VILAVideoSegmentationPipeline.segment(...)`** rejects disabled, image, T=1,
batch-mismatched, or invalid-anchor requests before executing VILA or SAM2.
VILA teacher forcing still observes the caller's complete video. Dense encoding
selects only the fixed anchor RGB frame, so the spatial decoder produces a
single predicted anchor mask rather than a redundant mask for every frame.
The default anchor is the first valid frame; callers may provide an explicit
`LongTensor[B]` of valid frame indices. Dynamic anchor selection is deferred.
The predicted anchor is then propagated in both directions when needed. This
hard prompt boundary is inference-only; future decoder training supervises the
coarse anchor directly and does not backpropagate through SAM2 propagation.

### 5.9 Narrow VILA Integration - S1

S1 adds request-local feature extraction around the repository-native VILA
model. Ordinary `forward`, `generate`, and `generate_content` signatures and
return values remain intact. The standalone `llava/remote_code` export is not
modified in S1; it must receive an equivalent isolated hook before extension
release through that distribution path.

**`fusion_observation(observer: Any) -> ContextManager[None]`** in
`llava/model/fusion_observer.py` stores the observer in a `ContextVar` and
restores the previous value on exit. **`current_fusion_observer() -> Any`**
returns `None` on every ordinary VILA request. This lightweight module imports
neither `llava.evo_seg` nor any dense dependency.

Both `_embed` implementations in `llava/model/llava_arch.py` read the current
observer once. When it is `None`, they execute the original indexing,
truncation, padding, return values, and side effects exactly. When present,
they additionally build per-sample rows containing `source_type` (`text`,
`image`, or `video`) and the original padded `input_ids` position for each
fused token (`-1` for expanded media). After the existing truncate/batchify
steps they call `observer.observe_fusion(...)`; the observer cannot replace or
mutate tensors returned by `_embed`.

**`FusedSequenceProvenance`** in `llava/evo_seg/provenance.py` is a frozen
validated dataclass containing `source_type`, `source_position [B,F]`,
`fused_position [B,F]`, `attention_mask [B,F]`, and `valid_length [B]`.

**`QueryStateBatch`** contains `states [B,L,D]`, `mask [B,L]`,
`source_positions [B,L]`, `fused_positions [B,L]`, and the provenance used to
produce it. Padding rows use position `-1` and zero states.

**`FusionCapture.observe_fusion(rows, padding_side, fused_attention_mask)`**
constructs exactly one provenance object and rejects reuse. Its
**`gather_query_states(hidden_states, query_token_mask) -> QueryStateBatch`**
maps every selected original text token to exactly one fused token and fails if
a selected position is padding, media, truncated, missing, or duplicated.

**`DenseFeatureBatch`** in `llava/evo_seg/vila_adapter.py` validates
`features [B,T,C,H,W]`, `frame_mask [B,T]`, and immutable diagnostics. A dense
provider is injected through a callable or an `encode(...)` method; S1 tests
use a fake provider and S2 later implements the real lazy SAM2 provider.

**`VILASegmentationAdapter.extract_query_states(input_ids, media,
media_config, query_token_mask, attention_mask=None, hidden_layer=-1) ->
QueryStateBatch`** enters a fusion observation scope and calls frozen VILA
teacher forcing with `packing=False`, `output_hidden_states=True`,
`return_dict=True`, and `use_cache=False`. Constructing this explicit adapter
puts a `torch.nn.Module` VILA model in eval mode and sets all of its parameters
to `requires_grad=False`; it does not change weight values, register `[SEG]`,
or reduce the selected expression to one token.

**`VILASegmentationAdapter.segment(input_ids, media, media_config,
query_token_mask, request, attention_mask=None, dense_input=None) ->
SegmentationResult`**
requires an enabled request, extracts query states, invokes the named injected
dense provider, builds `GroundingBatch`, calls `SegmentationCapability`, and
records VILA query, dense provider, decoder, and total time separately.
`dense_input` keeps raw SAM2 RGB pixels separate from VILA-preprocessed media.

### 5.10 Tests, Configuration, And Smoke Entry Points

**`tests/test_evo_seg_contracts.py`** constructs image (`T=1`) and padded video
batches, checks recursive request-option immutability, and asserts each invalid
rank/dtype/alignment/non-finite case fails with a specific error.

**`tests/test_evo_seg_decoder.py`** checks exact result shapes, zeroed invalid
frames, finite backward gradients confined to decoder parameters, deterministic
outputs at dropout zero, and changed masks after replacing the non-padding
query sequence with a different sequence while holding dense features fixed.

**`tests/test_evo_seg_losses.py`** verifies near-perfect predictions have lower
BCE/Dice than inverted predictions, invalid frames have no effect, objectness
uses valid-frame presence, query-swap loss is positive when the swapped query
scores better, and temporal loss is zero for `T=1`.

**`tests/test_evo_seg_capability.py`** verifies `None` and disabled requests do
not call the decoder, enabled requests do, request state does not leak across
calls, and importing `llava.evo_seg` does not add a `sam2` module to
`sys.modules`.

**`tests/test_evo_seg_baseline.py`** calls VILA's unchanged
`extract_media(..., draft=True)` prompt path for text-only, single-image,
multi-image, and video inputs. It verifies the original media ordering/token
contracts while the segmentation request is disabled, without opening media
files or loading weights.

**`configs/evo_seg/s0_decoder.yaml`** stores only synthetic tensor dimensions,
decoder hyperparameters, seed, and CPU device defaults. It contains no model,
dataset, checkpoint, or artifact paths.

**`scripts/evo_seg/smoke_decoder.py`** exposes
`main(argv: Optional[Sequence[str]] = None) -> int`. It loads the S0 config,
constructs deterministic synthetic image/video batches, performs forward and
backward passes through the public capability entry point, and writes the
Section 9 `smoke.json` schema only when an explicit output path outside the
repository is supplied. Without an output path it prints the JSON to stdout.

**`scripts/evo_seg/smoke_sam2_image.py`** requires explicit local source and
checkpoint CLI paths, constructs one deterministic synthetic RGB image, and
executes the real frozen SAM2 image encoder plus coarse-mask refinement. It
checks lazy import, finite features/masks, predictor state cleanup, and frozen
parameters, while reporting initialization, encoder, refinement-with-reencode,
and peak CUDA memory separately. It does not load VILA weights and therefore
does not by itself approve the full S2 image segmentation path.

**`scripts/evo_seg/smoke_image_segmentation.py`** requires explicit local VILA
model, SAM2 source, and SAM2 checkpoint paths. It constructs one deterministic
RGB image and a unique multi-token referring expression, preprocesses the same
image independently for VILA and SAM2, and runs the composed public image
pipeline. Before and after the extension call it executes the ordinary frozen
VILA path on the same inputs and requires identical final-token logits. The
JSON report distinguishes model initialization, VILA query encoding, SAM2
encoding, decoder, refinement-with-reencode, total time, and peak CUDA memory;
randomly initialized decoder masks are plumbing results, not quality claims.

### 5.11 `llava/evo_seg/training_contracts.py` - S4a

**`MaskTrainingRecord(...)`** validates one metadata-only sample without file
I/O. **`optimization_eligible`** is true for positive and no-object samples and
false for empty-query controls.

**`QuerySwapPair(left_sample_id, right_sample_id)`** stores a canonical pair of
sample IDs. **`MaskTrainingManifest(records, query_swap_pairs)`** validates
record uniqueness, per-media signature consistency, source-media split
isolation, pair semantics, and train/validation control coverage. Its immutable
`summary` reports counts by split, media type, and control kind.

**`RetentionProbe(...)`** validates ordinary VILA task cardinality and absolute
external media paths. **`RetentionProbeSet(baseline_model_id,
baseline_revision, probes)`** requires exactly one or more probes for each of
text, single-image, multi-image, and video and exposes an immutable summary.
Neither class imports VILA, SAM2, a dataset library, or torch.

**`scripts/evo_seg/smoke_video_segmentation.py`** requires explicit local VILA
model, SAM2 source, and SAM2 checkpoint paths. It constructs a deterministic
three-frame moving-object video unless an external MP4/JPEG directory is given,
feeds the same selected frames through VILA and raw RGB through the explicit
video pipeline, and uses no human or ground-truth mask prompt. The JSON report
records commit plus dirty-worktree state, config hash, source/fused query
positions, fixed anchor policy, exact ordinary-VILA retention, frozen/state
cleanup checks, tensor shapes, peak CUDA memory, and all required component
timings. `--anchor-index` can exercise an explicit middle anchor and therefore
both official forward and reverse propagation.

## 6 Freeze And Optimizer Policy

S0 trains only `QueryConditionedSpatialDecoder`. S1-S3 initially freeze the
VILA LLM, VILA vision tower, multimodal projector, and all SAM2 parameters.
Optimizer creation enumerates an allowlist and asserts that every trainable
parameter belongs to the decoder or explicitly approved capability adapter.

Capability-specific LoRA is not part of S0-S3. If later approved, its delta is
evaluated only inside `segment(...)`; default VILA calls must use the exact base
linear output and immutable base weights.

## 7 Validation Gates

| Gate | Required checks |
|---|---|
| S0 | Original text/image/multi-image/video prompt contracts; T=1/T>1 shapes; invalid-frame masking; gradients only in decoder; query-swap loss; no SAM2 import |
| S1 | correct query spans after media expansion/padding; text/single-image/multi-image/video baseline smoke unchanged |
| S2 | SAM2 lazy import; raw-image preprocessing metadata; frozen encoder; predictor state cleanup; real local-source encoder/refinement smoke; differentiable decoder output |
| S3 | fixed predicted mask accepted as the only video anchor; `[B,N,T,H,W]` output with padded frames zero; forward/reverse propagation and state cleanup; separate VILA vision/LLM, anchor encoder, decoder, frame I/O, state initialization, anchor prompt, and propagation timing |
| S4a | metadata-only image/video mask records; fixed sampled-frame/anchor alignment; same-media different-target query swaps; no-object and empty-query coverage in train/val; source-media split isolation; four-task ordinary-VILA retention probes |
| S4b | separately approved training recipe; same-image different-target generalization, no-object false positives, image IoU/Dice, video J/F, capability retention |

The first real-data go/no-go check must contain at least two targets from the
same image. A single-sample overfit can verify plumbing but cannot approve the
grounding representation.

## 8 Data Preparation

No data is copied into the repository. S4a validates metadata only; dataset
adapters later consume external manifests containing records such as:

```json
{
  "sample_id": "string",
  "media_id": "stable-source-id",
  "media_path": "/absolute/external/path",
  "media_type": "image_or_video",
  "frame_indices": [0],
  "query": "referring expression",
  "target_id": "object-id-or-null",
  "control_kind": "positive_or_no_object_or_empty_query",
  "mask_paths": ["/absolute/external/mask-or-null"],
  "target_presence": [true],
  "anchor_position": 0,
  "split": "train_or_val_or_test"
}
```

Frame indices, mask paths, and presence have identical length and order. Split
isolation is checked by stable `media_id` and exact `media_path`, not expression
or target. Same-media positive targets and negative controls remain in the same
split; query-swap pairs are declared explicitly rather than inferred by a
collator.

## 9 Result Formats

All generated results live outside Git under a user-selected artifact root.

`smoke.json` contains: Git commit, config hash, seed, execution path, tensor
shapes, finite/gradient checks, imported optional modules, and component timing
in milliseconds.

`metrics.jsonl` contains one JSON object per step with: step, total/BCE/Dice/
objectness/query-swap losses, IoU, foreground ratio, query-swap delta, memory
bytes, and elapsed milliseconds.

`retention.json` contains baseline and extension-disabled outputs for text,
single-image, multi-image, and video contracts, plus exact token equality or
task metric deltas.

## 10 Implementation Order

1. Add S0 contracts and tests.
2. Add S0 decoder and tensor/gradient tests.
3. Add losses and negative-control tests.
4. Add request-scoped capability public entry point and disabled-path test.
5. Run the complete no-weight suite, `py_compile`, and `git diff --check`.
6. Review S0 before adding any VILA-core hook.
7. Implement S1 provenance and local VILA wrapper. (complete)
8. Implement S2 lazy SAM2 image adapter and complete real VILA+SAM2 image plumbing smoke. (complete; decoder untrained)
9. Implement S3 predicted-anchor video propagation. (complete; decoder untrained)
10. Freeze S4a mask metadata, negative-control, query-swap, split-isolation, and retention contracts.
11. Design and separately approve S4b real-data training only after S4a passes.

## 11 Design Validation

- Experimental coverage: provisionally covers component and retention gates;
  formal benchmark coverage is pending `docs/idea_report.md` Part 3.
- Logic consistency: the common contract remains `[B,N,T,H,W]` from image
  (`T=1`) through video (`T>1`); language stays multi-token and dense features
  retain spatial axes.
- Completeness: every file in the proposed tree has a responsibility and its
  gate is identified; future-gate files are explicitly deferred.
- Baseline isolation: S0 makes no VILA-core changes, and later integration uses
  one inactive-by-default provenance observer plus a separate public entry
  point.
