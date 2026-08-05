# Implementation Guide - EvoVILA-Seg Spatiotemporal Segmentation

> Generated: 2026-08-05 | Strategy: extend the VILA baseline through opt-in adapters | Status: S0-S1_COMPLETE; S2_IN_PROGRESS; S3-S4 DEFERRED
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

S0 is the first coding milestone. S1-S4 are specified now so that S0 contracts
do not need to be rewritten later, but they are not authorized as formal
training runs by this document.

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
    └── training.py
tests/
├── test_evo_seg_contracts.py
├── test_evo_seg_decoder.py
├── test_evo_seg_losses.py
├── test_evo_seg_capability.py
├── test_evo_seg_provenance.py
├── test_evo_seg_vila_adapter.py
├── test_evo_seg_sam2_adapter.py
└── test_evo_seg_baseline.py
scripts/
└── evo_seg/
    ├── smoke_decoder.py
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
| `llava/evo_seg/sam2_adapter.py` | Lazy dense encoding, refinement, and mask-prompt propagation | RGB frames and anchor masks | SAM2 features/refined tubes | S2/S3 |
| `llava/evo_seg/training.py` | Freeze policy and optimizer parameter selection | VILA, decoder, optional SAM2 | checked parameter groups | S4 |
| `tests/test_evo_seg_contracts.py` | Reject malformed requests/batches/results and cover T=1/T>1 | synthetic tensors | pass/fail | S0 |
| `tests/test_evo_seg_decoder.py` | Check shapes, invalid frames, gradients, and query sensitivity | synthetic tensors | pass/fail | S0 |
| `tests/test_evo_seg_losses.py` | Check loss numerics and negative controls | synthetic logits/targets | pass/fail | S0 |
| `tests/test_evo_seg_capability.py` | Check explicit opt-in, public entry, timings, and import isolation | fake decoder/batch | pass/fail | S0 |
| `tests/test_evo_seg_provenance.py` | Check text/media expansion, left/right padding, truncation, and query gathering | fake fused rows/hidden states | pass/fail | S1 |
| `tests/test_evo_seg_vila_adapter.py` | Check frozen teacher forcing, multi-token extraction, dense provider bridge, and default isolation | no-weight VILA harness | pass/fail | S1 |
| `tests/test_evo_seg_sam2_adapter.py` | Check raw RGB validation, lazy local build, frozen encoder, padding reconstruction, and mask refinement | fake official predictor API | pass/fail | S2 |
| `tests/test_evo_seg_baseline.py` | Check original text/image/multi-image/video contracts with extension disabled | draft media wrappers | pass/fail | S0 |
| `scripts/evo_seg/smoke_decoder.py` | Reproducible no-weight S0 smoke entry point | CLI dimensions/seed/output path | JSON outside Git | S0 |
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

### 5.5 `llava/evo_seg/sam2_adapter.py` - S2/S3

SAM2 imports occur inside builder functions only.

**`RGBFrameBatch`** accepts only CPU `uint8` RGB pixels with shape
`[B,T,3,H,W]` and a boolean `[B,T]` frame mask. VILA-normalized tensors are
rejected so SAM2 always performs its own resize/normalization.

**`build_sam2_image_predictor(options) -> SAM2ImagePredictor`** lazily
validates a local official source tree, package-relative config, explicit local
checkpoint, and device. It never accepts a remote model ID or downloads an
asset. The returned model is put in eval mode and all parameters are frozen.

**`SAM2ImageFeatureProvider.encode_frames(batch) -> DenseFeatureBatch`** calls
the official `set_image_batch`, reads `image_embed [N,C,H,W]`, reconstructs
`[B,T,C,H,W]`, zeroes invalid frames, clones the result, and immediately clears
predictor image state under a lock.

**`SAM2ImageFeatureProvider.refine_masks(batch, coarse_masks) -> Tensor`**
resizes decoder logits to SAM2's mask-prompt size and calls `predict_batch`
without point/box/human prompts. The returned shape is `[B,N,T,Hrgb,Wrgb]`.

**`propagate_anchor_mask(video, anchor_index, anchor_mask) -> Tensor`** calls
the official SAM2 video predictor's mask-prompt path and returns `[T,H,W]`.

### 5.6 Narrow VILA Integration - S1

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
`return_dict=True`, and `use_cache=False`. It never registers `[SEG]`, changes
VILA parameters, or reduces the selected expression to one token.

**`VILASegmentationAdapter.segment(input_ids, media, media_config,
query_token_mask, request, attention_mask=None, dense_input=None) ->
SegmentationResult`**
requires an enabled request, extracts query states, invokes the named injected
dense provider, builds `GroundingBatch`, calls `SegmentationCapability`, and
records VILA query, dense provider, decoder, and total time separately.
`dense_input` keeps raw SAM2 RGB pixels separate from VILA-preprocessed media.

### 5.7 S0 Tests, Configuration, And Smoke Entry Point

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

## 6 Freeze And Optimizer Policy

S0 trains only `QueryConditionedSpatialDecoder`. S1/S2 initially freeze the
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
| S2 | SAM2 lazy import; raw-image preprocessing metadata; frozen encoder; differentiable decoder output |
| S3 | predicted mask accepted as video anchor; `[T,H,W]` output; separate encoder/decoder/propagation timing |
| S4 | same-image different-target generalization, no-object false positives, image IoU/Dice, video J/F, capability retention |

The first real-data go/no-go check must contain at least two targets from the
same image. A single-sample overfit can verify plumbing but cannot approve the
grounding representation.

## 8 Data Preparation

No data is copied into the repository. Dataset adapters later consume external
manifests containing:

```json
{
  "sample_id": "string",
  "media_path": "/absolute/external/path",
  "media_type": "image_or_video",
  "query": "referring expression",
  "mask_paths": ["one path per frame"],
  "object_present": [true],
  "split": "train_or_val_or_test"
}
```

Train/validation splits must be disjoint by source image or video, not merely by
expression. Same-image target groups stay in the same batch for shortcut tests.

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
8. Implement S2 lazy SAM2 image adapter. (boundary complete; real-asset smoke pending)
9. Implement S3 predicted-anchor video propagation.
10. Design and separately approve S4 real-data training.

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
