# M3 Implementation Design

## Capability registration

`llava/capabilities/registry.py` registers `video_segmentation` through a
factory that imports `llava.capabilities.video_segmentation` only when the
name is explicitly selected. The implementation module must not import SAM2
at module scope. Its predictor builder imports `sam2.build_sam` only when
`segment()` is called, so regular VILA loading remains independent of SAM2.

## Video capability

`llava/capabilities/video_segmentation.py` owns the M3 request lifecycle:

1. Validate a frame-directory or MP4 path and the anchor/prompt contract.
2. Lazily build `build_sam2_video_predictor` using a local config and
   checkpoint supplied in capability options.
3. Call `init_state`, add each anchor prompt with
   `add_new_points_or_box`, and propagate with `propagate_in_video`.
4. Reassemble predictor outputs into `[T,H,W]` or `[T,N,H,W]` masks.
5. Return a `VideoSegmentationResult` when requested and retain the last
   instrumentation in `get_outputs()`.

The default option values target the existing SAM2.1 tiny assets:

```text
checkpoint: /9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_tiny.pt
config: configs/sam2.1/sam2.1_hiera_t.yaml
```

The config can be absolute or relative to the SAM2 source root. A missing
SAM2 import, config, checkpoint, video path, or malformed prompt raises
`CapabilityError` with an actionable message. CUDA synchronization is used
around timed GPU sections when available.

## Model entry points

Add `segment_videos()` to both `LlavaMetaModel` and
`VILAPretrainedModel`. The entry point requires the explicit capability,
delegates to it, and does not use VILA's LLM or media-token fusion path. A
`return_result` flag exposes masks plus metadata and instrumentation while the
default return value is the mask tensor.

## Tests and smoke

`tests/test_capabilities.py` covers registration, lazy SAM2 import behavior,
missing-dependency errors, prompt/output normalization, and the model mixin
public entry point with a fake predictor. `scripts/evo/smoke_video_segmentation.py`
is the reproducible external-asset smoke command. It runs one DAVIS sequence
with SAM2.1 tiny and writes no repository artifacts.

## Documentation synchronization

Update `docs/ROADMAP.md` to mark M3 implementation status and append the
environment/asset and verification details to `docs/dev_log.md`. Historical
M0-M2 entries remain unchanged.
