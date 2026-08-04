# EvoVILA Project Scope

## Positioning

EvoVILA is a capability-preserving extension of the complete VILA codebase.
It keeps VILA as the general multimodal foundation and adds optional dense
spatiotemporal grounding behind explicit interfaces. The project is not a
replacement of VILA with a segmentation-only model.

The long-term research direction is capability-conditioned computation:
allocate visual perception, language reasoning, and dense reconstruction
compute according to the task, sample, and available budget while preserving
the original VILA behavior whenever dense perception is unnecessary.

## Capabilities

### Preserved VILA capabilities

- Image understanding
- Multi-image understanding
- Video understanding
- Image and video question answering
- Captioning
- Multimodal reasoning and text generation

### Optional EvoVILA capabilities

- Referring image segmentation
- Reasoning image segmentation
- Referring video object segmentation
- Reasoning video segmentation

### Long-term efficiency capabilities

- On-demand activation of dense segmentation
- Dynamic selection of reasoning frames
- Dynamic allocation of visual tokens and language-model compute
- Dynamic selection of segmentation anchor frames
- Dynamic control of SAM2 propagation, memory refresh, and mask refinement

These efficiency capabilities are research targets. M0 and M1 provide the
baseline protection and capability interface; M2 adds only a static image mask
warm-up and does not implement temporal propagation or conditional compute.

## Execution Model

At the conceptual capability level, an image is a video with `T = 1`. This
lets image and video dense-grounding interfaces share temporal contracts while
preserving the current VILA image and video input paths where their existing
behavior differs.

The ordinary text path uses the LLM directly. The ordinary image path loads
and preprocesses images, extracts visual tokens with the configured vision
tower, maps them through the multimodal projector, and inserts the resulting
embeddings at media-token positions before the LLM forward or generation call.
The ordinary video path samples frames and either represents them as image
media or uses the explicit video encoder/data-collator path, depending on the
entry point and dataset configuration.

The dense path is opt-in. M2 uses the following limited form:

```text
request capability
    -> select ordinary VILA or dense extension path
    -> reuse VILA visual/text context
    -> optional static image mask branch
    -> return masks or text plus an auxiliary mask loss
```

The M2 decoder consumes projected visual tokens arranged as a square grid. It
is a vision-only warm-up: it does not yet use referring expressions, reasoning
hidden states, special segmentation tokens, or temporal propagation.

The default branch must remain valid when no segmentation package is installed.

## Current Non-Goals

M0 through M2 do not solve or implement:

- SAM2 integration or propagation
- `[SEG]` or other segmentation special tokens
- Text-conditioned referring or reasoning segmentation
- Segmentation datasets, training recipes, or benchmark claims
- Dynamic or non-square visual-token mask decoding
- A capability router or conditional-compute policy
- Dynamic reasoning-frame or anchor-frame selection
- Memory refresh or mask-refinement policies
- Mixed capability training
- End-to-end efficiency response-surface analysis
- Final segmentation datasets, metrics, ablations, or benchmark claims
