# EvoVILA Development Rules

## Scope

EvoVILA is an extension of the imported VILA repository. The first obligation
of every change is to preserve VILA's existing image understanding,
multi-image understanding, video understanding, image/video question
answering, captioning, and multimodal reasoning behavior.

Segmentation is an optional capability. A normal VILA request must not import,
initialize, or execute SAM2 or any dense segmentation component. New dense
grounding behavior must be selected explicitly by capability and must have a
fallback path that leaves the original VILA path unchanged.

Prefer wrappers, adapters, registries, and narrow hooks around existing
components. Do not make broad edits to VILA core files when an extension point
can express the behavior. Preserve the upstream Apache-2.0 license headers,
copyright notices, and attribution in copied or modified files.

## Repository and Data Hygiene

- Keep the original VILA history and the `long_rl` submodule relationship.
- Do not commit model weights, checkpoints, training data, generated results,
  caches, local environments, credentials, or other private information.
- Do not download model weights or datasets as part of a code-only change.
- Keep `origin` pointed at EvoVILA and `upstream` pointed at the original VILA
  repository. Do not rewrite or remove `upstream`.
- Keep experiments and generated artifacts outside tracked source directories.

## Tests and Evaluation

- Every new feature must include a smoke test that exercises its public entry
  point without requiring a downloaded model or dataset where practical.
- Baseline smoke tests must cover text-only, single-image, multi-image, and
  video input contracts before capability extensions are enabled.
- Efficiency experiments must report vision encoder, LLM, mask decoder, and
  propagation time separately when those components exist.
- Do not use visual token count as a substitute for real end-to-end latency,
  memory, or FLOPs. Report the actual execution path and hardware settings.
- Record whether a test used the original VILA path or an optional extension
  path, and compare capability retention against the frozen baseline.

## Extension Boundaries

- Segmentation must be an opt-in module with no SAM2 dependency on ordinary
  VILA tasks.
- Image segmentation, video segmentation, and future conditional compute
  policies must remain separately measurable.
- A future capability router may choose whether to activate dense perception,
  but it must not silently change the default path or erase baseline outputs.
- Do not implement SAM2, `[SEG]` tokens, text-conditioned referring/reasoning
  segmentation, or dynamic routing as part of the M0-M2 warm-up work. A future
  training recipe must be added only after the mask data contract is frozen.
