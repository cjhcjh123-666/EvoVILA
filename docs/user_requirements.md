# EvoVILA-Seg User Requirements

## Repository And Environment

- Work from branch `EvoVILA-Seg`, based exactly on VILA `main` commit
  `0f1426e8da9181e6e6653e10bc15f62d515fa2f6`.
- Use the existing PyTorch VILA codebase and the environment at
  `/9950backfile/chenjiahui/.conda/envs/evovila`.
- Preserve the original `long_rl` submodule relationship. Do not include the
  user's local `long_rl` modification in EvoVILA-Seg commits.
- Keep model weights, datasets, caches, generated masks, logs, and experiment
  results outside Git.
- Do not download weights or datasets as part of code-only milestones.

## Confirmed Objective

Extend VILA with language-conditioned image and video spatiotemporal
segmentation while preserving its existing text, single-image, multi-image,
video, QA, captioning, and multimodal reasoning behavior.

The confirmed architecture assigns responsibilities as follows:

- VILA provides semantic understanding of the user query and media context.
- A trainable query-conditioned spatial decoder aligns language tokens with
  dense visual features and predicts an anchor mask, object confidence, and
  object representation.
- SAM2 refines image masks and propagates predicted anchor masks through
  video. SAM2 does not replace VILA's language or ordinary media path.
- An image is handled by the same segmentation contract as a one-frame video
  (`T=1`).

## Capability Preservation Invariants

- Segmentation is opt-in at request time. The default request executes the
  original VILA path.
- A normal VILA request must not import, initialize, or execute SAM2 or any
  dense decoder.
- The first training stages freeze all original VILA and SAM2 parameters.
- If capability-specific LoRA is later required, it must be disabled for
  normal requests and the base VILA weights must remain immutable.
- `[SEG]` may mark a mask-producing response, but a single `[SEG]` hidden state
  must not be the sole grounding representation.
- Do not use a direct LLM-hidden-to-SAM2-sparse-prompt projector as the main
  route. Predict an anchor mask from explicit language-spatial interaction and
  pass that mask to SAM2.
- Image and video segmentation must remain separately measurable even though
  they share contracts and decoder components.

## Implementation Strategy

- Use wrappers, adapters, registries, and narrow observation hooks around VILA.
- Build and validate the independent segmentation contracts and decoder before
  editing VILA core files.
- Use multi-token query states and cross-attention over dense spatial features.
- Include same-image different-target, query-swap, empty-query, and no-object
  controls in training and validation contracts.
- Treat dynamic anchor selection, conditional compute, and shared-parameter
  VILA fine-tuning as later milestones, not first-stage requirements.
- S1 query extraction must use an explicit original-token query mask and fused
  provenance. It must not infer the referring expression from a single marker
  token or keep model-global media/query state.
- S1 runs VILA under frozen teacher forcing with local sequence packing
  disabled so hidden states stay aligned with the observed fused sequence.

## Execution Policy

- Automatically run CPU/no-weight unit tests, static compilation, and Git
  hygiene checks.
- Do not start long GPU training or dataset downloads without a separate
  confirmed experiment plan.
- Keep the project README at the repository root and extension documentation
  under `docs/`.

## Coding Phase E Assumptions

The user confirmed starting implementation on 2026-08-05. The following
working assumptions apply to the S0 no-weight milestone and can be superseded
by a later user constraint:

- Reuse `/9950backfile/chenjiahui/.conda/envs/evovila`.
- Run S0 checks on CPU with synthetic tensors; do not start long GPU training.
- Do not download datasets, model weights, or SAM2 assets in this milestone.
- Automatically run fast unit tests, smoke tests, `py_compile`, and
  `git diff --check`; defer full training and benchmark runs.
- Preserve the existing `origin`/`upstream` remotes and Git identity; do not
  push during this implementation turn.
- Keep the existing VILA README at the repository root; extension notes remain
  under `docs/`.
