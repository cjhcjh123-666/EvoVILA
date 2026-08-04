# EvoVILA User Requirements

## Repository and environment

- Work in `/9950backfile/chenjiahui/EvoVILA`; do not use
  `/media/insslab/F0521D36521D0350` as the repository working directory.
- Keep the active environment under `/9950backfile`, at
  `/9950backfile/chenjiahui/.conda/envs/evovila`.
- Preserve the existing `long_rl` user modification. It must not be included
  in EvoVILA commits.
- Keep model weights, datasets, caches, generated masks, and experiment
  results outside Git.

## M3 scope

- Implement the next milestone, M3, as an explicit `video_segmentation`
  capability based on the local SAM2 source and SAM2.1 checkpoint.
- Preserve ordinary VILA image, multi-image, video, captioning, QA, and
  reasoning paths. They must not import, initialize, or execute SAM2.
- Keep SAM2 import and predictor construction lazy. An explicit capability
  configuration is required before the video branch can run.
- Support a directory of JPEG frames and an MP4 input accepted by SAM2.
- Support anchor-frame point and box prompts, including multiple object
  prompts.
- Return masks shaped `[T, H, W]` for one object or `[T, N, H, W]` for multiple
  objects, with stable object-ID metadata.
- Record anchor, propagation, and total latency for every video segmentation
  request.
- Add a DAVIS validation-sequence GPU smoke test using existing external
  assets. Do not download or copy assets into the repository.

## Explicitly deferred

- `[SEG]` tokens, text-conditioned referring segmentation, reasoning
  segmentation, dynamic anchor selection, capability routing, mixed training,
  retention/distillation, and full benchmark training.

## Verification

- Run focused unit tests without requiring SAM2 or external model assets.
- Run `py_compile`, `pip check`, and the existing VILA regression/smoke checks
  in `evovila`.
- Run the end-to-end GPU smoke with:
  `/9950backfile/zhangyafei/sam2` on `PYTHONPATH`, the existing SAM2.1 tiny
  checkpoint, and one DAVIS 2017 validation sequence.
