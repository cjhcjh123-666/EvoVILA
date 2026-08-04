# VILA Baseline Protocol

This protocol defines the reproducibility contract before optional EvoVILA
capabilities are enabled. It describes commands and measurements only; it does
not download a checkpoint or dataset as part of repository setup.

## Recommended checkpoint

Use one fixed official VILA-1.5 checkpoint for the first comparison. The
repository's existing inference examples use
`Efficient-Large-Model/VILA1.5-3b`, so it is the recommended lightweight
baseline when the checkpoint is available locally or through an approved
cache. Record the exact model identifier, revision, local path, and config
hash. Do not substitute an EvoVILA or segmentation checkpoint in baseline
measurements.

Required preconditions:

- VILA dependencies are installed at the versions required by `pyproject.toml`.
- The checkpoint is already available; this protocol does not fetch it.
- Input media and benchmark annotations are already available under approved,
  external paths; they are not committed to this repository.
- CUDA device, driver, and GPU model are recorded.

## Smoke commands

The commands below are templates. Replace paths with local files and keep
outputs outside tracked source directories.

### Image caption

```bash
vila-infer \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --text "Describe this image in one sentence." \
  --media /path/to/image.jpg
```

### Image QA

```bash
vila-infer \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --text "What is the main object in this image?" \
  --media /path/to/image.jpg
```

Both commands use `llava/cli/infer.py` -> `llava.load` ->
`llava.model.builder.load_pretrained_model` -> `model.generate_content`.

### Video caption

```bash
vila-infer \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --num_video_frames 8 \
  --text "Describe what happens in this video." \
  --media /path/to/video.mp4
```

The public path extracts video frames in `llava/utils/media.py`, repeats the
image media token for sampled frames, and preprocesses them in
`llava/model/llava_arch.py:generate_content`.

### Video QA

For benchmark-style JSON question files, use the existing evaluator template:

```bash
python llava/eval/model_vqa_video.py \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --video-folder /path/to/videos \
  --question-file /path/to/questions.json \
  --answers-file /tmp/vila-baseline/answers.jsonl \
  --num-chunks 1 \
  --chunk-idx 0
```

Check the installed script's `--help` for any local argument additions before
running. This path uses `LazySupervisedDataset._load_video`,
`process_images`, repeated `<image>` tokens, and `model.generate`; it is a
separate path from conversational `generate_content` and must be recorded as
such.

## Fixed settings

Record these values in every run:

| Field | Baseline requirement |
| --- | --- |
| Model | exact checkpoint ID, revision, config hash |
| Random seed | `0` unless the benchmark requires another fixed seed |
| Dtype | `float16` for the existing inference builder; record overrides |
| Sampling | record `do_sample`, temperature, top-p, and max new tokens |
| Video frames | `8` unless the benchmark protocol fixes another count |
| Video FPS | `0` for uniform sampling unless explicitly specified |
| Image resolution | processor/config value and aspect-ratio mode |
| Batch size | explicitly record, including distributed world size |
| Hardware | GPU model, count, driver, CUDA, and Transformers versions |

For deterministic smoke comparisons, use greedy decoding where supported and
record the generation config. Keep the same prompt, media bytes, frame count,
and preprocessing mode across baseline and extension runs.

## Output and measurements

Store one JSONL record per sample outside tracked source directories. Each
record should contain `sample_id`, `task`, `prompt`, `media_paths` (or stable
external IDs), `model_id`, `seed`, `generation_config`, `frame_count`,
`response`, and `status`. Do not commit the media or generated answers.

Measure warm-up separately from timed iterations and report:

- end-to-end wall time;
- media decode and preprocessing time;
- vision tower time;
- multimodal projector time;
- LLM prefill and decode time;
- peak GPU memory and host memory;
- batch size, sequence lengths, frame count, and output token count.

Future segmentation experiments must additionally report mask decoder and
propagation time separately. Visual-token count is not a substitute for these
measurements.

## Retention evaluation

Before M2, freeze the baseline outputs and environment metadata. Later
capability branches should rerun at least image QA/captioning, multi-image
understanding, video QA/captioning, and multimodal reasoning under the same
protocol. Report both task quality and retention deltas; do not compare a
segmentation-only run against a missing or changed baseline.
