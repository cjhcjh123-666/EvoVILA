# Image Segmentation Warm-Up

M2 adds a small, opt-in mask decoder to the existing image path. It is a
warm-up for capability integration, not the final EvoVILA segmentation model.

## Activation

Enable the capability in the model configuration:

```json
{
  "capabilities": {
    "names": ["image_segmentation"]
  }
}
```

Without this field, no segmentation module is imported or initialized. The
ordinary VILA image, video, and text paths keep their existing behavior.

## Data Contract

Images reach the capability as the already-prepared tensors used by VILA. The
M2 warm-up currently requires a static image path whose projected features have
the same number of visual tokens for every image and whose token count is a
perfect square. A feature tensor therefore has shape
`[num_images, num_visual_tokens, hidden_size]`, which is reshaped into a square
grid before decoding.

Training targets are optional and are passed as `segmentation_masks` with shape
`[num_images, height, width]`. A single `[height, width]` mask and
`[num_images, 1, height, width]` inputs are normalized by the collator and
capability. The batch collator requires one mask per image and rejects mixed
segmentation/non-segmentation batches.

## Interfaces

`model.segment_images(images)` runs the mask branch without an LLM call and
returns logits shaped `[num_images, 1, height, width]`. The normal model
`forward(..., segmentation_masks=...)` path computes BCE plus Dice as an
auxiliary loss and adds it to the language-model loss.

The decoder is vision-only. It does not consume referring expressions or LLM
hidden states, and therefore does not claim referring image segmentation or
reasoning image segmentation yet. It also does not implement `[SEG]` tokens,
video inputs, SAM2 propagation, dynamic tiling, or conditional routing.

## Checkpoints

When a model with enabled capability modules is saved, the module parameters
are written separately to `capabilities.bin`. The top-level config records
`capability_checkpoint`; loading resolves that file relative to the model
directory or the existing VILA `resume_path`. Base VILA weights are not moved
into this file.

No weights, datasets, generated masks, or experiment outputs belong in the
repository.
