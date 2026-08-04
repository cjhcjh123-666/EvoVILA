# EvoVILA Configurations

This directory is reserved for EvoVILA-specific experiment configuration. It
keeps future capability, retention, and efficiency settings separate from the
imported VILA `scripts/` and model defaults.

Use this directory for new configuration files only when the setting belongs
to EvoVILA. Existing VILA training and evaluation scripts should continue to
accept their original arguments and behavior.

## Planned organization

```text
configs/evo/
  capability/       # explicit optional capability requests
  baseline/         # frozen baseline protocol settings
  retention/        # future capability-retention mixtures
  efficiency/       # future component-level measurement settings
```

The model configuration field `capabilities` selects optional extensions. The
default is empty. For the M2 image warm-up, use:

```json
{
  "capabilities": {
    "names": ["image_segmentation"],
    "options": {
      "image_segmentation": {
        "hidden_channels": 128,
        "output_size": [224, 224]
      }
    }
  }
}
```

The warm-up requires equal-size square visual-token grids and consumes
preprocessed image tensors. Training targets are passed as the optional
`segmentation_masks` batch/model field with shape
`[num_images, height, width]`. The decoder is vision-only and is not a
referring/reasoning segmentation implementation. `capability_checkpoint` may
point to the generated `capabilities.bin` file.

SAM2, video, routing, retention, and efficiency configurations are not defined
here yet. Do not place model weights, datasets, caches, checkpoints, or
generated outputs under this directory.
