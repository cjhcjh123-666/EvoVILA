# Capability Interface

M1 adds an opt-in interface around the existing VILA multimodal embedding
path. The default request is empty, so ordinary model construction does not
instantiate an extension and does not import dense-perception dependencies.

## Contract

`llava.capabilities.CapabilityRequest` selects capability names and optional
per-capability settings. It accepts `None`, a name, a list of names, or a
mapping such as:

```python
{
    "names": ["example"],
    "options": {"example": {"mode": "audit"}},
}
```

`CapabilityRegistry` maps those explicit names to factories. A factory receives
`config=` and `options=` and must return a `Capability`. Unknown names fail in
strict mode rather than silently enabling a different implementation.

`MediaContext` is a read-only structural view of one `_embed` request. It
contains the input IDs, already-prepared image/video payloads, media
configuration, training state, and a stage label. Mapping and sequence
containers are frozen before dispatch. Capability return values are collected
by `CapabilityPipeline` for future consumers but are deliberately ignored by
VILA's fusion code in M1.

The M2 `image_segmentation` capability additionally observes projected image
features through `on_vision_features`. It returns `mask_logits` and, when
`segmentation_masks` are present in the context metadata, an auxiliary BCE plus
Dice loss. This extension is opt-in and does not replace VILA media-token
fusion.

M3 adds the separate `video_segmentation` capability. Its `segment_videos()`
entry point accepts a frame directory or MP4 and explicitly runs the local
SAM2.1 video predictor from anchor-frame point or box prompts. It returns
binary CPU masks shaped `[T,H,W]` for one object or `[T,N,H,W]` for multiple
objects. Predictor, anchor, propagation, and total latency are available from
`return_result=True`. SAM2 remains lazy and is not part of ordinary VILA
loading.

An extension can be registered for local use without changing VILA's registry:

```python
from llava.capabilities import Capability, DEFAULT_CAPABILITY_REGISTRY


class ExampleCapability(Capability):
    name = "example"

    def on_media_context(self, context):
        return {"media": context.media_count()}


DEFAULT_CAPABILITY_REGISTRY.register("example", ExampleCapability)
```

The model configuration field is `capabilities`. Both the normal
`llava/model/configuration_llava.py` path and the `trust_remote_code` VILA
configuration expose the same field. Both embedding implementations notify
the pipeline before media encoders run; no callback can replace embeddings,
consume media tokens, alter labels, or change sequence padding through the M1
interface.

## Invariants

- No configured capability means an empty no-op pipeline.
- Capability selection is explicit and deterministic.
- The default installation has no SAM2 or segmentation dependency.
- Image and video payloads retain their original container order and shapes.
- VILA remains responsible for media-token consumption and tensor alignment.
- A remote-code bundle without the EvoVILA package keeps the same no-op default;
  optional extensions require the shared package to be available.
