# VILA Code Map

This map describes the imported VILA baseline at commit `0f1426e` (the
original `main` branch at repository import time). Paths are relative to the
repository root. The map is based on the actual call sites, not only the
README.

## Construction and Loading

The normal library loading chain is:

```text
llava.load
  -> llava/entry.py:load
  -> auto_set_conversation_mode
  -> llava.model.builder:load_pretrained_model
  -> LlavaLlamaModel (or LlavaTopDownLlamaModel)
  -> LlavaMetaModel.init_vlm
       -> build_llm_and_tokenizer
       -> build_vision_tower
       -> build_mm_projector
       -> hydra.instantiate(image_encoder/video_encoder)
```

The model builder is in `llava/model/builder.py:27-156`. It distinguishes a
multimodal checkpoint from a language-only checkpoint, loads LoRA weights
when a base model is supplied, resizes the tokenizer embeddings, moves the
vision tower and projector to the requested device, and returns the tokenizer,
model, image processor, and context length.

`LlavaLlamaModel` is defined in
`llava/model/language_model/llava_llama.py:41-159`. Its constructor calls
`init_vlm`. That method is in `llava/model/llava_arch.py:56-95` and creates the
three major modules plus the `image` and `video` encoder wrappers from the
configuration in `llava/model/configuration_llava.py:26-85`.

## Vision Tower and Projector

`llava/model/multimodal_encoder/builder.py:30-76` selects the vision tower by
architecture/name. It supports CLIP, SigLIP, Intern, RADIO, and PS3 variants,
including fixed S2 and SigLIP dynamic-S2 variants. The tower wrapper in
`llava/model/multimodal_encoder/vision_encoder.py:32-204` calls the underlying
vision model with `output_hidden_states=True`, selects the configured hidden
state (`mm_vision_select_layer`), and removes the CLS token for `patch`
features.

The fixed multi-scale path is `VisionTowerS2.forward` at
`vision_encoder.py:207-248`. The dynamic-S2 wrapper is
`VisionTowerDynamicS2` at `vision_encoder.py:251-276`. Existing dynamic
preprocessing and chessboard merging live in
`llava/mm_utils.py:299-405` and `llava/model/llava_arch.py:298-394`.

`LlavaMetaModel.encode_images` at `llava/model/llava_arch.py:366-394` runs the
vision tower and then the multimodal projector. Dynamic-S2 additionally merges
tiles, projects them, and restores their spatial arrangement. The projector
registry is `llava/model/multimodal_projector/builder.py:27-42`; projector
implementations and downsampling choices are in
`llava/model/multimodal_projector/base_projector.py:126-252`.

## Image and Video Flow

### Public inference

`llava/cli/infer.py:100-175` accepts text and image/video paths, wraps paths in
`llava.media.Image` or `llava.media.Video`, and calls
`model.generate_content`.

`llava/utils/media.py:93-123:extract_media` extracts image objects and samples
video frames with `_extract_video` and `_load_video` at lines `32-90`. In this
public conversational path, a `Video` is expanded into sampled image frames
and repeated image media tokens. This is an important baseline behavior and
must not be silently changed by a future video-segmentation extension.

`llava/model/llava_arch.py:836-948:generate_content` performs the corresponding
preprocessing. It supports resize, dynamic tiling, dynamic-S2 tiling, and
video frame preprocessing, then tokenizes the conversation and delegates to
`generate`.

### Training and explicit video data

The training path starts at `llava/train/train.py:419-819`. Dataset selection is
made by `llava/data/builder.py:85-151:build_dataset`, which expands mixtures
from `llava/data/registry/mixtures.yaml` and instantiates dataset definitions
from `llava/data/registry/datasets/default.yaml`.

`llava/data/base.py:74-190:BaseDataset.__getitem__` calls a dataset's
`process`, extracts media, applies ordinary or dynamic image preprocessing,
converts conversations to input IDs and labels, and returns image/video
tensors. The legacy dataset implementations in `llava/data/dataset.py` also
contain explicit video sampling in `_load_video` and construct per-frame
image-token inputs. `DataCollator` at `llava/data/collate.py:13-100` gathers
media lists, pads text/labels, and emits `media` and `media_config`.

The explicit encoder wrappers are configured by `LlavaConfig`:

- `BasicImageEncoder` in `llava/model/encoders/image/basic.py:11-68` stacks
  image tensors, calls `parent.encode_images`, and optionally adds learned
  start/end text embeddings.
- `BasicVideoEncoder` in `llava/model/encoders/video/basic.py:11-59` flattens
  all frames, calls `parent.encode_images`, splits features back per video,
  adds optional frame-wise start/end embeddings, and flattens time and token
  dimensions.
- `TSPVideoEncoder` in `llava/model/encoders/video/tsp.py:15-73` adds temporal
  spatial pooling and separator embeddings.

## Media Tokens and Embedding Insertion

The token definitions are in `llava/constants.py:24-49`:
`<image>` and `<vila/video>` are the media tokens, with
`<vila/sentinel>` reserved by the original code. The tokenizer is created in
`llava/model/language_model/builder.py:64-215`. It records
`tokenizer.media_tokens`, adds each media token as a special token, and stores
their IDs in `tokenizer.media_token_ids`.

Conversation preprocessing is provided by
`llava/utils/tokenizer.py:72-172` and the repository's remote-code tokenizer
helpers. The data collator ensures the number of media tensors matches media
token positions before the model call.

The central fusion chain is:

```text
LlavaLlamaModel.forward
  -> LlavaMetaForCausalLM._embed
       -> __embed_media_tokens
            -> BasicImageEncoder/BasicVideoEncoder
                 -> encode_images
                      -> vision tower -> mm projector
       -> replace media-token embeddings with visual embeddings
       -> truncate and batchify inputs/labels
  -> self.llm(inputs_embeds=..., attention_mask=..., labels=...)
```

The generic implementation is in `llava/model/llava_arch.py:412-555`. It
removes padding, maps media token IDs back to `image`/`video`, consumes the
encoder outputs in order, assigns `IGNORE_INDEX` to inserted visual tokens,
truncates during training, and pads the fused sequence. A distributed-training
dummy call in `__embed_media_tokens` at lines `492-517` keeps encoder calls
balanced across ranks.

`LlavaLlamaModel.forward` at `llava/model/language_model/llava_llama.py:94-159`
then calls the underlying LLM. `output_hidden_states=True` is passed through
to the underlying Transformers model via `**kwargs`; its returned
`CausalLMOutputWithPast.hidden_states` is the current safe forward-time hook
for post-LLM feature consumers.

## Generation and Hidden States

`LlavaMetaForCausalLM.generate` at `llava/model/llava_arch.py:823-833` embeds
media and calls `self.llm.generate(inputs_embeds=..., attention_mask=...)`.
The current wrapper returns generated token IDs by default. A future consumer
that needs generation-time hidden states must explicitly request the supported
Transformers generation return structure, for example with
`return_dict_in_generate=True` and `output_hidden_states=True`, and verify the
installed Transformers version's output contract.

There is no current dedicated capture of hidden states for generated special
tokens. Media tokens are consumed during `_embed`, while generated IDs are
decoded in `generate_content`. A future hook should therefore either capture
the fused embedding/LLM forward output before decoding, or correlate returned
generation-step hidden states with generated IDs. It must not assume that the
number of visual tokens equals the number of generated tokens.

## Training, Freezing, LoRA, and Checkpoints

`llava/train/utils.py:82-127:prepare_config_for_training` copies model/data
arguments into the config and sets `tune_language_model`,
`tune_vision_tower`, and `tune_mm_projector`.

`llava/train/train.py:648-706` creates PEFT LoRA/DoRA adapters when enabled.
`find_all_linear_names` at `train.py:149-171` excludes modules according to
`lora_llm` and `lora_vt`. Without LoRA, lines `707-739` independently set
`requires_grad` for the LLM, vision tower, and projector. Training arguments
are declared in `llava/train/args.py:220-280`.

`LLaVATrainer.create_optimizer` at `llava/train/llava_trainer.py:667-829`
creates parameter groups and supports separate `mm_projector_lr` and
`vision_tower_lr`. `safe_save_model_for_hf_trainer` is in
`llava/train/train.py:174-185`; LoRA and non-LoRA checkpoint handling is also
implemented by `LLaVATrainer.save_model` around `llava/train/llava_trainer.py:808-829`.

## Inference and Evaluation Entrypoints

- `vila-infer` maps to `llava/cli/infer.py:100-175` and calls
  `llava.load` plus `model.generate_content`.
- `vila-eval` maps to `llava/cli/eval.py:29-...`, selects registered tasks,
  launches task scripts, and reads `results.json` or `metrics.json`.
- `llava/eval/model_vqa_video.py:51-116` is a direct video-QA evaluation
  path: sample frames, preprocess them, create repeated image tokens, and call
  `model.generate`.
- `llava/eval/model_refcoco.py:103-...` is an image grounding/evaluation
  entrypoint that evaluates generated box text. It is not a segmentation head.

## Candidate Extension Hooks

| Hook | Current contract | Extension constraint |
| --- | --- | --- |
| `LlavaMetaModel.init_vlm` | Builds LLM, vision tower, projector, and encoder registry | Add optional capability modules without making ordinary loading depend on them |
| `LlavaMetaModel.encode_images` | Returns projected visual features | Preserve shape/order and baseline outputs when extension is disabled |
| `BasicImageEncoder` / `BasicVideoEncoder` | Converts projected features to media embedding sequences | Keep media-token consumption and frame ordering stable |
| `LlavaMetaForCausalLM._embed` | Fuses text and media embeddings and labels | Do not alter default token alignment; use an explicit capability context |
| `LlavaLlamaModel.forward` | Runs the LLM and exposes standard output fields | Add optional outputs without changing ordinary loss/generation behavior |
| `LlavaMetaForCausalLM.generate_content` | Public multimodal generation path | Keep the default path usable without dense packages |
| `LlavaConfig` | Stores model, media, and efficiency configuration | Add namespaced optional capability config rather than unrelated global flags |
| `llava/data/builder.py` registry | Selects datasets through YAML/Hydra | Add dense datasets as opt-in registrations with explicit smoke tests |
