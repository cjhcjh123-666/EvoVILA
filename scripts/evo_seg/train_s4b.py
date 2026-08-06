"""S4b T1 image overfit: train decoder + [SEG] + projector + LoRA (+SAM2 mask decoder).

Outputs (checkpoints, retention baselines, summaries) are written only outside
the repository.  Ordinary VILA requests remain unchanged because LoRA/[SEG]
contribute only inside the explicit training scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import yaml
from PIL import Image
from tqdm import tqdm

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.contracts import GroundingBatch
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.losses import (
    binary_mask_loss,
    compute_segmentation_loss,
    dice_loss,
    objectness_loss,
)
from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2ImageFeatureProvider
from llava.evo_seg.training import (
    GroundingProjector,
    SegEmbeddingInjector,
    SegTrainingPolicy,
    apply_lora,
    seg_training_active,
)
from llava.evo_seg.vila_adapter import VILASegmentationAdapter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config must contain a mapping")
    return config


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(device: torch.device, function):
    _synchronize(device)
    started = time.perf_counter()
    value = function()
    _synchronize(device)
    return value, (time.perf_counter() - started) * 1000.0


def _setup_distributed(device_text: str) -> Tuple[torch.device, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device, rank, world_size
    device = torch.device(device_text)
    return device, 0, 1


def _sync_gradients(optimizer: torch.optim.Optimizer, world_size: int) -> None:
    if world_size <= 1:
        return
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = getattr(encoded, "input_ids", encoded)
    if isinstance(values, torch.Tensor):
        values = values.flatten().tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return tuple(int(value) for value in values)


def _prepare_inputs(model: Any, conversation: List[Dict[str, Any]]):
    from llava.mm_utils import process_images
    from llava.utils.media import extract_media
    from llava.utils.tokenizer import tokenize_conversation

    media = extract_media(conversation, config=model.config)
    device = next(model.parameters()).device
    if media["image"]:
        processed = process_images(media["image"], model.vision_tower.image_processor, model.config)
        processed = processed.to(device=device, dtype=torch.float16)
        media["image"] = [item for item in processed]
    else:
        media = {}
    input_ids = tokenize_conversation(
        conversation, model.tokenizer, add_generation_prompt=True
    ).unsqueeze(0).to(device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    media_config = defaultdict(dict) if media else {}
    return input_ids, media, media_config, attention_mask


def _query_span_mask(
    input_ids: torch.Tensor, tokenizer: Any, query: str, seg_id: int
) -> Tuple[torch.Tensor, int]:
    if input_ids.shape[0] != 1:
        raise ValueError("query span helper requires batch size one")
    row = input_ids[0].tolist()
    spans = set()
    for candidate_text in (query, f" {query}"):
        candidate = _token_ids(tokenizer, candidate_text)
        if not candidate:
            continue
        for start in range(len(row) - len(candidate) + 1):
            if tuple(row[start : start + len(candidate)]) == candidate:
                spans.add((start, start + len(candidate)))
    if len(spans) != 1:
        raise ValueError(f"query must map to exactly one token span, found {sorted(spans)}")
    start, end = next(iter(spans))
    try:
        seg_position = row.index(seg_id, end)
    except ValueError as error:
        raise ValueError("[SEG] token must appear after the query span") from error
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[0, start : seg_position + 1] = True
    return mask, seg_position


def _rgb_batch(image: Image.Image, sample_id: str) -> RGBFrameBatch:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    frames = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
    return RGBFrameBatch(
        frames=frames,
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        sample_ids=(sample_id,),
    )


def _target_tensor(mask_path: Optional[str], device: torch.device) -> Tuple[torch.Tensor, bool]:
    if mask_path is None:
        return torch.zeros(1, 1, 1, 1, 1, dtype=torch.bool, device=device), False
    array = np.asarray(Image.open(mask_path).convert("L"))
    tensor = torch.from_numpy((array > 0).astype(np.uint8)).to(device=device, dtype=torch.bool)
    return tensor[None, None, None], True


def _image_sample(
    record: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    template: str,
    seg_id: int,
    device: torch.device,
):
    image_path = Path(record["media_path"])
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    query = record["query"]
    instruction = template.format(query=query) + " [SEG]"
    conversation = [{"from": "human", "value": [image.copy(), instruction]}]
    input_ids, media, media_config, attention_mask = _prepare_inputs(model, conversation)
    query_mask, seg_position = _query_span_mask(input_ids, tokenizer, query, seg_id)
    rgb = _rgb_batch(image, record["sample_id"])
    target_mask, present = _target_tensor(record["mask_paths"][0], device)
    return {
        "input_ids": input_ids,
        "media": media,
        "media_config": media_config,
        "attention_mask": attention_mask,
        "query_mask": query_mask,
        "seg_position": seg_position,
        "rgb": rgb,
        "target_mask": target_mask,
        "target_presence": torch.tensor(
            [[[present]]], dtype=torch.bool, device=device
        ),
        "sample_id": record["sample_id"],
        "control_kind": record["control_kind"],
    }


def _video_sample(
    record: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    template: str,
    seg_id: int,
    device: torch.device,
):
    media_dir = Path(record["media_path"])
    frame_indices = [int(index) for index in record["frame_indices"]]
    frames = []
    for index in frame_indices:
        with Image.open(media_dir / f"{index:05d}.jpg") as handle:
            frames.append(handle.convert("RGB"))
    query = record["query"]
    instruction = template.format(query=query) + " [SEG]"
    conversation = [{"from": "human", "value": frames + [instruction]}]
    input_ids, media, media_config, attention_mask = _prepare_inputs(model, conversation)
    query_mask, seg_position = _query_span_mask(input_ids, tokenizer, query, seg_id)
    array = np.stack([np.asarray(frame, dtype=np.uint8).copy() for frame in frames])
    rgb_frames = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
    rgb = RGBFrameBatch(
        frames=rgb_frames,
        frame_mask=torch.ones(1, len(frames), dtype=torch.bool),
        sample_ids=(record["sample_id"],),
    )
    target_masks = []
    for mask_path in record["mask_paths"]:
        tensor, present = _target_tensor(mask_path, device)
        if not present:
            tensor = torch.zeros(1, 1, 1, 1, 1, dtype=torch.bool, device=device)
        target_masks.append(tensor)
    if target_masks:
        target = torch.cat(target_masks, dim=2)
    else:
        target = torch.zeros(1, 1, len(frames), 1, 1, dtype=torch.bool, device=device)
    presence = torch.tensor(record["target_presence"], dtype=torch.bool, device=device).view(1, 1, -1)
    return {
        "input_ids": input_ids,
        "media": media,
        "media_config": media_config,
        "attention_mask": attention_mask,
        "query_mask": query_mask,
        "seg_position": seg_position,
        "rgb": rgb,
        "target_mask": target,
        "target_presence": presence,
        "sample_id": record["sample_id"],
        "control_kind": record["control_kind"],
    }


def _probe_conversations(model: Any, probe_media: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_paths = probe_media["image_paths"]
    video_dir = probe_media["video_dir"]
    image_items = []
    for image_path in image_paths:
        with Image.open(image_path) as handle:
            image_items.append(handle.convert("RGB"))
    from llava.media import Video

    return [
        [{"from": "human", "value": ["What is the capital of France?"]}],
        [{"from": "human", "value": [image_items[0].copy(), "Describe this image in one sentence."]}],
        [
            {
                "from": "human",
                "value": [
                    image_items[0].copy(),
                    image_items[1].copy(),
                    "Are these two images from the same scene?",
                ],
            }
        ],
        [
            {
                "from": "human",
                "value": [Video(str(video_dir)), "Summarize the video in one sentence."],
            }
        ],
    ]


def _probe_logits(model: Any, conversation: List[Dict[str, Any]]) -> torch.Tensor:
    input_ids, media, media_config, attention_mask = _prepare_inputs(model, conversation)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            attention_mask=attention_mask,
            packing=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    return outputs.logits[:, -1].detach().to(device="cpu", dtype=torch.float32)


def _sam2_refine_trainable(
    provider: SAM2ImageFeatureProvider,
    rgb_frames: RGBFrameBatch,
    coarse_logits: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run SAM2 mask decoder with autograd enabled for the mask-prompt path."""

    predictor = provider._predictor
    images = []
    for batch_index, frame_index in rgb_frames.frame_mask.nonzero(as_tuple=False).tolist():
        image = rgb_frames.frames[batch_index, frame_index].permute(1, 2, 0).contiguous().numpy()
        images.append(image)
    prompt_encoder = predictor.model.sam_prompt_encoder
    mask_input_size = getattr(prompt_encoder, "mask_input_size", (256, 256))
    valid_logits = coarse_logits[0, :, 0].unsqueeze(0)  # [1,N,H,W]
    resized_logits = F.interpolate(
        valid_logits.to(dtype=torch.float32, device=device),
        size=(int(mask_input_size[0]), int(mask_input_size[1])),
        mode="bilinear",
        align_corners=False,
    )
    mask_inputs = [item.detach().to(dtype=torch.float32, device="cpu").numpy() for item in resized_logits]
    try:
        predictor.set_image_batch(images)
        prediction = predictor.predict_batch(
            mask_input_batch=mask_inputs,
            multimask_output=False,
            return_logits=True,
        )
    finally:
        predictor.reset_predictor()
    if not isinstance(prediction, (tuple, list)) or not prediction:
        raise RuntimeError("SAM2 predict_batch did not return mask logits")
    masks = prediction[0]
    if not isinstance(masks, (tuple, list)) or len(masks) != 1:
        raise RuntimeError("SAM2 refined mask batch does not match one image")
    tensor = torch.as_tensor(masks[0], device=device, dtype=coarse_logits.dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # [1,H,W]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)  # [N,H,W]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)  # [N,1,H,W]
    return tensor.unsqueeze(0)  # [1,N,1,H,W]


def _mask_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if target.numel() == 0 or target.dtype != torch.bool:
        return 0.0
    predicted = (prediction > 0).to(device=target.device, dtype=torch.bool)
    resized = F.interpolate(
        target.to(dtype=torch.float32)[None, None],
        size=prediction.shape[-2:],
        mode="nearest",
    ).squeeze(0).squeeze(0) > 0
    intersection = (predicted & resized).sum().float().item()
    union = (predicted | resized).sum().float().item()
    return intersection / union if union > 0 else 0.0


def _run(
    config: Dict[str, Any],
    config_path: Path,
    *,
    vila_model_path: Path,
    sam2_source_root: Path,
    sam2_checkpoint: Path,
    manifest_dir: Path,
    output_dir: Path,
    device_text: str,
    retention_image_paths: Optional[List[str]] = None,
    retention_video_dir: Optional[str] = None,
) -> Dict[str, Any]:
    device, rank, world_size = _setup_distributed(device_text)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("S4b training requires an available CUDA device")
    seed = int(config.get("seed", 0))
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats(device)

    import llava

    model, vila_initialization_ms = _timed_call(
        device,
        lambda: llava.load(
            str(vila_model_path), device=str(device), device_map={"": str(device)}
        ),
    )
    print(f"[train] model loaded in {vila_initialization_ms:.0f} ms", flush=True)
    model.eval()
    model.requires_grad_(False)

    tokenizer = model.tokenizer
    seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
    if seg_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SEG]"]})
        seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
        llm = model.get_llm()
        llm.resize_token_embeddings(len(tokenizer))
        print("[train] tokenizer resized with [SEG]", flush=True)
    llm_for_embeddings = model.get_llm()
    embed_module = getattr(getattr(llm_for_embeddings, "model", llm_for_embeddings), "embed_tokens")

    training = config.get("training", {})
    lora = config.get("lora", {})
    decoder_config = dict(config.get("decoder", {}))
    sam2_config = dict(config.get("sam2", {}))
    prompt_config = dict(config.get("prompt", {}))
    template = str(prompt_config.get("instruction_template", "Segment the object described by this referring expression: {query}."))

    probe_media = dict(config.get("retention_probes", {}))
    if retention_image_paths:
        probe_media["image_paths"] = retention_image_paths
    if retention_video_dir:
        probe_media["video_dir"] = retention_video_dir
    baseline_logits = None
    if rank == 0:
        probe_conversations = _probe_conversations(model, probe_media)
        print("[train] building retention baselines", flush=True)
        baseline_logits = [_probe_logits(model, conversation) for conversation in probe_conversations]
        print("[train] retention baselines captured", flush=True)
        torch.save(baseline_logits, output_dir / "retention_baseline.pt")
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = SAM2ImageFeatureProvider(
        {
            "source_root": str(sam2_source_root),
            "model_config": sam2_config["model_config"],
            "checkpoint_path": str(sam2_checkpoint),
            "device": str(device),
            "apply_postprocessing": bool(sam2_config.get("apply_postprocessing", True)),
        }
    )
    _, sam2_initialization_ms = _timed_call(device, provider.initialize)
    print("[train] SAM2 initialized", flush=True)

    decoder = QueryConditionedSpatialDecoder(
        query_dim=int(model.llm.config.hidden_size),
        feature_dim=int(decoder_config.get("feature_dim", 256)),
        model_dim=int(decoder_config.get("model_dim", 128)),
        num_heads=int(decoder_config.get("num_heads", 8)),
        num_layers=int(decoder_config.get("num_layers", 2)),
        num_object_queries=int(decoder_config.get("num_object_queries", 1)),
        dropout=float(decoder_config.get("dropout", 0.0)),
    ).to(device=device)
    projector = GroundingProjector(hidden_size=int(model.llm.config.hidden_size)).to(device=device)
    seg_injector = SegEmbeddingInjector(embed_module, seg_id, int(model.llm.config.hidden_size))
    seg_injector.seg_embedding = torch.nn.Parameter(seg_injector.seg_embedding.detach().to(device))
    print("[train] trainable components attached", flush=True)

    llm = model.get_llm()
    llm_layers = getattr(getattr(llm, "model", llm), "layers")
    lora_layers = list(llm_layers[-int(lora.get("layers", 8)) :])
    lora_adapters, lora_handles = apply_lora(
        lora_layers,
        list(lora.get("projections", ["self_attn.q_proj", "self_attn.v_proj"])),
        int(lora.get("rank", 8)),
        float(lora.get("alpha", 16)),
    )

    sam2_mask_decoder_trainable = bool(training.get("sam2_mask_decoder_trainable", False))
    sam2_mask_decoder_params: List[torch.nn.Parameter] = []
    if sam2_mask_decoder_trainable:
        mask_decoder = provider._predictor.model.sam_mask_decoder
        for parameter in mask_decoder.parameters():
            parameter.requires_grad_(True)
            sam2_mask_decoder_params.append(parameter)

    capability = SegmentationCapability(decoder)
    adapter = VILASegmentationAdapter(model, provider, capability)
    policy = SegTrainingPolicy(
        lora_adapters=lora_adapters,
        lora_handles=lora_handles,
        seg_injector=seg_injector,
        projector=projector,
        decoder=decoder,
        sam2_mask_decoder=sam2_mask_decoder_params,
        sam2_mask_decoder_trainable=sam2_mask_decoder_trainable,
    )
    optimizer = torch.optim.AdamW(
        policy.optimizer_param_groups(
            float(training.get("lora_lr", 1e-4)),
            float(training.get("lr", 1e-4)),
        ),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    train_records = [
        json.loads(line)
        for line in (manifest_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_records = [
        json.loads(line)
        for line in (manifest_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_records = [record for record in train_records if record["control_kind"] != "empty_query"]
    train_positives = [record for record in train_records if record["control_kind"] == "positive"]
    train_negatives = [record for record in train_records if record["control_kind"] == "no_object"]
    if not train_positives:
        raise ValueError("manifest contains no positive training records")
    task = str(config.get("task", "image"))
    sample_loader = _video_sample if task == "video" else _image_sample
    rng = random.Random(seed)
    fixed_subset = int(training.get("fixed_subset", 0))
    if fixed_subset > 0:
        rng.shuffle(train_positives)
        train_positives = train_positives[:fixed_subset]
        print(f"[train] fixed-subset overfit on {len(train_positives)} samples", flush=True)

    steps = int(training.get("steps", 500))
    log_every = int(training.get("log_every", 10))
    eval_every = int(training.get("eval_every", 25))
    loss_weights = dict(config.get("loss_weights", {"bce": 1.0, "dice": 1.0, "objectness": 0.1}))
    refined_weight = float(training.get("refined_mask_weight", 0.5))
    hidden_layer = int(training.get("hidden_layer", -1))

    history: List[Dict[str, Any]] = []
    if rank == 0:
        print("[train] starting training loop", flush=True)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    progress = tqdm(range(steps), desc=f"T1 rank{rank}", disable=(rank != 0), ncols=100)
    while step < steps:
        if fixed_subset > 0:
            positive = train_positives[(step * world_size + rank) % len(train_positives)]
        else:
            positive = rng.choice(train_positives)
        sample = sample_loader(positive, model, tokenizer, template, seg_id, device)
        decoder.train()
        projector.train()
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
        with autocast, seg_training_active(True):
            query_states = adapter.extract_query_states_training(
                sample["input_ids"],
                sample["media"],
                sample["media_config"],
                sample["query_mask"],
                sample["attention_mask"],
                hidden_layer=hidden_layer,
            )
            seg_positions = (
                sample["query_mask"][:, : sample["seg_position"] + 1].sum(dim=1) - 1
            ).to(dtype=torch.long, device=device)
            projected = projector(query_states.states, query_states.mask, seg_positions)
            dense = provider.encode_frames(sample["rgb"])
            batch = GroundingBatch(
                query_states=projected,
                query_mask=torch.ones(1, projected.shape[1], dtype=torch.bool, device=device),
                dense_features=dense.features,
                frame_mask=dense.frame_mask,
                target_masks=sample["target_mask"],
                target_presence=sample["target_presence"],
                sample_ids=(sample["sample_id"],),
            )
            result = capability(batch, {"enabled": True, "task": "image"})
            losses = compute_segmentation_loss(result, batch, loss_weights)
            total_loss = losses["total"]
            refined = None
            if sam2_mask_decoder_trainable and sample["control_kind"] == "positive":
                anchor_rgb = RGBFrameBatch(
                    frames=sample["rgb"].frames[:, :1],
                    frame_mask=torch.ones(1, 1, dtype=torch.bool),
                    sample_ids=(sample["sample_id"],),
                )
                refined = _sam2_refine_trainable(
                    provider, anchor_rgb, result.mask_logits[:, :, :1], device
                )
                refined_target = _align_targets_for_refine(sample["target_mask"][:, :, :1], refined, device)
                refined_valid = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
                refined_loss = (
                    binary_mask_loss(refined, refined_target, refined_valid)
                    + dice_loss(refined, refined_target, refined_valid)
                ) * refined_weight
                total_loss = total_loss + refined_loss
        total_loss.backward()
        _sync_gradients(optimizer, world_size)
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            float(training.get("grad_clip", 1.0)),
        )
        nan_grads = [
            f"{name}:{parameter.grad.abs().max().item():.3e}"
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if nan_grads:
            print(f"[train] step {step} NON-FINITE grads: {nan_grads[:5]}", flush=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        nan_params = [
            f"{group['name']}:{parameter.abs().max().item():.3e}"
            for group in optimizer.param_groups
            for parameter in group["params"]
            if not torch.isfinite(parameter).all()
        ]
        if nan_params:
            print(f"[train] step {step} NON-FINITE params: {nan_params[:5]}", flush=True)
        if not torch.isfinite(total_loss).all():
            print(f"[train] step {step} NON-FINITE loss {float(total_loss.detach())}", flush=True)

        anchor_iou = _mask_iou(
            result.mask_logits[0, 0, 0].detach(),
            sample["target_mask"][0, 0, 0].detach() if sample["control_kind"] == "positive" else torch.zeros(1, 1, dtype=torch.bool, device=device),
        )
        progress.update(1)
        progress.set_postfix(
            loss=float(total_loss.detach().item()),
            iou=anchor_iou,
            bce=float(losses["bce"].detach().item()),
            dice=float(losses["dice"].detach().item()),
        )
        history.append(
            {
                "step": step,
                "loss": float(total_loss.detach().item()),
                "bce": float(losses["bce"].detach().item()),
                "dice": float(losses["dice"].detach().item()),
                "anchor_iou": anchor_iou,
                "control_kind": sample["control_kind"],
                "sample_id": sample["sample_id"],
            }
        )
        if rank == 0 and (step % log_every == 0 or step == steps - 1):
            latest = history[-1]
            print(
                f"step {step:4d} loss {latest['loss']:.4f} bce {latest['bce']:.4f} "
                f"dice {latest['dice']:.4f} iou {latest['anchor_iou']:.3f} "
                f"kind {latest['control_kind']}"
            )
        if rank == 0 and (step % eval_every == 0 or step == steps - 1) and val_records:
            decoder.eval()
            projector.eval()
            ious = []
            for record in rng.sample([r for r in val_records if r["control_kind"] == "positive"], min(10, len(val_records))):
                sample = sample_loader(record, model, tokenizer, template, seg_id, device)
                with torch.no_grad(), seg_training_active(True), torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    query_states = adapter.extract_query_states_training(
                        sample["input_ids"], sample["media"], sample["media_config"],
                        sample["query_mask"], sample["attention_mask"], hidden_layer=hidden_layer,
                    )
                    seg_positions = (
                        sample["query_mask"][:, : sample["seg_position"] + 1].sum(dim=1) - 1
                    ).to(dtype=torch.long, device=device)
                    projected = projector(query_states.states, query_states.mask, seg_positions)
                    dense = provider.encode_frames(sample["rgb"])
                    batch = GroundingBatch(
                        query_states=projected,
                        query_mask=torch.ones(1, projected.shape[1], dtype=torch.bool, device=device),
                        dense_features=dense.features,
                        frame_mask=dense.frame_mask,
                        sample_ids=(sample["sample_id"],),
                    )
                    result = capability(batch, {"enabled": True, "task": "image"})
                ious.append(
                    _mask_iou(result.mask_logits[0, 0, 0].detach(), sample["target_mask"][0, 0, 0].detach())
                )
            val_iou = float(np.mean(ious)) if ious else 0.0
            print(f"step {step:4d} val IoU (n={len(ious)}): {val_iou:.3f}", flush=True)
            history[-1]["val_iou"] = val_iou
        step += 1
    progress.close()

    retention_checks: List[Dict[str, Any]] = []
    if rank == 0:
        decoder.eval()
        projector.eval()
        after_conversations = _probe_conversations(model, probe_media)
        after_logits = [_probe_logits(model, conversation) for conversation in after_conversations]
        for index, (before, after) in enumerate(zip(baseline_logits, after_logits)):
            equal = torch.equal(before, after)
            diff = float((before - after).abs().max().item())
            retention_checks.append({"probe": index, "exact": bool(equal), "max_abs_diff": diff})
            if not equal:
                print(f"WARNING retention probe {index} changed by {diff}")

    if rank == 0:
        checkpoint = {
            "decoder": decoder.state_dict(),
            "projector": projector.state_dict(),
            "seg_embedding": seg_injector.seg_embedding.detach().cpu(),
            "lora": {
                name: [adapter.state_dict() for adapter in adapters]
                for name, adapters in lora_adapters.items()
            },
            "sam2_mask_decoder": (
                {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in provider._predictor.model.sam_mask_decoder.named_parameters()
                }
                if sam2_mask_decoder_trainable
                else {}
            ),
        }
        torch.save(checkpoint, output_dir / "s4b_t1_checkpoint.pt")

        summary = {
            "git_commit": _git_commit(_repo_root()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "seed": seed,
            "world_size": world_size,
            "steps_completed": steps,
            "training": {
                "sam2_mask_decoder_trainable": sam2_mask_decoder_trainable,
                "lora_rank": int(lora.get("rank", 8)),
                "lora_layers": len(lora_layers),
                "freeze_summary": policy.freeze_state_summary(model),
            },
            "retention": retention_checks,
            "final_metrics": {
                "last_train_loss": history[-1]["loss"] if history else None,
                "last_train_iou": history[-1]["anchor_iou"] if history else None,
                "val_iou": history[-1].get("val_iou") if history else None,
            },
            "history": history,
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
            },
            "assets": {
                "vila_model_name": vila_model_path.name,
                "sam2_checkpoint_name": sam2_checkpoint.name,
            },
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "vila_initialization_ms": vila_initialization_ms,
            "sam2_initialization_ms": sam2_initialization_ms,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    return {}


def _align_targets_for_refine(target: torch.Tensor, refined: torch.Tensor, device: torch.device) -> torch.Tensor:
    squeezed = target[0].to(dtype=torch.float32, device=device)  # [1,1,H,W]
    aligned = F.interpolate(
        squeezed,
        size=refined.shape[-2:],
        mode="nearest",
    ).unsqueeze(0)
    return aligned


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_repo_root() / "configs/evo_seg/s4b_t1_image.yaml")
    parser.add_argument("--vila-model", type=Path, required=True)
    parser.add_argument("--sam2-source-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retention-image-paths", nargs="+", default=None)
    parser.add_argument("--retention-video-dir", default=None)
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    try:
        output_dir.relative_to(_repo_root().resolve())
    except ValueError:
        pass
    else:
        raise ValueError("training output must be outside the repository")
    _run(
        _load_config(config_path),
        config_path,
        vila_model_path=args.vila_model.expanduser().resolve(),
        sam2_source_root=args.sam2_source_root.expanduser().resolve(),
        sam2_checkpoint=args.sam2_checkpoint.expanduser().resolve(),
        manifest_dir=args.manifest_dir.expanduser().resolve(),
        output_dir=output_dir,
        device_text=args.device,
        retention_image_paths=args.retention_image_paths,
        retention_video_dir=args.retention_video_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
