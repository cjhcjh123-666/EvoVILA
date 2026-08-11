"""Visualize S4b referring-segmentation predictions (image + query + mask).

For each selected validation record this script renders three panels:
coarse decoder mask, SAM2-refined mask, and ground-truth mask, overlaid on the
source image, with the referring expression and per-panel IoU printed above.
No-object (control) records are rendered too, labelled with the hallucination
prediction.  Everything is read-only with respect to the VILA/SAM2 weights.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.contracts import GroundingBatch
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2ImageFeatureProvider
from llava.evo_seg.sam2_video_adapter import (
    build_sam2_video_image_predictor,
    SAM2VideoMaskPropagator,
)
from llava.evo_seg.training import (
    GroundingProjector,
    SegEmbeddingInjector,
    apply_lora,
    seg_training_active,
)
from llava.evo_seg.vila_adapter import VILASegmentationAdapter
from scripts.evo_seg.train_s4b import (
    _image_sample,
    _mask_iou,
    _sam2_refine_trainable,
    _upsample_dense,
)


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_records(manifest_dir: Path) -> List[Dict]:
    records = []
    path = manifest_dir / "val.jsonl"
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple, alpha: float = 0.45) -> Image.Image:
    overlay = image.convert("RGB")
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[mask] = color
    blended = np.asarray(overlay, dtype=np.float32) * (1.0 - alpha) + mask_rgb * alpha
    return Image.fromarray(blended.astype(np.uint8))


def _mask_from_logits(logits: torch.Tensor, size: tuple) -> np.ndarray:
    prob = torch.sigmoid(logits)
    mask = (prob > 0.5).detach().cpu().numpy()
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    return np.asarray(mask_img) > 127


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vila-model", type=Path, required=True)
    parser.add_argument("--sam2-source-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config = _load_config(args.config.expanduser().resolve())
    training = dict(config.get("training", {}))
    decoder_config = dict(config.get("decoder", {}))
    sam2_config = dict(config.get("sam2", {}))
    prompt_config = dict(config.get("prompt", {}))
    hidden_layer = int(training.get("hidden_layer", -1))
    spatial_scale = int(decoder_config.get("spatial_scale", 1))
    template = str(
        prompt_config.get(
            "instruction_template",
            "Segment the object described by this referring expression:\n{query}",
        )
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import llava

    model = llava.load(str(args.vila_model), device=str(device), device_map={"": str(device)})
    model.eval()
    model.requires_grad_(False)
    model.to(torch.bfloat16)
    tokenizer = model.tokenizer
    seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
    if seg_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SEG]"]})
        seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
        model.get_llm().resize_token_embeddings(len(tokenizer))
    embed_module = getattr(getattr(model.get_llm(), "model", model.get_llm()), "embed_tokens")
    seg_injector = SegEmbeddingInjector(embed_module, seg_id, int(model.llm.config.hidden_size))
    seg_injector.seg_embedding = torch.nn.Parameter(seg_injector.seg_embedding.detach().to(device))

    provider = SAM2ImageFeatureProvider(
        {
            "source_root": str(args.sam2_source_root),
            "model_config": sam2_config["model_config"],
            "checkpoint_path": str(args.sam2_checkpoint),
            "device": str(device),
            "apply_postprocessing": bool(sam2_config.get("apply_postprocessing", True)),
        },
        predictor_factory=build_sam2_video_image_predictor,
    )
    provider.initialize()

    decoder = QueryConditionedSpatialDecoder(
        query_dim=int(model.llm.config.hidden_size),
        feature_dim=int(decoder_config.get("feature_dim", 256)),
        model_dim=int(decoder_config.get("model_dim", 128)),
        num_heads=int(decoder_config.get("num_heads", 8)),
        num_layers=int(decoder_config.get("num_layers", 2)),
        num_object_queries=int(decoder_config.get("num_object_queries", 1)),
        dropout=float(decoder_config.get("dropout", 0.0)),
    ).to(device=device)
    projector = GroundingProjector(int(model.llm.config.hidden_size)).to(device=device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
    projector.load_state_dict(checkpoint["projector"])
    seg_injector.seg_embedding.data.copy_(checkpoint["seg_embedding"].to(device))
    lora = dict(config.get("lora", {}))
    llm = model.get_llm()
    llm_layers = getattr(getattr(llm, "model", llm), "layers")
    lora_layers = list(llm_layers[-int(lora.get("layers", 8)) :])
    lora_adapters, lora_handles = apply_lora(
        lora_layers,
        list(lora.get("projections", ["self_attn.q_proj", "self_attn.v_proj"])),
        int(lora.get("rank", 8)),
        float(lora.get("alpha", 16)),
    )
    for name, adapters in lora_adapters.items():
        for index, adapter in enumerate(adapters):
            adapter.load_state_dict(checkpoint["lora"][name][index])

    capability = SegmentationCapability(decoder)
    adapter = VILASegmentationAdapter(model, provider, capability)

    records = [r for r in _load_records(args.manifest_dir.expanduser().resolve()) if r["media_type"] == "image"]
    random.Random(args.seed).shuffle(records)
    records = records[: args.max_samples]
    if not records:
        raise SystemExit("no image records found")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for index, record in enumerate(records):
        try:
            sample = _image_sample(record, model, tokenizer, template, seg_id, device)
            with torch.no_grad(), seg_training_active(True), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                projected = projector(
                    query_states.states.to(dtype=torch.float32), query_states.mask, seg_positions
                )
                dense = _upsample_dense(provider.encode_frames(sample["rgb"]), spatial_scale)
                result = capability(
                    GroundingBatch(
                        query_states=projected,
                        query_mask=torch.ones(1, projected.shape[1], dtype=torch.bool, device=device),
                        dense_features=dense.features.to(dtype=torch.float32),
                        frame_mask=dense.frame_mask,
                        sample_ids=(record["sample_id"],),
                    ),
                    {"enabled": True, "task": "image"},
                )
                refined = _sam2_refine_trainable(provider, sample["rgb"], result.mask_logits, device)
        except Exception as error:  # noqa: BLE001
            print(f"sample {index} {record['sample_id']} FAILED: {error}", flush=True)
            continue

        source_path = Path(record["media_path"])
        try:
            source = Image.open(source_path).convert("RGB")
        except Exception as error:  # noqa: BLE001
            print(f"sample {index} image open failed: {error}", flush=True)
            continue
        size = source.size
        coarse_mask = _mask_from_logits(result.mask_logits[0, 0, 0].detach(), size)
        refined_mask = _mask_from_logits(refined[0, 0, 0].detach(), size)
        has_gt = bool(record["target_presence"][0]) if record.get("target_presence") else False
        if has_gt:
            gt_mask = sample["target_mask"][0, 0, 0].detach().cpu().numpy().astype(bool)
            gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8)).resize(size, Image.NEAREST)
            gt_mask = np.asarray(gt_img) > 127
            coarse_iou = _mask_iou(result.mask_logits[0, 0, 0].detach(), sample["target_mask"][0, 0, 0].detach())
            refined_iou = _mask_iou(refined[0, 0, 0].detach(), sample["target_mask"][0, 0, 0].detach())
        else:
            gt_mask = np.zeros((size[1], size[0]), dtype=bool)
            coarse_iou = refined_iou = None

        panels = [
            _overlay(source, coarse_mask, (255, 0, 0)),
            _overlay(source, refined_mask, (255, 128, 0)),
            _overlay(source, gt_mask, (0, 200, 0)),
        ]
        canvas = Image.new("RGB", (size[0] * 3, size[1] + 40), (255, 255, 255))
        for i, panel in enumerate(panels):
            canvas.paste(panel, (i * size[0], 40))
        draw = ImageDraw.Draw(canvas)
        labels = []
        if has_gt:
            labels.append(f"coarse IoU={coarse_iou:.3f}")
            labels.append(f"refined IoU={refined_iou:.3f}")
            labels.append("GT")
        else:
            labels.append("coarse (no-object)")
            labels.append("refined (no-object)")
            labels.append("GT: empty")
        for i, label in enumerate(labels):
            draw.text((i * size[0] + 6, 8), label, fill=(0, 0, 0), font=font)
        query = record["query"]
        draw.text((6, size[1] + 6), f"[{record['control_kind']}] {query}", fill=(0, 0, 0), font=font)
        out_path = output_dir / f"{index:02d}_{record['sample_id'].replace('.', '_')}.png"
        canvas.save(out_path)
        print(f"saved {out_path.name} (control={record['control_kind']}, coarse_iou={coarse_iou}, refined_iou={refined_iou})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
