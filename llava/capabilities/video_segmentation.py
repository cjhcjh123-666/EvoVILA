"""Optional SAM2-backed video segmentation capability.

SAM2 is deliberately imported only when ``segment`` is called. Importing this
module, building the capability registry, and loading an ordinary VILA model
therefore remain independent of the external SAM2 installation.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import torch

from .base import Capability, CapabilityError


DEFAULT_SAM2_ROOT = "/9950backfile/zhangyafei/sam2"
DEFAULT_SAM2_CHECKPOINT = os.path.join(DEFAULT_SAM2_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@dataclass(frozen=True)
class VideoSegmentationResult:
    """Masks and reproducibility metadata from one video request."""

    masks: torch.Tensor
    object_ids: Tuple[Any, ...]
    anchor_frame: int
    video_size: Tuple[int, int]
    instrumentation: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_ids", tuple(self.object_ids))
        object.__setattr__(self, "video_size", tuple(self.video_size))
        object.__setattr__(self, "instrumentation", MappingProxyType(dict(self.instrumentation)))


def _load_sam2_builder():
    """Load the official builder only for an explicit video request."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ModuleNotFoundError as exc:
        raise CapabilityError(
            "video_segmentation requires SAM2. Install the local SAM2 package "
            "or set PYTHONPATH=/9950backfile/zhangyafei/sam2 before calling "
            "segment_videos()."
        ) from exc
    except ImportError as exc:
        raise CapabilityError(f"SAM2 could not be imported for video_segmentation: {exc}") from exc
    return build_sam2_video_predictor


def _synchronize(device: Any) -> None:
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _scalar_object_id(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    return value


def _normalise_points(points: Any, labels: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (points is None) != (labels is None):
        raise CapabilityError("video_segmentation points and labels must be provided together")
    if points is None:
        return None, None
    points_tensor = torch.as_tensor(points, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.int32)
    if points_tensor.ndim == 1:
        if points_tensor.shape[0] != 2:
            raise CapabilityError("video_segmentation points must be shaped [2] or [K, 2]")
        points_tensor = points_tensor.unsqueeze(0)
    if points_tensor.ndim != 2 or points_tensor.shape[-1] != 2:
        raise CapabilityError("video_segmentation points must be shaped [2] or [K, 2]")
    if labels_tensor.ndim == 0:
        labels_tensor = labels_tensor.unsqueeze(0)
    if labels_tensor.ndim != 1 or labels_tensor.shape[0] != points_tensor.shape[0]:
        raise CapabilityError("video_segmentation labels must have one value per point")
    if not bool(torch.all((labels_tensor == 0) | (labels_tensor == 1))):
        raise CapabilityError("video_segmentation point labels must be 0 or 1")
    return points_tensor, labels_tensor


def _normalise_box(box: Any) -> Optional[torch.Tensor]:
    if box is None:
        return None
    box_tensor = torch.as_tensor(box, dtype=torch.float32)
    if box_tensor.numel() != 4:
        raise CapabilityError("video_segmentation box must contain [x0, y0, x1, y1]")
    return box_tensor.reshape(2, 2)


def _normalise_prompt(
    object_id: Any, prompt: Mapping[str, Any]
) -> Tuple[Any, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not isinstance(prompt, Mapping):
        raise CapabilityError(f"Prompt for object {object_id!r} must be a mapping")
    points, labels = _normalise_points(prompt.get("points"), prompt.get("labels"))
    box = _normalise_box(prompt.get("box"))
    if points is None and box is None:
        raise CapabilityError(f"Prompt for object {object_id!r} needs points or box")
    return object_id, points, labels, box


def _normalise_prompts(
    *,
    object_id: Any,
    points: Any,
    labels: Any,
    box: Any,
    prompts: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Tuple[Tuple[Any, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]], ...]:
    if prompts is not None:
        if points is not None or labels is not None or box is not None:
            raise CapabilityError("Use either prompts or the single-object points/labels/box arguments")
        if not isinstance(prompts, Mapping) or not prompts:
            raise CapabilityError("video_segmentation prompts must be a non-empty mapping")
        normalised = tuple(_normalise_prompt(obj_id, prompt) for obj_id, prompt in prompts.items())
    else:
        normalised = (_normalise_prompt(object_id, {"points": points, "labels": labels, "box": box}),)
    object_ids = [prompt[0] for prompt in normalised]
    if len(set(object_ids)) != len(object_ids):
        raise CapabilityError("video_segmentation object IDs must be unique")
    return normalised


def _normalise_output_masks(mask_logits: Any) -> torch.Tensor:
    masks = torch.as_tensor(mask_logits)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise CapabilityError(
            "SAM2 returned masks with an unsupported shape; expected [N, H, W] or [N, 1, H, W]"
        )
    return masks.detach().to(device="cpu") > 0


class VideoSegmentationCapability(Capability):
    """Run SAM2 video propagation from explicit anchor-frame prompts."""

    name = "video_segmentation"

    def __init__(self, config: Any = None, options: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(config=config, options=options)
        self._last_result: Optional[VideoSegmentationResult] = None

    def _sam2_root(self) -> Path:
        configured_root = self.options.get("sam2_root") or os.environ.get("EVOVILA_SAM2_ROOT") or DEFAULT_SAM2_ROOT
        return Path(configured_root).expanduser()

    def _resolve_config_name(self, config_file: Any, sam2_root: Path) -> str:
        config_path = Path(str(config_file)).expanduser()
        package_root = sam2_root / "sam2"
        if not config_path.is_absolute():
            candidates = (package_root / config_path, sam2_root / config_path)
            for candidate in candidates:
                if candidate.is_file():
                    try:
                        return candidate.relative_to(package_root).as_posix()
                    except ValueError:
                        break
            raise CapabilityError(
                f"SAM2 config was not found: {config_file}. Set options['sam2_root'] to the SAM2 source root."
            )
        if not config_path.is_file():
            raise CapabilityError(f"SAM2 config was not found: {config_path}")
        try:
            return config_path.relative_to(package_root).as_posix()
        except ValueError as exc:
            raise CapabilityError(
                f"SAM2 config {config_path} must be inside {package_root} so Hydra can compose it"
            ) from exc

    def _build_predictor(self, device: Any):
        # Import first so an absent optional dependency has a clear error even
        # when its configured checkpoint path is also unavailable.
        build_predictor = _load_sam2_builder()
        sam2_root = self._sam2_root()
        checkpoint = Path(
            str(self.options.get("checkpoint", self.options.get("checkpoint_path", DEFAULT_SAM2_CHECKPOINT)))
        ).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = sam2_root / checkpoint
        if not checkpoint.is_file():
            raise CapabilityError(f"SAM2 checkpoint was not found: {checkpoint}")
        config_file = self.options.get("config", self.options.get("config_file", DEFAULT_SAM2_CONFIG))
        config_name = self._resolve_config_name(config_file, sam2_root)
        predictor_options = self.options.get("predictor_kwargs", {})
        if not isinstance(predictor_options, Mapping):
            raise CapabilityError("video_segmentation predictor_kwargs must be a mapping")
        try:
            return build_predictor(
                config_name,
                str(checkpoint),
                device=device,
                vos_optimized=bool(self.options.get("vos_optimized", False)),
                **dict(predictor_options),
            )
        except Exception as exc:
            raise CapabilityError(f"SAM2 video predictor could not be built: {exc}") from exc

    def segment(
        self,
        video: Any,
        *,
        anchor_frame: int = 0,
        object_id: Any = 1,
        points: Any = None,
        labels: Any = None,
        box: Any = None,
        prompts: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        return_result: bool = False,
    ) -> Any:
        self._last_result = None
        video_path = Path(video).expanduser() if isinstance(video, (str, os.PathLike)) else None
        if video_path is None or not video_path.exists():
            raise CapabilityError("video_segmentation video must be an existing frame directory or MP4 path")
        if not video_path.is_dir() and video_path.suffix.lower() != ".mp4":
            raise CapabilityError("video_segmentation accepts a frame directory or an MP4 path")
        if isinstance(anchor_frame, bool) or not isinstance(anchor_frame, int) or anchor_frame < 0:
            raise CapabilityError("video_segmentation anchor_frame must be a non-negative integer")

        normalised_prompts = _normalise_prompts(
            object_id=object_id,
            points=points,
            labels=labels,
            box=box,
            prompts=prompts,
        )
        device = self.options.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise CapabilityError("video_segmentation requested CUDA but torch.cuda.is_available() is false")

        total_start = time.perf_counter()
        _synchronize(device)
        predictor_start = time.perf_counter()
        predictor = self._build_predictor(device)
        _synchronize(device)
        predictor_latency = time.perf_counter() - predictor_start

        init_options = self.options.get("init_state_kwargs", {})
        if not isinstance(init_options, Mapping):
            raise CapabilityError("video_segmentation init_state_kwargs must be a mapping")
        state_start = time.perf_counter()
        inference_state = predictor.init_state(video_path=str(video_path), **dict(init_options))
        _synchronize(device)
        state_latency = time.perf_counter() - state_start
        num_frames = int(inference_state.get("num_frames", 0))
        if num_frames <= 0:
            raise CapabilityError("SAM2 loaded no video frames")
        if anchor_frame >= num_frames:
            raise CapabilityError(f"anchor_frame {anchor_frame} is outside the {num_frames}-frame video")

        anchor_start = time.perf_counter()
        for current_object_id, current_points, current_labels, current_box in normalised_prompts:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=anchor_frame,
                obj_id=current_object_id,
                points=current_points,
                labels=current_labels,
                box=current_box,
            )
        _synchronize(device)
        anchor_latency = time.perf_counter() - anchor_start

        propagation_start = time.perf_counter()
        frame_masks = {}
        object_ids = tuple(prompt[0] for prompt in normalised_prompts)
        for frame_idx, output_object_ids, output_mask_logits in predictor.propagate_in_video(inference_state):
            masks = _normalise_output_masks(output_mask_logits)
            output_object_ids = tuple(_scalar_object_id(value) for value in output_object_ids)
            if masks.shape[0] != len(output_object_ids):
                raise CapabilityError("SAM2 returned a different number of object IDs and masks")
            frame_masks[int(frame_idx)] = {
                object_id: masks[index] for index, object_id in enumerate(output_object_ids)
            }
        _synchronize(device)
        propagation_latency = time.perf_counter() - propagation_start

        if set(frame_masks) != set(range(num_frames)):
            raise CapabilityError(
                f"SAM2 did not return exactly the expected {num_frames} frame indices; "
                "check the video input and predictor state"
            )
        first_frame = next(iter(frame_masks.values()))
        if any(object_id not in first_frame for object_id in object_ids):
            raise CapabilityError("SAM2 propagation did not return every requested object")
        height, width = tuple(first_frame[object_ids[0]].shape[-2:])
        masks = torch.zeros((num_frames, len(object_ids), height, width), dtype=torch.bool)
        for frame_idx in range(num_frames):
            current_frame = frame_masks[frame_idx]
            for object_index, current_object_id in enumerate(object_ids):
                current_mask = current_frame.get(current_object_id)
                if current_mask is None:
                    raise CapabilityError(
                        f"SAM2 propagation omitted object {current_object_id!r} on frame {frame_idx}"
                    )
                if tuple(current_mask.shape[-2:]) != (height, width):
                    raise CapabilityError("SAM2 returned inconsistent mask resolutions across frames")
                masks[frame_idx, object_index] = current_mask

        total_latency = time.perf_counter() - total_start
        output_masks = masks[:, 0] if len(object_ids) == 1 else masks
        result = VideoSegmentationResult(
            masks=output_masks,
            object_ids=object_ids,
            anchor_frame=anchor_frame,
            video_size=(
                int(inference_state.get("video_height", height)),
                int(inference_state.get("video_width", width)),
            ),
            instrumentation={
                "predictor_latency_s": predictor_latency,
                "state_init_latency_s": state_latency,
                "anchor_latency_s": anchor_latency,
                "propagation_latency_s": propagation_latency,
                "total_latency_s": total_latency,
                "num_frames": num_frames,
                "num_objects": len(object_ids),
                "anchor_frame": anchor_frame,
            },
        )
        self._last_result = result
        return result if return_result else result.masks

    def get_outputs(self) -> Mapping[str, Any]:
        if self._last_result is None:
            return {}
        return {
            "video_masks": self._last_result.masks,
            "video_segmentation_result": self._last_result,
            "video_segmentation_instrumentation": self._last_result.instrumentation,
        }

    def clear(self) -> None:
        self._last_result = None


__all__ = ["VideoSegmentationCapability", "VideoSegmentationResult"]
