import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import llava.evo_seg.sam2_video_adapter as sam2_video_module
from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2BuildOptions, SAM2ImageFeatureProvider
from llava.evo_seg.sam2_video_adapter import (
    SAM2VideoMaskPropagator,
    VideoPropagationResult,
    build_sam2_video_image_predictor,
)
from llava.evo_seg.video_pipeline import VideoSegmentationOutput, VILAVideoSegmentationPipeline
from llava.evo_seg.vila_adapter import VILASegmentationAdapter
from llava.model.fusion_observer import current_fusion_observer


class _FakeVILA(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        observer = current_fusion_observer()
        assert observer is not None
        rows = [
            {"source_type": ["text", "video", "video", "text"], "source_position": [0, -1, -1, 2]}
            for _ in range(batch_size)
        ]
        observer.observe_fusion(
            rows,
            padding_side="right",
            fused_attention_mask=torch.ones(batch_size, 4, dtype=torch.bool),
        )
        hidden = torch.arange(batch_size * 4 * 6, dtype=torch.float32).reshape(batch_size, 4, 6)
        return SimpleNamespace(hidden_states=(hidden + self.offset, hidden + self.offset + 1.0))


class _FakeVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.states = []
        self.paths = []
        self.prompts = []
        self.reset_calls = 0
        self.fail_propagation = False

    def init_state(self, video_path, **options):
        paths = sorted(Path(video_path).glob("*.jpg"))
        assert paths
        with Image.open(paths[0]) as image:
            width, height = image.size
        state = {
            "num_frames": len(paths),
            "height": height,
            "width": width,
            "obj_ids": [],
            "options": options,
        }
        self.states.append(state)
        self.paths.append(Path(video_path))
        return state

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        assert mask.dtype == torch.bool
        inference_state["obj_ids"].append(obj_id)
        self.prompts.append((frame_idx, obj_id, mask.clone()))
        return frame_idx, tuple(inference_state["obj_ids"]), None

    def propagate_in_video(self, inference_state, start_frame_idx, reverse):
        if reverse:
            indices = range(start_frame_idx, -1, -1)
        else:
            indices = range(start_frame_idx, inference_state["num_frames"])
        object_ids = tuple(reversed(inference_state["obj_ids"]))
        for frame_index in indices:
            if self.fail_propagation:
                raise RuntimeError("synthetic propagation failure")
            masks = torch.stack(
                [
                    torch.full(
                        (1, inference_state["height"], inference_state["width"]),
                        object_id * 10.0 + frame_index,
                    )
                    for object_id in object_ids
                ]
            )
            yield frame_index, object_ids, masks

    def reset_state(self, inference_state):
        assert inference_state
        self.reset_calls += 1


class _FakePredictor:
    def __init__(self, model=None):
        self.model = model or _FakeVideoModel()
        self._features = None
        self.set_calls = 0
        self.reset_calls = 0

    def set_image_batch(self, images):
        self.set_calls += 1
        values = torch.tensor([float(image[0, 0, 0]) for image in images])
        self._features = {"image_embed": values[:, None, None, None].expand(-1, 4, 2, 2).clone()}

    def get_image_embedding(self):
        return self._features["image_embed"]

    def predict_batch(self, **_kwargs):
        raise AssertionError("video path must not call the image mask refiner")

    def reset_predictor(self):
        self.reset_calls += 1
        self._features = None


def _provider(model=None):
    predictor = _FakePredictor(model)
    options = SAM2BuildOptions(
        source_root="/external/sam2",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint_path="/external/sam2.pt",
        device="cpu",
    )
    provider = SAM2ImageFeatureProvider(options, predictor_factory=lambda _options: predictor)
    return provider, predictor


def _rgb_batch(frames=3, frame_mask=None, batch_size=1):
    values = torch.arange(batch_size * frames, dtype=torch.uint8).reshape(batch_size, frames, 1, 1, 1)
    rgb = values.expand(batch_size, frames, 3, 6, 5).contiguous()
    if frame_mask is None:
        frame_mask = torch.ones(batch_size, frames, dtype=torch.bool)
    return RGBFrameBatch(
        frames=rgb,
        frame_mask=frame_mask,
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
    )


def _pipeline():
    model = _FakeVILA()
    provider, predictor = _provider()
    decoder = QueryConditionedSpatialDecoder(
        query_dim=6,
        feature_dim=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        num_object_queries=2,
    )
    adapter = VILASegmentationAdapter(model, provider, SegmentationCapability(decoder))
    propagator = SAM2VideoMaskPropagator(provider, {"jpeg_quality": 100})
    return VILAVideoSegmentationPipeline(adapter, propagator), model, predictor


def _inputs(batch_size=1):
    input_ids = torch.tensor([[11, 99, 12]], dtype=torch.long).expand(batch_size, -1).clone()
    query_mask = torch.tensor([[True, False, True]], dtype=torch.bool).expand(batch_size, -1).clone()
    return input_ids, query_mask


def test_composed_video_pipeline_uses_one_predicted_anchor_and_propagates_objects():
    pipeline, vila, predictor = _pipeline()
    input_ids, query_mask = _inputs()
    output = pipeline.segment(
        input_ids=input_ids,
        media={"video": [torch.zeros(3, 1)]},
        media_config={"video": {}},
        query_token_mask=query_mask,
        rgb_frames=_rgb_batch(),
        request={"enabled": True, "task": "video"},
    )
    assert isinstance(output, VideoSegmentationOutput)
    assert output.coarse_anchor_result.mask_logits.shape == (1, 2, 1, 2, 2)
    propagation = output.propagation_result
    assert isinstance(propagation, VideoPropagationResult)
    assert propagation.mask_logits.shape == (1, 2, 3, 6, 5)
    assert propagation.anchor_indices.tolist() == [0]
    assert torch.all(propagation.mask_logits[0, 0, 2] == 12.0)
    assert torch.all(propagation.mask_logits[0, 1, 2] == 22.0)
    assert output.diagnostics["anchor_policy"] == "first_valid"
    assert output.diagnostics["execution_path"] == "vila_sam2_predicted_anchor_video_pipeline"
    assert vila.calls == 1
    assert predictor.set_calls == 1
    assert len(predictor.model.prompts) == 2
    assert all(prompt[2].dtype == torch.bool for prompt in predictor.model.prompts)
    assert predictor.model.reset_calls == 1
    assert predictor.model.states == [{}]
    assert not predictor.model.paths[0].exists()
    pipeline.clear_request_state()
    assert not pipeline.adapter.capability.last_diagnostics
    assert predictor._features is None


def test_propagator_restores_padded_source_frames_and_explicit_middle_anchors():
    provider, predictor = _provider()
    propagator = SAM2VideoMaskPropagator(provider)
    frame_mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    batch = _rgb_batch(frames=4, frame_mask=frame_mask, batch_size=2)
    anchors = torch.tensor([2, 1], dtype=torch.long)
    logits = torch.ones(2, 2, 2, 2)
    result = propagator.propagate(batch, anchors, logits)
    assert result.mask_logits.shape == (2, 2, 4, 6, 5)
    assert result.anchor_indices.tolist() == [2, 1]
    assert result.diagnostics["compact_anchor_indices"] == (1, 0)
    assert torch.count_nonzero(result.mask_logits[0, :, 1]) == 0
    assert torch.count_nonzero(result.mask_logits[1, :, 0]) == 0
    assert torch.count_nonzero(result.mask_logits[1, :, 3]) == 0
    assert torch.all(result.mask_logits[0, 0, 0] == 10.0)
    assert torch.all(result.mask_logits[0, 0, 3] == 12.0)
    assert predictor.model.reset_calls == 2
    assert predictor.model.states == [{}, {}]
    assert all(not path.exists() for path in predictor.model.paths)


def test_propagator_clears_state_and_temporary_frames_after_failure():
    model = _FakeVideoModel()
    model.fail_propagation = True
    provider, predictor = _provider(model)
    propagator = SAM2VideoMaskPropagator(provider)
    with pytest.raises(RuntimeError, match="synthetic propagation failure"):
        propagator.propagate(
            _rgb_batch(),
            torch.tensor([0], dtype=torch.long),
            torch.ones(1, 1, 2, 2),
        )
    assert model.reset_calls == 1
    assert model.states == [{}]
    assert all(not path.exists() for path in model.paths)
    assert predictor._features is None


@pytest.mark.parametrize(
    ("request_value", "batch", "message"),
    [
        (None, _rgb_batch(), "disabled"),
        ({"enabled": True, "task": "image"}, _rgb_batch(), "task='video'"),
        ({"enabled": True, "task": "video"}, _rgb_batch(frames=1), "at least two"),
        (
            {"enabled": True, "task": "video"},
            _rgb_batch(frames=2, frame_mask=torch.tensor([[True, False]])),
            "at least two",
        ),
    ],
)
def test_video_pipeline_rejects_invalid_requests_before_components(request_value, batch, message):
    pipeline, vila, predictor = _pipeline()
    input_ids, query_mask = _inputs()
    with pytest.raises(ValueError, match=message):
        pipeline.segment(input_ids, {}, {}, query_mask, batch, request_value)
    assert vila.calls == 0
    assert predictor.set_calls == 0
    assert predictor.model.states == []


def test_video_pipeline_rejects_invalid_anchor_before_components():
    pipeline, vila, predictor = _pipeline()
    input_ids, query_mask = _inputs()
    batch = _rgb_batch(frames=3, frame_mask=torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="valid frame"):
        pipeline.segment(
            input_ids,
            {},
            {},
            query_mask,
            batch,
            {"enabled": True, "task": "video"},
            anchor_indices=torch.tensor([1], dtype=torch.long),
        )
    assert vila.calls == 0
    assert predictor.set_calls == 0


def test_video_pipeline_requires_shared_dense_provider_and_propagator():
    provider, _ = _provider()
    other_provider, _ = _provider()
    adapter = VILASegmentationAdapter(
        _FakeVILA(),
        provider,
        SegmentationCapability(QueryConditionedSpatialDecoder(6, 4, 8, 2, 1)),
    )
    with pytest.raises(ValueError, match="share one provider"):
        VILAVideoSegmentationPipeline(adapter, SAM2VideoMaskPropagator(other_provider))


def test_video_builder_wraps_the_same_frozen_model(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    model = _FakeVideoModel()
    calls = []

    def fake_build_video(**kwargs):
        calls.append(kwargs)
        return model

    class FakeImagePredictor:
        def __init__(self, wrapped_model):
            self.model = wrapped_model

    modules = {
        "sam2.build_sam": SimpleNamespace(build_sam2_video_predictor=fake_build_video),
        "sam2.sam2_image_predictor": SimpleNamespace(SAM2ImagePredictor=FakeImagePredictor),
    }
    monkeypatch.setattr(sam2_video_module, "_resolve_sam2_package", lambda _options: tmp_path)
    monkeypatch.setattr(sam2_video_module.importlib, "import_module", lambda name: modules[name])
    predictor = build_sam2_video_image_predictor(
        SAM2BuildOptions(
            model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
            checkpoint_path=str(checkpoint),
            device="cpu",
        )
    )
    assert predictor.model is model
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert calls[0]["ckpt_path"] == str(checkpoint)


def test_importing_video_pipeline_does_not_import_external_sam2():
    command = (
        "import sys; import llava.evo_seg.video_pipeline; "
        "assert not any(name == 'sam2' or name.startswith('sam2.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
