import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.image_pipeline import ImageSegmentationOutput, VILAImageSegmentationPipeline
from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2BuildOptions, SAM2ImageFeatureProvider
from llava.evo_seg.vila_adapter import VILASegmentationAdapter
from llava.model.fusion_observer import current_fusion_observer
from scripts.evo_seg.smoke_image_segmentation import _build_query_token_mask


class _FakeVILA(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, **_kwargs):
        self.calls += 1
        observer = current_fusion_observer()
        assert observer is not None
        observer.observe_fusion(
            [{"source_type": ["text", "image", "image", "text"], "source_position": [0, -1, -1, 2]}],
            padding_side="right",
            fused_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6) + self.offset
        return SimpleNamespace(hidden_states=(hidden, hidden + 1.0))


class _FakeSAM2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.sam_prompt_encoder = SimpleNamespace(mask_input_size=(4, 4))


class _FakePredictor:
    def __init__(self):
        self.model = _FakeSAM2Model()
        self._features = None
        self.images = []
        self.set_calls = 0
        self.predict_calls = 0
        self.reset_calls = 0

    def set_image_batch(self, images):
        self.set_calls += 1
        self.images = images
        values = torch.tensor([float(image[0, 0, 0]) for image in images])
        self._features = {"image_embed": values[:, None, None, None].expand(-1, 4, 2, 2).clone()}

    def predict_batch(self, mask_input_batch, multimask_output, return_logits):
        self.predict_calls += 1
        assert multimask_output is False
        assert return_logits is True
        outputs = []
        for mask_input, image in zip(mask_input_batch, self.images):
            assert isinstance(mask_input, np.ndarray)
            resized = F.interpolate(
                torch.from_numpy(mask_input).unsqueeze(0),
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            outputs.append(resized.numpy())
        return outputs, None, None

    def reset_predictor(self):
        self.reset_calls += 1
        self.images = []
        self._features = None


def _rgb_batch(frames=1):
    return RGBFrameBatch(
        frames=torch.full((1, frames, 3, 6, 5), 7, dtype=torch.uint8),
        frame_mask=torch.ones(1, frames, dtype=torch.bool),
        sample_ids=("sample",),
    )


def _provider():
    predictor = _FakePredictor()
    options = SAM2BuildOptions(
        source_root="/external/sam2",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint_path="/external/sam2.pt",
        device="cpu",
    )
    provider = SAM2ImageFeatureProvider(options, predictor_factory=lambda _options: predictor)
    return provider, predictor


def _pipeline():
    model = _FakeVILA()
    provider, predictor = _provider()
    decoder = QueryConditionedSpatialDecoder(
        query_dim=6,
        feature_dim=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        num_object_queries=1,
    )
    capability = SegmentationCapability(decoder)
    adapter = VILASegmentationAdapter(model, provider, capability)
    return VILAImageSegmentationPipeline(adapter, provider), model, predictor


def _inputs():
    input_ids = torch.tensor([[11, 99, 12]], dtype=torch.long)
    query_mask = torch.tensor([[True, False, True]], dtype=torch.bool)
    return input_ids, query_mask


def test_composed_image_pipeline_returns_coarse_and_refined_masks():
    pipeline, model, predictor = _pipeline()
    input_ids, query_mask = _inputs()
    output = pipeline.segment(
        input_ids=input_ids,
        media={"image": [torch.zeros(1)]},
        media_config={"image": {}},
        query_token_mask=query_mask,
        rgb_frames=_rgb_batch(),
        request={"enabled": True, "task": "image"},
    )
    assert isinstance(output, ImageSegmentationOutput)
    assert output.coarse_result.mask_logits.shape == (1, 1, 1, 2, 2)
    assert output.refined_mask_logits.shape == (1, 1, 1, 6, 5)
    assert output.diagnostics["execution_path"] == "vila_sam2_image_pipeline"
    assert set(("decoder", "sam2_mask_refinement_with_reencode", "image_pipeline_total")) <= set(
        output.diagnostics["component_timing_ms"]
    )
    assert model.calls == 1
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert predictor.set_calls == 2
    assert predictor.predict_calls == 1
    assert predictor._features is None
    pipeline.clear_request_state()
    assert not pipeline.adapter.capability.last_diagnostics


@pytest.mark.parametrize(
    ("request_value", "frames", "message"),
    [
        (None, 1, "disabled"),
        ({"enabled": True, "task": "video"}, 1, "task='image'"),
        ({"enabled": True, "task": "image"}, 2, "T=1"),
    ],
)
def test_pipeline_rejects_invalid_requests_before_vila_or_sam2(request_value, frames, message):
    pipeline, model, predictor = _pipeline()
    input_ids, query_mask = _inputs()
    with pytest.raises(ValueError, match=message):
        pipeline.segment(input_ids, {}, {}, query_mask, _rgb_batch(frames), request_value)
    assert model.calls == 0
    assert predictor.set_calls == 0


def test_pipeline_requires_one_provider_for_encoding_and_refinement():
    provider, _ = _provider()
    other_provider, _ = _provider()
    adapter = VILASegmentationAdapter(
        _FakeVILA(),
        provider,
        SegmentationCapability(
            QueryConditionedSpatialDecoder(6, 4, 8, 2, 1)
        ),
    )
    with pytest.raises(ValueError, match="same object"):
        VILAImageSegmentationPipeline(adapter, other_provider)


def test_importing_image_pipeline_does_not_import_external_sam2():
    command = (
        "import sys; import llava.evo_seg.image_pipeline; "
        "assert not any(name == 'sam2' or name.startswith('sam2.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


class _WordTokenizer:
    _vocabulary = {"bright": 21, "red": 22, "rectangle": 23}

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[self._vocabulary[word] for word in text.strip().split()])


def test_smoke_query_mask_selects_one_explicit_multi_token_span():
    input_ids = torch.tensor([[1, 8, 21, 22, 23, 9]], dtype=torch.long)
    mask = _build_query_token_mask(input_ids, _WordTokenizer(), "bright red rectangle")
    assert mask.tolist() == [[False, False, True, True, True, False]]
    with pytest.raises(ValueError, match="exactly one"):
        _build_query_token_mask(
            torch.tensor([[21, 22, 23, 21, 22, 23]], dtype=torch.long),
            _WordTokenizer(),
            "bright red rectangle",
        )
