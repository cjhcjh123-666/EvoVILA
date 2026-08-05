import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2BuildOptions, SAM2ImageFeatureProvider


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
        assert all(isinstance(image, np.ndarray) for image in images)
        assert all(image.dtype == np.uint8 and image.shape[-1] == 3 for image in images)
        values = torch.tensor([float(image[0, 0, 0]) for image in images])
        self._features = {"image_embed": values[:, None, None, None].expand(-1, 3, 2, 2).clone()}

    def predict_batch(self, mask_input_batch, multimask_output, return_logits):
        self.predict_calls += 1
        assert multimask_output is False
        assert return_logits is True
        outputs = []
        for mask_input, image in zip(mask_input_batch, self.images):
            assert mask_input.shape[-2:] == (4, 4)
            resized = F.interpolate(
                torch.from_numpy(mask_input).unsqueeze(0),
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            outputs.append(resized.numpy())
        return outputs, [np.ones(len(item)) for item in outputs], mask_input_batch

    def reset_predictor(self):
        self.reset_calls += 1
        self.images = []
        self._features = None


def _frames(frames=2):
    pixels = torch.zeros(2, frames, 3, 6, 5, dtype=torch.uint8)
    pixels[0, 0] = 1
    if frames > 1:
        pixels[0, 1] = 99
        pixels[1, 1] = 3
    pixels[1, 0] = 2
    frame_mask = torch.tensor([[True] + [False] * (frames - 1), [True] * frames])
    return RGBFrameBatch(pixels, frame_mask, sample_ids=("a", "b"))


def _options():
    return SAM2BuildOptions(
        source_root="/external/sam2",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint_path="/external/checkpoints/sam2.1_hiera_tiny.pt",
        device="cpu",
    )


def test_rgb_frame_contract_rejects_ambiguous_preprocessed_input():
    with pytest.raises(TypeError, match="uint8"):
        RGBFrameBatch(torch.zeros(1, 1, 3, 4, 4), torch.ones(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        RGBFrameBatch(
            torch.zeros(1, 1, 3, 4, 4, dtype=torch.uint8),
            torch.zeros(1, 1, dtype=torch.bool),
        )


def test_provider_is_lazy_freezes_model_and_reconstructs_padding():
    predictor = _FakePredictor()
    factory_calls = []

    def factory(options):
        factory_calls.append(options)
        return predictor

    provider = SAM2ImageFeatureProvider(_options(), predictor_factory=factory)
    assert not provider.initialized
    batch = _frames()
    dense = provider.encode_frames(batch)
    assert provider.initialized
    assert len(factory_calls) == 1
    assert dense.features.shape == (2, 2, 3, 2, 2)
    assert torch.equal(dense.frame_mask, batch.frame_mask)
    assert torch.all(dense.features[0, 0] == 1)
    assert torch.all(dense.features[0, 1] == 0)
    assert torch.all(dense.features[1, 0] == 2)
    assert torch.all(dense.features[1, 1] == 3)
    assert not predictor.model.training
    assert not any(parameter.requires_grad for parameter in predictor.model.parameters())
    assert predictor.reset_calls == 1
    assert predictor._features is None
    assert dense.diagnostics["preprocessing"] == "SAM2ImagePredictor.set_image_batch"


def test_provider_refines_anchor_logits_and_zeros_invalid_frames():
    predictor = _FakePredictor()
    provider = SAM2ImageFeatureProvider(_options(), predictor_factory=lambda _options: predictor)
    batch = _frames()
    coarse = torch.arange(2 * 2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2, 2)
    refined = provider.refine_masks(batch, coarse)
    assert refined.shape == (2, 2, 2, 6, 5)
    assert torch.all(refined[0, :, 1] == 0)
    assert not torch.all(refined[0, :, 0] == 0)
    assert predictor.predict_calls == 1
    assert predictor.reset_calls == 1
    assert predictor._features is None


def test_provider_protocol_fails_closed_before_initialization():
    predictor = _FakePredictor()
    provider = SAM2ImageFeatureProvider(_options(), predictor_factory=lambda _options: predictor)
    batch = _frames(frames=1)
    with pytest.raises(ValueError, match="disabled"):
        provider.encode(batch, None, None)
    assert not provider.initialized
    dense = provider.encode(batch, None, {"enabled": True, "task": "image"})
    assert dense.features.shape[:2] == (2, 1)


def test_image_request_rejects_multiple_raw_frames_without_building_sam2():
    calls = []
    provider = SAM2ImageFeatureProvider(_options(), predictor_factory=lambda options: calls.append(options))
    with pytest.raises(ValueError, match="T=1"):
        provider.encode(_frames(frames=2), None, {"enabled": True, "task": "image"})
    assert not calls
    assert not provider.initialized


def test_default_builder_rejects_missing_local_source_without_importing_sam2():
    provider = SAM2ImageFeatureProvider(_options())
    with pytest.raises(FileNotFoundError, match="source_root"):
        provider.encode_frames(_frames(frames=1))
    assert "sam2" not in sys.modules


def test_importing_optional_adapter_does_not_import_sam2():
    command = (
        "import sys; import llava.evo_seg.sam2_adapter; "
        "assert not any(name == 'sam2' or name.startswith('sam2.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
