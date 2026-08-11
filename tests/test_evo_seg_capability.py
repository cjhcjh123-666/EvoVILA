import sys
import subprocess

import pytest
import torch
from torch import nn

import llava.evo_seg as evo_seg
from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.contracts import GroundingBatch, SegmentationResult


def make_batch():
    return GroundingBatch(
        query_states=torch.randn(1, 3, 8),
        query_mask=torch.ones(1, 3, dtype=torch.bool),
        dense_features=torch.randn(1, 1, 4, 4, 4),
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
    )


class CountingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return SegmentationResult(
            mask_logits=torch.zeros(1, 1, 1, 2, 2),
            object_logits=torch.zeros(1, 1),
            object_embeddings=torch.zeros(1, 1, 3),
            frame_embeddings=torch.zeros(1, 1, 1, 3),
            frame_mask=batch.frame_mask,
        )


def test_disabled_requests_never_call_decoder():
    decoder = CountingDecoder()
    capability = SegmentationCapability(decoder)
    assert not capability.is_enabled(None)
    assert not capability.is_enabled({"enabled": False})
    with pytest.raises(ValueError, match="disabled"):
        capability(make_batch(), None)
    assert decoder.calls == 0


def test_enabled_request_calls_decoder_and_records_timing():
    decoder = CountingDecoder()
    capability = SegmentationCapability(decoder)
    result = capability(make_batch(), {"enabled": True, "task": "image"})
    assert decoder.calls == 1
    assert result.diagnostics["execution_path"] == "evo_seg_decoder"
    assert result.diagnostics["task"] == "image"
    assert result.diagnostics["component_timing_ms"]["decoder"] >= 0
    assert capability.last_diagnostics["component_timing_ms"]["total"] >= 0


def test_request_diagnostics_are_cleared_without_retaining_media_state():
    capability = SegmentationCapability(CountingDecoder())
    capability(make_batch(), {"enabled": True, "task": "video"})
    assert capability.last_diagnostics
    capability.clear_request_state()
    assert not capability.last_diagnostics


def test_public_import_has_no_sam2_side_effect():
    assert evo_seg.SegmentationCapability is SegmentationCapability
    command = (
        "import sys; import llava.evo_seg; "
        "assert 'llava.model' not in sys.modules; "
        "assert not any(name.lower().startswith('sam2') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_optional_providers_are_registered_modules():
    provider = nn.Linear(2, 2)
    capability = SegmentationCapability(CountingDecoder(), providers={"dense": provider})
    assert capability.providers["dense"] is provider
