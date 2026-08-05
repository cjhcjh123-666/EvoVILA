import subprocess
import sys
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.contracts import SegmentationResult
from llava.evo_seg.provenance import FusionCapture
from llava.evo_seg.vila_adapter import DenseFeatureBatch, VILASegmentationAdapter
from llava.model.fusion_observer import current_fusion_observer, fusion_observation
from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaTopDownMetaForCausalLM


class _TokenEmbedder:
    def __init__(self, dimension=4):
        self.dimension = dimension

    def __call__(self, input_ids):
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, self.dimension)


class _NativeHarness(LlavaMetaForCausalLM):
    def __init__(self, padding_side="right"):
        self.llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=_TokenEmbedder()))
        self.tokenizer = SimpleNamespace(
            media_token_ids={"image": 99, "video": 98},
            padding_side=padding_side,
            model_max_length=100,
        )
        self.training = False
        self.encoders = {}

    def _LlavaMetaForCausalLM__embed_media_tokens(self, media, media_config):
        embeds = defaultdict(deque)
        for name, values in media.items():
            embeds[name].extend(values)
        return embeds


class _TopDownHarness(LlavaTopDownMetaForCausalLM):
    def __init__(self, padding_side="right"):
        self.llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=_TokenEmbedder()))
        self.tokenizer = SimpleNamespace(
            media_token_ids={"image": 99, "video": 98},
            padding_side=padding_side,
            model_max_length=100,
        )
        self.training = False
        self.encoders = {}

    def _LlavaTopDownMetaForCausalLM__embed_media_tokens(
        self,
        media,
        media_config,
        image_num_each_sample=None,
        top_down_prompts=None,
        concat_low_high_res_features=False,
        smooth_selection_prob=False,
        num_look_close=None,
        gt_selection_maps=None,
        original_image_sizes=None,
    ):
        embeds = defaultdict(deque)
        for name, values in media.items():
            embeds[name].extend(values)
        return embeds, None, None


def _native_inputs():
    # The second sample contains two images and one video.  Media embeddings
    # are consumed globally in the same order as the native VILA path.
    input_ids = torch.tensor(
        [
            [1, 99, 2, 3, 0, 0],
            [4, 99, 5, 99, 98, 6],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]]
    )
    media = {
        "image": [torch.full((2, 4), 10.0), torch.full((1, 4), 20.0), torch.full((3, 4), 30.0)],
        "video": [torch.full((2, 4), 40.0)],
    }
    return input_ids, attention_mask, media


def test_native_embed_without_observer_preserves_baseline_contract():
    harness = _NativeHarness(padding_side="right")
    input_ids, attention_mask, media = _native_inputs()
    baseline = harness._embed(input_ids, media, {"image": {}, "video": {}}, None, attention_mask)
    assert len(baseline) == 3
    assert baseline[0].shape == (2, 9, 4)
    assert baseline[2].tolist() == [[True] * 5 + [False] * 4, [True] * 9]
    assert current_fusion_observer() is None


def test_native_embed_observer_maps_multi_image_video_and_padding():
    harness = _NativeHarness(padding_side="right")
    input_ids, attention_mask, media = _native_inputs()
    baseline = harness._embed(input_ids, media, {"image": {}, "video": {}}, None, attention_mask)
    capture = FusionCapture()
    with fusion_observation(capture):
        fused, _, fused_mask = harness._embed(input_ids, media, {"image": {}, "video": {}}, None, attention_mask)

    provenance = capture.provenance
    assert torch.equal(fused, baseline[0])
    assert torch.equal(fused_mask, baseline[2])
    assert fused.shape == (2, 9, 4)
    assert fused_mask.tolist() == [[True] * 5 + [False] * 4, [True] * 9]
    assert provenance.source_type[0][:5] == ("text", "image", "image", "text", "text")
    assert provenance.source_position[0, :5].tolist() == [0, -1, -1, 2, 3]
    assert provenance.source_type[1][:9] == (
        "text",
        "image",
        "text",
        "image",
        "image",
        "image",
        "video",
        "video",
        "text",
    )
    assert provenance.source_position[1, :9].tolist() == [0, -1, 2, -1, -1, -1, -1, -1, 5]

    hidden = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(2, 9, 4)
    query_mask = torch.tensor(
        [[True, False, True, False, False, False], [True, False, True, False, False, True]]
    )
    query = capture.gather_query_states(hidden, query_mask)
    assert query.states.shape == (2, 3, 4)
    assert query.source_positions.tolist() == [[0, 2, -1], [0, 2, 5]]
    assert query.fused_positions.tolist() == [[0, 3, -1], [0, 2, 8]]


def test_native_embed_observer_handles_left_padding_original_positions():
    harness = _NativeHarness(padding_side="left")
    input_ids = torch.tensor([[0, 0, 7, 99, 8], [0, 9, 99, 10, 11]], dtype=torch.long)
    attention_mask = torch.tensor([[False, False, True, True, True], [False, True, True, True, True]])
    media = {"image": [torch.full((2, 4), 10.0), torch.full((2, 4), 20.0)]}
    capture = FusionCapture()
    with fusion_observation(capture):
        _, _, fused_mask = harness._embed(input_ids, media, {"image": {}}, None, attention_mask)
    provenance = capture.provenance
    assert fused_mask.tolist() == [[False, True, True, True, True], [True, True, True, True, True]]
    assert provenance.source_position[0].tolist() == [-1, 2, -1, -1, 4]
    assert provenance.source_position[1].tolist() == [1, -1, -1, 3, 4]
    assert provenance.source_type[0] == ("padding", "text", "image", "image", "text")


def test_native_embed_truncates_provenance_with_fused_sequence():
    harness = _NativeHarness(padding_side="right")
    harness.training = True
    harness.tokenizer.model_max_length = 4
    input_ids = torch.tensor([[1, 99, 2, 3]], dtype=torch.long)
    media = {"image": [torch.full((3, 4), 10.0)]}
    capture = FusionCapture()
    with pytest.warns(UserWarning, match="Truncating sequences"):
        with fusion_observation(capture):
            fused, _, fused_mask = harness._embed(input_ids, media, {"image": {}}, None, None)
    assert fused.shape == (1, 4, 4)
    assert fused_mask.all()
    assert capture.provenance.source_type[0] == ("text", "image", "image", "image")
    truncated_query = torch.tensor([[False, False, True, False]])
    with pytest.raises(ValueError, match="did not map"):
        capture.gather_query_states(fused, truncated_query)


def test_top_down_embed_uses_same_provenance_observer_contract():
    harness = _TopDownHarness(padding_side="right")
    input_ids = torch.tensor([[1, 99, 2, 3]], dtype=torch.long)
    media = {"image": [torch.full((2, 4), 10.0)]}
    capture = FusionCapture()
    with fusion_observation(capture):
        result = harness._embed(
            input_ids,
            media,
            {"image": {}},
            None,
            None,
            instance_id_with_image=torch.tensor([0]),
        )
    assert len(result) == 5
    assert capture.provenance.source_type[0] == ("text", "image", "image", "text", "text")
    assert capture.provenance.source_position[0].tolist() == [0, -1, -1, 2, 3]


class _AdapterModel:
    def __init__(self, padding_side="right"):
        self.padding_side = padding_side
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        observer = current_fusion_observer()
        assert observer is not None
        rows = [
            {"source_type": ["text", "image", "image", "text"], "source_position": [0, -1, -1, 2]},
            {"source_type": ["text", "video", "video", "text", "text"], "source_position": [1, -1, -1, 3, 4]},
        ]
        mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
        observer.observe_fusion(rows, padding_side=self.padding_side, fused_attention_mask=mask)
        hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
        return SimpleNamespace(hidden_states=(hidden, hidden + 1.0))


class _Provider:
    def __init__(self):
        self.calls = []

    def encode(self, media, media_config, request):
        self.calls.append((media, media_config, request))
        return DenseFeatureBatch(
            features=torch.randn(2, 2, 4, 3, 3),
            frame_mask=torch.tensor([[True, True], [True, False]]),
            diagnostics={"provider": "fake"},
        )


class _AdapterDecoder(nn.Module):
    def forward(self, batch):
        return SegmentationResult(
            mask_logits=torch.zeros(batch.query_states.shape[0], 1, batch.dense_features.shape[1], 2, 2),
            object_logits=torch.zeros(batch.query_states.shape[0], 1),
            object_embeddings=torch.zeros(batch.query_states.shape[0], 1, 4),
            frame_embeddings=torch.zeros(batch.query_states.shape[0], 1, batch.dense_features.shape[1], 4),
            frame_mask=batch.frame_mask,
        )


def test_adapter_extracts_multi_token_states_without_grad_and_segments():
    model = _AdapterModel()
    provider = _Provider()
    adapter = VILASegmentationAdapter(model, provider, SegmentationCapability(_AdapterDecoder()))
    input_ids = torch.tensor([[10, 11, 99, 12, 13], [20, 21, 98, 22, 23]], dtype=torch.long)
    query_mask = torch.tensor(
        [[True, False, True, False, False], [False, True, False, True, True]], dtype=torch.bool
    )
    query = adapter.extract_query_states(input_ids, {}, {}, query_mask)
    assert query.states.shape == (2, 3, 4)
    assert not query.states.requires_grad
    assert len(model.calls) == 1
    assert model.calls[0]["packing"] is False
    result = adapter.segment(input_ids, {}, {}, query_mask, {"enabled": True, "task": "video"})
    assert result.diagnostics["execution_path"] == "vila_evo_seg_adapter"
    timings = result.diagnostics["component_timing_ms"]
    assert set(("vila_query_encoding", "dense_provider", "decoder", "total")) <= set(timings)
    assert result.diagnostics["dense_provider"]["provider"] == "fake"
    assert provider.calls[-1][2].task == "video"


def test_adapter_keeps_vila_media_separate_from_dense_input():
    model = _AdapterModel()
    provider = _Provider()
    adapter = VILASegmentationAdapter(model, provider, SegmentationCapability(_AdapterDecoder()))
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.long)
    query_mask = torch.tensor(
        [[True, False, True, False, False], [False, True, False, True, True]], dtype=torch.bool
    )
    raw_dense_input = object()
    adapter.segment(
        input_ids,
        {"image": "vila-preprocessed"},
        {},
        query_mask,
        {"enabled": True, "task": "video"},
        dense_input=raw_dense_input,
    )
    assert model.calls[-1]["media"] == {"image": "vila-preprocessed"}
    assert provider.calls[-1][0] is raw_dense_input


def test_adapter_disabled_request_does_not_call_model_or_provider():
    model = _AdapterModel()
    provider = _Provider()
    adapter = VILASegmentationAdapter(model, provider, SegmentationCapability(_AdapterDecoder()))
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    query_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with pytest.raises(ValueError, match="disabled"):
        adapter.segment(input_ids, {}, {}, query_mask, None)
    assert not model.calls
    assert not provider.calls


def test_adapter_rejects_multi_frame_image_request():
    model = _AdapterModel()

    def provider(*_args):
        return DenseFeatureBatch(torch.zeros(2, 2, 4, 2, 2), torch.ones(2, 2, dtype=torch.bool))

    adapter = VILASegmentationAdapter(model, provider, SegmentationCapability(_AdapterDecoder()))
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.long)
    query_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with pytest.raises(ValueError, match="exactly one"):
        adapter.segment(input_ids, {}, {}, query_mask, {"enabled": True, "task": "image"})


def test_default_import_path_has_no_evo_seg_or_sam2_import():
    command = (
        "import sys; import llava.model.llava_arch; "
        "assert 'llava.evo_seg' not in sys.modules; "
        "assert not any(name.lower().startswith('sam2') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
