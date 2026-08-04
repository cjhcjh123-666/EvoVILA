from dataclasses import dataclass

import pytest
from torch import nn

from llava.capabilities import (
    Capability,
    CapabilityError,
    CapabilityRegistry,
    CapabilityRequest,
    MediaContext,
    build_capability_pipeline,
)


@dataclass(frozen=True)
class FakeMedia:
    shape: tuple


class Recorder(Capability):
    name = "recorder"

    def __init__(self, config=None, options=None):
        super().__init__(config=config, options=options)
        self.contexts = []

    def on_media_context(self, context):
        self.contexts.append(context)
        return context.stage


def test_default_pipeline_is_noop_for_text_only_input():
    pipeline = build_capability_pipeline()
    input_ids = (101, 202, 303)
    context = MediaContext.from_inputs(input_ids, {}, {})

    assert pipeline.enabled_names == ()
    assert pipeline.on_media_context(context) == ()
    assert context.media_count() == 0
    assert context.input_ids == input_ids


def test_image_context_is_explicit_and_container_read_only():
    registry = CapabilityRegistry()
    recorders = []

    def make_recorder(**kwargs):
        recorder = Recorder(**kwargs)
        recorders.append(recorder)
        return recorder

    registry.register("recorder", make_recorder)
    pipeline = build_capability_pipeline(
        {"names": ["recorder"], "options": {"recorder": {"mode": "audit"}}},
        registry=registry,
    )
    input_ids = (101, 32000, 303)
    media = {"image": [FakeMedia((1, 3, 224, 224))]}
    media_config = {"image": {"block_sizes": [None]}}
    context = MediaContext.from_inputs(input_ids, media, media_config)

    assert pipeline.enabled_names == ("recorder",)
    assert pipeline.on_media_context(context) == ("embed",)
    assert context.media_count("image") == 1
    assert context.media["image"][0].shape == (1, 3, 224, 224)
    assert recorders[0].options["mode"] == "audit"
    assert input_ids == (101, 32000, 303)
    with pytest.raises(TypeError):
        context.media["image"] = ()


def test_video_context_preserves_frame_payload_and_token_alignment():
    registry = CapabilityRegistry()
    recorders = []

    def make_recorder(**kwargs):
        recorder = Recorder(**kwargs)
        recorders.append(recorder)
        return recorder

    registry.register("recorder", make_recorder)
    pipeline = build_capability_pipeline("recorder", registry=registry)
    input_ids = (101, 32001, 303)
    video = FakeMedia((8, 3, 224, 224))
    context = MediaContext.from_inputs(input_ids, {"video": [video]}, {"video": {}})

    pipeline.on_media_context(context)

    assert context.media_count("video") == 1
    assert context.media["video"][0].shape[0] == 8
    assert context.input_ids == input_ids
    assert recorders[0].contexts[0].media["video"][0] is video


def test_unknown_capability_is_not_silently_activated():
    with pytest.raises(CapabilityError, match="Unknown capability 'missing'"):
        build_capability_pipeline(CapabilityRequest(names=("missing",)))


def test_image_segmentation_is_explicit_and_returns_mask_loss():
    torch = pytest.importorskip("torch")
    pipeline = build_capability_pipeline("image_segmentation")
    features = torch.randn(2, 4, 8, requires_grad=True)
    targets = torch.zeros(2, 6, 6)
    targets[:, 1:4, 2:5] = 1
    context = MediaContext.from_inputs(
        input_ids=(101, 32000, 303),
        media={"image": [object(), object()]},
        media_config={"image": {}},
        metadata={"segmentation_masks": targets},
    )

    pipeline.on_media_context(context)
    pipeline.on_vision_features(context, features)
    outputs = pipeline.get_outputs()
    loss = pipeline.compute_loss()

    assert pipeline.enabled_names == ("image_segmentation",)
    assert outputs["mask_logits"].shape == (2, 1, 6, 6)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None


def test_image_segmentation_ignores_video_feature_notifications():
    torch = pytest.importorskip("torch")
    pipeline = build_capability_pipeline("image_segmentation")
    context = MediaContext.from_inputs(
        input_ids=None,
        media={"video": [object()]},
        media_config={"video": {}},
    )

    pipeline.on_media_context(context)
    pipeline.on_vision_features(context, torch.randn(1, 4, 8))

    assert pipeline.get_outputs() == {}


def test_capability_modules_can_be_registered_in_a_model_state_dict():
    torch = pytest.importorskip("torch")
    pipeline = build_capability_pipeline("image_segmentation")
    modules = nn.ModuleList([capability for capability in pipeline.capabilities if isinstance(capability, nn.Module)])

    assert len(modules) == 1
    assert any(key.startswith("0.mask_decoder") for key in modules.state_dict())


def test_data_collator_carries_segmentation_masks_when_dependencies_are_available():
    try:
        from llava.data.collate import DataCollator
    except ModuleNotFoundError as exc:
        pytest.skip(f"DataCollator dependencies are unavailable: {exc}")

    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        media_tokens = ("image", "video")
        media_token_ids = {"image": 32000, "video": 32001}
        pad_token_id = 0
        model_max_length = 32

    instance = {
        "input_ids": torch.tensor([101, 32000, 303]),
        "labels": torch.tensor([-100, -100, 303]),
        "image": [torch.zeros(3, 8, 8)],
        "segmentation_masks": torch.ones(6, 6),
    }
    batch = DataCollator(FakeTokenizer())([instance])

    assert batch["segmentation_masks"].shape == (1, 6, 6)


def test_model_mixin_segment_images_uses_the_capability_public_entry_point():
    torch = pytest.importorskip("torch")
    try:
        from llava.model.llava_arch import LlavaMetaModel
    except ModuleNotFoundError as exc:
        pytest.skip(f"full VILA model dependencies are unavailable: {exc}")

    class Harness(LlavaMetaModel, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.capabilities = build_capability_pipeline(
                {"names": ["image_segmentation"], "options": {"image_segmentation": {"output_size": (5, 7)}}}
            )
            self.capability_modules = nn.ModuleList(
                [capability for capability in self.capabilities.capabilities if isinstance(capability, nn.Module)]
            )
            self._active_capability_context = None

        def encode_images(self, images, block_sizes=None):
            features = images.mean(dim=(1, 2, 3))[:, None, None].expand(images.shape[0], 4, 8)
            self._notify_capability_vision_features(features)
            return features

    output = Harness().segment_images(torch.zeros(2, 3, 8, 8))

    assert output.shape == (2, 1, 5, 7)
