import pytest
import torch

from llava.evo_seg.contracts import GroundingBatch, SegmentationRequest, SegmentationResult


def make_batch(batch_size=2, frames=1, objects=2):
    query_states = torch.randn(batch_size, 4, 6)
    query_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    dense_features = torch.randn(batch_size, frames, 3, 5, 7)
    frame_mask = torch.ones(batch_size, frames, dtype=torch.bool)
    target_masks = torch.rand(batch_size, objects, frames, 5, 7)
    target_presence = torch.ones(batch_size, objects, frames, dtype=torch.bool)
    return GroundingBatch(
        query_states=query_states,
        query_mask=query_mask,
        dense_features=dense_features,
        frame_mask=frame_mask,
        target_masks=target_masks,
        target_presence=target_presence,
        sample_ids=tuple(f"sample-{idx}" for idx in range(batch_size)),
    )


def test_image_and_video_share_the_same_contract():
    image_batch = make_batch(frames=1)
    video_batch = make_batch(frames=3)
    assert image_batch.dense_features.shape == (2, 1, 3, 5, 7)
    assert video_batch.target_masks.shape == (2, 2, 3, 5, 7)


def test_padded_video_requires_one_valid_frame_per_sample():
    frame_mask = torch.tensor([[True, True, False], [True, False, False]])
    batch = make_batch(frames=3)
    batch = GroundingBatch(
        query_states=batch.query_states,
        query_mask=batch.query_mask,
        dense_features=batch.dense_features,
        frame_mask=frame_mask,
        target_masks=batch.target_masks,
        target_presence=batch.target_presence,
    )
    assert batch.frame_mask.tolist() == frame_mask.tolist()


def test_request_options_are_recursively_immutable():
    request = SegmentationRequest.from_value(
        {
            "enabled": True,
            "task": "video",
            "options": {"nested": {"levels": [1, 2]}, "flags": {"fast"}},
        }
    )
    assert request.enabled and request.task == "video"
    assert request.options["nested"]["levels"] == (1, 2)
    with pytest.raises(TypeError):
        request.options["nested"] = {}
    with pytest.raises(TypeError):
        request.options["nested"]["levels"] = ()


def test_disabled_request_is_the_default_and_unknown_fields_fail():
    assert SegmentationRequest.from_value(None) == SegmentationRequest.disabled()
    assert not SegmentationRequest.from_value({}).enabled
    with pytest.raises(ValueError, match="requires task"):
        SegmentationRequest.from_value({"enabled": True})
    with pytest.raises(ValueError, match="unknown"):
        SegmentationRequest.from_value({"enabled": False, "route": "dense"})


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("query_states", torch.randn(2, 6), "rank 3"),
        ("query_mask", torch.ones(2, 4), "dtype"),
        ("dense_features", torch.randn(2, 3, 5, 7), "rank 5"),
        ("frame_mask", torch.zeros(2, 1, dtype=torch.bool), "valid frame"),
        ("query_states", torch.full((2, 4, 6), float("nan")), "finite"),
    ],
)
def test_batch_validation_rejects_malformed_inputs(field, value, match):
    batch = make_batch()
    fields = {
        "query_states": batch.query_states,
        "query_mask": batch.query_mask,
        "dense_features": batch.dense_features,
        "frame_mask": batch.frame_mask,
        "target_masks": batch.target_masks,
        "target_presence": batch.target_presence,
    }
    fields[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        GroundingBatch(**fields)


def test_result_requires_zeroed_invalid_frames():
    frame_mask = torch.tensor([[True, False]])
    kwargs = dict(
        mask_logits=torch.zeros(1, 1, 2, 2, 2),
        object_logits=torch.zeros(1, 1),
        object_embeddings=torch.zeros(1, 1, 4),
        frame_embeddings=torch.zeros(1, 1, 2, 4),
        frame_mask=frame_mask,
    )
    SegmentationResult(**kwargs)
    invalid_logits = dict(kwargs, mask_logits=kwargs["mask_logits"].clone())
    invalid_logits["mask_logits"][0, 0, 1, 0, 0] = 1
    with pytest.raises(ValueError, match="mask_logits"):
        SegmentationResult(**invalid_logits)
