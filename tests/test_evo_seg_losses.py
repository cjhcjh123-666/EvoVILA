import torch

from llava.evo_seg.contracts import GroundingBatch, SegmentationResult
from llava.evo_seg.losses import (
    binary_mask_loss,
    compute_segmentation_loss,
    dice_loss,
    objectness_loss,
    query_swap_margin_loss,
    temporal_consistency_loss,
)


def test_perfect_masks_have_lower_bce_and_dice_than_inverted_masks():
    targets = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    good = torch.tensor([[[[[8.0, -8.0], [8.0, -8.0]]]]])
    bad = -good
    assert binary_mask_loss(good, targets, valid) < binary_mask_loss(bad, targets, valid)
    assert dice_loss(good, targets, valid) < dice_loss(bad, targets, valid)


def test_invalid_frames_do_not_contribute_to_mask_loss():
    targets = torch.zeros(1, 1, 2, 2, 2)
    targets[:, :, 0, 0, 0] = 1
    valid = torch.tensor([[[True, False]]])
    logits = torch.zeros_like(targets)
    changed = logits.clone()
    changed[:, :, 1] = 100
    assert torch.allclose(binary_mask_loss(logits, targets, valid), binary_mask_loss(changed, targets, valid))
    assert torch.allclose(dice_loss(logits, targets, valid), dice_loss(changed, targets, valid))


def test_objectness_reduces_presence_over_valid_frames():
    logits = torch.tensor([[8.0, -8.0]])
    presence = torch.tensor([[[False, True], [False, False]]])
    frame_mask = torch.tensor([[True, False]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[False, False]]).float())
    assert torch.allclose(objectness_loss(logits, presence, frame_mask), expected)


def test_query_swap_margin_detects_wrong_query():
    targets = torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    correct = torch.tensor([[[[[8.0, -8.0], [-8.0, -8.0]]]]])
    swapped = -correct
    assert query_swap_margin_loss(correct, swapped, targets, valid, margin=0.1) < 1e-5
    assert query_swap_margin_loss(swapped, correct, targets, valid, margin=0.1) > 0


def test_temporal_consistency_is_zero_for_image_and_identical_video_features():
    image_embeddings = torch.randn(2, 1, 1, 4)
    image_presence = torch.ones(2, 1, 1, dtype=torch.bool)
    image_frames = torch.ones(2, 1, dtype=torch.bool)
    assert temporal_consistency_loss(image_embeddings, image_presence, image_frames).item() == 0

    video_embeddings = image_embeddings.expand(2, 1, 3, 4).clone()
    video_presence = torch.ones(2, 1, 3, dtype=torch.bool)
    video_frames = torch.ones(2, 3, dtype=torch.bool)
    assert temporal_consistency_loss(video_embeddings, video_presence, video_frames).item() < 1e-6


def test_compute_segmentation_loss_requires_swap_control_when_enabled():
    batch = GroundingBatch(
        query_states=torch.randn(1, 2, 4),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
        dense_features=torch.randn(1, 1, 3, 2, 2),
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        target_masks=torch.zeros(1, 1, 1, 2, 2),
        target_presence=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    result = SegmentationResult(
        mask_logits=torch.zeros(1, 1, 1, 2, 2),
        object_logits=torch.zeros(1, 1),
        object_embeddings=torch.zeros(1, 1, 4),
        frame_embeddings=torch.zeros(1, 1, 1, 4),
        frame_mask=batch.frame_mask,
    )
    try:
        compute_segmentation_loss(result, batch, {"query_swap": 1.0})
    except ValueError as error:
        assert "swapped_logits" in str(error)
    else:
        raise AssertionError("missing query-swap control should fail closed")
