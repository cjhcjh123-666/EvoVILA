import torch

from llava.evo_seg.contracts import GroundingBatch
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder


def make_batch(frames=1, frame_mask=None):
    if frame_mask is None:
        frame_mask = torch.ones(2, frames, dtype=torch.bool)
    return GroundingBatch(
        query_states=torch.randn(2, 5, 8),
        query_mask=torch.tensor([[True, True, True, False, False], [True, True, True, True, False]]),
        dense_features=torch.randn(2, frames, 4, 6, 5),
        frame_mask=frame_mask,
    )


def make_decoder():
    torch.manual_seed(7)
    return QueryConditionedSpatialDecoder(
        query_dim=8,
        feature_dim=4,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        num_object_queries=2,
    )


def test_decoder_preserves_spatiotemporal_shape():
    result = make_decoder()(make_batch(frames=3))
    assert result.mask_logits.shape == (2, 2, 3, 6, 5)
    assert result.object_logits.shape == (2, 2)
    assert result.object_embeddings.shape == (2, 2, 16)
    assert result.frame_embeddings.shape == (2, 2, 3, 16)
    assert result.diagnostics["spatial_source_length"] == 3 * 6 * 5


def test_invalid_frames_are_zeroed_and_ignored_by_decoder():
    frame_mask = torch.tensor([[True, False, False], [True, True, False]])
    result = make_decoder()(make_batch(frames=3, frame_mask=frame_mask))
    invalid = ~frame_mask[:, None, :, None, None]
    invalid_logits = result.mask_logits.masked_select(invalid)
    assert torch.allclose(invalid_logits, torch.zeros_like(invalid_logits))
    invalid_embeddings = ~frame_mask[:, None, :, None]
    assert torch.allclose(
        result.frame_embeddings.masked_select(invalid_embeddings),
        torch.zeros_like(result.frame_embeddings.masked_select(invalid_embeddings)),
    )


def test_decoder_is_query_conditioned_for_different_token_sequences():
    decoder = make_decoder().eval()
    batch = make_batch(frames=1)
    changed_query = batch.query_states.clone()
    changed_query[:, 0, 0] += 4.0
    changed_batch = GroundingBatch(
        query_states=changed_query,
        query_mask=batch.query_mask,
        dense_features=batch.dense_features,
        frame_mask=batch.frame_mask,
    )
    first = decoder(batch)
    second = decoder(changed_batch)
    assert not torch.allclose(first.mask_logits, second.mask_logits)


def test_all_decoder_parameters_receive_finite_gradients():
    decoder = make_decoder()
    result = decoder(make_batch(frames=2))
    loss = result.mask_logits.square().mean() + result.object_logits.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in decoder.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_decoder_rejects_incompatible_dimensions():
    try:
        QueryConditionedSpatialDecoder(8, 4, 15, 4, 1)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected model/head dimension validation")
