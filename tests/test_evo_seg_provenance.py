import pytest
import torch

from llava.evo_seg.provenance import FusionCapture, FusedSequenceProvenance
from llava.model.fusion_observer import current_fusion_observer, fusion_observation


def rows():
    return [
        {"source_type": ["text", "image", "image", "text"], "source_position": [0, -1, -1, 3]},
        {"source_type": ["video", "video", "text"], "source_position": [-1, -1, 2]},
    ]


def test_right_and_left_padding_preserve_source_positions():
    attention = torch.tensor([[True, True, True, True], [True, True, True, False]])
    right = FusedSequenceProvenance.from_rows(rows(), padding_side="right", fused_attention_mask=attention)
    assert right.source_type[1] == ("video", "video", "text", "padding")
    assert right.source_position.tolist()[1] == [-1, -1, 2, -1]
    assert right.valid_length.tolist() == [4, 3]

    left_attention = torch.tensor([[True, True, True, True], [False, True, True, True]])
    left = FusedSequenceProvenance.from_rows(rows(), padding_side="left", fused_attention_mask=left_attention)
    assert left.source_type[1] == ("padding", "video", "video", "text")
    assert left.source_position.tolist()[1] == [-1, -1, -1, 2]


def test_gather_query_states_returns_all_selected_text_tokens():
    attention = torch.tensor([[True, True, True, True], [True, True, True, False]])
    provenance = FusedSequenceProvenance.from_rows(rows(), padding_side="right", fused_attention_mask=attention)
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    query_mask = torch.tensor([[True, False, False, True], [False, False, True, False]])
    result = provenance.gather_query_states(hidden, query_mask)
    assert result.states.shape == (2, 2, 3)
    assert result.mask.tolist() == [[True, True], [True, False]]
    assert result.source_positions.tolist() == [[0, 3], [2, -1]]
    assert result.fused_positions.tolist() == [[0, 3], [2, -1]]
    assert torch.equal(result.states[0, 0], hidden[0, 0])
    assert torch.equal(result.states[0, 1], hidden[0, 3])
    assert torch.equal(result.states[1, 1], torch.zeros(3))


@pytest.mark.parametrize("bad_mask", [
    torch.tensor([[False, True, False, False], [False, False, True, False]]),
    torch.tensor([[True, False, False, False], [False, False, False, True]]),
])
def test_query_selection_rejects_media_or_padding_positions(bad_mask):
    attention = torch.tensor([[True, True, True, True], [True, True, True, False]])
    provenance = FusedSequenceProvenance.from_rows(rows(), padding_side="right", fused_attention_mask=attention)
    hidden = torch.zeros(2, 4, 3)
    with pytest.raises(ValueError, match="did not map"):
        provenance.gather_query_states(hidden, bad_mask)


def test_capture_is_one_shot_and_scope_restores_previous_observer():
    capture = FusionCapture()
    assert current_fusion_observer() is None
    attention = torch.tensor([[True, True, True, True], [True, True, True, False]])
    with fusion_observation(capture):
        assert current_fusion_observer() is capture
        capture.observe_fusion(rows(), padding_side="right", fused_attention_mask=attention)
        with pytest.raises(RuntimeError, match="more than one"):
            capture.observe_fusion(rows(), padding_side="right", fused_attention_mask=attention)
    assert current_fusion_observer() is None


def test_provenance_rejects_unknown_source_and_mask_mismatch():
    bad_rows = [{"source_type": ["dense"], "source_position": [-1]}]
    with pytest.raises(ValueError, match="unknown"):
        FusedSequenceProvenance.from_rows(
            bad_rows,
            padding_side="right",
            fused_attention_mask=torch.ones(1, 1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="does not match"):
        FusedSequenceProvenance.from_rows(
            rows(),
            padding_side="right",
            fused_attention_mask=torch.ones(2, 4, dtype=torch.bool),
        )
