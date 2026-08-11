"""No-weight tests for the S4b training policy components."""

from __future__ import annotations

import torch
from torch import nn

from llava.evo_seg.training import (
    GroundingProjector,
    LoRAAdapter,
    SegEmbeddingInjector,
    SegTrainingPolicy,
    apply_lora,
    is_seg_training_active,
    remove_hooks,
    seg_training_active,
)


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(16, 16)
        self.self_attn.v_proj = nn.Linear(16, 16)


def test_lora_adapter_output_and_scale():
    adapter = LoRAAdapter(16, 16, rank=4, alpha=8)
    hidden = torch.randn(2, 16)
    output = adapter(hidden)
    assert output.shape == (2, 16)
    assert output.requires_grad
    assert adapter.lora_B.abs().max().item() == 0.0


def test_lora_hook_active_gate_and_frozen_base():
    layer = _FakeLayer()
    layer.requires_grad_(False)
    linear = layer.self_attn.q_proj
    original_weight = linear.weight.detach().clone()
    adapters, handles = apply_lora([layer], ["self_attn.q_proj", "self_attn.v_proj"], rank=4, alpha=8)
    for adapter in adapters["self_attn.q_proj"] + adapters["self_attn.v_proj"]:
        adapter.lora_B.data.normal_(0.0, 0.1)
    hidden = torch.randn(3, 16)
    with torch.no_grad():
        before = linear(hidden)
    assert torch.equal(before, linear(hidden))
    with seg_training_active(True):
        active = linear(hidden)
    assert not torch.equal(active, before)
    assert not is_seg_training_active()
    assert torch.equal(linear(hidden), before)
    assert torch.equal(linear.weight, original_weight)
    assert all(not parameter.requires_grad for parameter in linear.parameters())
    assert all(parameter.requires_grad for adapter in adapters["self_attn.q_proj"] for parameter in adapter.parameters())
    remove_hooks(handles)
    assert torch.equal(linear(hidden), before)


def test_seg_embedding_injector_active_gate():
    embed = nn.Embedding(10, 8)
    embed.requires_grad_(False)
    injector = SegEmbeddingInjector(embed, seg_id=9, hidden_size=8)
    input_ids = torch.tensor([[1, 9, 2]])
    base = embed(input_ids).detach().clone()
    with torch.no_grad():
        assert torch.equal(embed(input_ids), base)
    with seg_training_active(True):
        injected = embed(input_ids)
    assert not torch.equal(injected, base)
    assert not torch.equal(injected[0, 1], base[0, 1])
    assert torch.equal(injected[0, 0], base[0, 0])
    with torch.no_grad():
        assert torch.equal(embed(input_ids), base)
    injector.detach()


def test_grounding_projector_shape_and_grad():
    projector = GroundingProjector(hidden_size=8)
    states = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    seg_positions = torch.tensor([4, 3])
    output = projector(states, mask, seg_positions)
    assert tuple(output.shape) == (2, 6, 8)
    assert output.requires_grad
    output.sum().backward()
    assert projector.seg_projection.weight.grad is not None


def test_policy_optimizer_groups():
    layer = _FakeLayer()
    adapters, handles = apply_lora([layer], ["self_attn.q_proj"], rank=4, alpha=8)
    injector = SegEmbeddingInjector(nn.Embedding(10, 8), 9, 8)
    decoder = nn.Linear(8, 8)
    projector = GroundingProjector(8)
    policy = SegTrainingPolicy(
        lora_adapters=adapters,
        lora_handles=handles,
        seg_injector=injector,
        projector=projector,
        decoder=decoder,
        sam2_mask_decoder=[],
        sam2_mask_decoder_trainable=False,
    )
    groups = policy.optimizer_param_groups(lora_lr=2e-4, base_lr=1e-4)
    assert groups[0]["name"] == "decoder_projector_seg"
    assert groups[1]["name"] == "lora"
    assert groups[1]["lr"] == 2e-4
    assert all(parameter.requires_grad for parameter in groups[1]["params"])
    remove_hooks(handles)
    injector.detach()
