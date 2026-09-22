# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local/QA tests for generation-scoped H3 attention metadata."""

from types import SimpleNamespace

import pytest
import torch

from h3_prepared_attention import (
    PreparedMetadata,
    compiled_attention,
    eager_attention,
    prepare_offsets,
)


@pytest.mark.parametrize("batch,sequence", [(1, 43), (2, 257), (4, 513)])
@torch.inference_mode()
def test_compiled_attention(batch: int, sequence: int) -> None:
    packed = torch.randn(
        batch, sequence, 3, 4, 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v = packed.unbind(2)
    padding = torch.arange(batch, device="cuda") * 7
    mask = torch.arange(sequence, device="cuda")[None] >= padding[:, None]
    offsets, lengths = prepare_offsets(mask)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask[:, None, None, :],
    ).transpose(1, 2)
    fn = torch.compile(compiled_attention, fullgraph=True, dynamic=False)
    output = fn(q, k, v, offsets, lengths, 128**-0.5)
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
    eager = eager_attention(q, k, v, offsets, lengths, 128**-0.5)
    torch.testing.assert_close(output, eager, rtol=0, atol=0)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[~mask], changed_v[~mask] = 100, -100
    padded = fn(q, changed_k, changed_v, offsets, lengths, 128**-0.5)
    torch.testing.assert_close(padded, output, rtol=0, atol=0)
    if batch > 1:
        changed_v[-1] += 1
        isolated = fn(q, changed_k, changed_v, offsets, lengths, 128**-0.5)
        torch.testing.assert_close(isolated[:-1], output[:-1], rtol=0, atol=0)
        assert not torch.equal(isolated[-1], output[-1])
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled_attention(q, k, v, offsets, lengths, 128**-0.5)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        replayed = compiled_attention(q, k, v, offsets, lengths, 128**-0.5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, output, rtol=0, atol=0)


@pytest.mark.parametrize(
    "values", [[[True, False, True]], [[False, False]], [[True, False]]]
)
def test_invalid_masks(values: list[list[bool]]) -> None:
    with pytest.raises(ValueError, match="suffixes"):
        prepare_offsets(torch.tensor(values, device="cuda"))


@torch.inference_mode()
def test_metadata_lifecycle() -> None:
    bank = PreparedMetadata()
    tags = torch.tensor([0, 0, 1, 1], device="cuda")
    indices = torch.tensor([0, 1], device="cuda")
    context = SimpleNamespace(
        text_embeds=torch.empty(2, 2, 4, device="cuda"),
        text_attention_mask=torch.tensor([[True, True], [False, True]], device="cuda"),
    )
    kwargs = {"static_context": context, "token_tags": tags, "text_indices": indices}
    bank.before_transformer(None, (), kwargs)
    offsets = bank.offsets
    bank.before_transformer(None, (), kwargs)
    assert bank.builds == 1 and bank.offsets is offsets
    assert bank.lengths.tolist() == [4, 3]
    bank.clear()
    context.text_attention_mask = torch.ones(2, 2, device="cuda", dtype=torch.bool)
    bank.before_transformer(None, (), kwargs)
    assert bank.builds == 1 and bank.lengths.tolist() == [4, 4]
    replacement = SimpleNamespace(
        text_embeds=context.text_embeds, text_attention_mask=None
    )
    bank.before_transformer(None, (), dict(kwargs, static_context=replacement))
    assert bank.offsets is None and bank.lengths is None
